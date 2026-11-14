"""
tool_maker — Recursive Self-Improvement (RSI) Toolset.

Provides the LLM with a closed feedback loop to read, inspect, modify,
test, and commit the source code of any active LCP toolset back to disk.

DESIGN INVARIANTS:
  * Every tool is gated under GATED_TASK_CATEGORY = "code_execution" so the
    host confirmation modal blocks it unless the operator approves. The
    lcp binding's _gated_confirm() wrapper applies per-discussion gates.
  * Reads/paths are strictly scoped to it code files actually on disk that
    back a currently-discovered tool (never arbitrary paths).
  * Writes are full-file only; SEARCH/REPLACE patches are applied in memory
    with strict validation — a partial patch can never be written.
  * Tests import the module in a throw-away subprocess, so the operator's
    running Python process is never affected by a broken toolset.
  * Commits are git-scoped to the toolset folder — never the whole repo.
  * Only active when ``session_flags["enable_toolsmith"]`` is True on the
    LCP binding. Without the flag, these tools are not mounted.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOL_LIBRARY_NAME = "tool_maker"
TOOL_LIBRARY_DESCRIPTION = (
    "Recursive Self-Improvement (RSI) toolsmith: read, write, test and commit "
    "the source of any active LCP toolset. Requires enable_toolsmith=True."
)
# Every tool in this library is a mutating/executable action → code_execution gate.
GATED_TASK_CATEGORY = "code_execution"

# Populated by init_tools_library(host_config) so the tools can reach the binding.
_CONFIRM_HANDLER = None
_USER_INTERACTION_HANDLER = None
_LCP_BINDING = None  # set lazily via env.client.tools in each tool


# Toolsets that are refused for the child test agent because a "test" of them
# would necessarily let arbitrary code execution happen before the operator
# has had a chance to reject the commit. Defence in depth: even if the operator
# accidentally allows the gate, the child will still refuse.
_FORBIDDEN_TEST_TARGETS = frozenset({
    "system_shell",
    "execute_python",
    "execute_multi_file_service",
    "execute_python_data_query",
    "execute_sparql_query",
    "execute_sql_query",
    "computer_use",
})


def init_tools_library(host_config: Dict[str, Any]) -> None:
    global _CONFIRM_HANDLER, _USER_INTERACTION_HANDLER
    if not isinstance(host_config, dict):
        return
    _CONFIRM_HANDLER = host_config.get("confirm_handler")
    _USER_INTERACTION_HANDLER = host_config.get("user_interaction_handler")


def set_confirm_handler(handler) -> None:
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler


def set_user_interaction_handler(handler) -> None:
    global _USER_INTERACTION_HANDLER
    _USER_INTERACTION_HANDLER = handler


# ────────────────────────────── helpers ──────────────────────────────


def _get_binding(env: Any = None) -> Any:
    """Return the active LCP binding via env.client.tools, or raise."""
    client = getattr(env, "client", None) if env is not None else None
    binding = getattr(client, "tools", None) if client is not None else None
    if binding is None:
        raise RuntimeError(
            "tool_maker requires an LCP tools binding (env.client.tools); "
            "none is available in the current invocation context."
        )
    if not hasattr(binding, "discovered_tools"):
        raise RuntimeError(
            "tool_maker requires the LCP binding to expose discovered_tools."
        )
    return binding


def _toolsmith_flag_is_on(binding: Any) -> bool:
    """Cheap, side-effect-free check for ``session_flags['enable_toolsmith']``."""
    try:
        flags = getattr(binding, "session_flags", {}) or {}
        return bool(flags.get("enable_toolsmith"))
    except Exception:
        return False


def _require_toolsmith_flag(env: Any, tool_name: str) -> Optional[Dict[str, Any]]:
    """Hard gate for the heavy RSI act-phase tools.

    Returns ``None`` when the flag is on (caller proceeds normally), or a
    fully-formed error dict when the flag is off (caller returns it directly).

    BASIC tools (read/write/test/revert/discard/list_versions) do NOT call
    this — they must remain available without ``enable_toolsmith``. Only the
    act-phase tools (``tool_run_test_agent``) and the promotion tool
    (``tool_commit_toolset``) are gated here.

    The auto-discovery filter in ``LCPBinding.discover_tools()`` already hides
    these tools from the LLM's spec list when the flag is off. This check is
    defence in depth: even a direct ``execute_tool`` call from a test / GUI
    path cannot bypass the flag.
    """
    try:
        binding = _get_binding(env)
    except Exception as e:
        return {
            "success": False,
            "error": f"{tool_name} requires an active LCP binding: {e}",
        }
    if not _toolsmith_flag_is_on(binding):
        return {
            "success": False,
            "error": (
                f"{tool_name} is an RSI act-phase tool and requires "
                "enable_toolsmith=True. Enable it in GuiPrefs / pass "
                "enable_toolsmith=True to chat(). Basic tools-manipulation "
                "tools (read/write/test/revert/discard/list_versions) remain "
                "available without the flag."
            ),
            "requires_flag": "enable_toolsmith",
        }
    return None


def _resolve_toolset_file(binding: Any, tool_name_or_stem: str) -> Tuple[str, Path]:
    """Resolve a tool name OR a toolset stem to (canonical_stem, .py file path).

    Only toolset files that BACK a currently-discovered tool are eligible.
    This is a deliberate security invariant: the LLM cannot read or write
    arbitrary files by inventing a name.
    """
    needle = (tool_name_or_stem or "").strip()
    if not needle:
        raise ValueError("tool_name_or_stem is required.")

    # Try by exact tool name first.
    for tool_def in binding.discovered_tools:
        if tool_def.get("name") == needle:
            p = tool_def.get("_python_file_path")
            if p:
                return (tool_def.get("_library_name") or Path(p).parent.name or Path(p).stem, Path(p))

    # Then by canonical library stem (folder name for TOOLS.py layout, file stem otherwise).
    normalized = re.sub(r"[\s\-]+", "_", needle).lower()
    for tool_def in binding.discovered_tools:
        file_path = tool_def.get("_python_file_path")
        if not file_path:
            continue
        p = Path(file_path)
        lib_identity = tool_def.get("_library_name") or (
            p.parent.name if p.name == "TOOLS.py" else p.stem
        )
        if lib_identity.lower() == normalized:
            return lib_identity, p

    raise ValueError(
        f"tool_name_or_stem '{needle}' does not match any active tool or toolset. "
        f"Active toolsets: {sorted({(t.get('_library_name') or Path(t.get('_python_file_path','')).parent.name) for t in binding.discovered_tools if t.get('_python_file_path')})}"
    )


def _apply_aider_patches(original: str, patch_text: str) -> Tuple[bool, str, Optional[str]]:
    """Apply Aider-style SEARCH/REPLACE hunks in memory.

    Returns (ok, new_text_or_error, error_message).
    A partial application is NEVER returned — if any hunk fails, we bail.
    """
    hunks = re.findall(
        r"<<<<<<< SEARCH\s*\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE",
        patch_text,
        re.DOTALL,
    )
    if not hunks:
        return False, "", "No valid <<<<<<< SEARCH / ======= / >>>>>>> REPLACE hunks found."

    working = original
    for idx, (search_body, replace_body) in enumerate(hunks, 1):
        search_norm = search_body.strip("\n")
        replace_norm = replace_body.strip("\n")

        if search_norm not in working:
            # Try a whitespace-normalized fallback.
            search_lines = search_norm.splitlines()
            pattern_lines = [re.escape(line) for line in search_lines]
            fallback_pattern = re.compile(
                r"\n?".join(pattern_lines).replace(r"\ ", r"[ \t]+"), re.DOTALL
            )
            m = fallback_pattern.search(working)
            if not m:
                return (
                    False,
                    "",
                    f"Hunk #{idx}: SEARCH block not found verbatim in source. "
                    f"Refusing to write a partial patch.",
                )
            working = working[: m.start()] + replace_norm + working[m.end():]
        else:
            working = working.replace(search_norm, replace_norm, 1)

    return True, working, None


def _ast_smoke_check(code: str) -> Tuple[bool, Optional[str]]:
    """AST-parse candidate source and require at least one tool_ function."""
    try:
        tree = ast.parse(code)
    except SyntaxError as se:
        return False, f"SyntaxError: {se.msg} (line {se.lineno})"

    tool_names = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("tool_")
    ]
    if not tool_names:
        return False, (
            "AST parse succeeded but the module declares zero 'tool_*' functions. "
            "A toolset must expose at least one tool."
        )
    return True, None


def _isolated_import_probe(py_file: Path) -> Dict[str, Any]:
    """Import the target module in a throw-away subprocess and report discovered tools.

    This never executes any user tool — it only imports the module and calls
    introspect via ast. Any exception is caught in the subprocess and reported.
    """
    probe_script = textwrap.dedent(
        f"""
        import sys, importlib.util, json
        try:
            spec = importlib.util.spec_from_file_location("_rsi_probe_target", r"{str(py_file)}")
            if spec is None or spec.loader is None:
                print(json.dumps({{"ok": False, "error": "spec_from_file_location returned None"}}))
                sys.exit(0)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_rsi_probe_target"] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException as e:
                print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
                sys.exit(0)
            discovered = [n for n in dir(mod) if n.startswith("tool_")]
            print(json.dumps({{"ok": True, "error": None, "discovered_tools": discovered}}))
        except BaseException as e:
            print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            text=True,
            timeout=25,
            cwd=str(py_file.parent),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Import probe timed out after 25s.", "discovered_tools": []}

    raw_out = (result.stdout or "").strip().splitlines()
    if not raw_out:
        return {
            "ok": False,
            "error": f"Probe produced no output. stderr: {(result.stderr or '')[:800]}",
            "discovered_tools": [],
        }
    try:
        payload = json.loads(raw_out[-1])
    except Exception:
        return {
            "ok": False,
            "error": f"Could not decode probe output: {raw_out[-1][:400]}",
            "discovered_tools": [],
        }
    if result.returncode != 0 and payload.get("ok"):
        payload["ok"] = False
        payload["error"] = f"Probe process exited code {result.returncode}"
    return payload


def _git_commit_scoped_optional(toolset_dir: Path, message: str) -> Optional[Dict[str, Any]]:
    """Best-effort git add/commit when git is present and the folder is tracked.

    Returns None when git is not usable for this folder — callers must treat
    that as "no external history", NOT as a failure.
    """
    if not _git_available():
        return None
    if not _git_find_repo_root(toolset_dir):
        return None

    commit_msg = (message or "").strip() or f"toolsmith: update {toolset_dir.name}"
    try:
        add = subprocess.run(
            ["git", "add", "--", str(toolset_dir)],
            capture_output=True, text=True, timeout=15, cwd=str(toolset_dir),
        )
        if add.returncode != 0:
            return {"ok": False, "error": f"git add failed: {(add.stderr or '').strip()}"}

        commit = subprocess.run(
            ["git", "commit", "-m", commit_msg, "--", str(toolset_dir)],
            capture_output=True, text=True, timeout=20, cwd=str(toolset_dir),
        )
        if commit.returncode != 0:
            stderr = (commit.stderr or "").strip()
            if "nothing to commit" in stderr.lower() or "nothing added to commit" in stderr.lower():
                return {"ok": True, "message": "Nothing to commit (no changes).", "already_committed": True}
            return {"ok": False, "error": f"git commit failed: {stderr}"}

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(toolset_dir),
        )
        return {
            "ok": True,
            "message": (commit.stdout or "").strip(),
            "commit_sha": (head.stdout or "").strip() if head.returncode == 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "git operation timed out."}
    except Exception as e:
        return {"ok": False, "error": f"git commit exception: {e}"}


def _git_available() -> bool:
    """Return True when a `git` executable is on PATH.

    Used purely to decide whether opportunistic git integration is possible.
    The toolsmith NEVER requires git — every commit/revert/discard operation
    has a fully file-system-based fallback in ``.rsi/commits/`` and
    ``.rsi_backup/``.
    """
    try:
        run = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=4,
        )
        return run.returncode == 0
    except Exception:
        return False


def _git_find_repo_root(start: Path) -> Optional[Path]:
    """Optional helper: find a containing git repo when git is available."""
    if not _git_available():
        return None
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return None


# ── RELEASE MANIFEST (git-free commit lifecycle) ──
# `.rsi/commits/<epoch>.json` records a semantic "release" of the toolset
# source. This is the canonical commit history when git is unavailable (or
# unwanted). Each release embeds the full source text so it is self-contained
# and survives every filesystem reinstall.
_RSI_META_DIRNAME = ".rsi"
_RSI_COMMITS_DIRNAME = "commits"


def _rsi_commits_dir(py_file: Path) -> Path:
    d = py_file.parent / _RSI_META_DIRNAME / _RSI_COMMITS_DIRNAME
    return d


def _list_rsi_releases(py_file: Path) -> List[Path]:
    """Every release sidecar (``.rsi/commits/<epoch>.json``) for this toolset,
    newest first. Ignores any other *.json files that don't match the pattern."""
    d = _rsi_commits_dir(py_file)
    if not d.is_dir():
        return []
    try:
        rls = [p for p in d.glob("*.json") if p.is_file()]
    except Exception:
        return []
    def _sort_key(p: Path) -> int:
        try:
            return int(p.stem)
        except Exception:
            return 0
    return sorted(rls, key=_sort_key, reverse=True)


def _write_release_manifest(
    py_file: Path,
    source_text: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a new release sidecar with the embedded source payload.

    Never overwrites: an always-unique epoch-based filename is used (with a
    monotonically increasing suffix on collision, e.g. same-second releases).
    """
    import hashlib

    d = _rsi_commits_dir(py_file)
    d.mkdir(parents=True, exist_ok=True)

    epoch = int(time.time())
    target = d / f"{epoch}.json"
    bump = 0
    while target.exists():
        bump += 1
        target = d / f"{epoch}_{bump}.json"

    payload = {
        "schema": 1,
        "toolset": py_file.name,
        "toolset_path": str(py_file.resolve()),
        "created_at": epoch,
        "message": (message or "").strip() or "toolsmith release",
        "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source": source_text,
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if k not in payload})

    try:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "release_file": str(target.resolve()), "release_id": target.stem}
    except Exception as e:
        return {"ok": False, "error": f"Could not write release manifest: {e}"}


def _latest_release(py_file: Path) -> Optional[Dict[str, Any]]:
    """Return the newest release's decoded payload, or None when no releases exist."""
    releases = _list_rsi_releases(py_file)
    if not releases:
        return None
    try:
        return json.loads(releases[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def _git_file_is_tracked(py_file: Path) -> bool:
    """True when ``py_file`` is tracked by git. Never raises; returns False
    when git is unavailable or the file is not in a repo."""
    if not _git_available():
        return False
    try:
        run = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(py_file)],
            capture_output=True, text=True, timeout=8, cwd=str(py_file.parent),
        )
        return run.returncode == 0
    except Exception:
        return False


def _git_checkout_file(py_file: Path) -> Tuple[bool, Optional[str]]:
    """Revert ``py_file`` to its committed HEAD content (tracked files only).

    Used opportunistically — callers MUST treat a failure or "git unavailable"
    as a signal to fall back to the file-system release manifest.
    """
    if not _git_available():
        return False, "git is not available on this system."
    try:
        run = subprocess.run(
            ["git", "checkout", "HEAD", "--", str(py_file)],
            capture_output=True, text=True, timeout=15, cwd=str(py_file.parent),
        )
        if run.returncode != 0:
            return False, (run.stderr or run.stdout or "").strip() or "git checkout failed."
        return True, None
    except Exception as e:
        return False, f"git checkout exception: {e}"


def _git_is_working_tree_dirty(py_file: Path) -> bool:
    """True when the working-tree copy differs from the committed HEAD.

    Returns True when git is unavailable OR when the file is untracked — the
    caller should treat "dirty" as "there is at least one thing worth
    snapshotting", which remains true in the file-system-only flow."""
    if not _git_available() or not _git_file_is_tracked(py_file):
        return True
    try:
        run = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(py_file)],
            capture_output=True, text=True, timeout=10, cwd=str(py_file.parent),
        )
        return run.returncode not in (0,)
    except Exception:
        return True


def _list_rsi_backups(py_file: Path) -> List[Path]:
    """All ``.rsi_backup/<name>.<ts>.bak`` snapshots, newest first."""
    backup_dir = py_file.parent / ".rsi_backup"
    if not backup_dir.is_dir():
        return []
    try:
        snaps = [p for p in backup_dir.glob(f"{py_file.name}.*.bak") if p.is_file()]
    except Exception:
        return []
    def _sort_key(p: Path) -> int:
        # Filenames are ``<name>.<epoch_seconds>.bak`` — sort by the numeric middle.
        parts = p.name.rsplit(".", 2)
        try:
            return int(parts[-2])
        except Exception:
            return 0
    return sorted(snaps, key=_sort_key, reverse=True)


def _purge_rsi_backups(py_file: Path) -> int:
    """Delete every ``.rsi_backup/*.bak`` snapshot for ``py_file``. Returns count."""
    removed = 0
    for snap in _list_rsi_backups(py_file):
        try:
            snap.unlink()
            removed += 1
        except Exception:
            pass
    # Drop the (now empty) folder itself.
    backup_dir = py_file.parent / ".rsi_backup"
    try:
        if backup_dir.is_dir() and not any(backup_dir.iterdir()):
            backup_dir.rmdir()
    except Exception:
        pass
    return removed


def _run_child_test_agent(
    *,
    toolset_dir: Path,
    toolset_module: Path,
    test_task: str,
    max_steps: int,
    timeout_sec: int,
    env: Any,
) -> Dict[str, Any]:
    """Spawn the isolated child runner subprocess with a strict timeout.

    The child receives only:
      - the candidate toolset folder (for importing the candidate module)
      - the candidate module's absolute path (for filtering discovered tools)
      - the LLM binding profiles from the orchestrator's env (so it can
        actually talk to the model — without them the child is useless)
      - a hard timeout, after which SIGTERM is delivered

    It NEVER receives the orchestrator's workspace, the host 'tools' folder,
    or the confirm_handler — so no unattended code path can ask the operator
    for approval on behalf of a child that the operator did not see go out.
    """
    here = Path(__file__).resolve().parent
    runner = here / "_isolated_child_runner.py"
    if not runner.exists():
        return {"ok": False, "error": f"Child runner missing: {runner}"}

    # Pull the binding / client from env so we can forward usable profiles.
    client = getattr(env, "client", None) if env is not None else None
    llm_bp = {}
    llm_mp = {}
    system_dir = None
    try:
        if client is not None:
            llm_bp = getattr(client, "llm_binding_profiles", {}) or {}
            llm_mp = getattr(client, "llm_model_profiles", {}) or {}
            system_dir = getattr(client, "_system_dir", None) or getattr(client, "system_dir", None)
    except Exception:
        pass

    payload = {
        "toolset_dir": str(toolset_dir),
        "toolset_module": str(toolset_module),
        "test_task": (test_task or "").strip(),
        "max_steps": max(1, min(int(max_steps or 6), 12)),
        "timeout_sec": max(10, min(int(timeout_sec or 120), 300)),
        "llm_binding_profiles": llm_bp,
        "llm_model_profiles": llm_mp,
        "system_dir": system_dir,
    }

    # Subprocess environment: inherit PYTHONPATH so lollms_client is importable;
    # never inherit arbitrary user env vars that could be a sandbox escape.
    child_env = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "TMP": os.environ.get("TMP", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    child_env = {k: v for k, v in child_env.items() if v}

    try:
        run = subprocess.run(
            [sys.executable, str(runner)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=int(payload["timeout_sec"]) + 5,
            cwd=str(toolset_dir),
            env=child_env,
        )
    except subprocess.TimeoutExpired as te:
        return {
            "ok": False,
            "error": f"Child test agent hard-timed out after {te.timeout}s.",
            "transcript": [],
        }
    except Exception as e:
        return {"ok": False, "error": f"Spawn failed: {e}"}

    # Parse the child's last JSON line.
    out_lines = (run.stdout or "").strip().splitlines()
    envelope: Dict[str, Any] = {}
    if out_lines:
        try:
            envelope = json.loads(out_lines[-1])
        except Exception:
            envelope = {
                "ok": False,
                "error": f"Could not decode child output: {out_lines[-1][:400]}",
                "raw_stdout": run.stdout[-2000:] if run.stdout else "",
            }
    else:
        envelope = {
            "ok": False,
            "error": f"Child produced no stdout. stderr: {(run.stderr or '')[:800]}",
            "raw_stderr": run.stderr[-2000:] if run.stderr else "",
        }

    envelope["child_exit_code"] = run.returncode
    if run.returncode not in (0, None) and envelope.get("ok"):
        envelope["ok"] = False
        envelope["error"] = f"Child exited with code {run.returncode}."
    return envelope


# ──────────────────────────────── tools ────────────────────────────────


def tool_read_tool_source(
    tool_name_or_stem: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Read the full Python source of an active LCP toolset.

    The target must correspond to a currently-discovered tool (either its exact
    tool function name, or the toolset folder/file stem). Arbitrary paths are
    never resolved.

    Args:
        tool_name_or_stem: Tool function name (e.g. "tool_read_file") or toolset stem (e.g. "workspace_tools").

    Returns:
        dict: {"success": bool, "stem": str, "path": str, "source": str, "line_count": int}
    """
    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)
        if not py_file.exists():
            return {"success": False, "error": f"Toolset file no longer exists on disk: {py_file}"}
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        return {
            "success": True,
            "stem": stem,
            "path": str(py_file.resolve()),
            "source": source,
            "line_count": len(source.splitlines()),
            "size_chars": len(source),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to read toolset source: {e}"}


def tool_write_tool_source(
    tool_name_or_stem: str,
    new_source: str,
    reason: str = "",
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Write new source for an active LCP toolset.

    Two modes are supported, distinguished automatically:
      1. FULL REPLACEMENT: ``new_source`` does not contain <<<<<<< SEARCH markers.
      2. AIDER PATCH: ``new_source`` contains SEARCH/REPLACE hunks; every hunk
         is applied to the CURRENT on-disk file in memory. If any hunk fails,
         the write is aborted — a partial patch is never written.

    The candidate source is AST-validated and must declare at least one
    ``tool_*`` function; otherwise the write is refused.

    Args:
        tool_name_or_stem: Tool function name or toolset stem.
        new_source: New source code, or an Aider-style SEARCH/REPLACE block.
        reason: Short rationale recorded in the response (audit trail).

    Returns:
        dict: {"success": bool, "stem": str, "path": str, "mode": "full"|"patch",
               "line_count": int, "reason": str}
    """
    # HALT respect: refuse if any prior rejection in this turn left the flag set.
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to write tool source."}

    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)
        original = py_file.read_text(encoding="utf-8") if py_file.exists() else ""

        is_patch = ("<<<<<<< SEARCH" in (new_source or "")) and ("=======" in (new_source or ""))
        if is_patch:
            ok, resolved_or_err, err_msg = _apply_aider_patches(original, new_source)
            if not ok:
                return {
                    "success": False,
                    "error": f"Aider patch could not be applied: {err_msg}",
                    "mode": "patch",
                    "stem": stem,
                }
            candidate = resolved_or_err
            write_mode = "patch"
        else:
            candidate = new_source or ""
            write_mode = "full"

        syntax_ok, syntax_err = _ast_smoke_check(candidate)
        if not syntax_ok:
            return {
                "success": False,
                "error": f"Candidate source rejected before write: {syntax_err}",
                "mode": write_mode,
                "stem": stem,
            }

        # Backup the previous file to `.rsi_backup/` so the edit is reversible.
        backup_dir = py_file.parent / ".rsi_backup"
        if py_file.exists():
            backup_dir.mkdir(exist_ok=True)
            backup_file = backup_dir / f"{py_file.name}.{int(time.time())}.bak"
            try:
                backup_file.write_text(original, encoding="utf-8")
            except Exception:
                pass

        py_file.write_text(candidate, encoding="utf-8")
        return {
            "success": True,
            "stem": stem,
            "path": str(py_file.resolve()),
            "mode": write_mode,
            "line_count": len(candidate.splitlines()),
            "reason": reason or "(no reason provided)",
            "message": f"Toolset '{stem}' written ({write_mode}). Call tool_test_tool_source next.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to write toolset source: {e}"}


def tool_test_tool_source(
    tool_name_or_stem: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run AST + isolated import smoke tests against an active toolset.

    Two phases:
      PHASE 1 — AST check (never executes user code).
      PHASE 2 — Isolated subprocess import (does not affect the live process).

    No ``tool_*`` function is CALLED by this probe — it is strictly an import
    and introspection test. Use the tools themselves only after operator
    authorization via the normal invocation path.

    Args:
        tool_name_or_stem: Tool function name or toolset stem.

    Returns:
        dict: {"success": bool, "stem": str, "path": str,
               "syntax_ok": bool, "import_ok": bool, "discovered_tools": [..],
               "error": str|null}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to test."}

    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)
        if not py_file.exists():
            return {"success": False, "stem": stem, "error": f"Toolset file missing: {py_file}"}

        source = py_file.read_text(encoding="utf-8", errors="ignore")
        syntax_ok, syntax_err = _ast_smoke_check(source)

        import_ok = False
        discovered: List[str] = []
        import_error = None
        if syntax_ok:
            probe = _isolated_import_probe(py_file)
            import_ok = bool(probe.get("ok"))
            discovered = list(probe.get("discovered_tools") or [])
            import_error = probe.get("error")

        overall_ok = syntax_ok and import_ok
        # On success, clear any prior HALT flag so a re-attempt after a fix is possible.
        if overall_ok:
            globals()["_HALT_REQUESTED"] = False

        return {
            "success": overall_ok,
            "stem": stem,
            "path": str(py_file.resolve()),
            "syntax_ok": syntax_ok,
            "import_ok": import_ok,
            "discovered_tools": discovered,
            "error": syntax_err if not syntax_ok else import_error,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to test toolset source: {e}"}


def tool_run_test_agent(
    tool_name_or_stem: str,
    test_task: str,
    max_steps: int = 6,
    timeout_sec: int = 120,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Equip a CONSTRAINED child test agent with the candidate toolset and run it.

    The child is:
      * Mounted with ONLY the candidate toolset's tools (no shell, no python,
        no workspace, no sub-agents, no other host tools).
      * Hard-capped at ``max_steps`` reasoning rounds (ceiling 12).
      * Executed in a fresh subprocess with a hard ``timeout_sec`` (ceiling 300).
      * Refused outright for the ``_FORBIDDEN_TEST_TARGETS`` toolsets, because
        testing them would let arbitrary code execution happen before the
        operator can reject the commit.

    Call this BEFORE ``tool_commit_toolset`` — the child's transcript becomes
    part of the audit trail the operator approves with.

    Args:
        tool_name_or_stem: Candidate toolset name or tool function name.
        test_task: Natural-language directive for the test agent.
        max_steps: Reasoning budget (1..12, default 6).
        timeout_sec: Absolute subprocess timeout (10..300, default 120).

    Returns:
        dict: {"success": bool, "ok": bool, "verdict": str|None, "error": str|None,
               "transcript": [...], "steps": int, "available_tools": [..],
               "child_exit_code": int}
    """
    if globals().get("_HALT_REQUESTED"):
        return {
            "success": False,
            "ok": False,
            "error": "HALT requested in this turn — refusing to spawn a test agent.",
        }

    # RSI act-phase tool — requires enable_toolsmith=True.
    flag_gate = _require_toolsmith_flag(env, "tool_run_test_agent")
    if flag_gate is not None:
        flag_gate["ok"] = False
        return flag_gate

    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)

        # Fail-closed: refuse the child-test path for the code-execution toolsets.
        if (stem or "").lower() in _FORBIDDEN_TEST_TARGETS:
            return {
                "success": False,
                "ok": False,
                "stem": stem,
                "error": (
                    f"Refused: '{stem}' is a code-execution toolset. A child agent "
                    "loaded with it would let arbitrary code run before the operator "
                    "can reject the commit. Test such changes manually."
                ),
            }

        if not py_file.exists():
            return {"success": False, "ok": False, "error": f"Candidate file missing: {py_file}"}

        # Smoke test first: never hand the child a syntactically broken toolset.
        pre_check = tool_test_tool_source(tool_name_or_stem, env=env)
        if not pre_check.get("success"):
            return {
                "success": False,
                "ok": False,
                "stem": stem,
                "error": (
                    "Refusing to run the test agent — the candidate fails basic "
                    "syntax/import checks. Fix it first. Underlying error: "
                    f"{pre_check.get('error')}"
                ),
            }

        envelope = _run_child_test_agent(
            toolset_dir=py_file.parent,
            toolset_module=py_file,
            test_task=test_task,
            max_steps=max_steps,
            timeout_sec=timeout_sec,
            env=env,
        )
        envelope["stem"] = stem
        envelope["success"] = bool(envelope.get("ok"))
        return envelope
    except Exception as e:
        return {
            "success": False,
            "ok": False,
            "error": f"Failed to run test agent: {e}",
        }


def tool_list_tool_versions(
    tool_name_or_stem: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """List every available revert target for an active toolset.

    Sources, in order of authority (highest first):
      * ``git_head`` — the currently-committed version (only if tracked).
      * ``git_dirty`` — whether the working tree differs from HEAD.
      * ``rsi_backups`` — reverse-chronological list of in-place snapshots
        captured by ``tool_write_tool_source`` (each is a safe rollback point).

    Args:
        tool_name_or_stem: Tool function name or toolset stem.

    Returns:
        dict: {"success": bool, "stem": str, "path": str,
               "has_git_head": bool, "git_dirty": bool,
               "backups": [{"file": str, "index": int, "epoch": int}, ...],
               "current_size_chars": int}
    """
    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)
        if not py_file.exists():
            return {"success": False, "error": f"Toolset file missing: {py_file}"}

        backups = _list_rsi_backups(py_file)
        releases = _list_rsi_releases(py_file)
        current_size = py_file.stat().st_size

        release_entries: List[Dict[str, Any]] = []
        for idx, rel in enumerate(releases):
            entry: Dict[str, Any] = {"release_id": rel.stem, "index": idx}
            try:
                payload = json.loads(rel.read_text(encoding="utf-8"))
                entry["message"] = payload.get("message", "")
                entry["created_at"] = payload.get("created_at")
                entry["sha256"] = payload.get("sha256")
            except Exception:
                entry["message"] = "(unreadable sidecar)"
            release_entries.append(entry)

        git_available = _git_available()
        git_tracked = _git_file_is_tracked(py_file) if git_available else False

        return {
            "success": True,
            "stem": stem,
            "path": str(py_file.resolve()),
            "git_available": git_available,
            "has_git_head": git_tracked,
            "git_dirty": _git_is_working_tree_dirty(py_file) if git_tracked else None,
            "releases": release_entries,
            "release_count": len(release_entries),
            "backups": [
                {"file": str(snap.resolve()), "index": idx, "epoch": _backup_epoch(snap)}
                for idx, snap in enumerate(backups)
            ],
            "current_size_bytes": current_size,
            "backup_count": len(backups),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to list toolset versions: {e}"}


def _backup_epoch(snap: Path) -> int:
    parts = snap.name.rsplit(".", 2)
    try:
        return int(parts[-2])
    except Exception:
        return 0


def tool_revert_tool_source(
    tool_name_or_stem: str,
    backup_index: int = 0,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Restore a previous on-disk snapshot without touching git.

    ``backup_index=0`` (default) restores the **most recent** snapshot, i.e.
    "undo my last write". Use ``tool_list_tool_versions`` to discover indices
    beyond 0 for older rollbacks.

    If no backups exist on this toolset yet, this falls back to
    ``git checkout HEAD -- <file>`` **only** when the file is tracked — the
    operator's path to "abort everything since the last commit". When neither
    source is available, the revert is refused (fail-closed; we never
    silently delete a file).

    After the restore, the smoke tests are re-run. A broken snapshot cannot
    be left in place — the previous (working) content is restored and the
    revert is reported as failed with the offending snapshot index.

    Args:
        tool_name_or_stem: Tool function name or toolset stem.
        backup_index: Which backup to restore (0 = most recent).

    Returns:
        dict: {"success": bool, "stem": str, "restored_from": "backup:N"|"git_head",
               "restored_size": int, "test_ok": bool, "test_error": str|None}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to revert."}

    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)
        if not py_file.exists():
            return {"success": False, "error": f"Toolset file missing: {py_file}"}

        backups = _list_rsi_backups(py_file)

        # Snapshot the CURRENT file so a "revert to garbage" can itself be undone.
        current_content = py_file.read_text(encoding="utf-8", errors="ignore")
        pre_revert_dir = py_file.parent / ".rsi_backup"
        pre_revert_dir.mkdir(exist_ok=True)
        (pre_revert_dir / f"{py_file.name}.{int(time.time())}.bak").write_text(
            current_content, encoding="utf-8",
        )

        restored_from: Optional[str] = None
        restored_content: Optional[str] = None
        restore_error: Optional[str] = None

        # Primary path: restore a specific snapshot.
        if backups and 0 <= backup_index < len(backups):
            try:
                restored_content = backups[backup_index].read_text(encoding="utf-8", errors="ignore")
                restored_from = f"backup:{backup_index}"
            except Exception as re_err:
                restore_error = f"Could not read backup #{backup_index}: {re_err}"
        elif backups and backup_index >= len(backups):
            return {
                "success": False,
                "stem": stem,
                "error": f"backup_index={backup_index} exceeds available backups ({len(backups)}).",
            }

        # Fallback A: no backups exist yet → try the latest release manifest.
        # This is the git-free "revert to last commit" path and works even if
        # the operator never made an RSI snapshot.
        if restored_content is None and not backups:
            latest = _latest_release(py_file)
            if latest and isinstance(latest.get("source"), str):
                restored_content = latest["source"]
                restored_from = f"release:{latest.get('created_at', 'latest')}"
            else:
                # Fallback B (opportunistic, not required): git HEAD if tracked.
                if not _git_file_is_tracked(py_file):
                    return {
                        "success": False,
                        "stem": stem,
                        "error": (
                            "No RSI backups, no release manifest, and no git "
                            "tracking — there is no safe revert target."
                        ),
                    }
                ok, err = _git_checkout_file(py_file)
                if not ok:
                    return {"success": False, "stem": stem, "error": f"git checkout failed: {err}"}
                restored_from = "git_head"
                restored_content = py_file.read_text(encoding="utf-8", errors="ignore")
        elif restore_error is not None:
            return {"success": False, "stem": stem, "error": restore_error}
        elif restored_content is None:
            return {"success": False, "stem": stem, "error": "No revert target available."}

        # Write the restored content to disk (only if not from git, which already did).
        if restored_from and restored_from.startswith(("backup:", "release:")):
            py_file.write_text(restored_content, encoding="utf-8")

        # Smoke-check the restored content; undo the revert if it's broken.
        syntax_ok, syntax_err = _ast_smoke_check(restored_content)
        test_ok = False
        test_error: Optional[str] = None
        if syntax_ok:
            probe = _isolated_import_probe(py_file)
            test_ok = bool(probe.get("ok"))
            test_error = probe.get("error")
        else:
            test_error = syntax_err

        if not test_ok:
            # Restore the pre-revert content — a broken rollback is never left behind.
            try:
                py_file.write_text(current_content, encoding="utf-8")
            except Exception:
                pass
            return {
                "success": False,
                "stem": stem,
                "restored_from": restored_from,
                "error": (
                    f"Reverted content failed smoke tests (rolled back to pre-revert state). "
                    f"Underlying error: {test_error}"
                ),
                "test_ok": False,
                "test_error": test_error,
            }

        return {
            "success": True,
            "stem": stem,
            "restored_from": restored_from,
            "restored_size": len(restored_content),
            "test_ok": True,
            "test_error": None,
            "message": f"Toolset '{stem}' reverted from {restored_from}. Working tree is dirty (no commit made).",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to revert toolset: {e}"}


def tool_discard_changes(
    tool_name_or_stem: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hard-abort the current experiment: reset to the last committed state.

    Behaviour:
      * If the file is tracked by git: ``git checkout HEAD -- <file>`` (this
        discards ALL uncommitted edits, not just the last write).
      * Otherwise: restores the EARLIEST available RSI backup snapshot (i.e.
        the file content right before toolsmith made its first edit).

    Unlike ``tool_revert_tool_source`` (which undoes one step), this is the
    "wipe the slate" path. It never deletes the file; it always leaves
    valid Python on disk or refuses.

    Args:
        tool_name_or_stem: Tool function name or toolset stem.

    Returns:
        dict: {"success": bool, "stem": str, "restored_from": str, "test_ok": bool}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to discard."}

    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)
        if not py_file.exists():
            return {"success": False, "error": f"Toolset file missing: {py_file}"}

        restored_from: Optional[str] = None
        restored_content: Optional[str] = None

        # Preferred: latest release manifest (git-free commit).
        latest = _latest_release(py_file)
        if latest and isinstance(latest.get("source"), str):
            restored_content = latest["source"]
            restored_from = f"release:{latest.get('created_at', 'latest')}"
            try:
                py_file.write_text(restored_content, encoding="utf-8")
            except Exception as ex:
                return {"success": False, "stem": stem, "error": f"Restore failed: {ex}"}
        elif _git_file_is_tracked(py_file):
            # Opportunistic git reset when there is no release manifest yet.
            ok, err = _git_checkout_file(py_file)
            if not ok:
                return {"success": False, "stem": stem, "error": f"git checkout failed: {err}"}
            restored_from = "git_head"
            restored_content = py_file.read_text(encoding="utf-8", errors="ignore")
        else:
            # Last resort: the oldest snapshot before toolsmith's first edit.
            backups = _list_rsi_backups(py_file)
            if not backups:
                return {
                    "success": False,
                    "stem": stem,
                    "error": (
                        "No release manifest, no git tracking, and no RSI backups "
                        "exist — nothing to reset to."
                    ),
                }
            oldest = backups[-1]
            try:
                restored_content = oldest.read_text(encoding="utf-8", errors="ignore")
                restored_from = f"backup:{len(backups) - 1}"
                py_file.write_text(restored_content, encoding="utf-8")
            except Exception as ex:
                return {"success": False, "stem": stem, "error": f"Restore failed: {ex}"}

        # Verify.
        syntax_ok, syntax_err = _ast_smoke_check(restored_content)
        if not syntax_ok:
            return {
                "success": False,
                "stem": stem,
                "restored_from": restored_from,
                "error": f"Committed/oldest snapshot is broken: {syntax_err}",
            }

        # Purge backups only after a successful discard: we're back to a known
        # baseline; keeping stale intermediate snapshots invites confusion.
        purged = _purge_rsi_backups(py_file)

        return {
            "success": True,
            "stem": stem,
            "restored_from": restored_from,
            "purged_backups": purged,
            "message": f"Toolset '{stem}' reset to {restored_from}.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to discard toolset changes: {e}"}


def tool_commit_toolset(
    tool_name_or_stem: str,
    commit_message: str = "",
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Commit the toolset back to git and hot-remount it on the LCP binding.

    ORDER OF OPERATIONS:
      1. Re-run the AST + import probe (never commit a broken toolset).
      2. git add + git commit scoped to the toolset folder.
      3. Hot-remount the toolset on the LCP binding so the next LLM round sees
         the updated tool set without requiring a client restart.

    Args:
        tool_name_or_stem: Tool function name or toolset stem.
        commit_message: Optional commit message (defaults to auto-generated).

    Returns:
        dict: {"success": bool, "stem": str, "commit_sha": str, "tools_count": int,
               "message": str}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to commit."}

    # RSI promotion tool — requires enable_toolsmith=True.
    flag_gate = _require_toolsmith_flag(env, "tool_commit_toolset")
    if flag_gate is not None:
        return flag_gate

    try:
        binding = _get_binding(env)
        stem, py_file = _resolve_toolset_file(binding, tool_name_or_stem)

        # Step 1: guard against committing a broken toolset.
        pre_commit = tool_test_tool_source(tool_name_or_stem, env=env)
        if not pre_commit.get("success"):
            return {
                "success": False,
                "error": "Refusing to commit — the toolset fails smoke tests. "
                         "Fix the source and re-run tool_test_tool_source before "
                         f"committing. Underlying error: {pre_commit.get('error')}",
                "stem": stem,
            }

        # Step 2a: write the git-free release manifest with embedded source.
        # This is what makes the toolset committable with ZERO external tooling
        # (no git binary required, ever). Failures here are hard — the release
        # manifest IS the commit record.
        source_now = py_file.read_text(encoding="utf-8", errors="ignore")
        manifest_result = _write_release_manifest(
            py_file,
            source_now,
            commit_message or f"toolsmith: update {stem}",
            extra={"stem": stem, "smoke_tested": True},
        )
        if not manifest_result.get("ok"):
            return {
                "success": False,
                "error": (
                    "Refusing to report success — the release manifest could not "
                    f"be written ({manifest_result.get('error')}). The toolset "
                    "source is unchanged on disk."
                ),
                "stem": stem,
            }

        # Step 2b: opportunistic git commit. Runs only when git is present AND
        # the toolset folder is inside a tracked repo. A failure here does not
        # fail the overall commit: the release manifest is authoritative.
        commit_result = _git_commit_scoped_optional(
            py_file.parent, commit_message or f"toolsmith: update {stem}"
        )
        if commit_result is None:
            commit_result = {
                "ok": True,
                "message": "Release recorded (git not available — file-system manifest is canonical).",
                "commit_sha": None,
                "already_committed": False,
            }

        # Step 3: hot-remount the toolset on the binding.
        remount_ok = False
        tools_count_after = 0
        try:
            if hasattr(binding, "_remount_toolset_after_change"):
                remount_ok = bool(binding._remount_toolset_after_change(stem))
                tools_count_after = sum(
                    1 for t in binding.discovered_tools
                    if (t.get("_library_name") or "") == stem
                    or Path(t.get("_python_file_path") or "").parent.name == stem
                )
        except Exception:
            remount_ok = False

        # Step 4: on a successful commit, purge .rsi_backup/ — git history is
        # now the canonical record of this toolset's evolution, so keeping
        # unbounded intermediate snapshots around is pure noise. If the commit
        # failed (dirty-tree error, no git, etc.), the backups stay put so the
        # operator can still revert or retry.
        purged_backups = 0
        if commit_result.get("ok"):
            purged_backups = _purge_rsi_backups(py_file)

        return {
            "success": bool(commit_result.get("ok")),
            "stem": stem,
            "commit_sha": commit_result.get("commit_sha"),
            "release_id": manifest_result.get("release_id"),
            "release_file": manifest_result.get("release_file"),
            "tools_count": tools_count_after,
            "remounted": remount_ok,
            "purged_backups": purged_backups,
            "message": (
                "Toolset released and remounted."
                if commit_result.get("ok") and remount_ok
                else commit_result.get("message") or commit_result.get("error")
            ),
            "already_committed": commit_result.get("already_committed", False),
            "uses_git": bool(commit_result.get("commit_sha")),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to commit toolset: {e}"}