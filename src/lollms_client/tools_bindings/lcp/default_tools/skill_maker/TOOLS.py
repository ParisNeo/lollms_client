"""
skill_maker — RSI Toolset for LLM *skills* (SKILL.md files).

Mirrors ``tool_maker`` exactly (same two-tier split, same lifecycle, same
`.rsi_backup/` + `.rsi/commits/` stores) but operates on a skill folder's
``SKILL.md`` instead of a Python ``TOOLS.py``.

DESIGN INVARIANTS (mirror tool_maker):
  * Every tool declares ``GATED_TASK_CATEGORY = "code_execution"`` so the host
    confirmation modal blocks it until the operator approves it (skill files
    can scope tool permissions, so they deserve the same gate).
  * Read/write paths are strictly scoped to skills that either (a) live inside
    a configured ``skills_dirs`` root AND (b) are registered on the
    personality's ``skills_manager``. Arbitrary paths never resolve.
  * Writes are full-file or Aider-style SEARCH/REPLACE patches applied in
    memory; a partial patch is never written.
  * Tests parse the YAML frontmatter and validate its required shape (name,
    description). No user content is executed.
  * Child test agent mounts ONLY the candidate skill's declared ``tools:``.
  * Release manifests under ``.rsi/commits/<epoch>.json`` make commits work
    with NO git. git runs opportunistically when present.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOL_LIBRARY_NAME = "skill_maker"
TOOL_LIBRARY_DESCRIPTION = (
    "RSI toolset for skills: read, write, test, revert, discard and commit "
    "SKILL.md files. RSI act-phase requires enable_skillmaker=True."
)
# Skill content dictates tool permissions — treat as code_execution gate.
GATED_TASK_CATEGORY = "code_execution"

_CONFIRM_HANDLER = None
_USER_INTERACTION_HANDLER = None


# Skills whose declared ``tools:`` include these are refused for the child
# test agent — testing them would let arbitrary code run before the operator
# can reject the commit.
_FORBIDDEN_TEST_SKILL_TOOLS = frozenset({
    "system_shell",
    "execute_python",
    "execute_multi_file_service",
    "execute_python_data_query",
    "execute_sparql_query",
    "execute_sql_query",
    "computer_use",
    "tool_execute_python",
    "tool_execute_shell_command",
    "tool_execute_command",
    "tool_run_terminal",
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
    client = getattr(env, "client", None) if env is not None else None
    binding = getattr(client, "tools", None) if client is not None else None
    if binding is None:
        raise RuntimeError(
            "skill_maker requires an LCP tools binding (env.client.tools)."
        )
    if not hasattr(binding, "discovered_tools"):
        raise RuntimeError("Binding must expose discovered_tools.")
    return binding


def _skillmaker_flag_is_on(binding: Any) -> bool:
    try:
        flags = getattr(binding, "session_flags", {}) or {}
        return bool(flags.get("enable_skillmaker"))
    except Exception:
        return False


def _require_skillmaker_flag(env: Any, tool_name: str) -> Optional[Dict[str, Any]]:
    """Hard gate for the heavy skill RSI act-phase tools."""
    try:
        binding = _get_binding(env)
    except Exception as e:
        return {"success": False, "error": f"{tool_name} requires an active LCP binding: {e}"}
    if not _skillmaker_flag_is_on(binding):
        return {
            "success": False,
            "error": (
                f"{tool_name} is an RSI act-phase tool and requires "
                "enable_skillmaker=True. Enable it in GuiPrefs or pass "
                "enable_skillmaker=True to chat(). Basic skill-manipulation "
                "tools remain available without the flag."
            ),
            "requires_flag": "enable_skillmaker",
        }
    return None


def _iter_registered_skills(personality: Any) -> List[Dict[str, Any]]:
    """Every skill registered on the personality's skills_manager, or []."""
    if personality is None:
        return []
    mgr = getattr(personality, "skills_manager", None)
    if mgr is None:
        return []
    try:
        if hasattr(mgr, "list_skills"):
            return list(mgr.list_skills() or [])
    except Exception:
        pass
    return []


def _resolve_skill_file(env: Any, skill_name_or_path: str) -> Tuple[str, Path]:
    """Resolve a skill name (or slug) to (canonical_slug, SKILL.md path).

    Containment invariants:
      1. The resolved file MUST be inside a directory that appears in the
         personality's skills_manager skill list (canonical identity).
      2. The resolved file MUST be named ``SKILL.md``.
    Arbitrary user-supplied absolute paths never resolve.
    """
    needle = (skill_name_or_path or "").strip()
    if not needle:
        raise ValueError("skill_name_or_path is required.")

    personality = None
    client = getattr(env, "client", None) if env is not None else None
    if client is not None:
        personality = getattr(client, "_active_personality", None) or getattr(client, "personality", None)

    # Best-effort: prefer skills_manager from env.discussion if present
    if personality is None and env is not None:
        disc = getattr(env, "discussion", None)
        personality = getattr(disc, "_personality", None)

    skills = _iter_registered_skills(personality)
    if not skills:
        # Fallback: scan the standard skills roots so a bare CLI/test path
        # still functions (no personality attached). Files must still be
        # *somewhere recognised*.
        candidate_roots: List[Path] = []
        for p in [
            Path.home() / ".lollms_client" / "skills",
            Path.cwd() / ".lollms_code" / "skills",
        ]:
            if p.exists():
                candidate_roots.append(p.resolve())
        for root in candidate_roots:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                sfile = child / "SKILL.md"
                if sfile.exists() and (child.name == needle or sfile.name == needle):
                    return child.name, sfile.resolve()
        raise ValueError(
            f"No skill matching '{needle}' is registered in this session and no "
            f"standard skill root contains it."
        )

    normalized = re.sub(r"[\s\-]+", "_", needle).lower()
    for s in skills:
        title = (s.get("title") or "").strip()
        fp = s.get("file_path") or ""
        if not fp:
            continue
        p = Path(fp).resolve()
        if p.name != "SKILL.md":
            continue
        slug = p.parent.name
        variants = {
            title.lower(),
            slug.lower(),
            re.sub(r"[\s\-]+", "_", title).lower(),
            re.sub(r"[\s\-]+", "_", slug).lower(),
        }
        if needle in (str(p), title, slug) or normalized in variants:
            return slug, p

    raise ValueError(
        f"Skill '{needle}' matched none of the {len(skills)} registered skills."
    )


def _parse_frontmatter(content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract the YAML frontmatter between leading '---' markers. Returns
    (dict, error). Uses PyYAML when available; falls back to a trivial
    key: value parser otherwise."""
    if not content.startswith("---"):
        return None, "SKILL.md must start with a YAML frontmatter block (--- ... ---)."
    try:
        end = content.index("\n---", 3)
    except ValueError:
        return None, "SKILL.md frontmatter block was not closed with '---'."
    raw_fm = content[3:end].strip()
    body = content[end + 4:]
    if not raw_fm:
        return None, "SKILL.md frontmatter block is empty."
    try:
        import yaml  # type: ignore
        fm = yaml.safe_load(raw_fm) or {}
    except Exception as ye:
        # Pragmatic fallback: parse only simple `key: value` lines.
        fm = {}
        for line in raw_fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"').strip("'")
        if not fm:
            return None, f"Could not parse frontmatter as YAML: {ye}"
    if not isinstance(fm, dict):
        return None, "Frontmatter did not decode to a mapping."
    fm["_body"] = body
    return fm, None


def _validate_skill_body(content: str) -> Tuple[bool, Optional[str]]:
    """Validate SKILL.md structural invariants."""
    fm, err = _parse_frontmatter(content)
    if err is not None:
        return False, err
    if not fm.get("name"):
        return False, "Frontmatter is missing the required 'name' field."
    if not fm.get("description"):
        return False, "Frontmatter is missing the required 'description' field."
    body = (fm.get("_body") or "").strip()
    if len(body) < 30:
        return False, (
            "SKILL.md body is too short (<30 chars). A skill must describe "
            "its workflow so the LLM can act on it."
        )
    return True, None


def _apply_aider_patches(original: str, patch_text: str) -> Tuple[bool, str, Optional[str]]:
    hunks = re.findall(
        r"<<<<<<< SEARCH\s*\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE",
        patch_text, re.DOTALL,
    )
    if not hunks:
        return False, "", "No valid SEARCH/REPLACE hunks found."
    working = original
    for idx, (search_body, replace_body) in enumerate(hunks, 1):
        search_norm = search_body.strip("\n")
        replace_norm = replace_body.strip("\n")
        if search_norm not in working:
            return False, "", (
                f"Hunk #{idx}: SEARCH block not found verbatim in source. "
                "Refusing to write a partial patch."
            )
        working = working.replace(search_norm, replace_norm, 1)
    return True, working, None


# ── Storage: backups + release manifests (mirrors tool_maker) ──
_RSI_META_DIRNAME = ".rsi"
_RSI_COMMITS_DIRNAME = "commits"


def _rsi_commits_dir(skill_file: Path) -> Path:
    return skill_file.parent / _RSI_META_DIRNAME / _RSI_COMMITS_DIRNAME


def _list_rsi_releases(skill_file: Path) -> List[Path]:
    d = _rsi_commits_dir(skill_file)
    if not d.is_dir():
        return []
    try:
        rls = [p for p in d.glob("*.json") if p.is_file()]
    except Exception:
        return []
    def _key(p: Path) -> int:
        try:
            return int(p.stem.split("_")[0])
        except Exception:
            return 0
    return sorted(rls, key=_key, reverse=True)


def _write_release_manifest(skill_file: Path, source_text: str, message: str) -> Dict[str, Any]:
    import hashlib
    d = _rsi_commits_dir(skill_file)
    d.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    target = d / f"{epoch}.json"
    bump = 0
    while target.exists():
        bump += 1
        target = d / f"{epoch}_{bump}.json"
    payload = {
        "schema": 1,
        "skill_slug": skill_file.parent.name,
        "skill_file": str(skill_file.resolve()),
        "created_at": epoch,
        "created_at_human": datetime.utcfromtimestamp(epoch).isoformat() + "Z",
        "message": (message or "").strip() or "skillmaker release",
        "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source": source_text,
    }
    try:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "release_file": str(target.resolve()), "release_id": target.stem}
    except Exception as e:
        return {"ok": False, "error": f"Could not write release manifest: {e}"}


def _latest_release(skill_file: Path) -> Optional[Dict[str, Any]]:
    releases = _list_rsi_releases(skill_file)
    if not releases:
        return None
    try:
        return json.loads(releases[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_rsi_backups(skill_file: Path) -> List[Path]:
    d = skill_file.parent / ".rsi_backup"
    if not d.is_dir():
        return []
    try:
        snaps = [p for p in d.glob(f"{skill_file.name}.*.bak") if p.is_file()]
    except Exception:
        return []
    def _key(p: Path) -> int:
        parts = p.name.rsplit(".", 2)
        try:
            return int(parts[-2])
        except Exception:
            return 0
    return sorted(snaps, key=_key, reverse=True)


def _purge_rsi_backups(skill_file: Path) -> int:
    removed = 0
    for snap in _list_rsi_backups(skill_file):
        try:
            snap.unlink()
            removed += 1
        except Exception:
            pass
    d = skill_file.parent / ".rsi_backup"
    try:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    except Exception:
        pass
    return removed


def _backup_epoch(snap: Path) -> int:
    parts = snap.name.rsplit(".", 2)
    try:
        return int(parts[-2])
    except Exception:
        return 0


def _git_available() -> bool:
    try:
        return subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=4
        ).returncode == 0
    except Exception:
        return False


def _git_file_is_tracked(p: Path) -> bool:
    if not _git_available():
        return False
    try:
        return subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(p)],
            capture_output=True, text=True, timeout=8, cwd=str(p.parent),
        ).returncode == 0
    except Exception:
        return False


def _git_commit_scoped_optional(skill_dir: Path, message: str) -> Optional[Dict[str, Any]]:
    if not _git_available():
        return None
    # Only commit when the skill folder is inside a tracked repo.
    p = skill_dir.resolve()
    root = None
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            root = parent
            break
    if root is None:
        return None
    try:
        add = subprocess.run(
            ["git", "add", "--", str(skill_dir)],
            capture_output=True, text=True, timeout=15, cwd=str(skill_dir),
        )
        if add.returncode != 0:
            return {"ok": False, "error": (add.stderr or "").strip()}
        commit = subprocess.run(
            ["git", "commit", "-m", message, "--", str(skill_dir)],
            capture_output=True, text=True, timeout=20, cwd=str(skill_dir),
        )
        if commit.returncode != 0:
            stderr = (commit.stderr or "").strip()
            if "nothing" in stderr.lower():
                return {"ok": True, "message": "Nothing to commit.", "already_committed": True}
            return {"ok": False, "error": stderr}
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(skill_dir),
        )
        return {
            "ok": True,
            "message": (commit.stdout or "").strip(),
            "commit_sha": (head.stdout or "").strip() if head.returncode == 0 else None,
        }
    except Exception as e:
        return {"ok": False, "error": f"git commit exception: {e}"}


# ── Child runner (spawn subprocess; mirrors tool_maker) ──


def _run_child_skill_agent(
    *,
    skill_file: Path,
    declared_tools: List[str],
    test_task: str,
    max_steps: int,
    timeout_sec: int,
    env: Any,
) -> Dict[str, Any]:
    here = Path(__file__).resolve().parent
    runner = here / "_isolated_skill_runner.py"
    if not runner.exists():
        return {"ok": False, "error": f"Child runner missing: {runner}"}

    client = getattr(env, "client", None) if env is not None else None
    llm_bp: Dict[str, Any] = {}
    llm_mp: Dict[str, Any] = {}
    system_dir = None
    try:
        if client is not None:
            llm_bp = getattr(client, "llm_binding_profiles", {}) or {}
            llm_mp = getattr(client, "llm_model_profiles", {}) or {}
            system_dir = getattr(client, "_system_dir", None) or getattr(client, "system_dir", None)
    except Exception:
        pass

    payload = {
        "skill_file": str(skill_file),
        "declared_tools": declared_tools,
        "candidate_skills_dirs": [str(skill_file.parent.parent)],
        "test_task": (test_task or "").strip(),
        "max_steps": max(1, min(int(max_steps or 6), 12)),
        "timeout_sec": max(10, min(int(timeout_sec or 120), 300)),
        "llm_binding_profiles": llm_bp,
        "llm_model_profiles": llm_mp,
        "system_dir": system_dir,
    }

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
            capture_output=True, text=True,
            timeout=int(payload["timeout_sec"]) + 5,
            cwd=str(skill_file.parent),
            env=child_env,
        )
    except subprocess.TimeoutExpired as te:
        return {"ok": False, "error": f"Child hard-timed out after {te.timeout}s.", "transcript": []}
    except Exception as e:
        return {"ok": False, "error": f"Spawn failed: {e}"}

    out_lines = (run.stdout or "").strip().splitlines()
    envelope: Dict[str, Any]
    if out_lines:
        try:
            envelope = json.loads(out_lines[-1])
        except Exception:
            envelope = {
                "ok": False,
                "error": f"Could not decode child output: {out_lines[-1][:400]}",
                "raw_stdout": (run.stdout or "")[-2000:],
            }
    else:
        envelope = {
            "ok": False,
            "error": f"Child produced no stdout. stderr: {(run.stderr or '')[:800]}",
            "raw_stderr": (run.stderr or "")[-2000:],
        }
    envelope["child_exit_code"] = run.returncode
    if run.returncode not in (0, None) and envelope.get("ok"):
        envelope["ok"] = False
        envelope["error"] = f"Child exited code {run.returncode}."
    return envelope


# ──────────────────────────────── tools ────────────────────────────────


def tool_read_skill_source(
    skill_name_or_path: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Read the SKILL.md of a registered skill.

    Args:
        skill_name_or_path: Skill slug, title, or its SKILL.md path (must be registered).

    Returns:
        dict: {"success": bool, "slug": str, "path": str, "source": str, "line_count": int}
    """
    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        if not skill_file.exists():
            return {"success": False, "error": f"SKILL.md missing on disk: {skill_file}"}
        source = skill_file.read_text(encoding="utf-8", errors="ignore")
        return {
            "success": True,
            "slug": slug,
            "path": str(skill_file.resolve()),
            "source": source,
            "line_count": len(source.splitlines()),
            "size_chars": len(source),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to read skill source: {e}"}


def tool_write_skill_source(
    skill_name_or_path: str,
    new_source: str,
    reason: str = "",
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Write new SKILL.md content for a registered skill.

    Supports FULL content or Aider-style SEARCH/REPLACE patches. Applied in
    memory; validates the resulting SKILL.md before touching disk.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.
        new_source: New SKILL.md content or SEARCH/REPLACE block.
        reason: Rationale recorded in the response (audit trail).

    Returns:
        dict: {"success": bool, "slug": str, "path": str, "mode": "full"|"patch"}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to write skill source."}

    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        original = skill_file.read_text(encoding="utf-8") if skill_file.exists() else ""

        is_patch = ("<<<<<<< SEARCH" in (new_source or "")) and ("=======" in (new_source or ""))
        if is_patch:
            ok, resolved_or_err, err_msg = _apply_aider_patches(original, new_source)
            if not ok:
                return {"success": False, "mode": "patch", "slug": slug,
                        "error": f"Aider patch could not be applied: {err_msg}"}
            candidate, mode = resolved_or_err, "patch"
        else:
            candidate, mode = new_source or "", "full"

        ok, err = _validate_skill_body(candidate)
        if not ok:
            return {
                "success": False,
                "slug": slug,
                "mode": mode,
                "error": f"Candidate SKILL.md rejected before write: {err}",
            }

        # Snapshot current content so the edit is reversible.
        if skill_file.exists():
            d = skill_file.parent / ".rsi_backup"
            d.mkdir(exist_ok=True)
            try:
                (d / f"{skill_file.name}.{int(time.time())}.bak").write_text(original, encoding="utf-8")
            except Exception:
                pass

        skill_file.write_text(candidate, encoding="utf-8")

        return {
            "success": True,
            "slug": slug,
            "path": str(skill_file.resolve()),
            "mode": mode,
            "line_count": len(candidate.splitlines()),
            "reason": reason or "(no reason provided)",
            "message": f"Skill '{slug}' written ({mode}).",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to write skill source: {e}"}


def tool_test_skill_source(
    skill_name_or_path: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Validate the SKILL.md frontmatter + structural shape. No execution.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.

    Returns:
        dict: {"success": bool, "slug": str, "frontmatter_ok": bool,
               "structure_ok": bool, "declared_tools": [...], "error": str|null}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to test."}

    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        if not skill_file.exists():
            return {"success": False, "slug": slug, "error": f"SKILL.md missing: {skill_file}"}

        source = skill_file.read_text(encoding="utf-8", errors="ignore")
        fm, fm_err = _parse_frontmatter(source)
        structure_ok, struct_err = (False, fm_err) if fm_err else _validate_skill_body(source)

        declared_tools: List[str] = []
        if fm:
            raw = fm.get("tools")
            if isinstance(raw, list):
                declared_tools = [str(x).strip() for x in raw if x]
            elif isinstance(raw, str):
                declared_tools = [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]

        overall = bool(fm) and structure_ok
        if overall:
            globals()["_HALT_REQUESTED"] = False

        return {
            "success": overall,
            "slug": slug,
            "path": str(skill_file.resolve()),
            "frontmatter_ok": bool(fm is not None and fm_err is None),
            "structure_ok": structure_ok,
            "declared_tools": declared_tools,
            "error": fm_err or struct_err,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to test skill source: {e}"}


def tool_run_skill_test_agent(
    skill_name_or_path: str,
    test_task: str,
    max_steps: int = 6,
    timeout_sec: int = 120,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Equip a CONSTRAINED child agent with the skill's declared tools and run it.

    The child is:
      * Mounted with ONLY the tools declared in the skill's ``tools:`` list.
      * Hard-capped at ``max_steps`` rounds (ceiling 12).
      * Executed in a fresh subprocess with a hard ``timeout_sec`` (ceiling 300).
      * Refused if the skill declares any tool from ``_FORBIDDEN_TEST_SKILL_TOOLS``.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.
        test_task: Natural-language directive for the child.
        max_steps: Reasoning budget (1..12, default 6).
        timeout_sec: Subprocess timeout (10..300, default 120).

    Returns:
        dict: {"success": bool, "ok": bool, "verdict": str|null, "transcript": [...],
               "steps": int, "available_tools": [...]}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "ok": False,
                "error": "HALT requested in this turn — refusing to spawn a skill test agent."}

    flag_gate = _require_skillmaker_flag(env, "tool_run_skill_test_agent")
    if flag_gate is not None:
        flag_gate["ok"] = False
        return flag_gate

    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        source = skill_file.read_text(encoding="utf-8", errors="ignore")
        fm, fm_err = _parse_frontmatter(source)
        if fm_err:
            return {"success": False, "ok": False, "slug": slug,
                    "error": f"Refusing — the skill frontmatter is broken: {fm_err}"}

        declared_tools: List[str] = []
        raw = fm.get("tools")
        if isinstance(raw, list):
            declared_tools = [str(x).strip() for x in raw if x]
        elif isinstance(raw, str):
            declared_tools = [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]

        forbidden_hits = [t for t in declared_tools if t in _FORBIDDEN_TEST_SKILL_TOOLS]
        if forbidden_hits:
            return {
                "success": False,
                "ok": False,
                "slug": slug,
                "error": (
                    f"Refused: the skill declares code-execution tools "
                    f"{forbidden_hits}, which the child test agent must never be "
                    "equipped with. Test such skills manually."
                ),
            }

        if not declared_tools:
            return {
                "success": False,
                "ok": False,
                "slug": slug,
                "error": "Refusing — the skill declares no 'tools:', so there is nothing for a child agent to test.",
            }

        envelope = _run_child_skill_agent(
            skill_file=skill_file,
            declared_tools=declared_tools,
            test_task=test_task,
            max_steps=max_steps,
            timeout_sec=timeout_sec,
            env=env,
        )
        envelope["slug"] = slug
        envelope["success"] = bool(envelope.get("ok"))
        return envelope
    except Exception as e:
        return {"success": False, "ok": False, "error": f"Failed to run skill test agent: {e}"}


def tool_list_skill_versions(
    skill_name_or_path: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """List every available revert target for a registered skill.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.

    Returns:
        dict: {"success": bool, "slug": str, "releases": [...], "backups": [...],
               "git_available": bool, "has_git_head": bool, "git_dirty": bool|null}
    """
    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        if not skill_file.exists():
            return {"success": False, "error": f"SKILL.md missing: {skill_file}"}

        backups = _list_rsi_backups(skill_file)
        releases = _list_rsi_releases(skill_file)

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

        git_ok = _git_available()
        tracked = _git_file_is_tracked(skill_file) if git_ok else False
        git_dirty: Optional[bool] = None
        if tracked:
            try:
                r = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", str(skill_file)],
                    capture_output=True, text=True, timeout=8, cwd=str(skill_file.parent),
                )
                git_dirty = r.returncode != 0
            except Exception:
                git_dirty = None

        return {
            "success": True,
            "slug": slug,
            "path": str(skill_file.resolve()),
            "git_available": git_ok,
            "has_git_head": tracked,
            "git_dirty": git_dirty,
            "releases": release_entries,
            "release_count": len(release_entries),
            "backups": [
                {"file": str(b.resolve()), "index": idx, "epoch": _backup_epoch(b)}
                for idx, b in enumerate(backups)
            ],
            "backup_count": len(backups),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to list skill versions: {e}"}


def tool_revert_skill(
    skill_name_or_path: str,
    backup_index: int = 0,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Revert a skill to a prior snapshot (0 = undo last write).

    Snapshots current content first; falls back to release-manifest or git HEAD.
    Post-restore validation runs and a broken snapshot is auto-rolled-back to
    the pre-revert state (never leaves a broken skill on disk).

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.
        backup_index: 0 = most recent backup; use tool_list_skill_versions for others.

    Returns:
        dict: {"success": bool, "slug": str, "restored_from": str, "test_ok": bool}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to revert."}

    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        if not skill_file.exists():
            return {"success": False, "error": f"SKILL.md missing: {skill_file}"}

        backups = _list_rsi_backups(skill_file)
        current_content = skill_file.read_text(encoding="utf-8", errors="ignore")

        # Snapshot pre-revert content (so revert-of-a-revert is possible).
        d = skill_file.parent / ".rsi_backup"
        d.mkdir(exist_ok=True)
        (d / f"{skill_file.name}.{int(time.time())}.bak").write_text(current_content, encoding="utf-8")

        restored_from: Optional[str] = None
        restored_content: Optional[str] = None

        if backups and 0 <= backup_index < len(backups):
            try:
                restored_content = backups[backup_index].read_text(encoding="utf-8", errors="ignore")
                restored_from = f"backup:{backup_index}"
            except Exception as re_err:
                return {"success": False, "slug": slug, "error": f"Could not read backup #{backup_index}: {re_err}"}
        elif backups and backup_index >= len(backups):
            return {"success": False, "slug": slug,
                    "error": f"backup_index={backup_index} exceeds available backups ({len(backups)})."}

        if restored_content is None and not backups:
            latest = _latest_release(skill_file)
            if latest and isinstance(latest.get("source"), str):
                restored_content = latest["source"]
                restored_from = f"release:{latest.get('created_at', 'latest')}"
            elif _git_file_is_tracked(skill_file):
                try:
                    r = subprocess.run(
                        ["git", "checkout", "HEAD", "--", str(skill_file)],
                        capture_output=True, text=True, timeout=15, cwd=str(skill_file.parent),
                    )
                    if r.returncode != 0:
                        return {"success": False, "slug": slug,
                                "error": f"git checkout failed: {(r.stderr or '').strip()}"}
                except Exception as e:
                    return {"success": False, "slug": slug, "error": f"git checkout exception: {e}"}
                restored_from = "git_head"
                restored_content = skill_file.read_text(encoding="utf-8", errors="ignore")
            else:
                return {"success": False, "slug": slug,
                        "error": "No backups, no release manifest, and no git tracking — nothing to revert to."}

        if restored_from and restored_from.startswith(("backup:", "release:")):
            skill_file.write_text(restored_content, encoding="utf-8")

        test_ok, test_err = _validate_skill_body(restored_content)
        if not test_ok:
            # Broken snapshot → restore pre-revert content, never leave broken state.
            try:
                skill_file.write_text(current_content, encoding="utf-8")
            except Exception:
                pass
            return {
                "success": False,
                "slug": slug,
                "restored_from": restored_from,
                "error": f"Reverted content failed validation (rolled back). Underlying: {test_err}",
                "test_ok": False,
            }

        return {
            "success": True,
            "slug": slug,
            "restored_from": restored_from,
            "test_ok": True,
            "message": f"Skill '{slug}' reverted from {restored_from}.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to revert skill: {e}"}


def tool_discard_skill_changes(
    skill_name_or_path: str,
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hard-abort the current experiment: reset skill to last committed state.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.

    Returns:
        dict: {"success": bool, "slug": str, "restored_from": str, "purged_backups": int}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to discard."}

    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)
        if not skill_file.exists():
            return {"success": False, "error": f"SKILL.md missing: {skill_file}"}

        restored_from: Optional[str] = None

        latest = _latest_release(skill_file)
        if latest and isinstance(latest.get("source"), str):
            skill_file.write_text(latest["source"], encoding="utf-8")
            restored_from = f"release:{latest.get('created_at', 'latest')}"
        elif _git_file_is_tracked(skill_file):
            try:
                r = subprocess.run(
                    ["git", "checkout", "HEAD", "--", str(skill_file)],
                    capture_output=True, text=True, timeout=15, cwd=str(skill_file.parent),
                )
                if r.returncode != 0:
                    return {"success": False, "slug": slug,
                            "error": f"git checkout failed: {(r.stderr or '').strip()}"}
            except Exception as e:
                return {"success": False, "slug": slug, "error": f"git checkout exception: {e}"}
            restored_from = "git_head"
        else:
            backups = _list_rsi_backups(skill_file)
            if not backups:
                return {"success": False, "slug": slug,
                        "error": "No release, no git tracking, no backups — nothing to reset to."}
            try:
                skill_file.write_text(backups[-1].read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                restored_from = f"backup:{len(backups) - 1}"
            except Exception as e:
                return {"success": False, "slug": slug, "error": f"Restore failed: {e}"}

        ok, err = _validate_skill_body(skill_file.read_text(encoding="utf-8", errors="ignore"))
        if not ok:
            return {"success": False, "slug": slug, "error": f"Reset target is broken: {err}"}

        purged = _purge_rsi_backups(skill_file)
        return {
            "success": True,
            "slug": slug,
            "restored_from": restored_from,
            "purged_backups": purged,
            "message": f"Skill '{slug}' reset to {restored_from}.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to discard skill changes: {e}"}


def tool_commit_skill(
    skill_name_or_path: str,
    commit_message: str = "",
    env: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Record a release of the skill (git-free), opportunistically git-commit,
    then remount the personality's skill so the next LLM round sees the update.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path.
        commit_message: Optional commit message.

    Returns:
        dict: {"success": bool, "slug": str, "release_id": str, "commit_sha": str|null,
               "remounted": bool, "uses_git": bool}
    """
    if globals().get("_HALT_REQUESTED"):
        return {"success": False, "error": "HALT requested in this turn — refusing to commit."}

    flag_gate = _require_skillmaker_flag(env, "tool_commit_skill")
    if flag_gate is not None:
        return flag_gate

    try:
        slug, skill_file = _resolve_skill_file(env, skill_name_or_path)

        # Re-validate before committing.
        pre = tool_test_skill_source(skill_name_or_path, env=env)
        if not pre.get("success"):
            return {"success": False, "slug": slug,
                    "error": f"Refusing to commit — the skill fails validation. "
                             f"Underlying error: {pre.get('error')}"}

        source_now = skill_file.read_text(encoding="utf-8", errors="ignore")
        manifest = _write_release_manifest(
            skill_file, source_now, commit_message or f"skillmaker: update {slug}"
        )
        if not manifest.get("ok"):
            return {"success": False, "slug": slug,
                    "error": f"Could not write release manifest: {manifest.get('error')}"}

        commit_result = _git_commit_scoped_optional(
            skill_file.parent, commit_message or f"skillmaker: update {slug}"
        )
        if commit_result is None:
            commit_result = {
                "ok": True,
                "message": "Release recorded (git not available — file-system manifest is canonical).",
                "commit_sha": None,
                "already_committed": False,
            }

        # Hot-remount the skill on the personality if reachable.
        remount_ok = False
        client = getattr(env, "client", None) if env is not None else None
        personality = getattr(client, "_active_personality", None) if client else None
        if personality is None:
            disc = getattr(env, "discussion", None) if env is not None else None
            personality = getattr(disc, "_personality", None)
        if personality is not None and hasattr(personality, "skills_manager"):
            try:
                personality.skills_manager.reload()
                remount_ok = True
            except Exception:
                remount_ok = False

        purged_backups = 0
        if commit_result.get("ok"):
            purged_backups = _purge_rsi_backups(skill_file)

        return {
            "success": bool(commit_result.get("ok")),
            "slug": slug,
            "release_id": manifest.get("release_id"),
            "release_file": manifest.get("release_file"),
            "commit_sha": commit_result.get("commit_sha"),
            "remounted": remount_ok,
            "purged_backups": purged_backups,
            "uses_git": bool(commit_result.get("commit_sha")),
            "message": (
                "Skill released and remounted."
                if commit_result.get("ok") and remount_ok
                else commit_result.get("message") or commit_result.get("error")
            ),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to commit skill: {e}"}