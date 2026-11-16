"""toolset_staging — discover, stage, and edit one toolset source at a time.

Why this exists
---------------
The ``extra_tools`` argument on ``LollmsPersonality.from_handbag`` mounts
toolsets as **usable** (callable) tools. It deliberately does *not* imply
"these sources are also editable" — because editing changes the contract, and
a mounted toolset whose source is silently patchable turns a routine
conversation into a mutation risk.

When the host wires ``toolset_staging`` alongside ``extra_tools``, the agent
gets a *separate*, opt-in editing surface:

    list_stageable  →  stage(name)  →  read / patch / rewrite  →  unstage

The staged toolset's current source is re-rendered into a mutable prompt
region (see ``lollms_personality/toolset_staging_state.py``) EVERY turn. So
after ``tool_stage_patch`` the very next reasoning round sees the updated
source — no manual reload, no personality restart, no editor/target split.

Only ONE toolset is staged at a time
------------------------------------
The prompt region is meant to be a focus area. If two toolsets were staged
simultaneously the LLM would confuse patches that belong to different files,
and any non-trivial conversation would drown in source text. ``tool_stage``
enforces a hard single-target invariant:

    calling `tool_stage(name=B)` while A is staged → **error**, with the
    guidance to call `tool_stage_unstage` first.

That error text is the exact iteration loop the agent needs.

What the tools DO and DO NOT do
-------------------------------
* ``list_stageable``: enumerate toolset folders that are *mount-eligible*
  (workspace rglob + live binding's ``discovered_tools``). No source read.
* ``stage``: register one folder in ``<workspace>/.toolset_staging/
  staged.json``. **Reads nothing, writes nothing.** The prompt region is
  populated by the personality on the next turn.
* ``read``: return the full ``TOOLS.py`` source of the staged toolset.
* ``patch``: apply one or more Aider SEARCH/REPLACE blocks atomically to the
  staged toolset's ``TOOLS.py``. Validates with ``ast.parse`` + ``def tool_*``.
  Refreshes the personality binding on success so the edited function is
  immediately callable.
* ``rewrite``: replace the entire ``TOOLS.py`` with validated source.
* ``unstage``: remove the staging entry. **The edited file remains on disk**
  — it was always the deliverable.
* ``status``: non-mutating view.

Invariants
----------
* All writes go through ``ast.parse`` + ``def tool_*`` guardrails. A broken
  source never reaches disk.
* The registry only stores identity (name + folder + timestamp). The prompt
  builder reads the source from disk every turn.
* No hidden copy of the source exists. The deliverable is ``TOOLS.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lollms_client.lollms_personality.toolset_staging_state import (
    add_staged,
    build_staged_prompt_block,
    clear_staged,
    load_registry,
    read_toolset_source,
    remove_staged,
    save_registry,
    staging_dir,
)

TOOL_LIBRARY_NAME = "Toolset Staging"
TOOL_LIBRARY_DESC = (
    "Discover toolsets, stage ONE at a time, and edit its TOOLS.py in place. "
    "Staged source is re-rendered into the system prompt every turn."
)
TOOL_LIBRARY_ICON = "🗂️"

# ---------------------------------------------------------------------------
# Aider patch markers, assembled from primitives so this file never contains
# a line that starts with a literal patch marker.
# ---------------------------------------------------------------------------
_LT = chr(60) * 7       # 7 x '<'
_EQ = chr(61) * 7       # 7 x '='
_GT = chr(62) * 7       # 7 x '>'
_PATCH_BLOCK_RE = re.compile(
    re.escape(_LT) + r"[ \t]*SEARCH[ \t]*\n(.*?)\n?[ \t]*" + re.escape(_EQ)
    + r"[ \t]*\n(.*?)\n?[ \t]*" + re.escape(_GT) + r"[ \t]*REPLACE",
    re.DOTALL,
)
_TOOL_FUNC_RE = re.compile(r"^\s*def\s+(tool_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)

_ENTRYPOINT = "TOOLS.py"
_MAX_SOURCE_CHARS = 2_000_000


def init_tools_library(config: Optional[dict] = None) -> None:
    """No global state to initialize. State lives in <workspace>/.toolset_staging."""
    return None


# ---------------------------------------------------------------------------
# Environment + workspace helpers
# ---------------------------------------------------------------------------

def _workspace_root(env: Any) -> Path:
    """Resolve the caller's workspace root, defensively."""
    candidates: List[Any] = []
    if env is not None:
        tc = getattr(env, "tool_context", None)
        if tc is not None:
            candidates.extend(
                [
                    getattr(tc, "workspace_path", None),
                    getattr(tc, "workspace_root", None),
                ]
            )
        candidates.extend(
            [
                getattr(env, "workspace_path", None),
                getattr(env, "workspace_root", None),
            ]
        )
    for candidate in candidates:
        if candidate:
            try:
                return Path(candidate).resolve()
            except (OSError, ValueError):
                continue
    return Path.cwd().resolve()


def _resolve_personality(env: Any) -> Any:
    """Best-effort lookup of the hosting LollmsPersonality on ``env``."""
    if env is None:
        return None
    candidates = [
        env,
        getattr(env, "personality", None),
        getattr(env, "host_personality", None),
    ]
    tc = getattr(env, "tool_context", None)
    if tc is not None:
        candidates.extend(
            [
                tc,
                getattr(tc, "personality", None),
                getattr(tc, "host_personality", None),
                getattr(tc, "parent", None),
            ]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        if callable(getattr(candidate, "refresh_tools_and_skills", None)):
            return candidate
    return None


def _reload_personality(env: Any) -> Dict[str, Any]:
    personality = _resolve_personality(env)
    if personality is None:
        return {
            "reloaded": False,
            "reason": (
                "No personality handle exposing refresh_tools_and_skills() "
                "was found on env. Call it manually to make the edited tool "
                "reachable through the live binding."
            ),
        }
    try:
        personality.refresh_tools_and_skills()
        return {"reloaded": True, "at": _utc_now()}
    except Exception as exc:  # noqa: BLE001
        return {"reloaded": False, "error": f"{type(exc).__name__}: {exc}"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Toolset discovery — workspace + live binding
# ---------------------------------------------------------------------------

def _list_workspace_toolsets(workspace: Path) -> List[Dict[str, Any]]:
    """Every toolset folder under the workspace that contains TOOLS.py.

    Excludes:
      * the staging toolset itself (never an edit target),
      * hidden folders RELATIVE to the workspace (absolute dot-prefixed
        ancestors are fine — workspaces often live under e.g.
        ``~/.aitiana/tmp/...``).
    """
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    try:
        for entry in workspace.rglob(_ENTRYPOINT):
            folder = entry.parent
            resolved = str(folder.resolve())
            if resolved in seen:
                continue
            if folder.name == "toolset_staging":
                continue
            try:
                rel = folder.relative_to(workspace)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            seen.add(resolved)
            candidates.append({
                "name": folder.name,
                "folder": resolved,
                "relative_path": str(rel).replace("\\", "/"),
                "source": "workspace",
            })
    except (OSError, ValueError):
        pass
    candidates.sort(key=lambda c: c["relative_path"])
    return candidates


def _list_external_toolsets(env: Any, workspace: Path) -> List[Dict[str, Any]]:
    """Every toolset folder mounted on the live binding but outside the workspace.

    ``extra_tools`` on ``LollmsPersonality`` produces a single LCPBinding whose
    ``discovered_tools`` entries expose ``_python_file_path``. For folder-layout
    toolsets (``TOOLS.py``) the parent folder is a legitimate staging target
    even when it lives outside ``workspace``.
    """
    if env is None:
        return []
    client = getattr(env, "client", None)
    if client is None:
        return []
    binding = getattr(client, "tools", None)
    if binding is None:
        return []
    discovered = getattr(binding, "discovered_tools", None) or []

    workspace_set = {c["folder"] for c in _list_workspace_toolsets(workspace)}
    external: Dict[str, Dict[str, Any]] = {}
    for tool_def in discovered:
        py_path = tool_def.get("_python_file_path")
        if not py_path:
            continue
        try:
            p = Path(py_path).resolve()
        except (OSError, ValueError):
            continue
        if p.name != _ENTRYPOINT:
            continue
        folder = p.parent
        if folder.name == "toolset_staging":
            continue
        resolved = str(folder)
        if resolved in workspace_set or resolved in external:
            continue
        external[resolved] = {
            "name": folder.name,
            "folder": resolved,
            "relative_path": resolved,
            "source": "external",
        }
    return sorted(external.values(), key=lambda c: c["name"])


def _list_candidate_toolsets(workspace: Path, env: Any = None) -> List[Dict[str, Any]]:
    candidates = _list_workspace_toolsets(workspace)
    if env is not None:
        candidates.extend(_list_external_toolsets(env, workspace))
    return candidates


def _resolve_toolset_folder(
    workspace: Path,
    name: Optional[str],
    folder: Optional[str],
    env: Any = None,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve a name/folder pair to a concrete target folder containing TOOLS.py."""
    if folder:
        raw = folder.strip()
        if raw:
            try:
                p = Path(raw)
                if not p.is_absolute():
                    p = (workspace / p).resolve()
                else:
                    p = p.resolve()
            except (OSError, ValueError) as exc:
                return None, {"success": False, "error": f"Invalid folder: {exc}"}
            if not p.is_dir():
                return None, {"success": False, "error": f"Not a directory: {p}"}
            return p, None
    if not name:
        return None, {
            "success": False,
            "error": "Provide either name or folder.",
        }
    target = name.strip()

    matches: List[Path] = []
    for candidate in _list_workspace_toolsets(workspace):
        if candidate["name"] == target:
            matches.append(Path(candidate["folder"]))
    if not matches and env is not None:
        for candidate in _list_external_toolsets(env, workspace):
            if candidate["name"] == target:
                try:
                    matches.append(Path(candidate["folder"]).resolve())
                except (OSError, ValueError):
                    continue
    if not matches:
        return None, {
            "success": False,
            "error": (
                f"No toolset named '{target}' was found under the workspace "
                f"({workspace}) or on the live binding. Call "
                "tool_stage_list_stageable to enumerate candidates."
            ),
        }
    if len(matches) > 1:
        listed = ", ".join(str(p) for p in matches)
        return None, {
            "success": False,
            "error": (
                f"Ambiguous name '{target}': multiple matches ({listed}). "
                "Pass folder explicitly."
            ),
        }
    return matches[0], None


# ---------------------------------------------------------------------------
# Source validation + Aider patch application
# ---------------------------------------------------------------------------

def _validate_source(source: str) -> Tuple[bool, str, List[str]]:
    if not source.strip():
        return False, "Source is empty.", []
    if len(source) > _MAX_SOURCE_CHARS:
        return False, f"Source exceeds the {_MAX_SOURCE_CHARS}-character safety cap.", []
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})", []
    tool_names = _TOOL_FUNC_RE.findall(source)
    if not tool_names:
        return (
            False,
            "Source defines no top-level 'def tool_*(...)' function. "
            "Every LCP toolset must expose at least one.",
            [],
        )
    return True, "", tool_names


def _parse_aider_patches(
    text: str,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[str]]:
    blocks: List[Tuple[str, str]] = []
    for match in _PATCH_BLOCK_RE.finditer(text or ""):
        search = match.group(1)
        replace = match.group(2)
        if not search.strip():
            return None, (
                "A SEARCH block is empty. Empty blocks are rejected — use "
                "tool_stage_rewrite for full-source replacement."
            )
        blocks.append((search, replace))
    if not blocks:
        return None, (
            "No valid Aider patch block found. Expected at least one "
            "SEARCH ... separator ... REPLACE block."
        )
    return blocks, None


def _apply_patches_atomically(
    source: str, blocks: List[Tuple[str, str]]
) -> Tuple[Optional[str], Optional[str]]:
    working = source
    for idx, (search, replace) in enumerate(blocks, start=1):
        count = working.count(search)
        if count == 0:
            return None, (
                f"Patch block #{idx}: SEARCH text was not found in the current "
                "source. Aborting — no changes were written."
            )
        if count > 1:
            return None, (
                f"Patch block #{idx}: SEARCH text matched {count} times; "
                "ambiguous. Aborting — no changes were written."
            )
        working = working.replace(search, replace, 1)
    return working, None


def _load_module_from_path(entry: Path) -> Any:
    """Import a fresh module object from ``entry`` for direct call testing."""
    module_name = f"_toolset_staging_test_{uuid.uuid4().hex}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(entry))
        if spec is None or spec.loader is None:
            return {"error": f"Could not build import spec for {entry}"}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Public tool API — one toolset at a time
# ---------------------------------------------------------------------------

def tool_stage_list_stageable(env: Optional[Any] = None) -> Dict[str, Any]:
    """Enumerate every toolset the agent may stage for editing.

    Combines two sources:
      * toolsets under the workspace (``source: "workspace"``),
      * toolsets mounted on the live binding but outside the workspace
        (``source: "external"``) — this is how ``extra_tools`` entries are
        editable without a copy-in step.

    Use this first. Then call ``tool_stage(name=...)`` with exactly ONE
    candidate. Multi-select is refused by design: the staged region is a
    focus area, and two simultaneously staged toolsets corrupt the context.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, workspace, count, staged (list of names currently
        staged), candidates.
    """
    workspace = _workspace_root(env)
    candidates = _list_candidate_toolsets(workspace, env)
    reg = load_registry(workspace)
    staged_names = [s["name"] for s in reg["staged"]]

    # ── Zero-candidate diagnostic ──────────────────────────────────────────
    # An empty `candidates` list is NOT benign: it means the agent has no
    # target and will fall back to improvising (doing the edit by hand, which
    # then trips the execute_python sandbox and looks like a "block after
    # spawning a test agent"). Two causes, and the agent cannot tell them
    # apart from a bare `count: 0`:
    #   (a) the toolset lives OUTSIDE `workspace` and its host binding is not
    #       visible on `env` (e.g. this tool runs inside a sub-agent child
    #       process whose `env` carries no client/`discovered_tools`), or
    #   (b) nothing containing TOOLS.py exists under the workspace at all.
    # Return the precise reason so the caller can act instead of retrying.
    if not candidates:
        external_hint = None
        client = getattr(env, "client", None) if env is not None else None
        binding = getattr(client, "tools", None) if client is not None else None
        if env is None:
            external_hint = (
                "No `env` handle was provided, so toolsets mounted via "
                "extra_tools could not be enumerated. If the target toolset "
                "lives outside the workspace, it can only be found through the "
                "live binding — ensure toolset_staging is mounted in the SAME "
                "agent that owns the binding (not a sub-agent whose env lacks it)."
            )
        elif binding is None:
            external_hint = (
                "The provided `env` exposes no .client.tools binding, so "
                "external (out-of-workspace) toolsets could not be enumerated. "
                "A sub-agent spawn does not inherit the parent's binding: run "
                "toolset_staging on the agent that was constructed with "
                "from_handbag(extra_tools=[..., toolset_staging])."
            )
        else:
            external_hint = (
                "The live binding is reachable but advertised no TOOLS.py "
                "entries (discovered_tools[*]['_python_file_path']). The target "
                "toolset may be a single-file toolset (not a folder with "
                "TOOLS.py) — staging only operates on folder-layout toolsets."
            )
        return {
            "success": True,
            "workspace": str(workspace),
            "count": 0,
            "candidates": [],
            "staged": staged_names,
            "diagnosis": (
                "No stageable toolset found under the workspace, and the live "
                "binding yielded none either. " + external_hint
            ),
            "next_step": (
                "If the target is a folder with a TOOLS.py that sits outside "
                "the current workspace, re-run the agent with toolset_staging "
                "mounted AND `workspace_path` pointing at a folder that "
                "CONTAINS the target (the editor enumerates the workspace via "
                "rglob). Do NOT attempt to edit the toolset by hand in "
                "execute_python — that path is sandboxed and will block."
            ),
        }

    return {
        "success": True,
        "workspace": str(workspace),
        "count": len(candidates),
        "candidates": candidates,
        "staged": staged_names,
        "next_step": (
            "Call tool_stage(name=\"<candidate>\") — ONE toolset only. If you "
            "already have one staged and want another, call tool_stage_unstage "
            "first."
        ),
    }


def tool_stage(
    name: str = "",
    folder: str = "",
    env: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Stage ONE toolset folder for in-place editing.

    Staging pins the target folder in ``<workspace>/.toolset_staging/
    staged.json``. It reads nothing and writes nothing — the personality
    populates the STAGED TOOLSETS prompt region on the NEXT turn from disk.

    Single-target invariant: if a different toolset is already staged, this
    call returns an error asking the caller to ``tool_stage_unstage`` first.

    Args:
        name: Toolset name (folder name). Required when ``folder`` is omitted.
        folder: Explicit folder path. Recommended when names collide.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, name, folder, staged (full registry),
        and next_step — or an error dict on failure.
    """
    workspace = _workspace_root(env)
    reg = load_registry(workspace)

    stale = []
    for entry in reg["staged"]:
        try:
            exists = (Path(entry["folder"]) / _ENTRYPOINT).is_file()
        except (OSError, ValueError):
            exists = False
        if not exists:
            stale.append(entry["name"])
    if stale:
        reg["staged"] = [e for e in reg["staged"] if e["name"] not in stale]
        save_registry(workspace, reg)

    resolved, err = _resolve_toolset_folder(workspace, name, folder, env=env)
    if err is not None:
        return err
    entry_file = resolved / _ENTRYPOINT
    if not entry_file.is_file():
        return {
            "success": False,
            "error": f"Target folder contains no {_ENTRYPOINT}: {resolved}",
        }

    target_name = resolved.name
    if reg["staged"]:
        already = reg["staged"][0]
        if already["name"] == target_name and already["folder"] == str(resolved):
            return {
                "success": True,
                "name": target_name,
                "folder": str(resolved),
                "already_staged": True,
                "staged": [already],
                "message": (
                    f"'{target_name}' is already staged. Use tool_stage_read / "
                    "tool_stage_patch / tool_stage_rewrite."
                ),
            }
        return {
            "success": False,
            "error": (
                f"Cannot stage '{target_name}': '{already['name']}' is already "
                "staged. Only ONE toolset may be staged at a time (the prompt "
                "region is a focus area). Call "
                f"tool_stage_unstage(name=\"{already['name']}\") first, then "
                f"call tool_stage(name=\"{target_name}\") again."
            ),
            "currently_staged": [already],
        }

    entry = add_staged(workspace, target_name, str(resolved))
    return {
        "success": True,
        "name": entry["name"],
        "folder": entry["folder"],
        "staged": [entry],
        "next_step": (
            f"'{target_name}' is staged. Its full TOOLS.py source will appear "
            "in your next round's STAGED TOOLSETS prompt region. Use "
            "tool_stage_read to inspect it now, tool_stage_patch (Aider "
            "blocks) or tool_stage_rewrite to modify it, and tool_stage_unstage "
            "to release it."
        ),
    }


def tool_stage_unstage(
    name: str = "",
    env: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Remove one (or all) toolset(s) from the STAGED TOOLSETS prompt region.

    The edited ``TOOLS.py`` remains on disk — unstaging only removes the
    read-only prompt view; the deliverable is always the on-disk file.

    Args:
        name: Toolset name to unstage. Empty string clears ALL staged entries
            (useful when the agent wants a clean focus slot).
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, removed (bool|int), staged (remaining entries).
    """
    workspace = _workspace_root(env)
    if name:
        removed = remove_staged(workspace, name)
        reg = load_registry(workspace)
        return {
            "success": True,
            "removed": removed,
            "staged": reg["staged"],
            "message": (
                f"'{name}' unstaged (TOOLS.py remains on disk at the target "
                "folder)." if removed else f"'{name}' was not staged."
            ),
        }
    count = clear_staged(workspace)
    return {
        "success": True,
        "removed": count,
        "staged": [],
        "message": f"Cleared {count} staged toolset(s).",
    }


def tool_stage_read(env: Optional[Any] = None) -> Dict[str, Any]:
    """Return the full source of the currently staged toolset's TOOLS.py.

    The same text is injected into the STAGED TOOLSETS prompt region every
    turn — this tool exists so the LLM can pull it on demand too (e.g. right
    after ``tool_stage``, before the next round has been assembled).

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, name, folder, tools, source — or an error dict.
    """
    workspace = _workspace_root(env)
    reg = load_registry(workspace)
    if not reg["staged"]:
        return {
            "success": False,
            "error": (
                "Nothing is staged. Call tool_stage(name=\"...\") first. "
                "Use tool_stage_list_stageable to enumerate candidates."
            ),
        }
    entry = reg["staged"][0]
    source = read_toolset_source(entry["folder"])
    if source is None:
        return {
            "success": False,
            "error": (
                f"Staged toolset '{entry['name']}' has no {_ENTRYPOINT} on disk "
                f"at: {entry['folder']}. Call tool_stage_unstage to clear it."
            ),
        }
    ok, verr, tool_names = _validate_source(source)
    return {
        "success": True,
        "name": entry["name"],
        "folder": entry["folder"],
        "valid": ok,
        "validation_error": verr or None,
        "tools": tool_names,
        "source": source,
    }


def tool_stage_patch(
    patches: str,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply one or more Aider SEARCH/REPLACE blocks to the staged toolset.

    Atomic: all blocks apply or none. Validates with ``ast.parse`` + ``def
    tool_*`` guardrails. On success, writes ``TOOLS.py`` and refreshes the
    personality binding so the edited function is immediately callable.

    The STAGED TOOLSETS prompt region is refreshed from disk on the very next
    round — the LLM automatically sees the new source without any additional
    call.

    Args:
        patches: One or more Aider blocks. The markers are the standard
            7-character runs.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, blocks_applied, name, tools, reloaded — or an
        error dict (nothing is written when this returns an error).
    """
    workspace = _workspace_root(env)
    reg = load_registry(workspace)
    if not reg["staged"]:
        return {
            "success": False,
            "error": (
                "Nothing is staged. Call tool_stage(name=\"...\") first."
            ),
        }
    entry = reg["staged"][0]
    target = Path(entry["folder"]) / _ENTRYPOINT
    if not target.is_file():
        return {
            "success": False,
            "error": f"Staged toolset's TOOLS.py is missing on disk: {target}",
        }

    blocks, perr = _parse_aider_patches(patches or "")
    if perr is not None:
        return {"success": False, "error": perr}

    try:
        current = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to read {target}: {exc}"}

    new_source, aerr = _apply_patches_atomically(current, blocks)
    if aerr is not None:
        return {"success": False, "error": aerr}

    ok, verr, tool_names = _validate_source(new_source)
    if not ok:
        return {
            "success": False,
            "error": f"Patched source failed validation ({verr}). Nothing written.",
        }

    try:
        target.write_text(new_source, encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to write {target}: {exc}"}

    reload_result = _reload_personality(env)
    return {
        "success": True,
        "blocks_applied": len(blocks),
        "name": entry["name"],
        "folder": entry["folder"],
        "tools": tool_names,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
        "message": (
            f"'{entry['name']}' patched ({len(blocks)} block(s)). The next "
            "round's STAGED TOOLSETS region will show the updated source."
        ),
    }


def tool_stage_rewrite(
    new_source: str,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Replace the entire TOOLS.py of the currently staged toolset.

    Same validation + hot-reload path as ``tool_stage_patch``.

    Args:
        new_source: The complete new TOOLS.py content.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, name, tools, reloaded — or an error dict.
    """
    workspace = _workspace_root(env)
    reg = load_registry(workspace)
    if not reg["staged"]:
        return {
            "success": False,
            "error": "Nothing is staged. Call tool_stage(name=\"...\") first.",
        }
    entry = reg["staged"][0]
    target = Path(entry["folder"]) / _ENTRYPOINT

    ok, verr, tool_names = _validate_source(new_source or "")
    if not ok:
        return {
            "success": False,
            "error": f"Rewrite failed validation ({verr}). Nothing written.",
        }

    try:
        target.write_text(new_source, encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to write {target}: {exc}"}

    reload_result = _reload_personality(env)
    return {
        "success": True,
        "name": entry["name"],
        "folder": entry["folder"],
        "tools": tool_names,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
        "message": (
            f"'{entry['name']}' rewritten. The next round's STAGED TOOLSETS "
            "region will show the updated source."
        ),
    }


def tool_stage_test(
    tool_name: str,
    arguments_json: str = "{}",
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a ``tool_*`` function from the staged toolset's on-disk TOOLS.py.

    Runs the CURRENT disk content in a fresh import, so it does not rely on
    the live binding having been reloaded. Use this to validate patches
    against the source the LLM is editing.

    Args:
        tool_name: The tool_* callable name in the staged toolset, e.g.
            "tool_echo".
        arguments_json: A JSON object of keyword arguments.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, name, tool_name, output — or an error dict.
    """
    workspace = _workspace_root(env)
    reg = load_registry(workspace)
    if not reg["staged"]:
        return {
            "success": False,
            "error": "Nothing is staged. Call tool_stage(name=\"...\") first.",
        }
    entry = reg["staged"][0]
    target = Path(entry["folder"]) / _ENTRYPOINT
    if not target.is_file():
        return {"success": False, "error": f"Missing on disk: {target}"}

    try:
        raw = json.loads(arguments_json) if arguments_json else {}
    except ValueError as exc:
        return {"success": False, "error": f"arguments_json is not valid JSON: {exc}"}
    if not isinstance(raw, dict):
        return {"success": False, "error": "arguments_json must decode to an object."}

    module = _load_module_from_path(target)
    if isinstance(module, dict):
        return {"success": False, "error": module["error"]}

    func = getattr(module, tool_name, None)
    if not callable(func):
        available = sorted(_TOOL_FUNC_RE.findall(target.read_text(encoding="utf-8", errors="ignore")))
        return {
            "success": False,
            "error": (
                f"'{tool_name}' is not a callable in the staged TOOLS.py. "
                f"Available: {', '.join(n for n in available if n != tool_name) or '(none)'}."
            ),
        }

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        sig = None
    kwargs = dict(raw)
    if sig is not None and "env" in sig.parameters and "env" not in kwargs:
        kwargs["env"] = env

    try:
        output = func(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "tool_name": tool_name,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "success": True,
        "name": entry["name"],
        "folder": entry["folder"],
        "tool_name": tool_name,
        "output": output,
    }


def tool_stage_status(env: Optional[Any] = None) -> Dict[str, Any]:
    """Report the current staging state (no side effects).

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, workspace, staged (list of {name, folder,
        staged_at}), prompt_block_present (bool).
    """
    workspace = _workspace_root(env)
    reg = load_registry(workspace)
    block = build_staged_prompt_block(workspace)
    return {
        "success": True,
        "workspace": str(workspace),
        "staged": reg["staged"],
        "count": len(reg["staged"]),
        "prompt_block_present": bool(block),
    }


__all__ = [
    "tool_stage_list_stageable",
    "tool_stage",
    "tool_stage_unstage",
    "tool_stage_read",
    "tool_stage_patch",
    "tool_stage_rewrite",
    "tool_stage_test",
    "tool_stage_status",
]