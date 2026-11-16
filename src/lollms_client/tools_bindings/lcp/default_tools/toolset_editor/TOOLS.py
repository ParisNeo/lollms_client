"""toolset_editor — in-place live editor for an LCP toolset.

Why this exists
---------------
The legacy flow required the host app to close and reopen a ``LollmsPersonality``
after every edit to pick up new ``tool_*`` functions -- because
``tool_load_generated_toolset`` does not hot-reload and
``refresh_tools_and_skills()`` is a host-only entry point. That made an in-agent
"edit -> observe -> re-edit" loop impossible.

This toolset delivers the whole loop *inside* the agent turn::

    activate  ->  edit (patch | rewrite)  ->  auto-reload  ->  test  ->  repeat

Each editing session tracks one target ``TOOLS.py`` (inside one toolset folder)
in a *special editing area* -- an authoritative JSON session file plus an
in-process cache. The on-disk source is the source of truth: after every
successful write the session also attempts
``personality.refresh_tools_and_skills()`` (duck-typed from the host ``env``),
so the freshly edited ``tool_*`` callables become immediately reachable through
the same LCP binding that hosted this editor.

AUTO-BOOTSTRAP -- you may not need ``tool_edit_toolset_activate`` at all
---------------------------------------------------------------------
When any editing tool (read / patch / rewrite / reload / test / close) is
called without an active session, it AUTO-ACTIVATES the sole toolset folder
found under the workspace. This is the single most common case for an AI
Builder session, and it removes the ordering burden from the LLM:

    tool_edit_toolset_read()      # auto-activates; returns the source
    tool_edit_toolset_patch(...)  # applies patch; auto-reloads

If ZERO or MORE THAN ONE toolset candidates exist, auto-activation returns a
structured error listing the candidates; call ``tool_edit_toolset_list()`` to
see them and ``tool_edit_toolset_activate(toolset_name=...)`` to pick one.

Because writes land **directly in the target TOOLS.py**, the host app can, at
any point, take the toolset folder as-is and commit it to the caller's own
repository. Nothing is hidden in a shadow copy.

Editing primitives
------------------
* ``patch``   -- one or more Aider SEARCH/REPLACE blocks applied atomically
                 (all succeed or none).
* ``rewrite`` -- the full replacement source.

Every write runs ``ast.parse`` and requires at least one ``def tool_<verb>(``
so a broken edit never reaches disk. Reload is best-effort: when no personality
handle is present on ``env`` the tool returns ``reloaded: False`` and the caller
must invoke ``refresh_tools_and_skills()`` itself.

Naming
------
All public tools are top-level ``tool_edit_toolset_*`` functions, following the
same convention as every other default LCP toolset.

Implementation note on marker literals
--------------------------------------
Aider patch markers are built at runtime from ``chr(60)`` / ``chr(61)`` /
``chr(62)`` runs so this source file never contains a line that begins with the
literal marker sequence. That keeps generic "apply-patch" tooling from
misinterpreting this file's own constants as a live diff.
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

TOOL_LIBRARY_NAME = "Toolset Editor"
TOOL_LIBRARY_DESC = (
    "In-place live editor for a single LCP toolset: Aider patches, full "
    "rewrites, hot-reload, and direct tool validation."
)
TOOL_LIBRARY_ICON = "✏️"

# ---------------------------------------------------------------------------
# Aider patch markers, assembled from primitives so this file never contains
# a line that starts with a literal patch marker (which would confuse
# generic apply-patch tooling that treats this module as its own diff).
# ---------------------------------------------------------------------------
_LT = chr(60) * 7       # marker start:  7 x '<'
_EQ = chr(61) * 7       # separator:     7 x '='
_GT = chr(62) * 7       # marker end:    7 x '>'
_PATCH_BLOCK_RE = re.compile(
    re.escape(_LT) + r"[ \t]*SEARCH[ \t]*\n(.*?)\n?[ \t]*" + re.escape(_EQ)
    + r"[ \t]*\n(.*?)\n?[ \t]*" + re.escape(_GT) + r"[ \t]*REPLACE",
    re.DOTALL,
)
_TOOL_FUNC_RE = re.compile(r"^\s*def\s+(tool_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)

_ENTRYPOINT = "TOOLS.py"
_SESSION_DIRNAME = ".toolset_editor"
_SESSION_FILENAME = "session.json"
_MAX_SOURCE_CHARS = 2_000_000


def init_tools_library(config: Optional[dict] = None) -> None:
    """No global state to initialize. Sessions are per-workspace on disk."""
    return None


# ---------------------------------------------------------------------------
# Environment + workspace helpers
# ---------------------------------------------------------------------------

def _workspace_root(env: Any) -> Path:
    """Resolve the caller's workspace root, defensively.

    Mirrors ``tool_maker._get_workspace_root``: prefer
    ``env.tool_context.workspace_path``, fall back to any common alias,
    else the process CWD.
    """
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


def _session_dir(workspace: Path) -> Path:
    d = workspace / _SESSION_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(workspace: Path) -> Path:
    return _session_dir(workspace) / _SESSION_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_session(workspace: Path) -> Optional[Dict[str, Any]]:
    path = _session_path(workspace)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, ValueError):
        return None


def _save_session(workspace: Path, data: Dict[str, Any]) -> None:
    path = _session_path(workspace)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _list_workspace_toolsets(workspace: Path) -> List[Dict[str, Any]]:
    """Enumerate every toolset folder under the workspace containing TOOLS.py.

    Bounded to a reasonable depth so a giant workspace cannot stall a call.
    Excludes our own editor folder and any ``.toolset_editor`` session dir.
    """
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    try:
        for entry in workspace.rglob(_ENTRYPOINT):
            folder = entry.parent
            resolved = str(folder.resolve())
            if resolved in seen:
                continue
            # Skip our own toolset folder.
            if folder.name == "toolset_editor":
                continue
            # Skip HIDDEN folders RELATIVE to the workspace only. We must NOT
            # inspect absolute parts, because the workspace itself may sit
            # under a dot-prefixed ancestor (e.g. ~/.aitiana_studio/tmp/...).
            try:
                rel = folder.relative_to(workspace)
            except ValueError:
                # Entry resolves outside the workspace (symlink); skip it.
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            seen.add(resolved)
            candidates.append({
                "toolset_name": folder.name,
                "toolset_folder": resolved,
                "relative_path": str(rel).replace("\\", "/"),
                "source": "workspace",
            })
    except (OSError, ValueError):
        pass
    candidates.sort(key=lambda c: c["relative_path"])
    return candidates


def _list_external_toolsets(env: Any, workspace: Path) -> List[Dict[str, Any]]:
    """Enumerate toolsets from the live binding's ``discovered_tools``.

    This is the "extra_tools exception": a folder the caller wired through
    ``extra_tools`` (or the handbag's own ``tools/`` folders) is a legitimate
    editing target **even when it lives outside the workspace**. The live LCP
    binding is the authoritative source for those folders — every discovered
    tool exposes ``_python_file_path``, whose parent (for a folder-layout
    ``TOOLS.py``) is the mount folder.

    Excludes:
      * folders already covered by ``_list_workspace_toolsets`` (dedup),
      * our own editor folder (self-editing is refused),
      * any folder whose ``.py`` file is not ``TOOLS.py`` (single-file toolsets
        are not the target of this toolset-*folder* editor; use the file path
        directly through the caller instead),
      * folders under a HIDDEN segment relative to their *own* parent — but
        NOT absolute dot-prefixed ancestors of the workspace, since there is
        no workspace relationship here.
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

    workspace_set = {
        c["toolset_folder"] for c in _list_workspace_toolsets(workspace)
    }
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
        # Skip our own editor folder.
        if folder.name == "toolset_editor":
            continue
        resolved = str(folder)
        if resolved in workspace_set or resolved in external:
            continue
        external[resolved] = {
            "toolset_name": folder.name,
            "toolset_folder": resolved,
            "relative_path": resolved,   # no workspace-relative form exists
            "source": "external",
        }
    return sorted(external.values(), key=lambda c: c["toolset_name"])


def _list_candidate_toolsets(workspace: Path, env: Any = None) -> List[Dict[str, Any]]:
    """Enumerate editing candidates: workspace-first, then external.

    Workspace candidates take precedence because they are the caller's
    explicit deliverable location. External toolsets — those mounted through
    ``extra_tools`` on the live binding but living outside the workspace — are
    appended so an editor persona wired with ``extra_tools=[EDITOR, TARGET]``
    can edit ``TARGET`` directly without copying it into the workspace.
    """
    candidates = _list_workspace_toolsets(workspace)
    if env is not None:
        candidates.extend(_list_external_toolsets(env, workspace))
    return candidates


def _try_auto_activate(
    workspace: Path,
    env: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Bootstrap a session against the sole candidate toolset.

    Candidates are workspace toolsets first, then external toolsets from the
    live binding (the ``extra_tools`` exception). If exactly ONE candidate
    exists, auto-activate it. If zero or more than one, return an error that
    lists the candidates and tells the caller how to disambiguate.

    Returns (session, error_message).
    """
    candidates = _list_candidate_toolsets(workspace, env)
    if not candidates:
        return None, (
            f"No toolset containing {_ENTRYPOINT} was found under the workspace "
            f"({workspace}) and no external toolset is mounted on the live "
            "binding. Either the target toolset folder is not inside the "
            "workspace, or it is missing TOOLS.py, or it was not passed via "
            "extra_tools. Pass toolset_folder explicitly in "
            "tool_edit_toolset_activate(toolset_folder=...)."
        )
    if len(candidates) > 1:
        listed = ", ".join(
            f"'{c['toolset_name']}' ({c['relative_path']}, {c['source']})"
            for c in candidates
        )
        return None, (
            f"{len(candidates)} toolset candidates were found; auto-activation is "
            f"ambiguous. Call tool_edit_toolset_activate(toolset_name=..., "
            f"toolset_folder=...) with one of: {listed}."
        )
    target = candidates[0]
    result = tool_edit_toolset_activate(
        toolset_name=target["toolset_name"],
        toolset_folder=target["toolset_folder"],
        env=env,
    )
    if not result.get("success"):
        return None, result.get("error") or "Auto-activation failed."
    return _load_session(workspace), None


def _require_session(
    env: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Dict[str, Any]]]:
    """Returns (session, workspace, error_dict).

    If no session exists, this AUTO-BOOTSTRAPS against the sole toolset
    folder in the workspace. That removes the hard requirement to call
    ``tool_edit_toolset_activate`` first when there is exactly one obvious
    target — the single most common case for an AI Builder session.

    When zero or multiple candidates exist, a structured error is returned
    with the list of candidates so the caller can disambiguate.
    """
    workspace = _workspace_root(env)
    session = _load_session(workspace)
    if session is not None:
        return session, workspace, None
    session, err = _try_auto_activate(workspace, env)
    if session is None:
        return None, workspace, {"success": False, "error": err}
    return session, workspace, None


# ---------------------------------------------------------------------------
# Personality resolution for hot-reload
# ---------------------------------------------------------------------------

def _resolve_personality(env: Any) -> Any:
    """Best-effort lookup of the hosting LollmsPersonality on ``env``.

    Duck-typed: look for any object exposing a callable
    ``refresh_tools_and_skills`` so we stay agnostic to the exact
    ``tool_context`` shape while remaining compatible with the documented
    LollmsPersonality API.
    """
    if env is None:
        return None
    direct_candidates = [
        env,
        getattr(env, "personality", None),
        getattr(env, "host_personality", None),
    ]
    tc = getattr(env, "tool_context", None)
    if tc is not None:
        direct_candidates.extend(
            [
                tc,
                getattr(tc, "personality", None),
                getattr(tc, "host_personality", None),
                getattr(tc, "parent", None),
            ]
        )
    for candidate in direct_candidates:
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
                "was found on env. The caller must call it manually before "
                "the edited tool is reachable through the live binding."
            ),
        }
    try:
        personality.refresh_tools_and_skills()
        return {"reloaded": True, "at": _utc_now()}
    except Exception as exc:  # noqa: BLE001 - surface any reload failure as data
        return {"reloaded": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Toolset path + source validation
# ---------------------------------------------------------------------------

def _resolve_toolset_folder(
    workspace: Path,
    toolset_name: Optional[str],
    toolset_folder: Optional[str],
    env: Any = None,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Return the folder containing the target TOOLS.py, or an error dict.

    Resolution order:
      1. ``toolset_folder`` when given (absolute, or workspace-relative).
      2. ``toolset_name`` searched in the workspace.
      3. ``toolset_name`` searched in the live binding's mounted toolsets
         (the ``extra_tools`` exception — folders outside the workspace).
    """
    if toolset_folder:
        raw = toolset_folder.strip()
        if raw:
            try:
                folder = Path(raw)
                if not folder.is_absolute():
                    folder = (workspace / folder).resolve()
                else:
                    folder = folder.resolve()
            except (OSError, ValueError) as exc:
                return None, {
                    "success": False,
                    "error": f"Invalid toolset_folder: {exc}",
                }
            if not folder.is_dir():
                return None, {
                    "success": False,
                    "error": f"toolset_folder is not a directory: {folder}",
                }
            return folder, None
    if not toolset_name:
        return None, {
            "success": False,
            "error": "Provide either toolset_folder or toolset_name.",
        }
    target = toolset_name.strip()

    matches: List[Path] = []
    for candidate in workspace.rglob(_ENTRYPOINT):
        if candidate.parent.name == target:
            matches.append(candidate.parent)
    if not matches and env is not None:
        # Fall back to the live binding (extra_tools) — folders outside the
        # workspace are legitimate targets.
        for candidate in _list_external_toolsets(env, workspace):
            if candidate["toolset_name"] == target:
                try:
                    matches.append(Path(candidate["toolset_folder"]).resolve())
                except (OSError, ValueError):
                    continue
    if not matches:
        return None, {
            "success": False,
            "error": (
                f"No folder named '{target}' containing {_ENTRYPOINT} was "
                f"found under the workspace ({workspace}) or on the live "
                "binding. Check the name, or pass toolset_folder explicitly."
            ),
        }
    if len(matches) > 1:
        listed = ", ".join(str(p) for p in matches)
        return None, {
            "success": False,
            "error": (
                f"Ambiguous toolset_name '{target}': multiple matches ({listed}). "
                "Pass toolset_folder explicitly."
            ),
        }
    return matches[0], None


def _validate_source(source: str) -> Tuple[bool, str, List[str]]:
    """Return (ok, error_message, discovered_tool_names)."""
    if not source.strip():
        return False, "Source is empty.", []
    if len(source) > _MAX_SOURCE_CHARS:
        return (
            False,
            f"Source exceeds the {_MAX_SOURCE_CHARS}-character safety cap.",
            [],
        )
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


# ---------------------------------------------------------------------------
# Aider SEARCH/REPLACE patch handling
# ---------------------------------------------------------------------------

def _parse_aider_patches(
    text: str,
) -> Tuple[Optional[List[Tuple[str, str]]], Optional[str]]:
    """Return (blocks, error). Each block is (search, replace)."""
    blocks: List[Tuple[str, str]] = []
    for match in _PATCH_BLOCK_RE.finditer(text or ""):
        search = match.group(1)
        replace = match.group(2)
        if not search.strip():
            return None, (
                "A SEARCH block is empty. Empty SEARCH blocks are rejected -- "
                "use tool_edit_toolset_rewrite for full-source replacement."
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
    """Apply every block or none. Returns (new_source, error)."""
    working = source
    for idx, (search, replace) in enumerate(blocks, start=1):
        count = working.count(search)
        if count == 0:
            return None, (
                f"Patch block #{idx}: SEARCH text was not found in the current "
                "source. Aborting - no changes were written."
            )
        if count > 1:
            return None, (
                f"Patch block #{idx}: SEARCH text matched {count} times; "
                "ambiguous. Aborting - no changes were written."
            )
        working = working.replace(search, replace, 1)
    return working, None


def _load_module_from_path(entry: Path) -> Any:
    """Import a fresh module object from ``entry`` without touching sys.modules."""
    module_name = f"_toolset_editor_test_{uuid.uuid4().hex}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(entry))
        if spec is None or spec.loader is None:
            return {"error": f"Could not build an import spec for {entry}"}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Public tool API
# ---------------------------------------------------------------------------

def tool_edit_toolset_activate(
    toolset_name: str,
    toolset_folder: Optional[str] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
        """Open a live editing session against one target LCP toolset.

    Loads the toolset's TOOLS.py into the session's editing area and pins the
    target folder. Every subsequent patch/rewrite writes to this exact TOOLS.py.

    Args:
        toolset_name: Logical name (folder name). Used to disambiguate when
            toolset_folder is omitted.
        toolset_folder: Explicit path to the toolset folder (absolute, or
            relative to the workspace root). Recommended when the folder name
            is not unique.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, session_id, toolset_name, toolset_folder,
        entrypoint, tools, reloaded, reload_detail, next_step -- or an error
        dict on failure.
    """
    workspace = _workspace_root(env)
    folder, err = _resolve_toolset_folder(
        workspace, toolset_name, toolset_folder, env=env
    )
    if err is not None:
        return err
    entry = folder / _ENTRYPOINT
    if not entry.is_file():
        return {
            "success": False,
            "error": f"Target folder does not contain {_ENTRYPOINT}: {folder}",
        }
    try:
        source = entry.read_text(encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to read {entry}: {exc}"}
    ok, verr, tool_names = _validate_source(source)
    if not ok:
        return {
            "success": False,
            "error": (
                f"Existing source failed validation ({verr}). Fix it manually "
                "before activating an editing session."
            ),
        }

    session = {
        "session_id": uuid.uuid4().hex,
        "toolset_name": folder.name,
        "toolset_folder": str(folder),
        "entrypoint": _ENTRYPOINT,
        "opened_at": _utc_now(),
        "last_write_at": None,
        "last_reload": None,
        "write_count": 0,
        "draft": source,
    }
    _save_session(workspace, session)

    reload_result = _reload_personality(env)
    if reload_result.get("reloaded"):
        session["last_reload"] = reload_result
        _save_session(workspace, session)

    return {
        "success": True,
        "session_id": session["session_id"],
        "toolset_name": session["toolset_name"],
        "toolset_folder": session["toolset_folder"],
        "entrypoint": session["entrypoint"],
        "tools": tool_names,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
        "next_step": (
            "Use tool_edit_toolset_patch (Aider blocks) or "
            "tool_edit_toolset_rewrite (full source). Each successful write "
            "auto-validates, writes to disk, and attempts a live reload."
        ),
    }


def tool_edit_toolset_read(env: Optional[Any] = None) -> Dict[str, Any]:
    """Return the FULL source of the target TOOLS.py -- no file tools required.

    This is the canonical way for an editor agent to inspect the toolset
    source. It does NOT rely on ``workspace_tools``, ``execute_python``, a
    sub-agent, or any other capability: the returned ``source`` field IS the
    complete file content, ready to be reasoned about and patched.

    When called with no active session it auto-activates the sole toolset
    folder found in the workspace -- so a typical AI Builder session is:

        tool_edit_toolset_read()   # → {"source": "<full TOOLS.py content>", ...}

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, toolset_name, toolset_folder, entrypoint, valid,
        validation_error, tools, source -- or an error dict on failure.
    """
    session, _workspace, err = _require_session(env)
    if err is not None:
        return err
    source = session.get("draft", "")
    ok, verr, tool_names = _validate_source(source)
    return {
        "success": True,
        "toolset_name": session.get("toolset_name"),
        "toolset_folder": session.get("toolset_folder"),
        "entrypoint": session.get("entrypoint"),
        "valid": ok,
        "validation_error": verr or None,
        "tools": tool_names,
        "source": source,
    }


def tool_edit_toolset_patch(
    patches: str,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply one or more Aider SEARCH/REPLACE blocks to the target TOOLS.py.

    All blocks are applied atomically (all succeed -> one write -> one reload;
    any failure -> no change). After a successful write the session attempts a
    live reload via personality.refresh_tools_and_skills() so the new tool_*
    callables are reachable immediately.

    Args:
        patches: One or more Aider blocks of the form
            SEARCH marker / old exact lines / separator / new lines / REPLACE
            marker. The markers are the standard 7-character runs; each block
            must contain exactly one SEARCH section and one REPLACE section.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, blocks_applied, write_count, toolset_folder,
        tools, reloaded, reload_detail -- or an error dict on failure (nothing
        is written when this returns an error).
    """
    session, workspace, err = _require_session(env)
    if err is not None:
        return err
    blocks, perr = _parse_aider_patches(patches or "")
    if perr is not None:
        return {"success": False, "error": perr}

    current = session.get("draft", "")
    new_source, aerr = _apply_patches_atomically(current, blocks)
    if aerr is not None:
        return {"success": False, "error": aerr}
    ok, verr, tool_names = _validate_source(new_source)
    if not ok:
        return {
            "success": False,
            "error": f"Patched source failed validation ({verr}). Nothing written.",
        }

    target = Path(session["toolset_folder"]) / _ENTRYPOINT
    try:
        target.write_text(new_source, encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to write {target}: {exc}"}

    session["draft"] = new_source
    session["last_write_at"] = _utc_now()
    session["write_count"] = int(session.get("write_count", 0)) + 1
    reload_result = _reload_personality(env)
    session["last_reload"] = reload_result
    _save_session(workspace, session)

    return {
        "success": True,
        "blocks_applied": len(blocks),
        "write_count": session["write_count"],
        "toolset_folder": session["toolset_folder"],
        "tools": tool_names,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
    }


def tool_edit_toolset_rewrite(
    new_source: str,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Replace the entire target TOOLS.py with new_source.

    Args:
        new_source: The complete new file content. Must parse and must define
            at least one top-level tool_* function.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, write_count, toolset_folder, tools, reloaded,
        reload_detail -- or an error dict on failure.
    """
    session, workspace, err = _require_session(env)
    if err is not None:
        return err
    ok, verr, tool_names = _validate_source(new_source or "")
    if not ok:
        return {
            "success": False,
            "error": f"Rewrite failed validation ({verr}). Nothing written.",
        }

    target = Path(session["toolset_folder"]) / _ENTRYPOINT
    try:
        target.write_text(new_source, encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to write {target}: {exc}"}

    session["draft"] = new_source
    session["last_write_at"] = _utc_now()
    session["write_count"] = int(session.get("write_count", 0)) + 1
    reload_result = _reload_personality(env)
    session["last_reload"] = reload_result
    _save_session(workspace, session)

    return {
        "success": True,
        "write_count": session["write_count"],
        "toolset_folder": session["toolset_folder"],
        "tools": tool_names,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
    }


def tool_edit_toolset_reload(env: Optional[Any] = None) -> Dict[str, Any]:
    """Explicitly rebuild the live tool binding from the target TOOLS.py.

    Normally patch/rewrite auto-reload; call this only when reload was
    deferred (no personality handle on env during the write) or after an
    external edit to TOOLS.py.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, reloaded, reload_detail.
    """
    session, workspace, err = _require_session(env)
    if err is not None:
        return err
    reload_result = _reload_personality(env)
    session["last_reload"] = reload_result
    _save_session(workspace, session)
    return {
        "success": True,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
    }


def tool_edit_toolset_test(
    tool_name: str,
    arguments_json: str = "{}",
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a tool_* function directly from the on-disk TOOLS.py.

    This runs the *current disk content* in a fresh import, so it is a true
    independent test (it does not rely on the live LCP binding having been
    reloaded). Use it to validate the edit before continuing the cycle.

    Args:
        tool_name: The tool_* callable name, e.g. "tool_count_rows".
        arguments_json: A JSON object of keyword arguments for the callable.
            Inspected at call time to inject env where the signature requires it.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, tool_name, output -- or an error dict on failure.
    """
    session, _workspace, err = _require_session(env)
    if err is not None:
        return err
    entry = Path(session["toolset_folder"]) / _ENTRYPOINT
    if not entry.is_file():
        return {"success": False, "error": f"Target entrypoint missing: {entry}"}

    try:
        raw = json.loads(arguments_json) if arguments_json else {}
    except ValueError as exc:
        return {"success": False, "error": f"arguments_json is not valid JSON: {exc}"}
    if not isinstance(raw, dict):
        return {
            "success": False,
            "error": "arguments_json must decode to a JSON object.",
        }

    module = _load_module_from_path(entry)
    if isinstance(module, dict):
        return {"success": False, "error": module["error"]}

    func = getattr(module, tool_name, None)
    if not callable(func):
        available = [
            n
            for n in _TOOL_FUNC_RE.findall(session.get("draft", ""))
            if n != tool_name
        ]
        return {
            "success": False,
            "error": (
                f"'{tool_name}' was not found as a callable in the target "
                f"TOOLS.py. Available tools: "
                f"{', '.join(sorted(available)) or '(none)'}."
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
    except Exception as exc:  # noqa: BLE001 - surface tool errors as data
        return {
            "success": False,
            "tool_name": tool_name,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"success": True, "tool_name": tool_name, "output": output}


def tool_edit_toolset_list(env: Optional[Any] = None) -> Dict[str, Any]:
    """List every editable toolset folder — workspace + externally mounted.

    Workspace-resident toolsets are listed first (``source: "workspace"``);
    then any additional toolset that is mounted on the live binding but lives
    outside the workspace (``source: "external"``). The latter is what makes
    the ``extra_tools`` wiring directly editable without copying the folder
    into the workspace first.

    Use this first when the target toolset's exact folder name is not known.
    Each entry provides ``toolset_name`` (folder name) and ``toolset_folder``
    (absolute path) — pass either to ``tool_edit_toolset_activate``.

    The editor's own folder is excluded from the result.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, workspace, count, workspace_count,
        external_count, candidates (a list of {"toolset_name",
        "toolset_folder", "relative_path", "source"} entries).
    """
    workspace = _workspace_root(env)
    workspace_candidates = _list_workspace_toolsets(workspace)
    external_candidates = _list_external_toolsets(env, workspace) if env else []
    candidates = workspace_candidates + external_candidates
    return {
        "success": True,
        "workspace": str(workspace),
        "count": len(candidates),
        "workspace_count": len(workspace_candidates),
        "external_count": len(external_candidates),
        "candidates": candidates,
        "active_session": bool(_load_session(workspace)),
        "next_step": (
            "Call tool_edit_toolset_activate(toolset_name=...) with one of the "
            "listed names (auto-activation also works when exactly one candidate "
            "exists, e.g. for tool_edit_toolset_read). External candidates are "
            "mount-on-the-live-binding toolsets living outside the workspace — "
            "they are editable directly."
        ),
    }


def tool_edit_toolset_status(env: Optional[Any] = None) -> Dict[str, Any]:
    """Return the current session state -- no side effects.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success and either {active: False} or {active: True, session}.
    """
    workspace = _workspace_root(env)
    session = _load_session(workspace)
    if session is None:
        return {"success": True, "active": False}
    return {
        "success": True,
        "active": True,
        "session": {
            "session_id": session.get("session_id"),
            "toolset_name": session.get("toolset_name"),
            "toolset_folder": session.get("toolset_folder"),
            "entrypoint": session.get("entrypoint"),
            "opened_at": session.get("opened_at"),
            "last_write_at": session.get("last_write_at"),
            "write_count": session.get("write_count", 0),
            "last_reload": session.get("last_reload"),
        },
    }


def tool_edit_toolset_close(env: Optional[Any] = None) -> Dict[str, Any]:
    """Delete the editing session file (the edited TOOLS.py stays untouched).

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success (and error on failure to remove).
    """
    workspace = _workspace_root(env)
    path = _session_path(workspace)
    removed = False
    try:
        if path.is_file():
            path.unlink()
            removed = True
    except OSError as exc:
        return {"success": False, "error": f"Failed to remove session: {exc}"}
    return {"success": True, "session_removed": removed}