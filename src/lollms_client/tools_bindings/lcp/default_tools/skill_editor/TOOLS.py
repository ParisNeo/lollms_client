"""skill_editor — in-place live editor for an LCP skill (SKILL.md).

Why this exists
---------------
The `skill_maker` RSI toolset provides an "act phase" (patch / revert / commit
/ test-agent). That is powerful but heavy: every tool is gated behind
`GATED_TASK_CATEGORY = "code_execution"`, and the RSI act-phase requires
`enable_skillmaker=True`. For a competent editor agent that simply needs to
iterate on a `SKILL.md` within a single turn, that machinery is more friction
than value — and worse, `skill_maker` offers no in-turn read→patch→validate
loop backed by a live `skills_manager.reload()`.

This toolset delivers the whole loop *inside* the agent turn::

    activate  ->  edit (patch | rewrite)  ->  auto-reload  ->  test  ->  repeat

Design mirrors `toolset_editor` exactly:
  * One target (a ``SKILL.md`` inside one skill folder) per editing session.
  * The session is mirrored to a JSON file under
    ``<workspace>/.skill_editor/session.json`` so state survives between
    calls / processes.
  * Every write runs an AST-free frontmatter validation (``name``,
    ``description``, non-trivial body) so a broken SKILL.md never reaches
    disk.
  * Every successful write attempts an immediate
    ``skills_manager.reload()`` via the hosting personality so the edited
    skill is visible to the same LCP binding that hosts this editor.

AUTO-BOOTSTRAP — you may not need ``tool_edit_skill_activate`` at all
---------------------------------------------------------------------
When any editing tool (read / patch / rewrite / reload / test / close) is
called without an active session, it AUTO-ACTIVATES the sole skill folder
found under the workspace's skills root. If exactly one candidate exists the
most common case becomes::

    tool_edit_skill_read()      # auto-activates; returns the SKILL.md source
    tool_edit_skill_patch(...)  # applies patch; auto-reloads

Zero or multiple candidates → a structured error lists them and tells the
caller how to disambiguate (``tool_edit_skill_list`` +
``tool_edit_skill_activate(skill_name=...)``).

Implementation note on marker literals
--------------------------------------
Aider patch markers are built at runtime from ``chr(60)`` / ``chr(61)`` /
``chr(62)`` runs so this source file never contains a line that begins with
the literal marker sequence. That keeps generic "apply-patch" tooling from
misinterpreting this file's own constants as a live diff.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

TOOL_LIBRARY_NAME = "Skill Editor"
TOOL_LIBRARY_DESC = (
    "In-place live editor for a single LCP skill: Aider patches, full "
    "rewrites, hot-reload via skills_manager, and direct skill validation."
)
TOOL_LIBRARY_ICON = "📝"

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

_SKILL_FILENAME = "SKILL.md"
_SESSION_DIRNAME = ".skill_editor"
_SESSION_FILENAME = "session.json"
_DEFAULT_SKILLS_SUBDIR = "skills"
_MAX_SOURCE_CHARS = 500_000
_MIN_BODY_CHARS = 30


def init_tools_library(config: Optional[dict] = None) -> None:
    """No global state to initialize. Sessions are per-workspace on disk."""
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


def _list_workspace_skills(workspace: Path) -> List[Dict[str, Any]]:
    """Enumerate every skill folder (containing SKILL.md) under the workspace.

    Bounded to a reasonable depth. Excludes our own ``.skill_editor`` session
    dir and any HIDDEN folder RELATIVE to the workspace (the workspace itself
    may sit under a dot-prefixed ancestor, so absolute parts are never
    inspected).
    """
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    try:
        for entry in workspace.rglob(_SKILL_FILENAME):
            folder = entry.parent
            resolved = str(folder.resolve())
            if resolved in seen:
                continue
            # Skip our own session dir.
            if folder.name == _SESSION_DIRNAME:
                continue
            # Skip HIDDEN folders RELATIVE to the workspace only.
            try:
                rel = folder.relative_to(workspace)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            seen.add(resolved)
            candidates.append({
                "skill_name": folder.name,
                "skill_folder": resolved,
                "relative_path": str(rel).replace("\\", "/"),
                "source": "workspace",
            })
    except (OSError, ValueError):
        pass
    candidates.sort(key=lambda c: c["relative_path"])
    return candidates

def _list_external_skills(env: Any, workspace: Path) -> List[Dict[str, Any]]:
    """Enumerate skills registered on the live SkillsManager but OUTSIDE the workspace.

    This is the "extra_skills_dirs exception": a skill wired through
    ``extra_skills_dirs`` (or the handbag's own ``skills/`` folder) is a
    legitimate editing target **even when its folder lives outside the
    workspace**. Mirroring ``toolset_editor._list_external_toolsets``, the
    live binding is the authoritative source: we duck-type to the hosting
    personality's ``skills_manager`` and read ``list_skills()``.

    Excludes:
      * skills already covered by ``_list_workspace_skills`` (dedup),
      * our own ``.skill_editor`` session dir (guard, even if one slipped in),
      * any entry whose ``file_path`` is not a ``SKILL.md``.

    ``relative_path`` is set to the absolute folder because there is no
    workspace-relative form for an external folder.
    """
    candidates: List[Dict[str, Any]] = []
    personality = _resolve_personality(env)
    if personality is None:
        return candidates
    mgr = getattr(personality, "skills_manager", None)
    if mgr is None:
        return candidates
    try:
        entries = mgr.list_skills() or []
    except Exception:
        return candidates

    workspace_set = {
        c["skill_folder"] for c in _list_workspace_skills(workspace)
    }
    seen: set = set()
    for entry in entries:
        fp = _skill_entry_file_path(entry)
        if not fp:
            continue
        try:
            p = Path(fp).resolve()
        except (OSError, ValueError):
            continue
        if p.name != _SKILL_FILENAME:
            continue
        folder = p.parent
        if folder.name == _SESSION_DIRNAME:
            continue
        resolved = str(folder)
        if resolved in workspace_set or resolved in seen:
            continue
        seen.add(resolved)
        candidates.append({
            "skill_name": folder.name,
            "skill_folder": resolved,
            "relative_path": resolved,
            "source": "external",
        })
    candidates.sort(key=lambda c: c["skill_name"])
    return candidates

def _list_candidate_skills(
    workspace: Path, env: Any = None,
) -> List[Dict[str, Any]]:
    """Enumerate editing candidates: workspace-first, then external.

    Workspace-resident skill folders take precedence because they are the
    caller's explicit deliverable location. External skills — those registered
    on the live ``skills_manager`` but living outside the workspace — are
    appended so an editor persona wired with ``extra_skills_dirs=[ROOT]`` can
    edit ``ROOT/<slug>/SKILL.md`` directly without copying it into the
    workspace.
    """
    candidates = _list_workspace_skills(workspace)
    if env is not None:
        candidates.extend(_list_external_skills(env, workspace))
    return candidates

def _try_auto_activate(
    workspace: Path,
    env: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Bootstrap a session against the sole candidate skill.

    Candidates are workspace skill folders first, then external skills
    registered on the live ``skills_manager``. If exactly ONE candidate
    exists, auto-activate it. Zero or multiple candidates → structured error
    listing them.
    """
    candidates = _list_candidate_skills(workspace, env)
    if not candidates:
        return None, (
            f"No skill folder containing {_SKILL_FILENAME} was found under the "
            f"workspace ({workspace}) and no skill is registered on the live "
            "skills_manager. Either the target skill is outside the workspace "
            "and not registered, or it is missing SKILL.md. Pass skill_folder "
            "explicitly in tool_edit_skill_activate(skill_folder=...)."
        )
    if len(candidates) > 1:
        listed = ", ".join(
            f"'{c['skill_name']}' ({c['relative_path']}, {c['source']})"
            for c in candidates
        )
        return None, (
            f"{len(candidates)} skill candidates were found; auto-activation is "
            f"ambiguous. Call tool_edit_skill_activate(skill_name=..., "
            f"skill_folder=...) with one of: {listed}."
        )
    target = candidates[0]
    result = tool_edit_skill_activate(
        skill_name=target["skill_name"],
        skill_folder=target["skill_folder"],
        env=env,
    )
    if not result.get("success"):
        return None, result.get("error") or "Auto-activation failed."
    return _load_session(workspace), None


def _require_session(
    env: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[Dict[str, Any]]]:
    """Returns (session, workspace, error_dict).

    Auto-bootstraps against the sole skill folder in the workspace when no
    session exists. Zero or multiple candidates → structured error listing.
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

    Duck-typed: look for any object exposing a callable ``skills_manager``
    with a ``reload()`` method (that is the definitive skill-reload handle).
    ``refresh_tools_and_skills()`` is accepted too — it is the canonical
    public entry point on the personality.
    """
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
        mgr = getattr(candidate, "skills_manager", None)
        if mgr is not None and callable(getattr(mgr, "reload", None)):
            return candidate
    return None


def _reload_skills(env: Any) -> Dict[str, Any]:
    """Attempt a live skill reload against the hosting personality."""
    personality = _resolve_personality(env)
    if personality is None:
        return {
            "reloaded": False,
            "reason": (
                "No personality handle exposing refresh_tools_and_skills() or "
                "a skills_manager with reload() was found on env. The caller "
                "must call skills_manager.reload() manually before the edited "
                "skill is visible."
            ),
        }
    try:
        # Prefer the narrow skills_manager.reload() when available. It is
        # what actually re-scans skills roots and is safe to call without
        # rebuilding the tool binding (which would drop session_flags).
        mgr = getattr(personality, "skills_manager", None)
        if mgr is not None and callable(getattr(mgr, "reload", None)):
            mgr.reload()
            return {"reloaded": True, "at": _utc_now(), "via": "skills_manager.reload"}
        # Fallback to the broader personality entry point.
        aware = getattr(personality, "refresh_tools_and_skills", None)
        if callable(aware):
            aware()
            return {"reloaded": True, "at": _utc_now(), "via": "refresh_tools_and_skills"}
    except Exception as exc:  # noqa: BLE001 - surface reload failures as data
        return {"reloaded": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"reloaded": False, "reason": "Personality exposes no reload hook."}


# ---------------------------------------------------------------------------
# Registered / loaded status resolution
# ---------------------------------------------------------------------------
#
# Three distinct states a SKILL.md can be in, all of which the LLM conflates
# when the user says "load that skill into memory":
#
#   in_tree     — a SKILL.md file exists on disk at a path the caller can name.
#   registered  — the file is inside a skills root the personality's
#                 SkillsManager has scanned (i.e. tool_load_skill would see it).
#   loaded      — the skill is promoted into the === ACTIVE SKILLS === prompt
#                 section (i.e. it is currently in the session's context).
#
# `tool_edit_skill_read` returns SOURCE TEXT; it does not perform any of the
# three transitions. These helpers make the state explicit so the agent can
# stop guessing which verb to use.

def _skill_entry_file_path(entry: Any) -> Optional[str]:
    """Extract the absolute SKILL.md path from either a dict or a Skill-like object."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        fp = entry.get("file_path")
        return str(fp) if fp else None
    fp = getattr(entry, "file_path", None)
    return str(fp) if fp else None


def _skill_entry_visibility(entry: Any) -> Optional[str]:
    """Extract the visibility state ('loadable'|'searchable'|'visible') from an entry."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        # SkillsManager.list_skills() emits dicts shaped like Skill.to_dict().
        return entry.get("visibility") or entry.get("status") or None
    return getattr(entry, "visibility", None)


def _iter_manager_skills(mgr: Any) -> List[Any]:
    """Return every skill entry the manager knows about, safely."""
    if mgr is None:
        return []
    try:
        entries = mgr.list_skills()
    except Exception:
        return []
    return list(entries or [])


def _resolve_skill_status(env: Any, target_path: Optional[Path]) -> Dict[str, Any]:
    """Return {in_tree, registered, registered_path, loaded, guidance}.

    ``target_path`` is the SKILL.md path the caller is operating on. When
    it is ``None`` the caller did not yet resolve a target — we report
    ``in_tree=False`` and skip manager lookups.

    ``registered`` is True when the target path is present in
    ``skills_manager.list_skills()`` (path-resolved match).
    ``loaded`` is True when that same entry has visibility == "loadable"
    (i.e. it is inside the === ACTIVE SKILLS === section on the next turn).
    """
    status: Dict[str, Any] = {
        "in_tree": False,
        "registered": False,
        "registered_path": None,
        "loaded": False,
        "guidance": None,
    }
    if target_path is None:
        status["guidance"] = (
            "No target SKILL.md was resolved. Pass skill_name or skill_folder, "
            "or place exactly one SKILL.md under the workspace and rely on "
            "auto-activation."
        )
        return status

    try:
        resolved = target_path.resolve()
    except (OSError, ValueError):
        resolved = target_path
    status["in_tree"] = resolved.is_file()
    status["registered_path"] = str(resolved)

    personality = _resolve_personality(env)
    mgr = getattr(personality, "skills_manager", None) if personality is not None else None
    if mgr is None:
        status["guidance"] = _status_guidance(status, manager_found=False)
        return status

    for entry in _iter_manager_skills(mgr):
        fp = _skill_entry_file_path(entry)
        if not fp:
            continue
        try:
            entry_path = Path(fp).resolve()
        except (OSError, ValueError):
            continue
        if entry_path == resolved:
            status["registered"] = True
            status["loaded"] = _skill_entry_visibility(entry) == "loadable"
            break

    status["guidance"] = _status_guidance(status, manager_found=True)
    return status


def _status_guidance(status: Dict[str, Any], *, manager_found: bool) -> str:
    """Produce a one-line action hint matching the observed state."""
    if not status["in_tree"]:
        return (
            "This SKILL.md does not exist on disk. Check the path, or use "
            "tool_edit_skill_list to see what skills are visible."
        )
    if not manager_found:
        return (
            "The file exists, but no SkillsManager is attached to env, so its "
            "'registered' state cannot be determined. The source can still be "
            "read and edited."
        )
    if status["registered"] and status["loaded"]:
        return (
            "Registered AND loaded. It is currently in ACTIVE SKILLS — its "
            "instructions already reach the LLM every turn. To stop including "
            "it, call tool_unload_skill(title=...)."
        )
    if status["registered"] and not status["loaded"]:
        return (
            "Registered but NOT loaded. It is discoverable via tool_load_skill "
            "but is not in the prompt. To activate it, call "
            "tool_load_skill(title=...). Reading its source with this tool does "
            "NOT load it."
        )
    return (
        "Not registered with the personality's SkillsManager. tool_load_skill "
        "will not find it. Either (a) place the skill under one of the "
        "personality's skills roots (handbag/skills or an extra_skills_dirs "
        "entry) and reload, or (b) treat it as a plain editable source file — "
        "editing it here is fine, but no activation is available."
    )


# ---------------------------------------------------------------------------
# Skill path + source validation
# ---------------------------------------------------------------------------

def _resolve_skill_folder(
    workspace: Path,
    skill_name: Optional[str],
    skill_folder: Optional[str],
    env: Any = None,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Return the folder containing the target SKILL.md, or an error dict.

    Resolution order:
      1. ``skill_folder`` when given (absolute, or workspace-relative).
      2. ``skill_name`` searched in the workspace.
      3. ``skill_name`` searched in the live ``skills_manager`` (the
         ``extra_skills_dirs`` exception — skills whose folder is outside
         the workspace but registered on the personality).
    """
    if skill_folder:
        raw = skill_folder.strip()
        if raw:
            try:
                folder = Path(raw)
                if not folder.is_absolute():
                    folder = (workspace / folder).resolve()
                else:
                    folder = folder.resolve()
            except (OSError, ValueError) as exc:
                return None, {"success": False, "error": f"Invalid skill_folder: {exc}"}
            if not folder.is_dir():
                return None, {
                    "success": False,
                    "error": f"skill_folder is not a directory: {folder}",
                }
            return folder, None
    if not skill_name:
        return None, {
            "success": False,
            "error": "Provide either skill_folder or skill_name.",
        }
    target = skill_name.strip()

    matches: List[Path] = []
    for candidate in workspace.rglob(_SKILL_FILENAME):
        if candidate.parent.name == target:
            matches.append(candidate.parent)
    if not matches and env is not None:
        # Fall back to the live skills_manager (extra_skills_dirs) — skill
        # folders outside the workspace are legitimate targets when they are
        # registered on the personality.
        for candidate in _list_external_skills(env, workspace):
            if candidate["skill_name"] == target:
                try:
                    matches.append(Path(candidate["skill_folder"]).resolve())
                except (OSError, ValueError):
                    continue
    if not matches:
        return None, {
            "success": False,
            "error": (
                f"No folder named '{target}' containing {_SKILL_FILENAME} was "
                f"found under the workspace ({workspace}) or on the live "
                "skills_manager. Check the name, or pass skill_folder explicitly."
            ),
        }
    if len(matches) > 1:
        listed = ", ".join(str(p) for p in matches)
        return None, {
            "success": False,
            "error": (
                f"Ambiguous skill_name '{target}': multiple matches ({listed}). "
                "Pass skill_folder explicitly."
            ),
        }
    return matches[0], None


def _parse_frontmatter(source: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract YAML frontmatter between leading '---' markers.

    Returns (frontmatter_dict, error_message). The dict always includes the
    ``_body`` key holding the post-frontmatter body (used for body length
    checks). PyYAML is optional: a minimal ``key: value`` fallback covers the
    common case when it is not installed.
    """
    if not source.startswith("---"):
        return None, "SKILL.md must start with a YAML frontmatter block (--- ... ---)."
    try:
        end = source.index("\n---", 3)
    except ValueError:
        return None, "SKILL.md frontmatter block was not closed with '---'."
    raw_fm = source[3:end].strip()
    body = source[end + 4:]
    if not raw_fm:
        return None, "SKILL.md frontmatter block is empty."

    fm: Dict[str, Any] = {}
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(raw_fm) or {}
        if isinstance(loaded, dict):
            fm = loaded
    except Exception:
        pass
    if not fm:
        # Pragmatic fallback: only simple ``key: value`` lines.
        for line in raw_fm.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                fm[key.strip()] = value.strip().strip('"').strip("'")
    if not fm:
        return None, "Could not parse frontmatter as YAML or simple key: value pairs."
    fm["_body"] = body
    return fm, None


# ── Skill self-protection (author opt-out) ──
# A skill opts out of agent-driven edits by declaring EITHER:
#   * in its frontmatter:  SKILL_LIBRARY_READONLY: true
#   * in its body:         SKILL_LIBRARY_READONLY = True
# The detection is a plain regex over both regions; a false is the default.
_READONLY_FM_RE = re.compile(
    r"^SKILL_LIBRARY_READONLY\s*:\s*(?:true|yes|on|1)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_READONLY_BODY_RE = re.compile(
    r"^\s*SKILL_LIBRARY_READONLY\s*=\s*(?:True|1|'true'|\"true\"|'True'|\"True\")\s*$",
    re.MULTILINE,
)


declared: List[str] = []
raw_tools = fm.get("tools")
if isinstance(raw_tools, list):
    declared = [str(x).strip() for x in raw_tools if x]
elif isinstance(raw_tools, str):
    declared = [t.strip() for t in re.split(r"[,\s]+", raw_tools) if t.strip()]
return True, "", declared

# ---------------------------------------------------------------------------
# Skill self-protection (author opt-out)
# ---------------------------------------------------------------------------
# A skill opts out of agent-driven edits by declaring EITHER:
#
#   * ``modifiable: false`` in its frontmatter - the contract already
#     published by ``SkillsManager`` (update_skill, append_to_skill,
#     remove_skill, set_skill_visibility) and produced automatically when
#     a skill loads without a ``modifiable:`` key. This is the canonical,
#     project-wide flag; skill_editor honours it here so direct-to-disk
#     writes cannot bypass the manager's own read-only gate.
#
#   * ``SKILL_LIBRARY_READONLY: true`` in frontmatter, or
#     ``SKILL_LIBRARY_READONLY = True`` in the body - the explicit,
#     toolset-style opt-out for authors who want an unmistakable marker in
#     the file (mirrors TOOL_LIBRARY_READONLY in toolset_staging).
#
# The check is re-run at WRITE time so a protection added after activation
# still blocks a patch.
_READONLY_FM_RE = re.compile(
r"^SKILL_LIBRARY_READONLY\s*:\s*(?:true|yes|on|1)\s*$",
re.IGNORECASE | re.MULTILINE,
)
_READONLY_BODY_RE = re.compile(
r"^\s*SKILL_LIBRARY_READONLY\s*=\s*(?:True|1|'true'|\"true\"|'True'|\"True\")\s*$",
re.MULTILINE,
)
_MODIFIABLE_FM_RE = re.compile(
r"^modifiable\s*:\s*(?:false|no|off|0)\s*$",
re.IGNORECASE | re.MULTILINE,
)


def _is_skill_protected(skill_folder: Union[str, Path]) -> bool:
"""Return True when a skill's SKILL.md declares itself read-only.

Detects BOTH the canonical SkillsManager contract (``modifiable: false``,
absent-flag => read-only at the manager layer) and the explicit toolset-
style opt-out (``SKILL_LIBRARY_READONLY``). skill_editor writes to disk
directly, bypassing the manager's own gate, so this check is the ONLY
thing standing between the agent and an edit to a read-only skill.

This is an author intent declaration, NOT a security boundary - a tool
with filesystem access can bypass it. It exists to prevent well-meaning
but destructive agent edits, not to defeat an adversarial tool.

Args:
    skill_folder: Skill folder, or the SKILL.md path itself.

Returns:
    True when either flag is present, when the file is missing, or when
    the frontmatter omits ``modifiable:`` (absent => read-only, matching
    SkillsManager). False ONLY when ``modifiable:`` is explicitly truthy
    and ``SKILL_LIBRARY_READONLY`` is absent.
"""
try:
    entry = Path(skill_folder)
    if entry.is_dir():
        entry = entry / _SKILL_FILENAME
    if not entry.is_file():
        return False
    text = entry.read_text(encoding="utf-8", errors="ignore")
except (OSError, ValueError):
    return False

if _READONLY_BODY_RE.search(text):
    return True
if _READONLY_FM_RE.search(text):
    return True
if _MODIFIABLE_FM_RE.search(text):
    return True

# Canonical SkillsManager default: absent ``modifiable:`` => read-only.
# Only treat as read-only when a frontmatter block actually exists - a
# free-form markdown file with no frontmatter is legacy and predates the
# contract, so we do not refuse it here (display-only skills do not need
# to be explicitly writable).
if text.startswith("---"):
    fm_match = re.match(r"^---\s*?\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if fm_match:
        has_modifiable = re.search(
            r"^\s*modifiable\s*:", fm_match.group(1), re.MULTILINE
        )
        if not has_modifiable:
            return True
return False

# ---------------------------------------------------------------------------
# Skill self-protection (author opt-out)
# ---------------------------------------------------------------------------
# A skill opts out of agent-driven edits by declaring EITHER:
#
#   * ``modifiable: false`` in its frontmatter — the contract already
#     published by ``SkillsManager`` (_reconcile_skill_metadata,
#     update_skill, append_to_skill, remove_skill, set_skill_visibility)
#     and produced automatically when a skill loads without a
#     ``modifiable:`` key. This is the canonical, project-wide flag;
#     skill_editor honours it here so direct-to-disk writes cannot
#     bypass the manager's own read-only gate.
#
#   * ``SKILL_LIBRARY_READONLY: true`` in frontmatter, or
#     ``SKILL_LIBRARY_READONLY = True`` in the body — the explicit,
#     toolset-style opt-out for authors who want an unmistakable marker
#     in the file (mirrors TOOL_LIBRARY_READONLY in toolset_staging).
#
# The check is re-run at WRITE time so a protection added after
# activation still blocks a patch.
_READONLY_FM_RE = re.compile(
r"^SKILL_LIBRARY_READONLY\s*:\s*(?:true|yes|on|1)\s*$",
re.IGNORECASE | re.MULTILINE,
)
_READONLY_BODY_RE = re.compile(
r"^\s*SKILL_LIBRARY_READONLY\s*=\s*(?:True|1|'true'|\"true\"|'True'|\"True\")\s*$",
re.MULTILINE,
)
_MODIFIABLE_FM_RE = re.compile(
r"^modifiable\s*:\s*(?:false|no|off|0)\s*$",
re.IGNORECASE | re.MULTILINE,
)


def _is_skill_protected(skill_folder: Union[str, Path]) -> bool:
"""Return True when a skill's SKILL.md declares itself read-only.

Detects BOTH the canonical SkillsManager contract (``modifiable: false``,
absent-flag ⇒ read-only at the manager layer) and the explicit toolset-
style opt-out (``SKILL_LIBRARY_READONLY``). skill_editor writes to disk
directly, bypassing the manager's own gate, so this check is the ONLY
thing standing between the agent and an edit to a read-only skill.

This is an author intent declaration, NOT a security boundary — a tool
with filesystem access can bypass it. It exists to prevent well-meaning
but destructive agent edits, not to defeat an adversarial tool.

Args:
    skill_folder: Skill folder, or the SKILL.md path itself.

Returns:
    True when either flag is present, when the file is missing, or when
    the frontmatter omits ``modifiable:`` (absent ⇒ read-only, matching
    SkillsManager). False ONLY when ``modifiable:`` is explicitly truthy
    and ``SKILL_LIBRARY_READONLY`` is absent.
"""
try:
    entry = Path(skill_folder)
    if entry.is_dir():
        entry = entry / _SKILL_FILENAME
    if not entry.is_file():
        return False
    text = entry.read_text(encoding="utf-8", errors="ignore")
except (OSError, ValueError):
    return False

if _READONLY_BODY_RE.search(text):
    return True
if _READONLY_FM_RE.search(text):
    return True
if _MODIFIABLE_FM_RE.search(text):
    return True

# Canonical SkillsManager default: absent ``modifiable:`` ⇒ read-only.
# Only treat as read-only when a frontmatter block actually exists — a
# free-form markdown file with no frontmatter is legacy and predates the
# contract, so we do not refuse it here (display-only skills do not need
# to be explicitly writable).
if text.startswith("---"):
    fm_match = re.match(r"^---\s*?\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if fm_match:
        has_modifiable = re.search(
            r"^\s*modifiable\s*:", fm_match.group(1), re.MULTILINE
        )
        if not has_modifiable:
            return True
return False
def _validate_source(source: str) -> Tuple[bool, str, List[str]]:
    ok, msg = _validate_skill_body(source)
    return ok, msg, []
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
                "use tool_edit_skill_rewrite for full-source replacement."
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


# ---------------------------------------------------------------------------
# Public tool API
# ---------------------------------------------------------------------------

def tool_edit_skill_list(env: Optional[Any] = None) -> Dict[str, Any]:
    """List every editable skill folder — workspace + externally registered.

    Workspace-resident skill folders are listed first (``source: "workspace"``);
    then any additional skill that is registered on the live ``skills_manager``
    but lives outside the workspace (``source: "external"``). The latter is
    what makes the ``extra_skills_dirs`` wiring directly editable without
    copying the skill folder into the workspace first.

    Use this first when the target skill's exact folder name is not known.
    Each entry provides ``skill_name`` (folder name) and ``skill_folder``
    (absolute path) — pass either to ``tool_edit_skill_activate``.

    The editor's own session directory is excluded from the result.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, workspace, count, workspace_count,
        external_count, candidates, active_session.
    """
    workspace = _workspace_root(env)
    workspace_candidates = _list_workspace_skills(workspace)
    external_candidates = _list_external_skills(env, workspace) if env else []
    candidates = workspace_candidates + external_candidates

    # ── Read-only stamping ──
    # Protected skills are still LISTED (so the agent understands why a
    # candidate cannot be activated) AND surfaced under `protected` for
    # easy filtering. `editable_count` excludes them so the agent can see
    # at a glance how many candidates it may actually edit.
    protected_names: List[str] = []
    for cand in candidates:
        try:
            is_protected = _is_skill_protected(cand.get("skill_folder", ""))
        except Exception:
            is_protected = False
        cand["protected"] = is_protected
        if is_protected:
            protected_names.append(cand["skill_name"])

    editable_count = sum(1 for c in candidates if not c.get("protected"))
    return {
        "success": True,
        "workspace": str(workspace),
        "count": len(candidates),
        "editable_count": editable_count,
        "workspace_count": len(workspace_candidates),
        "external_count": len(external_candidates),
        "candidates": candidates,
        "protected": protected_names,
        "active_session": bool(_load_session(workspace)),
        "next_step": (
            "Call tool_edit_skill_activate(skill_name=...) with one of the "
            "listed names (auto-activation also works when exactly one "
            "candidate exists, e.g. for tool_edit_skill_read). Candidates "
            "marked `protected: true` decline editing by author declaration "
            "(`modifiable: false` or `SKILL_LIBRARY_READONLY`) and will be "
            "refused by tool_edit_skill_activate."
        ),
    }


def tool_edit_skill_activate(
    skill_name: str,
    skill_folder: Optional[str] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Open a live editing session against one target skill folder.

    Loads the target's SKILL.md into the session's editing area and pins the
    folder. Every subsequent patch/rewrite writes to this exact SKILL.md.

    Args:
        skill_name: Logical slug (folder name). Used to disambiguate when
            skill_folder is omitted.
        skill_folder: Explicit path to the skill folder (absolute, or
            relative to the workspace root).
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, session_id, skill_name, skill_folder, entrypoint
        (always SKILL.md), declared_tools, reloaded, reload_detail --
        or an error dict on failure.
    """
    workspace = _workspace_root(env)
    folder, err = _resolve_skill_folder(
        workspace, skill_name, skill_folder, env=env
    )
    if err is not None:
        return err
    entry = folder / _SKILL_FILENAME
    if not entry.is_file():
        return {
            "success": False,
            "error": f"Target folder does not contain {_SKILL_FILENAME}: {folder}",
        }

    # ── Self-protection: refuse to activate a read-only skill ──
    if _is_skill_protected(folder):
        return {
            "success": False,
            "protected": True,
            "skill_name": folder.name,
            "skill_folder": str(folder),
            "error": (
                f"'{folder.name}' is READ-ONLY. Its {_SKILL_FILENAME} declares "
                "`modifiable: false` or `SKILL_LIBRARY_READONLY`, so it is "
                "protected against agent-driven edits. The protection must be "
                "removed from the source by a human operator — this is not "
                "something an agent may do."
            ),
        }

    try:
        source = entry.read_text(encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to read {entry}: {exc}"}
    ok, verr, declared_tools = _validate_source(source)
    if not ok:
        return {
            "success": False,
            "error": (
                f"Existing SKILL.md failed validation ({verr}). Fix it manually "
                "before activating an editing session."
            ),
        }

    session = {
        "session_id": uuid.uuid4().hex,
        "skill_name": folder.name,
        "skill_folder": str(folder),
        "entrypoint": _SKILL_FILENAME,
        "opened_at": _utc_now(),
        "last_write_at": None,
        "last_reload": None,
        "write_count": 0,
        "draft": source,
    }
    _save_session(workspace, session)

    reload_result = _reload_skills(env)
    if reload_result.get("reloaded"):
        session["last_reload"] = reload_result
        _save_session(workspace, session)

    status = _resolve_skill_status(env, entry)
    return {
        "success": True,
        "session_id": session["session_id"],
        "skill_name": session["skill_name"],
        "skill_folder": session["skill_folder"],
        "entrypoint": session["entrypoint"],
        "declared_tools": declared_tools,
        "in_tree": status["in_tree"],
        "registered": status["registered"],
        "loaded": status["loaded"],
        "load_status_guidance": status["guidance"],
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
        "next_step": (
            "Use tool_edit_skill_patch (Aider blocks) or "
            "tool_edit_skill_rewrite (full source). Each successful write "
            "auto-validates, writes to disk, and attempts a live reload."
        ),
    }


def tool_edit_skill_read(env: Optional[Any] = None) -> Dict[str, Any]:
    """Return the FULL source of the target SKILL.md -- no file tools required.

    This is the canonical way for an editor agent to inspect the skill source.
    It does NOT rely on ``workspace_tools``, a sub-agent, or any other
    capability: the returned ``source`` field IS the complete file content,
    ready to be reasoned about and patched.

    When called with no active session it auto-activates the sole skill
    folder found in the workspace.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, skill_name, skill_folder, entrypoint, valid,
        validation_error, declared_tools, source -- or an error dict on
        failure.
    """
    session, _workspace, err = _require_session(env)
    if err is not None:
        return err
    source = session.get("draft", "")
    ok, verr, declared_tools = _validate_source(source)

    target = Path(session["skill_folder"]) / _SKILL_FILENAME
    status = _resolve_skill_status(env, target)

    return {
        "success": True,
        "skill_name": session.get("skill_name"),
        "skill_folder": session.get("skill_folder"),
        "entrypoint": session.get("entrypoint"),
        "valid": ok,
        "validation_error": verr or None,
        "declared_tools": declared_tools,
        "in_tree": status["in_tree"],
        "registered": status["registered"],
        "loaded": status["loaded"],
        "load_status_guidance": status["guidance"],
        "source": source,
    }


def tool_edit_skill_load_status(
    skill_name_or_path: str = "",
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Report exactly *which* of the three "skill" states the target is in.

    This closes the ambiguity that arises whenever a user says "load that skill
    into memory": reading its source is not the same as registering it, and
    registering it is not the same as promoting it into the active prompt.

    States:
      * ``in_tree``    — a SKILL.md exists at the resolved absolute path.
      * ``registered`` — the file is inside a skills root that the personality's
                        SkillsManager has scanned (so ``tool_load_skill`` can
                        find it).
      * ``loaded``     — the skill is currently promoted into the
                        ``=== ACTIVE SKILLS ===`` section (its instructions
                        reach the LLM every turn).

    The returned ``guidance`` field is a one-line, actionable next step that
    tells the caller what tool to call for their intent.

    Args:
        skill_name_or_path: Skill slug, title, or SKILL.md path. Empty
            (default) → resolve against the current session, auto-activating
            the sole workspace skill if no session exists.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, skill_name, skill_folder, path, in_tree,
        registered, loaded, guidance.
    """
    workspace = _workspace_root(env)
    target: Optional[Path] = None

    # Prefer the active editing session (its target may differ from the sole
    # workspace candidate if the user explicitly activated a specific skill).
    session = _load_session(workspace)
    if session is not None and not (skill_name_or_path or "").strip():
        try:
            target = Path(session["skill_folder"]) / _SKILL_FILENAME
        except (KeyError, OSError, ValueError):
            target = None

    if target is None:
        needle = (skill_name_or_path or "").strip()
        folder: Optional[Path] = None
        if needle:
            # Try the needle as a folder path first, then as a slug.
            folder, resolve_err = _resolve_skill_folder(
                workspace, None, needle, env=env
            )
            if folder is None:
                folder, resolve_err = _resolve_skill_folder(
                    workspace, needle, None, env=env
                )
                if folder is None:
                    return {
                        "success": False,
                        "error": resolve_err.get("error") if resolve_err else (
                            f"Could not resolve '{needle}'."
                        ),
                        "workspace": str(workspace),
                    }
        else:
            candidates = _list_candidate_skills(workspace, env)
            if not candidates:
                return {
                    "success": True,
                    "workspace": str(workspace),
                    "path": None,
                    "in_tree": False,
                    "registered": False,
                    "loaded": False,
                    "guidance": (
                        "No SKILL.md was found under the workspace. Place target "
                        "at <workspace>/<slug>/SKILL.md or pass skill_name_or_path."
                    ),
                }
            if len(candidates) > 1:
                listed = ", ".join(
                    f"'{c['skill_name']}' ({c['relative_path']})" for c in candidates
                )
                return {
                    "success": True,
                    "workspace": str(workspace),
                    "path": None,
                    "in_tree": False,
                    "registered": False,
                    "loaded": False,
                    "guidance": (
                        f"{len(candidates)} skills visible; pass skill_name_or_path "
                        f"to disambiguate: {listed}."
                    ),
                }
            folder = Path(candidates[0]["skill_folder"])
        target = folder / _SKILL_FILENAME

    status = _resolve_skill_status(env, target)
    return {
        "success": True,
        "workspace": str(workspace),
        "path": status["registered_path"],
        "in_tree": status["in_tree"],
        "registered": status["registered"],
        "loaded": status["loaded"],
        "guidance": status["guidance"],
    }


def tool_edit_skill_patch(
    patches: str,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply one or more Aider SEARCH/REPLACE blocks to the target SKILL.md.

    All blocks are applied atomically (all succeed -> one write -> one reload;
    any failure -> no change). After a successful write the session attempts a
    live reload via ``skills_manager.reload()`` so the new SKILL.md content is
    visible immediately.

    Args:
        patches: One or more Aider blocks of the form
            SEARCH marker / old exact lines / separator / new lines / REPLACE
            marker. The markers are the standard 7-character runs.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, blocks_applied, write_count, skill_folder,
        declared_tools, reloaded, reload_detail -- or an error dict on
        failure (nothing is written when this returns an error).
    """
    session, workspace, err = _require_session(env)
    if err is not None:
        return err

    # Re-check protection at WRITE time. The flag may have been added to the
    # source after the skill was activated (or after the last successful
    # write) — a patch must honour it.
    if _is_skill_protected(session["skill_folder"]):
        return {
            "success": False,
            "protected": True,
            "skill_name": session.get("skill_name"),
            "skill_folder": session.get("skill_folder"),
            "error": (
                f"'{session.get('skill_name')}' is READ-ONLY "
                f"(`modifiable: false` or `SKILL_LIBRARY_READONLY` in its "
                f"{_SKILL_FILENAME}). Nothing was written."
            ),
        }

    blocks, perr = _parse_aider_patches(patches or "")
    if perr is not None:
        return {"success": False, "error": perr}

    current = session.get("draft", "")
    new_source, aerr = _apply_patches_atomically(current, blocks)
    if aerr is not None:
        return {"success": False, "error": aerr}
    ok, verr, declared_tools = _validate_source(new_source)
    if not ok:
        return {
            "success": False,
            "error": f"Patched source failed validation ({verr}). Nothing written.",
        }

    target = Path(session["skill_folder"]) / _SKILL_FILENAME
    try:
        target.write_text(new_source, encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to write {target}: {exc}"}

    session["draft"] = new_source
    session["last_write_at"] = _utc_now()
    session["write_count"] = int(session.get("write_count", 0)) + 1
    reload_result = _reload_skills(env)
    session["last_reload"] = reload_result
    _save_session(workspace, session)

    return {
        "success": True,
        "blocks_applied": len(blocks),
        "write_count": session["write_count"],
        "skill_folder": session["skill_folder"],
        "declared_tools": declared_tools,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
    }


def tool_edit_skill_rewrite(
    new_source: str,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Replace the entire target SKILL.md with new_source.

    Args:
        new_source: The complete new file content. Must start with a valid
            frontmatter block (name, description), and must contain a
            non-trivial body.
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, write_count, skill_folder, declared_tools,
        reloaded, reload_detail -- or an error dict on failure.
    """
    session, workspace, err = _require_session(env)
    if err is not None:
        return err

    # Re-check protection at WRITE time (see tool_edit_skill_patch).
    if _is_skill_protected(session["skill_folder"]):
        return {
            "success": False,
            "protected": True,
            "skill_name": session.get("skill_name"),
            "skill_folder": session.get("skill_folder"),
            "error": (
                f"'{session.get('skill_name')}' is READ-ONLY "
                f"(`modifiable: false` or `SKILL_LIBRARY_READONLY` in its "
                f"{_SKILL_FILENAME}). Nothing was written."
            ),
        }

    ok, verr, declared_tools = _validate_source(new_source or "")
    if not ok:
        return {
            "success": False,
            "error": f"Rewrite failed validation ({verr}). Nothing written.",
        }

    target = Path(session["skill_folder"]) / _SKILL_FILENAME
    try:
        target.write_text(new_source, encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to write {target}: {exc}"}

    session["draft"] = new_source
    session["last_write_at"] = _utc_now()
    session["write_count"] = int(session.get("write_count", 0)) + 1
    reload_result = _reload_skills(env)
    session["last_reload"] = reload_result
    _save_session(workspace, session)

    return {
        "success": True,
        "write_count": session["write_count"],
        "skill_folder": session["skill_folder"],
        "declared_tools": declared_tools,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
    }


def tool_edit_skill_reload(env: Optional[Any] = None) -> Dict[str, Any]:
    """Explicitly rebuild the skills index from the on-disk SKILL.md.

    Normally patch/rewrite auto-reload; call this only when reload was
    deferred (no personality handle on env during the write) or after an
    external edit.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, reloaded, reload_detail.
    """
    session, workspace, err = _require_session(env)
    if err is not None:
        return err
    reload_result = _reload_skills(env)
    session["last_reload"] = reload_result
    _save_session(workspace, session)
    return {
        "success": True,
        "reloaded": reload_result.get("reloaded", False),
        "reload_detail": reload_result,
    }


def tool_edit_skill_test(env: Optional[Any] = None) -> Dict[str, Any]:
    """Validate the current SKILL.md against the structural invariants.

    Runs the same frontmatter + body checks the write path already enforces,
    reading the CURRENT DRAFT (not disk), so it is a true "pre-commit" check.

    Args:
        env: Host tool-context handle (injected by LCP).

    Returns:
        A dict with success, skill_name, skill_folder, frontmatter_ok,
        body_ok, declared_tools, error.
    """
    session, _workspace, err = _require_session(env)
    if err is not None:
        return err
    source = session.get("draft", "")
    ok, verr, declared_tools = _validate_source(source)

    # Report the two sub-invariants separately for richer feedback.
    frontmatter_ok = False
    body_ok = False
    fm = None
    fm_err = None
    if source.strip():
        fm, fm_err = _parse_frontmatter(source)
        frontmatter_ok = fm is not None and fm_err is None and bool(fm.get("name")) and bool(fm.get("description"))
        if fm is not None:
            body_ok = len((fm.get("_body") or "").strip()) >= _MIN_BODY_CHARS

    return {
        "success": ok,
        "skill_name": session.get("skill_name"),
        "skill_folder": session.get("skill_folder"),
        "frontmatter_ok": frontmatter_ok,
        "body_ok": body_ok,
        "declared_tools": declared_tools,
        "error": verr or fm_err or None,
    }


def tool_edit_skill_status(env: Optional[Any] = None) -> Dict[str, Any]:
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
            "skill_name": session.get("skill_name"),
            "skill_folder": session.get("skill_folder"),
            "entrypoint": session.get("entrypoint"),
            "opened_at": session.get("opened_at"),
            "last_write_at": session.get("last_write_at"),
            "write_count": session.get("write_count", 0),
            "last_reload": session.get("last_reload"),
        },
    }


def tool_edit_skill_close(env: Optional[Any] = None) -> Dict[str, Any]:
    """Delete the editing session file (the edited SKILL.md stays untouched).

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