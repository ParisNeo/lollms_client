"""
toolset_staging_state — shared on-disk state for the Toolset Staging mechanism.

Two consumers rely on this module:

  * the ``toolset_staging`` LCP toolset (``tools_bindings/lcp/default_tools/
    toolset_staging/TOOLS.py``), which *writes* the staging registry and
    *mutates* the target ``TOOLS.py`` source; and

  * ``LollmsPersonality._build_staged_toolsets_context`` (in
    ``lollms_personality.py``), which *reads* the registry every turn and
    re-renders each staged toolset's CURRENT on-disk source into a mutable
    prompt region.

Design
------
The registry is a single JSON file at ``<workspace>/.toolset_staging/staged.json``.
Its payload shape::

    {
      "version": 1,
      "staged": [
        {"name": "file_tools", "folder": "/abs/path/file_tools", "staged_at": "..."},
        ...                                    # typically 0 or 1 entries
      ]
    }

The source of truth for the *content* of each staged toolset is always the
``TOOLS.py`` file on disk — never the registry. The prompt region is therefore
re-rendered from disk every turn; if the LLM patches ``TOOLS.py`` via
``tool_stage_patch`` the next round's prompt shows the new content
automatically.

Invariants
----------
* Never raises. All failure modes return safe defaults.
* The registry path is fixed and workspace-scoped so a single discussion
  cannot leak staged entries into another.
* The registry never contains the toolset source text — only identity. A
  stale entry whose ``TOOLS.py`` no longer exists is silently skipped by the
  prompt builder (with a ``missing: true`` flag in its manifest).

The module is deliberately dependency-free (stdlib only) so it can be
imported from both the toolset and the personality without circular imports.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ENTRYPOINT = "TOOLS.py"
_STAGING_DIRNAME = ".toolset_staging"
_REGISTRY_FILENAME = "staged.json"
_MAX_SOURCE_CHARS = 2_000_000  # safety cap per toolset


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def staging_dir(workspace: Path) -> Path:
    """Return (and create) ``<workspace>/.toolset_staging/``."""
    d = Path(workspace) / _STAGING_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry_path(workspace: Path) -> Path:
    return staging_dir(workspace) / _REGISTRY_FILENAME


def load_registry(workspace: Path) -> Dict[str, Any]:
    """Load the staging registry. Always returns a dict with a 'staged' list."""
    path = registry_path(workspace)
    if not path.is_file():
        return {"version": 1, "staged": []}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"version": 1, "staged": []}
    if not isinstance(data, dict):
        return {"version": 1, "staged": []}
    staged = data.get("staged")
    if not isinstance(staged, list):
        staged = []
    # Filter any malformed entries.
    cleaned: List[Dict[str, Any]] = []
    for item in staged:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        folder = item.get("folder")
        if isinstance(name, str) and isinstance(folder, str) and name and folder:
            cleaned.append({
                "name": name,
                "folder": folder,
                "staged_at": item.get("staged_at") or _utc_now(),
            })
    return {"version": 1, "staged": cleaned}


def save_registry(workspace: Path, data: Dict[str, Any]) -> None:
    """Atomically persist the staging registry."""
    path = registry_path(workspace)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def add_staged(workspace: Path, name: str, folder: str) -> Dict[str, Any]:
    """Stage a toolset (idempotent — re-staging is a no-op on the payload)."""
    reg = load_registry(workspace)
    for item in reg["staged"]:
        if item["name"] == name:
            item["folder"] = folder
            save_registry(workspace, reg)
            return item
    entry = {"name": name, "folder": folder, "staged_at": _utc_now()}
    reg["staged"].append(entry)
    save_registry(workspace, reg)
    return entry


def remove_staged(workspace: Path, name: str) -> bool:
    """Unstage a toolset by name. Returns True when an entry was removed."""
    reg = load_registry(workspace)
    before = len(reg["staged"])
    reg["staged"] = [item for item in reg["staged"] if item["name"] != name]
    if len(reg["staged"]) == before:
        return False
    save_registry(workspace, reg)
    return True


def clear_staged(workspace: Path) -> int:
    """Remove ALL staged entries. Returns how many were removed."""
    reg = load_registry(workspace)
    count = len(reg["staged"])
    if count:
        reg["staged"] = []
        save_registry(workspace, reg)
    return count


def read_toolset_source(folder: str) -> Optional[str]:
    """Read the current ``TOOLS.py`` source of a staged toolset, or None.

    The prompt builder calls this every turn — the on-disk file is the single
    source of truth for what the LLM sees.
    """
    try:
        entry = Path(folder) / _ENTRYPOINT
    except (TypeError, ValueError):
        return None
    if not entry.is_file():
        return None
    try:
        source = entry.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if len(source) > _MAX_SOURCE_CHARS:
        # Truncate giant files with an explicit marker so the prompt stays
        # bounded. The LLM can still patch beyond the visible tail.
        source = source[:_MAX_SOURCE_CHARS] + (
            "\n\n# [... source truncated at "
            f"{_MAX_SOURCE_CHARS} chars — use tool_stage_read to fetch more]"
        )
    return source


def build_staged_prompt_block(workspace: Path) -> str:
    """Render the current staging region for injection into the system prompt.

    Returns an empty string when nothing is staged. Otherwise, returns::

        === STAGED TOOLSETS (read-only view — edit via tool_stage_*) ===
        ... (per-toolset source block) ...
        === END STAGED TOOLSETS ===

    The block is deliberately explicit about being *read-only* from the LLM's
    perspective: mutations flow exclusively through the ``toolset_staging``
    tools so every write is validated + atomic. Letting the model directly
    emit ``<artifact>`` writes over ``TOOLS.py`` outside of ``tool_stage_patch``
    would bypass the toolset's AST + `def tool_*` guardrails.
    """
    reg = load_registry(workspace)
    staged = reg.get("staged") or []
    if not staged:
        return ""

    lines: List[str] = [
        "=== STAGED TOOLSETS (read-only view — edit via tool_stage_* tools) ===",
        (
            "These toolset source files are staged for in-place editing. The "
            "text below is refreshed from disk EVERY TURN: patches applied "
            "through `tool_stage_patch` / `tool_stage_rewrite` appear here on "
            "your next round automatically. You cannot write to them by "
            "emitting an `<artifact>` tag — the only sanctioned flow is "
            "`tool_stage_patch` (Aider blocks) or `tool_stage_rewrite` (full "
            "source). Stage just ONE toolset at a time; call "
            "`tool_stage_unstage` before staging another."
        ),
    ]
    for entry in staged:
        name = entry.get("name") or "<unnamed>"
        folder = entry.get("folder") or ""
        source = read_toolset_source(folder)
        if source is None:
            lines.append(
                f"\n--- Staged toolset: {name} ---\n"
                f"(TOOLS.py is no longer present on disk at: {folder or '<unknown>'}; "
                f"call `tool_stage_unstage(name=\"{name}\")` to clear this entry.)\n"
                f"--- End staged toolset: {name} ---"
            )
            continue
        lines.append(f"\n--- Staged toolset: {name} (folder: {folder}) ---")
        lines.append("```python")
        lines.append(source.rstrip("\n"))
        lines.append("```")
        lines.append(f"--- End staged toolset: {name} ---")
    lines.append("=== END STAGED TOOLSETS ===")
    return "\n".join(lines)


__all__ = [
    "staging_dir",
    "registry_path",
    "load_registry",
    "save_registry",
    "add_staged",
    "remove_staged",
    "clear_staged",
    "read_toolset_source",
    "build_staged_prompt_block",
]