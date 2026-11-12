"""
tool_maker.py
LCP toolset for authoring new LCP tool libraries from inside an active agent.

This toolset provides two public entry points:

  - ``tool_tool_maker``            — Generates a new LCP toolset folder
                                      (``<name>/TOOLS.py`` + ``<name>/description.yaml``)
                                      inside the workspace, following the
                                      canonical LCP toolset layout.
  - ``tool_load_generated_toolset``— Loads a previously generated toolset back
                                      into the current personality so its tools
                                      become callable immediately.

The write path is deliberately conservative:
  * the toolset name is sanitized to a bare identifier (no traversal, no
    separators, no reserved names);
  * the target folder MUST resolve inside the workspace, otherwise the call
    is rejected;
  * overwriting an existing non-empty folder requires ``overwrite=True``;
  * the generated ``TOOLS.py`` is AST-compiled before being written so a
    malformed snippet never lands on disk.
"""

from __future__ import annotations

import ast
import re
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors


TOOL_LIBRARY_NAME = "Tool Maker"
TOOL_LIBRARY_DESC = (
    "Authors new LCP tool libraries inside the workspace. "
    "Use tool_tool_maker(...) to scaffold a complete toolset folder "
    "(<name>/TOOLS.py + <name>/description.yaml) from a natural-language "
    "specification and a Python implementation, and tool_load_generated_toolset(...) "
    "to hot-load the freshly generated toolset into the current personality so "
    "its tool_* functions become callable without a restart."
)
TOOL_LIBRARY_ICON = "🛠️"

# Reserved names that would collide with the LCP loader's discovery rules,
# the tool's own persistence, or Python import semantics.
_RESERVED_TOOLSET_NAMES = frozenset({
    "__pycache__", "conftest", "setup", "tests", "test",
    "lollms_client", "lcp", "default_tools",
})

# Strict identifier for the generated folder name and the toolset title.
# ASCII letters/digits/underscore only — no separators, no traversal, no bytes.
_TOOLSET_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# A tool definition is expected to be a bare identifier prefixed with `tool_`.
_TOOL_FUNC_RE = re.compile(r"^tool_([A-Za-z0-9_]+)$")


# ──────────────────────────────────────────────────────────────────────────────
# Path / name resolution helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_workspace_root(env: Optional[Any] = None) -> Path:
    """Resolves the workspace root from the injected runtime context.

    Preference order:
      1. ``env.tool_context.workspace_path`` (host-injected, authoritative);
      2. CWD fallback for agnostic contexts (matches execute_python semantics).
    """
    if env is not None:
        tool_ctx = getattr(env, "tool_context", None)
        if tool_ctx is not None:
            ctx_ws = getattr(tool_ctx, "workspace_path", None)
            if ctx_ws:
                try:
                    ws = Path(ctx_ws).resolve()
                    if ws.exists() and ws.is_dir():
                        return ws
                except Exception:
                    pass
    return Path.cwd().resolve()


def _sanitize_toolset_name(name: str) -> str:
    """Normalizes a user-supplied toolset name into a bare identifier.

    Raises ``ValueError`` on anything that cannot be safely mapped onto a bare
    folder name — we never guess, we never strip the meaningful part. The
    caller converts this to a structured tool error.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Toolset 'name' is required and must be a non-empty string.")
    cleaned = name.strip().lower()
    # Replace common separators with underscore so "file organizer" → "file_organizer".
    cleaned = re.sub(r"[\s\-./\\]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(
            f"Toolset name '{name}' has no usable characters after sanitization."
        )
    if cleaned[0].isdigit():
        cleaned = f"toolset_{cleaned}"
    if cleaned in _RESERVED_TOOLSET_NAMES:
        raise ValueError(
            f"Toolset name '{cleaned}' is reserved; choose a different name."
        )
    if not _TOOLSET_NAME_RE.match(cleaned):
        raise ValueError(
            f"Toolset name '{cleaned}' is not a valid Python identifier / folder name."
        )
    return cleaned


def _resolve_toolset_dir(toolset_name: str, workspace_root: Path) -> Path:
    """Returns the absolute path of ``<workspace>/<toolset_name>/``.

    The resolved path is verified to sit inside ``workspace_root`` — the
    (already sanitized) name cannot escape, but defense-in-depth catches any
    future regression in the sanitizer without ever writing outside the box.
    """
    target = (workspace_root / toolset_name).resolve()
    try:
        target.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Resolved toolset path '{target}' escapes the workspace root."
        ) from exc
    return target


# ──────────────────────────────────────────────────────────────────────────────
# Source-generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_python_source(source: str, label: str) -> None:
    """AST-compiles ``source`` and raises ``ValueError`` on syntax errors.

    We do NOT execute the source here — that would be a security and
    correctness hazard on partial drafts. We only ensure it parses so a broken
    ``TOOLS.py`` never lands on disk. Runtime errors are the user's to fix
    via a follow-up call with ``overwrite=True``.
    """
    try:
        ast.parse(source, filename=label)
    except SyntaxError as exc:
        raise ValueError(
            f"{label} contains a syntax error at line {exc.lineno}: {exc.msg}"
        ) from exc


def _extract_tool_functions(source: str) -> List[str]:
    """Returns the ``tool_*`` function names defined at module top level.

    Used by ``tool_load_generated_toolset`` to confirm that a generated toolset
    actually exposes at least one discoverable tool, and to surface the exact
    names in the success message so the agent can call them immediately.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _TOOL_FUNC_RE.match(node.name):
                names.append(node.name)
    return names


def _render_description_yaml(
    toolset_name: str,
    title: str,
    description: str,
    author: str,
    version: str,
) -> str:
    """Renders the canonical ``description.yaml`` for an LCP toolset.

    Strictly matches the LCP loader's schema: ``title``/``author``/``description``
    are top-level fields; any extra metadata lives under ``model_input_parameters``
    for forward compatibility with the loader.
    """
    safe_description = (description or "").replace("\n", " ").strip()
    return textwrap.dedent(f"""\
        title: {title}
        author: {author}
        creation_date: 2026-01-01
        last_update_date: 2026-01-01
        version: {version}
        description: {safe_description}
        icon: "🛠️"
        global_input_parameters: []
        model_input_parameters: []
    """)


def _render_tools_module(
    toolset_name: str,
    title: str,
    description: str,
    implementation: str,
) -> str:
    """Assembles a complete, self-consistent ``TOOLS.py``.

    The module structure follows the LCP canonical layout:
      * ``TOOL_LIBRARY_NAME`` / ``TOOL_LIBRARY_DESC`` / ``TOOL_LIBRARY_ICON``;
      * ``init_tools_library(config=None)`` — must never raise;
      * one or more top-level ``tool_*`` functions supplied by the caller.

    The user's implementation is inserted verbatim after the header so line
    numbers preserved in their tracebacks match what they wrote.
    """
    header = textwrap.dedent(f"""\
        \"\"\"
        {title}

        {description}
        \"\"\"

        from __future__ import annotations

        from typing import Any, Dict, List, Optional


        TOOL_LIBRARY_NAME = {title!r}
        TOOL_LIBRARY_DESC = {description!r}
        TOOL_LIBRARY_ICON = "🛠️"


        def init_tools_library(config: Optional[dict] = None) -> None:
            \"\"\"LCP initialization hook — must never raise, must be idempotent.\"\"\"
            return None


    """)
    body = implementation.rstrip() + "\n"
    return header + body


# ──────────────────────────────────────────────────────────────────────────────
# Public tools
# ──────────────────────────────────────────────────────────────────────────────

def tool_tool_maker(
    name: str = "",
    description: str = "",
    implementation: str = "",
    title: Optional[str] = None,
    author: str = "Lollms Agent",
    version: str = "0.1.0",
    overcode_if_exists: bool = False,
    overwrite: Optional[bool] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Scaffolds a complete LCP toolset folder inside the workspace.

    Creates ``<workspace>/<name>/TOOLS.py`` and ``<workspace>/<name>/description.yaml``,
    following the canonical LCP toolset layout. The generated module exposes
    ``TOOL_LIBRARY_NAME`` / ``TOOL_LIBRARY_DESC`` / ``TOOL_LIBRARY_ICON`` and
    ``init_tools_library(config=None)``, and contains the caller's ``tool_*``
    functions verbatim.

    Args:
        name: Bare toolset identifier (e.g. ``file_organizer``). Sanitized to
            a lowercase identifier; separators are converted to underscores.
        description: One-line, human-readable summary for the description file
            and the library descriptor.
        implementation: Complete Python source containing one or more top-level
            ``tool_<verb>`` functions. Must be syntactically valid.
        title: Optional human-readable title for the descriptor. Defaults to a
            title-cased version of ``name``.
        author: Author string recorded in ``description.yaml``.
        version: Semantic version recorded in ``description.yaml``.
        overcode_if_exists: Legacy spelling of ``overwrite`` kept for backward
            compat with external callers.
        overwrite: When True, replaces an existing toolset folder. When False
            (or unset with a non-empty existing folder), the call is refused.
        env: Optional host-injected runtime context. ``env.tool_context.workspace_path``
            determines the workspace root.

    Returns:
        Structured result dict with ``success``, ``toolset_path``, ``tool_names``,
        and (on failure) ``error``.
    """
    # Accept legacy spelling. Explicit `overwrite` wins over the legacy flag.
    if overwrite is None:
        overwrite = bool(overcode_if_exists)
    if not isinstance(implementation, str) or not implementation.strip():
        return {
            "success": False,
            "error": "The 'implementation' parameter is required and must be non-empty Python source.",
        }

    try:
        toolset_name = _sanitize_toolset_name(name)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    workspace_root = _get_workspace_root(env)
    try:
        toolset_dir = _resolve_toolset_dir(toolset_name, workspace_root)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    resolved_title = (title or toolset_name.replace("_", " ").title()).strip()
    safe_description = (description or f"Generated toolset '{toolset_name}'.").strip()

    tools_source = _render_tools_module(
        toolset_name=toolset_name,
        title=resolved_title,
        description=safe_description,
        implementation=implementation,
    )
    try:
        _validate_python_source(tools_source, label=f"{toolset_name}/TOOLS.py")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    tool_names = _extract_tool_functions(tools_source)
    if not tool_names:
        return {
            "success": False,
            "error": (
                "The 'implementation' contains no top-level 'tool_*' function. "
                "Define at least one function named 'tool_<verb>(...)' at module "
                "scope so the LCP loader can discover it."
            ),
        }

    # Refuse silent overwrite of an existing, non-empty folder.
    if toolset_dir.exists() and any(toolset_dir.iterdir()) and not overwrite:
        return {
            "success": False,
            "error": (
                f"Toolset folder '{toolset_name}' already exists and is not empty. "
                "Pass overwrite=True to replace it, or choose a different name."
            ),
        }

    description_yaml = _render_description_yaml(
        toolset_name=toolset_name,
        title=resolved_title,
        description=safe_description,
        author=author,
        version=version,
    )

    try:
        toolset_dir.mkdir(parents=True, exist_ok=True)
        (toolset_dir / "TOOLS.py").write_text(tools_source, encoding="utf-8")
        (toolset_dir / "description.yaml").write_text(description_yaml, encoding="utf-8")
    except OSError as exc:
        return {
            "success": False,
            "error": f"Failed to write toolset '{toolset_name}' to disk: {exc}",
        }

    ASCIIColors.success(
        f"[tool_maker] Generated toolset '{toolset_name}' with {len(tool_names)} tool(s): "
        f"{', '.join(tool_names)}"
    )

    try:
        relative_path = str(toolset_dir.relative_to(workspace_root))
    except ValueError:
        relative_path = str(toolset_dir)

    return {
        "success": True,
        "toolset_name": toolset_name,
        "toolset_path": relative_path,
        "tool_names": tool_names,
        "message": (
            f"Created toolset '{toolset_name}' at '{relative_path}'. "
            f"Call tool_load_generated_toolset(name='{toolset_name}') to hot-load it."
        ),
    }


def tool_load_generated_toolset(
    name: str = "",
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """Hot-loads a previously generated toolset into the active personality.

    Resolves ``<workspace>/<name>/`` and, when a personality is reachable via
    ``env.discussion`` or ``env.personality``, triggers a tool-library refresh
    so the newly generated ``tool_*`` functions become callable without a
    process restart.

    The loader is defensive: if no personality is accessible from ``env``, the
    tool still confirms the toolset exists on disk and returns the discovered
    tool names so the caller knows what to wire up next. It never raises.
    """
    try:
        toolset_name = _sanitize_toolset_name(name)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    workspace_root = _get_workspace_root(env)
    try:
        toolset_dir = _resolve_toolset_dir(toolset_name, workspace_root)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    tools_py = toolset_dir / "TOOLS.py"
    if not tools_py.is_file():
        return {
            "success": False,
            "error": (
                f"Toolset '{toolset_name}' has no TOOLS.py at "
                f"'{tools_py}'. Generate it first via tool_tool_maker(...)."
            ),
        }

    try:
        source = tools_py.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "success": False,
            "error": f"Failed to read '{tools_py}': {exc}",
        }

    tool_names = _extract_tool_functions(source)
    if not tool_names:
        return {
            "success": False,
            "error": (
                f"Toolset '{toolset_name}' has no top-level 'tool_*' functions; "
                "nothing to load."
            ),
        }

    # Resolve the active personality through the host context, if present.
    # Both shapes are supported: discussion.personality (chat agents) and
    # env.personality (raw personality sessions).
    personality = None
    if env is not None:
        personality = getattr(env, "personality", None)
        if personality is None:
            discussion = getattr(env, "discussion", None)
            if discussion is not None:
                personality = getattr(discussion, "personality", None)

    loaded = False
    load_error: Optional[str] = None
    if personality is not None:
        refresh = getattr(personality, "refresh_toolset", None)
        if callable(refresh):
            try:
                refresh(extra_tool_paths=[toolset_dir])
                loaded = True
            except Exception as exc:
                load_error = f"Personality refresh failed: {exc}"
                ASCIIColors.warning(f"[tool_maker] {load_error}")
                traceback.print_exc()
        else:
            load_error = (
                "Personality has no 'refresh_toolset' hook; the toolset exists "
                "but cannot be hot-loaded. Restart the session or attach the "
                "toolset path manually."
            )

    if loaded:
        ASCIIColors.success(
            f"[tool_maker] Hot-loaded toolset '{toolset_name}' "
            f"({len(tool_names)} tool(s))."
        )
    elif load_error is None:
        load_error = "No active personality found in the runtime context."

    return {
        "success": True,
        "toolset_name": toolset_name,
        "tool_names": tool_names,
        "loaded": loaded,
        "message": (
            f"Toolset '{toolset_name}' is ready with {len(tool_names)} tool(s): "
            f"{', '.join(tool_names)}."
            if loaded
            else f"Toolset '{toolset_name}' exists with {len(tool_names)} tool(s): "
                 f"{', '.join(tool_names)}. {load_error}"
        ),
    }