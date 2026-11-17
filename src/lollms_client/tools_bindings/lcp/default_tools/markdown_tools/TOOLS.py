"""
markdown_tools.py
=================
Structured scouting and section retrieval for Markdown files in the workspace.

Two tools:

  * tool_get_markdown_structure(file_name)
        Returns the heading hierarchy (title / chapters / sections /
        subsections) parsed from ATX headings (#, ##, ###, ...). Fenced code
        blocks are respected so `#` inside code is never treated as a heading.

  * tool_load_markdown_section(file_name, section_title)
        Returns the full content of a single section identified by its heading
        title. Loading a chapter (##) returns the chapter body INCLUDING all
        nested sections (###) and subsections (####) beneath it, up to the next
        heading of equal-or-higher level.

The library auto-mounts whenever a .md / .markdown file is present in the
workspace (see TOOL_LIBRARY_FILE_TYPES below).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOL_LIBRARY_NAME = "Markdown Tools"
TOOL_LIBRARY_DESC = (
    "Scout the heading hierarchy of a Markdown file and load individual "
    "sections (chapters, sections, subsections) with their full nested content."
)
TOOL_LIBRARY_ICON = "📑"
# File types this library handles. The LCP binding auto-mounts this toolset
# whenever one of these extensions is present in the workspace.
TOOL_LIBRARY_FILE_TYPES = [".md", ".markdown"]

# ATX heading: 1-6 '#' followed by whitespace and a title.
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')
# Fenced code block delimiters (``` or ~~~, optionally with a language tag).
_FENCE_RE = re.compile(r'^\s*(`{3,}|~{3,})')


def init_tools_library(config: dict = None) -> None:
    """No external dependencies required."""
    pass


def _resolve_markdown_path(file_name: str) -> Path:
    """Resolves a workspace-relative path and validates it is a Markdown file."""
    path = Path(file_name)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"File '{file_name}' not found.")
    if not path.is_file():
        raise ValueError(f"Path '{file_name}' is not a file.")
    if path.suffix.lower() not in (".md", ".markdown"):
        raise ValueError(
            f"File '{file_name}' is not a Markdown file (got '{path.suffix}')."
        )
    return path


def _parse_headings(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts ATX headings from a list of lines, skipping fenced code blocks.

    Returns a flat list of dicts:
        {"level": int, "title": str, "line": int (0-based), "end_line": int}
    where `end_line` is the line index of the next heading of equal-or-higher
    level (exclusive), or len(lines) if none.
    """
    headings: List[Dict[str, Any]] = []
    in_fence = False
    fence_marker: Optional[str] = None

    for idx, raw_line in enumerate(lines):
        fence_match = _FENCE_RE.match(raw_line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0]
            elif fence_marker and marker[0] == fence_marker:
                in_fence = False
                fence_marker = None
            continue
        if in_fence:
            continue

        m = _HEADING_RE.match(raw_line)
        if m:
            headings.append({
                "level": len(m.group(1)),
                "title": m.group(2).strip(),
                "line": idx,
                "end_line": len(lines),
            })

    # Compute end_line for each heading: next heading with level <= current.
    for i, h in enumerate(headings):
        for j in range(i + 1, len(headings)):
            if headings[j]["level"] <= h["level"]:
                h["end_line"] = headings[j]["line"]
                break

    return headings


def _build_tree(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds a nested tree from a flat heading list (by level)."""
    roots: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []

    for h in headings:
        node = {
            "level": h["level"],
            "title": h["title"],
            "line": h["line"],
            "end_line": h["end_line"],
            "children": [],
        }
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def _render_tree(nodes: List[Dict[str, Any]], depth: int = 0) -> List[str]:
    """Renders the tree as an indented outline with line ranges."""
    out: List[str] = []
    for node in nodes:
        indent = "  " * depth
        marker = "#" * node["level"]
        out.append(
            f"{indent}{marker} {node['title']}  "
            f"[lines {node['line'] + 1}-{node['end_line']}]"
        )
        out.extend(_render_tree(node["children"], depth + 1))
    return out


def _flatten(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flattens a tree into a list of nodes (pre-order)."""
    flat: List[Dict[str, Any]] = []
    for node in nodes:
        flat.append(node)
        flat.extend(_flatten(node["children"]))
    return flat


def _find_section(
    flat: List[Dict[str, Any]], section_title: str
) -> Optional[Dict[str, Any]]:
    """
    Locates a section by title. Matching strategy:
      1. Exact case-insensitive match.
      2. Case-insensitive substring match (first hit).
    """
    needle = section_title.strip().lower()
    if not needle:
        return None

    for node in flat:
        if node["title"].strip().lower() == needle:
            return node
    for node in flat:
        if needle in node["title"].strip().lower():
            return node
    return None


def tool_get_markdown_structure(file_name: str) -> Dict[str, Any]:
    """
    Returns the structured heading hierarchy of a Markdown file.

    Parses ATX headings (#, ##, ###, ...) while ignoring '#' characters inside
    fenced code blocks. The result is a nested outline (title / chapters /
    sections / subsections) plus a flat list for easy lookup.

    Args:
        file_name (str): Relative path to the Markdown file in the workspace.
    """
    try:
        path = _resolve_markdown_path(file_name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        headings = _parse_headings(lines)
        if not headings:
            return {
                "success": True,
                "file_name": file_name,
                "total_lines": len(lines),
                "heading_count": 0,
                "tree": [],
                "flat": [],
                "output": f"'{file_name}' contains no ATX headings.",
            }

        tree = _build_tree(headings)
        flat = _flatten(tree)

        outline_lines = [f"# Structure of '{file_name}' ({len(headings)} heading(s)):"]
        outline_lines.extend(_render_tree(tree))
        output = "\n".join(outline_lines)

        return {
            "success": True,
            "file_name": file_name,
            "total_lines": len(lines),
            "heading_count": len(headings),
            "tree": tree,
            "flat": [
                {
                    "level": n["level"],
                    "title": n["title"],
                    "line": n["line"] + 1,
                    "end_line": n["end_line"],
                }
                for n in flat
            ],
            "output": output,
        }
    except Exception as e:
        import traceback as _tb
        return {
            "success": False,
            "error": f"Failed to parse Markdown structure of '{file_name}': {e}",
            "traceback": _tb.format_exc(),
        }


def tool_load_markdown_section(
    file_name: str, section_title: str
) -> Dict[str, Any]:
    """
    Loads the full content of a single Markdown section identified by its
    heading title.

    Loading a chapter (##) returns the chapter body INCLUDING all nested
    sections (###) and subsections (####) beneath it, up to the next heading
    of equal-or-higher level. Loading a subsection (###) returns only that
    subsection's body.

    Args:
        file_name (str): Relative path to the Markdown file in the workspace.
        section_title (str): The heading title to load (case-insensitive;
            exact match preferred, then substring match).
    """
    try:
        path = _resolve_markdown_path(file_name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        headings = _parse_headings(lines)
        if not headings:
            return {
                "success": False,
                "error": f"'{file_name}' contains no ATX headings to load.",
            }

        tree = _build_tree(headings)
        flat = _flatten(tree)
        node = _find_section(flat, section_title)
        if node is None:
            available = ", ".join(f"'{n['title']}'" for n in flat[:20])
            return {
                "success": False,
                "error": (
                    f"Section '{section_title}' not found in '{file_name}'. "
                    f"Available headings: {available}"
                    + (" ..." if len(flat) > 20 else "")
                ),
            }

        start = node["line"]
        end = node["end_line"]
        content = "\n".join(lines[start:end]).rstrip()

        return {
            "success": True,
            "file_name": file_name,
            "section_title": node["title"],
            "level": node["level"],
            "line_start": start + 1,
            "line_end": end,
            "content": content,
            "output": content,
        }
    except Exception as e:
        import traceback as _tb
        return {
            "success": False,
            "error": f"Failed to load section '{section_title}' from '{file_name}': {e}",
            "traceback": _tb.format_exc(),
        }


def get_tools() -> List[Dict[str, Any]]:
    """Return the tool descriptors exposed by this library."""
    return [
        {
            "name": "tool_get_markdown_structure",
            "description": (
                "Returns the structured heading hierarchy of a Markdown file "
                "(title / chapters / sections / subsections parsed from ATX "
                "headings). Fenced code blocks are respected. Use this FIRST "
                "to scout a document before loading individual sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Relative path to the Markdown file in the workspace.",
                    },
                },
                "required": ["file_name"],
            },
            "output": {"type": "object"},
            "callable": tool_get_markdown_structure,
            "_source": "markdown_tools",
            "_source_file": __file__,
        },
        {
            "name": "tool_load_markdown_section",
            "description": (
                "Loads the full content of a single Markdown section by its "
                "heading title. Loading a chapter (##) returns the chapter body "
                "INCLUDING all nested sections (###) and subsections (####) "
                "beneath it. Loading a subsection (###) returns only that "
                "subsection's body."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Relative path to the Markdown file in the workspace.",
                    },
                    "section_title": {
                        "type": "string",
                        "description": (
                            "The heading title to load (case-insensitive; "
                            "exact match preferred, then substring match)."
                        ),
                    },
                },
                "required": ["file_name", "section_title"],
            },
            "output": {"type": "object"},
            "callable": tool_load_markdown_section,
            "_source": "markdown_tools",
            "_source_file": __file__,
        },
    ]