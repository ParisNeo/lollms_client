"""
inspect_text.py
LCP toolset for targeted inspection of persisted text files in the sandboxed
workspace. Complements the tool-output windowing doctrine: when a large tool
output is offloaded to a .log file, the LLM uses these tools to read exactly
the region it needs instead of re-loading the whole payload.

Tools:
  - tool_read_lines:  1-based inclusive line window reader.
  - tool_read_chars:  character-offset window reader.
  - tool_grep_file:   regex search returning matches with surrounding
                      context lines (grep -C behavior).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Inspect Text"
TOOL_LIBRARY_DESC = "Targeted inspection of workspace text files: read a line range, a character-offset window, or grep with surrounding context. Designed for examining stripped tool outputs and large logs without reloading them entirely."
TOOL_LIBRARY_ICON = "🔍"

_DEFAULT_CONTEXT_LINES = 3
_MAX_CONTEXT_LINES = 50
_MAX_RETURN_CHARS = 20000
_HIDDEN_PART_RE = re.compile(r"(?:^|/)(?:\.versions|__pycache__|\.git)(?:/|$)")


def init_tools_library(config: dict = None) -> None:
    return None


def _get_workspace_root() -> Path:
    """
    Resolves the sandbox root. The orchestrator chdirs into the workspace
    before invoking tools, so CWD is authoritative; standalone execution
    falls back to ./data_workspace.
    """
    cwd = Path.cwd()
    if cwd.name in ("workspace_data", "data_workspace") or (cwd / "data_workspace").exists():
        return cwd
    return Path("./data_workspace").resolve()


def _resolve_workspace_path(file_name: str) -> Optional[Path]:
    """
    Confines a user-supplied path to the workspace sandbox.
    Blocks traversal ('..'), absolute escapes, and hidden/system directories.
    Returns None when the path is unsafe or the file does not exist.
    """
    if not file_name or not isinstance(file_name, str):
        return None

    root = _get_workspace_root()
    clean = file_name.replace("\\", "/").lstrip("/")
    if not clean or ".." in Path(clean).parts:
        return None
    if _HIDDEN_PART_RE.search(clean):
        return None

    try:
        candidate = (root / clean).resolve()
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _windowed_header(file_name: str, total_units: int, unit_name: str, start: int, end: int) -> str:
    stripped = total_units - (end - start + 1)
    suffix = f" ({stripped} {unit_name} outside window)" if stripped > 0 else " (full file)"
    return f"[{file_name} | {unit_name} {start}-{end} of {total_units}{suffix}]\n"


def _bounded_preview(text: str) -> str:
    if len(text) <= _MAX_RETURN_CHARS:
        return text
    head = text[:_MAX_RETURN_CHARS // 2]
    tail = text[-_MAX_RETURN_CHARS // 2:]
    stripped = len(text) - _MAX_RETURN_CHARS
    return (
        f"{head}\n... [result window limit: {stripped} middle characters omitted — "
        f"narrow your range or use tool_read_chars for precise offsets] ...\n{tail}"
    )


def tool_read_lines(
    file_name: str = "",
    start_line: int = 1,
    end_line: int = 0,
) -> Dict[str, Any]:
    """
    Reads an inclusive 1-based line range from a text file in the workspace.

    Use this to inspect a specific region of a stripped tool output (.log)
    or any workspace text file without loading the whole file.

    Args:
        file_name (str): Name of the text file to read (relative to the workspace root). Path traversal is blocked.
        start_line (int): First line to return (1-based). Values below 1 are clamped to 1.
        end_line (int): Last line to return (inclusive). 0 or values below start_line mean 'start_line + 49' (a 50-line window).
    """
    resolved = _resolve_workspace_path(file_name)
    if resolved is None:
        return {
            "success": False,
            "error": (
                f"File '{file_name}' not found in the workspace or the path was "
                f"refused (traversal/hidden directories are blocked)."
            ),
        }

    try:
        text = _safe_read_text(resolved)
    except Exception as read_err:
        return {"success": False, "error": f"Failed to read '{file_name}': {read_err}"}

    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return {"success": True, "output": f"[{file_name} is empty]"}

    start = max(1, int(start_line))
    if not end_line or int(end_line) < start:
        end = start + 49
    else:
        end = int(end_line)
    end = min(end, total)

    selected = lines[start - 1:end]
    header = _windowed_header(file_name, total, "lines", start, end)
    body = "\n".join(selected)
    return {"success": True, "output": _bounded_preview(header + body)}


def tool_read_chars(
    file_name: str = "",
    char_start: int = 0,
    char_end: int = 0,
) -> Dict[str, Any]:
    """
    Reads a character-offset window from a text file in the workspace.

    Use this when you need a byte-precise slice (e.g. the stripped middle of
    a windowed tool output, or a specific offset reported by a previous tool).

    Args:
        file_name (str): Name of the text file to read (relative to the workspace root). Path traversal is blocked.
        char_start (int): First character offset to return (0-based). Negative values are clamped to 0.
        char_end (int): Last character offset (exclusive). 0 or values below char_start mean 'char_start + 20000'.
    """
    resolved = _resolve_workspace_path(file_name)
    if resolved is None:
        return {
            "success": False,
            "error": (
                f"File '{file_name}' not found in the workspace or the path was "
                f"refused (traversal/hidden directories are blocked)."
            ),
        }

    try:
        text = _safe_read_text(resolved)
    except Exception as read_err:
        return {"success": False, "error": f"Failed to read '{file_name}': {read_err}"}

    total = len(text)
    if total == 0:
        return {"success": True, "output": f"[{file_name} is empty]"}

    start = max(0, int(char_start))
    if not char_end or int(char_end) <= start:
        end = start + _MAX_RETURN_CHARS
    else:
        end = int(char_end)
    end = min(end, total)

    selected = text[start:end]
    header = _windowed_header(file_name, total, "characters", start, end)
    return {"success": True, "output": _bounded_preview(header + selected)}


def tool_grep_file(
    file_name: str = "",
    pattern: str = "",
    context_lines: int = _DEFAULT_CONTEXT_LINES,
) -> Dict[str, Any]:
    """
    Searches a workspace text file for a regex pattern and returns each match
    with surrounding context lines (grep -C behavior).

    Use this to locate a symbol, error, or value inside a large stripped log,
    then follow up with tool_read_lines for the exact region.

    Args:
        file_name (str): Name of the text file to search (relative to the workspace root). Path traversal is blocked.
        pattern (str): Python regular expression to search for. Required.
        context_lines (int): Number of context lines before and after each match. Default 3, max 50.
    """
    resolved = _resolve_workspace_path(file_name)
    if resolved is None:
        return {
            "success": False,
            "error": (
                f"File '{file_name}' not found in the workspace or the path was "
                f"refused (traversal/hidden directories are blocked)."
            ),
        }

    if not pattern or not isinstance(pattern, str):
        return {"success": False, "error": "The 'pattern' parameter is empty. Provide a regular expression to search for."}

    try:
        regex = re.compile(pattern)
    except re.error as compile_err:
        return {"success": False, "error": f"Invalid regular expression '{pattern}': {compile_err}"}

    try:
        text = _safe_read_text(resolved)
    except Exception as read_err:
        return {"success": False, "error": f"Failed to read '{file_name}': {read_err}"}

    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return {"success": True, "output": f"[{file_name} is empty]"}

    ctx = max(0, min(int(context_lines), _MAX_CONTEXT_LINES))
    match_line_numbers: List[int] = []
    for idx, line in enumerate(lines):
        if regex.search(line):
            match_line_numbers.append(idx)

    if not match_line_numbers:
        return {
            "success": True,
            "output": (
                f"[no matches for pattern '{pattern}' in {file_name} "
                f"({total} lines searched)]"
            ),
        }

    merged_spans: List[List[int]] = []
    for line_idx in match_line_numbers:
        span_start = max(0, line_idx - ctx)
        span_end = min(total - 1, line_idx + ctx)
        if merged_spans and span_start <= merged_spans[-1][1] + 1:
            merged_spans[-1][1] = max(merged_spans[-1][1], span_end)
        else:
            merged_spans.append([span_start, span_end])

    blocks: List[str] = [f"[{file_name} | {len(match_line_numbers)} match(es) for '{pattern}']\n"]
    for span_start, span_end in merged_spans[:20]:
        blocks.append(f"--- lines {span_start + 1}-{span_end + 1} ---")
        for line_idx in range(span_start, span_end + 1):
            marker = ">" if line_idx in set(match_line_numbers) else " "
            blocks.append(f"{marker}{line_idx + 1:>6}| {lines[line_idx]}")
    if len(merged_spans) > 20:
        blocks.append(f"... [{len(merged_spans) - 20} more match region(s) omitted — narrow the pattern] ...")

    return {"success": True, "output": _bounded_preview("\n".join(blocks))}