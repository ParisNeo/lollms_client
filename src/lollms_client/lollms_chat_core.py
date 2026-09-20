"""
lollms_chat_core.py
Unified core execution layer for Lollms multi-tier agentic architecture.

Enforces two distinct context paradigms:
1. Disposable Worker Context: Fresh, atomic sandbox containing only the specific task
   and explicitly assigned files. Completely discarded upon report generation.
2. Persistent Orchestrator Context with Two Views:
   - Orchestrator/Model View: Detailed coordination trace (plans, delegations, verification verdicts,
     and plain-data worker reports).
   - User/UI View: Natural conversational prose with structured event telemetry or processing tags.

Factorizes shared logic:
- Structural symbol and metadata detection across all supported languages
- Tool execution in sandboxed CWD with path sanitization and head/tail output windowing
- Resilient JSON repair and parameter normalization for LLM tool invocations
- Context visibility operations (<unlock_file>, <lock_file>, <pin_file>, etc.)
- Workspace snapshotting and change reconciliation
- Error forensic dumping with host path redaction
"""

from __future__ import annotations

import base64
import hashlib
import html
import inspect
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ascii_colors import ASCIIColors, trace_exception


# ── Lazy Type Resolvers (Zero Circular Import Dependency) ─────────────────

try:
    from lollms_client.lollms_types import MSG_TYPE, EventMode
except ImportError:
    class EventMode:
        PROCESSING_TAG_MODE = 0
        FULL_CALLBACK_MODE = 1
        MIXED_MODE = 2
        SILENT_MODE = 3

    class MSG_TYPE:
        MSG_TYPE_CHUNK = 0
        MSG_TYPE_INFO = 8
        MSG_TYPE_TOOL_START = 50
        MSG_TYPE_TOOL_END = 51
        MSG_TYPE_ARTEFACT_BUILD_START = 52
        MSG_TYPE_ARTEFACT_BUILD_END = 53
        MSG_TYPE_CONTEXT_UPDATE = 54
        MSG_TYPE_ARTEFACT_SYMBOL_DETECTED = 55
        MSG_TYPE_SCRATCHPAD_UPDATE = 56
        MSG_TYPE_ROUND_INFO = 61


def _get_artefact_visibility():
    try:
        from lollms_client.lollms_artefact.lollms_artefact import ArtefactVisibility
        return ArtefactVisibility
    except Exception:
        class _FallbackVisibility:
            HIDDEN = "hidden"
            TREE_LOCKED = "tree_locked"
            TREE_UNLOCKABLE = "tree_unlockable"
            FOLDER_COLLAPSED = "folder_collapsed"
            PINNED = "pinned"
            METADATA = "metadata"
            FULL = "full"
        return _FallbackVisibility


def _is_tool_binding(obj: Any) -> bool:
    """Agnostic check whether an object is a tool binding."""
    return (
        obj is not None
        and not isinstance(obj, (list, str))
        and hasattr(obj, "discover_tools")
        and hasattr(obj, "execute_tool")
        and hasattr(obj, "to_chat_tool_specs")
    )

is_tool_binding = _is_tool_binding


# ── Constants ──────────────────────────────────────────────────────────────

_BASE64_RE = re.compile(r'^[A-Za-z0-9+/=\s]{500,}$')

_BINARY_BLOB_KEYS = {
    "plot_b64", "image_b64", "audio_b64", "video_b64", "file_b64",
    "screenshot_b64", "pdf_b64", "thumbnail_b64", "base64",
    "binary", "raw_image", "image_data", "raw_data",
    "images", "image_media_types",
}

_TOOL_UI_PREVIEW_WINDOW = 6000
_TOOL_UI_PREVIEW_HALF = _TOOL_UI_PREVIEW_WINDOW // 2

_HOST_ROOT_RE = re.compile(
    r'(?:[A-Za-z]:\\(?:Users|home|Documents|Program Files|Windows)[\\/][^\s"\'<>|]*'
    r'|/(?:home|Users|root|var|opt|usr/local)/[^\s"\'<>|]*'
    r'|(?:[A-Za-z]:)?\.[\\/][^\s"\']*\.versions[\\/][^\s"\'<>|]*)'
)
_USER_PREFIX_RE = re.compile(
    r'(?:[A-Za-z]:\\)?(?:Users|home)[\\/][^\s"\'<>|]*?[\\/]'
)

_IGNORED_WS_DIRS = {
    "__pycache__", ".venv", "venv", ".git", ".idea", ".vscode", "node_modules",
    ".lollms", "build", "dist", ".next", "env", ".env", ".lollms_code",
    ".lollms_metadata", "egg-info", "dist-info", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "htmlcov", "site-packages", "artefacts_metadata", "discussions",
    ".versions", "versions"
}
_IGNORED_WS_EXTS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib"}


# ── Context Opacity & String Sanitization ───────────────────────────────────

def sanitize_unicode(text: str) -> str:
    """Removes invisible Unicode formatting characters that corrupt functional XML parsing."""
    if not text:
        return text

    invisible_chars = [
        '\u200b', '\u200c', '\u200d', '\ufeff', '\u200e', '\u200f',
        '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u2060',
        '\u2061', '\u2062', '\u2063', '\u2064',
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    return text

_sanitize_unicode = sanitize_unicode


def sanitize_host_paths(text: str) -> str:
    """Strips absolute host filesystem paths to maintain sandbox opacity."""
    if not text:
        return text
    try:
        cleaned = _HOST_ROOT_RE.sub("<host-path>", text)
        cleaned = _USER_PREFIX_RE.sub("<host>/", cleaned)
        cleaned = cleaned.replace("\\\\?\\", "")
        return cleaned
    except Exception:
        return text

_sanitize_host_paths = sanitize_host_paths


def is_large_base64(v: str) -> bool:
    """Checks whether a string is a large base64 payload."""
    sample = v.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(sample) < 500:
        return False
    return bool(_BASE64_RE.match(sample[:1000]))

_is_large_base64 = is_large_base64


# ── JSON Repair & Tool Result Sanitization ──────────────────────────────────

def repair_llm_json(raw_text: str) -> str:
    """
    Repairs common LLM JSON malformations (unescaped literal control characters,
    multiline strings, and trailing unclosed brackets).
    """
    if not raw_text:
        return raw_text

    for candidate in (raw_text, raw_text.strip().strip("`")):
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    repaired_chars: List[str] = []
    in_string = False
    i = 0
    n = len(raw_text)

    while i < n:
        ch = raw_text[i]
        if in_string:
            if ch == "\\":
                repaired_chars.append(ch)
                if i + 1 < n:
                    repaired_chars.append(raw_text[i + 1])
                i += 2
                continue
            if ch == '"':
                j = i + 1
                while j < n and raw_text[j] in " \t\r\n":
                    j += 1
                if j >= n or raw_text[j] in ",}]:":
                    in_string = False
                    repaired_chars.append(ch)
                    i += 1
                    continue
                repaired_chars.append('\\"')
                i += 1
                continue
            if ch == "\n":
                repaired_chars.append("\\n")
                i += 1
                continue
            if ch == "\r":
                repaired_chars.append("\\r")
                i += 1
                continue
            if ch == "\t":
                repaired_chars.append("\\t")
                i += 1
                continue
            if ord(ch) < 0x20:
                i += 1
                continue
            repaired_chars.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = True
            repaired_chars.append(ch)
            i += 1
            continue

        repaired_chars.append(ch)
        i += 1

    repaired = "".join(repaired_chars)
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        pass

    balanced = repaired
    balanced += "}" * max(0, balanced.count("{") - balanced.count("}"))
    balanced += "]" * max(0, balanced.count("[") - balanced.count("]"))
    try:
        json.loads(balanced)
        return balanced
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        start = balanced.find("{")
        if start != -1:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(balanced[start:])
            return json.dumps(obj, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        pass

    return repaired

_repair_llm_json = repair_llm_json
repair_llm_tool_json = repair_llm_json
_repair_llm_tool_json = repair_llm_json


def calculate_dynamic_tool_char_limit(client: Optional[Any] = None) -> int:
    """Calculates max allowed characters for a tool result (25% of context window, clamped to 16k-90k)."""
    if client and hasattr(client, 'get_ctx_size'):
        try:
            ctx_size = client.get_ctx_size() or 0
            if ctx_size > 0:
                dynamic_limit = int((ctx_size * 0.25) * 4)
                return min(max(dynamic_limit, 16000), 90000)
        except Exception:
            pass
    return 24000

_calculate_dynamic_tool_char_limit = calculate_dynamic_tool_char_limit


def sanitize_tool_result(
    tool_res: Any,
    max_chars: Optional[int] = None,
    client: Optional[Any] = None,
) -> str:
    """Sanitizes tool outputs by unwrapping nested payloads, stripping binary blobs, and formatting failures."""
    if max_chars is None:
        max_chars = calculate_dynamic_tool_char_limit(client)

    def _find_prompt_injection(obj: Any, depth: int = 0) -> Optional[str]:
        if depth > 4:
            return None
        if isinstance(obj, dict):
            pinj = obj.get("prompt_injection")
            if isinstance(pinj, str) and pinj.strip():
                return pinj.strip()
            for v in obj.values():
                hit = _find_prompt_injection(v, depth + 1)
                if hit:
                    return hit
        elif isinstance(obj, list):
            for v in obj:
                hit = _find_prompt_injection(v, depth + 1)
                if hit:
                    return hit
        return None

    def _walk(obj: Any, depth: int = 0) -> Any:
        if depth > 6:
            return "[truncated: depth limit]"
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            if is_large_base64(obj):
                approx_kb = len(obj) * 3 / 4 / 1024
                return f"[base64 blob stripped: {approx_kb:.1f}KB]"
            if len(obj) > max_chars:
                return obj[:max_chars] + f"\n... [truncated, {len(obj) - max_chars} more chars]"
            return obj
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in _BINARY_BLOB_KEYS:
                    if isinstance(v, str) and v:
                        approx_kb = len(v) * 3 / 4 / 1024
                        cleaned[k] = f"[base64 blob stripped: {approx_kb:.1f}KB]"
                    elif isinstance(v, (list, tuple)) and v:
                        approx_kb = sum(len(x) for x in v if isinstance(x, str)) * 3 / 4 / 1024
                        cleaned[k] = f"[list of {len(v)} base64 blobs stripped: {approx_kb:.1f}KB]"
                    else:
                        cleaned[k] = None
                else:
                    cleaned[k] = _walk(v, depth + 1)
            return cleaned
        if isinstance(obj, (list, tuple)):
            walked = [_walk(v, depth + 1) for v in obj[:50]]
            if len(obj) > 50:
                walked.append(f"... [truncated, {len(obj) - 50} more items]")
            return walked
        return str(obj)

    if isinstance(tool_res, str):
        if len(tool_res) > max_chars:
            return tool_res[:max_chars] + f"\n... [truncated, {len(tool_res) - max_chars} more chars]"
        return tool_res

    if isinstance(tool_res, dict):
        inner_dict = tool_res.get("output") if isinstance(tool_res.get("output"), dict) else {}
        if not inner_dict and tool_res.get("success") is False:
            inner_dict = tool_res

        is_fail = (
            tool_res.get("success") is False
            or (inner_dict and inner_dict.get("success") is False)
            or tool_res.get("status_code", 200) not in (200, 201)
            or (inner_dict and inner_dict.get("status_code", 200) not in (200, 201))
            or bool(tool_res.get("error"))
            or (inner_dict and bool(inner_dict.get("error")))
            or (tool_res.get("return_code") is not None and tool_res.get("return_code") != 0)
            or (inner_dict and inner_dict.get("return_code") is not None and inner_dict.get("return_code") != 0)
        )

        if is_fail:
            error_parts = ["⚠️ **Tool Execution Failed**"]
            error_msg = tool_res.get("error") or (inner_dict.get("error") if inner_dict else None)
            if not error_msg:
                error_msg = "Tool returned success=False with no error message."
            error_parts.append(f"**Error Details:**\n{error_msg}")

            stderr = tool_res.get("stderr") or (inner_dict.get("stderr") if inner_dict else None)
            if stderr and str(stderr).strip():
                error_parts.append(f"**Standard Error (stderr):**\n```\n{str(stderr).strip()}\n```")

            out_val = tool_res.get("output")
            if isinstance(out_val, str) and out_val.strip() and out_val.strip() != str(error_msg).strip():
                error_parts.append(f"**Output before failure:**\n{out_val.strip()}")

            tb = tool_res.get("traceback") or (inner_dict.get("traceback") if inner_dict else None)
            if tb and str(tb).strip() and str(tb).strip() not in str(error_msg):
                error_parts.append(f"**Stack Trace:**\n```\n{str(tb).strip()}\n```")

            rc = tool_res.get("return_code") if tool_res.get("return_code") is not None else (inner_dict.get("return_code") if inner_dict else None)
            if rc is not None and rc != 0:
                error_parts.append(f"**Exit Code:** {rc}")

            pinj = _find_prompt_injection(tool_res)
            if pinj:
                error_parts.append(f"\n{pinj}")

            error_text = "\n\n".join(error_parts)
            if len(error_text) > max_chars:
                error_text = error_text[:max_chars] + f"\n... [truncated, {len(error_text) - max_chars} more chars]"
            return error_text

    pinj = _find_prompt_injection(tool_res)
    if pinj:
        return f"✓ Success\n{pinj}"

    unwrapped = tool_res
    if isinstance(tool_res, dict):
        if "output" in tool_res:
            unwrapped = tool_res["output"]
            if isinstance(unwrapped, dict):
                for key in ("content", "text", "result", "data", "page_content", "summary"):
                    if key in unwrapped:
                        unwrapped = unwrapped[key]
                        break
        elif "content" in tool_res:
            unwrapped = tool_res["content"]
        elif "result" in tool_res:
            unwrapped = tool_res["result"]
        elif "data" in tool_res:
            unwrapped = tool_res["data"]

    if unwrapped is None:
        return "Tool executed successfully but returned no output content."

    def _replace_none(obj):
        if obj is None:
            return "[No output returned by tool]"
        if isinstance(obj, dict):
            return {k: _replace_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace_none(v) for v in obj]
        return obj

    sanitized = _walk(_replace_none(unwrapped))
    if isinstance(sanitized, str):
        if len(sanitized) > max_chars:
            return sanitized[:max_chars] + f"\n... [truncated, {len(sanitized) - max_chars} more chars]"
        return sanitized

    try:
        text = json.dumps(sanitized, indent=2, default=str, ensure_ascii=False)
    except Exception:
        text = str(sanitized)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
    return text

_sanitize_tool_result = sanitize_tool_result


def build_windowed_output_preview(text: str) -> str:
    """Builds a head/tail windowed preview of a tool output for UI display."""
    if not isinstance(text, str) or len(text) <= _TOOL_UI_PREVIEW_WINDOW:
        return text if isinstance(text, str) else str(text)
    head = text[:_TOOL_UI_PREVIEW_HALF]
    tail = text[-_TOOL_UI_PREVIEW_HALF:]
    stripped_chars = len(text) - _TOOL_UI_PREVIEW_WINDOW
    return (
        f"{head}\n"
        f"... [stripped for brevity — {stripped_chars} middle characters omitted] ...\n"
        f"{tail}"
    )

_build_windowed_output_preview = build_windowed_output_preview


def build_progressive_continuation_prompt(stall_count: int, recent_tools: Optional[List[str]] = None) -> str:
    """Standardized progressive prompt when the model stalls without action tags or <done/>."""
    recent_ctx = f" Recent actions executed: {recent_tools}." if recent_tools else ""
    if stall_count <= 1:
        return (
            f"[SYSTEM DIRECTIVE: You wrote conversational text without executing an action tag or emitting `<done/>`.{recent_ctx}\n"
            "Conversational declarations and apologies DO NOT execute tools or create files.\n"
            "MANDATORY: Output the functional XML tag (`<tool>`, `<artifact>`, `<unlock_file>`, `<generate_image>`) as your next step.\n"
            "If your task is completely finished, write your answer and emit `<done/>` on a new line.]"
        )
    elif stall_count == 2:
        return (
            f"[SYSTEM: ACTION REQUIRED — You have produced conversational text without action tags or `<done/>` for 2 consecutive turns.{recent_ctx}\n"
            "You MUST output the functional tag IMMEDIATELY, or output `<done/>` to terminate.\n"
            "Do NOT output conversational apologies or introductory preambles.]"
        )
    else:
        return (
            f"[SYSTEM: CRITICAL — You have stalled {stall_count} times without producing an action tag or `<done/>`.\n"
            "Emit `<done/>` on a new line NOW to terminate the turn.]"
        )

_build_progressive_continuation_prompt = build_progressive_continuation_prompt


def inject_tool_images_for_vlm(tool_result: Dict[str, Any], client: Optional[Any] = None) -> List[Dict[str, str]]:
    """Extracts base64 images from a tool result and formats them as image_url content blocks for VLM models."""
    if not client:
        return []

    vision_capable = False
    try:
        if hasattr(client, 'llm') and hasattr(client.llm, 'supports_vision'):
            vision_capable = bool(client.llm.supports_vision)
        elif hasattr(client, 'supports_vision'):
            vision_capable = bool(client.supports_vision)
        elif hasattr(client, 'has_vision_capability'):
            vision_capable = bool(client.has_vision_capability())
    except Exception:
        pass

    if not vision_capable:
        return []

    raw_images = tool_result.get("images") or tool_result.get("image_b64") or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not raw_images or not isinstance(raw_images, list):
        return []

    media_types = tool_result.get("image_media_types") or []
    content_blocks: List[Dict[str, str]] = []

    for idx, img_b64 in enumerate(raw_images):
        if not isinstance(img_b64, str) or not img_b64.strip():
            continue
        mtype = media_types[idx] if idx < len(media_types) else "image/png"
        if ";" in img_b64 and "base64," in img_b64:
            url = img_b64
        else:
            url = f"data:{mtype};base64,{img_b64}"
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    return content_blocks

_inject_tool_images_for_vlm = inject_tool_images_for_vlm


def dump_error(
    error: Exception,
    context_desc: str,
    round_count: int,
    workspace_dir: Optional[Path] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False,
) -> None:
    """Unified error forensic dumper writing to _debug_dumps/."""
    if not debug_mode or not workspace_dir:
        return

    try:
        debug_dir = workspace_dir / "_debug_dumps"
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_context = re.sub(r"[^a-z0-9_]+", "_", context_desc.lower()).strip("_") or "error"
        error_log_path = debug_dir / f"error_round_{round_count}_{safe_context}.log"

        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"🐛 [DEBUG] ERROR DUMP - ROUND {round_count}\n")
            f.write(f"Context: {context_desc}\n")
            f.write("=" * 80 + "\n\n")

            f.write("--- EXCEPTION ---\n")
            f.write(f"Type: {type(error).__name__}\n")
            f.write(sanitize_host_paths(str(error)) + "\n\n")

            f.write("--- TRACEBACK ---\n")
            f.write(sanitize_host_paths(traceback.format_exc()))
            f.write("\n\n")

            if extra_data:
                f.write("--- EXTRA DATA ---\n")
                try:
                    f.write(sanitize_host_paths(
                        json.dumps(extra_data, indent=2, default=str, ensure_ascii=False)
                    ))
                except Exception:
                    f.write(sanitize_host_paths(str(extra_data)))
                f.write("\n\n")

        ASCIIColors.error(f"[Lollms] 🐛 Error dumped to: {error_log_path}")
    except Exception as dump_err:
        ASCIIColors.warning(f"[Lollms] Failed to write error dump: {dump_err}")

_dump_error = dump_error


# ── Structural Symbol Detection ────────────────────────────────────────────

def detect_structural_symbols(buffer: str, language: Optional[str] = None, art_type: str = "code") -> List[Dict[str, Any]]:
    """
    Parses a code or text buffer and extracts high-level structural symbols
    across Markdown, Python, JS/TS, Rust, Go, C/C++/Java, HTML, CSS, and SQL.
    """
    if not buffer:
        return []

    lines = buffer.splitlines()
    symbols: List[Dict[str, Any]] = []
    lang = (language or "").lower()
    in_py_class: Optional[str] = None

    for idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        line_num = idx + 1

        # 1. Markdown / Headings
        if lang in ("markdown", "md") or art_type in ("document", "note", "skill", "scratchpad", "presentation") or not lang:
            m = re.match(r'^(#{1,6})\s+(.+)$', line_str)
            if m:
                level = len(m.group(1))
                h_type = "heading" if level > 3 else ("major_section" if level == 1 else ("section" if level == 2 else "subsection"))
                name = m.group(2).strip()
                symbols.append({
                    "symbol_type": h_type,
                    "symbol_name": name,
                    "level": level,
                    "line": line_num,
                    "detail": f"{h_type.replace('_', ' ').capitalize()}: {name}",
                    "signature": line_str
                })
                continue

        # 2. Python
        if lang == "python" or art_type in ("code", "tool"):
            m_class = re.match(r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\((.*?)\))?\s*:', line_str)
            if m_class:
                c_name = m_class.group(1)
                bases = m_class.group(2) or ""
                in_py_class = c_name
                symbols.append({
                    "symbol_type": "class",
                    "symbol_name": c_name,
                    "line": line_num,
                    "detail": f"Class {c_name}" + (f"({bases})" if bases else ""),
                    "signature": line_str.rstrip(":")
                })
                continue

            m_func = re.match(r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', line_str)
            if m_func:
                f_name = m_func.group(1)
                args = m_func.group(2)
                is_async = line_str.startswith("async ")
                indent = len(line) - len(line.lstrip())
                is_method = bool(indent > 0 and in_py_class) or "self" in args or "cls" in args

                if is_method:
                    sym_type = "async_method" if is_async else "method"
                    parent_ctx = f" in {in_py_class}" if in_py_class else ""
                    detail = f"{'Async Method' if is_async else 'Method'} {f_name}{parent_ctx}"
                else:
                    in_py_class = None
                    sym_type = "async_function" if is_async else "function"
                    detail = f"{'Async Function' if is_async else 'Function'} {f_name}"

                symbols.append({
                    "symbol_type": sym_type,
                    "symbol_name": f_name,
                    "parent_class": in_py_class if is_method else None,
                    "line": line_num,
                    "detail": detail,
                    "signature": f"{'async ' if is_async else ''}def {f_name}({args})"
                })
                continue

        # 3. JavaScript / TypeScript
        if lang in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
            m_ts = re.match(r'^(?:export\s+)?(interface|type|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_ts:
                kind, name = m_ts.group(1), m_ts.group(2)
                symbols.append({
                    "symbol_type": kind,
                    "symbol_name": name,
                    "line": line_num,
                    "detail": f"{kind.capitalize()} {name}",
                    "signature": line_str
                })
                continue

            m_class = re.match(r'^(?:export\s+)?(?:default\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_class:
                c_name = m_class.group(1)
                symbols.append({
                    "symbol_type": "class",
                    "symbol_name": c_name,
                    "line": line_num,
                    "detail": f"Class {c_name}",
                    "signature": line_str
                })
                continue

            m_func = re.match(r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line_str)
            if m_func:
                f_name = m_func.group(1)
                is_async = "async " in line_str
                symbols.append({
                    "symbol_type": "async_function" if is_async else "function",
                    "symbol_name": f_name,
                    "line": line_num,
                    "detail": f"{'Async Function' if is_async else 'Function'} {f_name}",
                    "signature": line_str
                })
                continue

            m_arrow = re.match(r'^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_]*)\s*=>', line_str)
            if m_arrow:
                f_name = m_arrow.group(1)
                is_hook = f_name.startswith("use") and len(f_name) > 3 and f_name[3].isupper()
                is_component = f_name[0].isupper()
                sym_type = "react_hook" if is_hook else ("react_component" if is_component else "arrow_function")
                detail = f"Hook {f_name}" if is_hook else (f"Component <{f_name} />" if is_component else f"Function {f_name}")
                symbols.append({
                    "symbol_type": sym_type,
                    "symbol_name": f_name,
                    "line": line_num,
                    "detail": detail,
                    "signature": line_str
                })
                continue

        # 4. Rust
        if lang in ("rust", "rs"):
            m_rust = re.match(r'^(?:pub\s+)?(struct|enum|trait|union|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_rust:
                kind, name = m_rust.group(1), m_rust.group(2)
                symbols.append({
                    "symbol_type": kind,
                    "symbol_name": name,
                    "line": line_num,
                    "detail": f"Rust {kind.capitalize()} {name}",
                    "signature": line_str
                })
                continue

            m_impl = re.match(r'^impl(?:\s*<[^>]*>)?\s+(?:([a-zA-Z_][a-zA-Z0-9_]*)\s+for\s+)?([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_impl:
                trait_name, target = m_impl.group(1), m_impl.group(2)
                desc = f"Impl {trait_name} for {target}" if trait_name else f"Impl {target}"
                symbols.append({
                    "symbol_type": "impl",
                    "symbol_name": target,
                    "line": line_num,
                    "detail": desc,
                    "signature": line_str
                })
                continue

            m_fn = re.match(r'^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_fn:
                f_name = m_fn.group(1)
                symbols.append({
                    "symbol_type": "function",
                    "symbol_name": f_name,
                    "line": line_num,
                    "detail": f"Function fn {f_name}()",
                    "signature": line_str
                })
                continue

        # 5. Go
        if lang in ("go", "golang"):
            m_go_type = re.match(r'^type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(struct|interface)', line_str)
            if m_go_type:
                name, kind = m_go_type.group(1), m_go_type.group(2)
                symbols.append({
                    "symbol_type": kind,
                    "symbol_name": name,
                    "line": line_num,
                    "detail": f"Go {kind.capitalize()} {name}",
                    "signature": line_str
                })
                continue

            m_go_func = re.match(r'^func\s+(?:\((?:[^)]+)\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line_str)
            if m_go_func:
                f_name = m_go_func.group(1)
                symbols.append({
                    "symbol_type": "function",
                    "symbol_name": f_name,
                    "line": line_num,
                    "detail": f"Go Func {f_name}()",
                    "signature": line_str
                })
                continue

        # 6. C / C++ / C# / Java
        if lang in ("c", "cpp", "c++", "csharp", "cs", "java"):
            m_oop = re.match(r'^(?:public|private|protected|internal|static|abstract|sealed|final|\s)*\s*(class|struct|interface|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_oop:
                kind, name = m_oop.group(1), m_oop.group(2)
                symbols.append({
                    "symbol_type": kind,
                    "symbol_name": name,
                    "line": line_num,
                    "detail": f"{kind.capitalize()} {name}",
                    "signature": line_str
                })
                continue

        # 7. HTML
        if lang == "html":
            m_html = re.match(r'<(\w+)(?:\s+[^>]*)?(?:id|class)=["\']([^"\']*)["\']', line_str, re.IGNORECASE)
            if m_html and m_html.group(1).lower() in ("section", "article", "main", "nav", "header", "footer", "form", "table", "dialog", "aside"):
                tag, id_or_cls = m_html.group(1), m_html.group(2)
                symbols.append({
                    "symbol_type": "html_element",
                    "symbol_name": f"<{tag} {id_or_cls}>",
                    "line": line_num,
                    "detail": f"HTML <{tag}> ({id_or_cls})",
                    "signature": line_str
                })
                continue

        # 8. CSS / SCSS
        if lang in ("css", "scss", "sass", "less"):
            m_css = re.match(r'^([.#@][a-zA-Z0-9_\-:\s,>+~]+)\s*\{', line_str)
            if m_css:
                sel = m_css.group(1).strip()
                symbols.append({
                    "symbol_type": "css_selector",
                    "symbol_name": sel,
                    "line": line_num,
                    "detail": f"CSS {sel}",
                    "signature": line_str
                })
                continue

        # 9. SQL
        if lang == "sql":
            m_sql = re.match(r'^(CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|PROCEDURE|FUNCTION|INDEX)|ALTER\s+TABLE)\s+([a-zA-Z0-9_\."`]+)', line_str, re.IGNORECASE)
            if m_sql:
                stmt_type, target = m_sql.group(1).upper(), m_sql.group(2)
                symbols.append({
                    "symbol_type": "sql_statement",
                    "symbol_name": f"{stmt_type} {target}",
                    "line": line_num,
                    "detail": f"SQL {stmt_type} {target}",
                    "signature": line_str
                })
                continue

    return symbols

_detect_structural_symbols = detect_structural_symbols


def extract_artefact_meta(buffer: str, language: Optional[str] = None, art_type: str = "code") -> Dict[str, Any]:
    """Extracts line counts, tokens, patch hunks, sections, and previews from an artifact buffer."""
    if not buffer:
        return {
            "line_count": 0,
            "size_chars": 0,
            "estimated_tokens": 0,
            "is_patch": False,
            "current_section": None,
            "sections": [],
            "sections_count": 0,
            "patch_stats": None,
            "preview": ""
        }

    lines = buffer.splitlines()
    line_count = len(lines)
    size_chars = len(buffer)
    estimated_tokens = size_chars // 4

    is_patch = "<<<<<<< SEARCH" in buffer
    patch_stats = None
    if is_patch:
        search_count = len(re.findall(r'^<{6,8}(?:\s*\w+)?\s*$', buffer, re.MULTILINE))
        replace_count = len(re.findall(r'^={5,}\s*$', buffer, re.MULTILINE))
        has_end_replace = bool(re.search(r'^>{6,8}(?:\s*\w+)?\s*$', buffer, re.MULTILINE))
        patch_stats = {
            "hunks_count": search_count,
            "is_complete_hunk": search_count > 0 and search_count == replace_count and has_end_replace
        }

    detected_symbols = detect_structural_symbols(buffer, language, art_type)
    current_section = detected_symbols[-1]["detail"] if detected_symbols else None

    sections = [
        {"type": s["symbol_type"], "name": s["symbol_name"], "line": s["line"], "detail": s["detail"]}
        for s in detected_symbols
    ]
    capped_sections = sections[-20:] if len(sections) > 20 else sections
    preview = lines[-1].strip()[:120] if lines else ""

    return {
        "line_count": line_count,
        "size_chars": size_chars,
        "estimated_tokens": estimated_tokens,
        "is_patch": is_patch,
        "current_section": current_section,
        "sections": capped_sections,
        "sections_count": len(sections),
        "patch_stats": patch_stats,
        "preview": preview,
    }

_extract_artefact_meta = extract_artefact_meta


# ── Workspace Snapshotting & Diffing ───────────────────────────────────────

def take_workspace_snapshot(workspace_dir: Path) -> Dict[Path, Dict[str, Any]]:
    """Takes a non-recursive/depth-capped snapshot of files and hashes."""
    snapshot: Dict[Path, Dict[str, Any]] = {}
    if not workspace_dir or not workspace_dir.exists():
        return snapshot

    _MAX_SNAPSHOT_FILES = 5000
    _MAX_HASH_SIZE = 512 * 1024
    files_scanned = 0

    try:
        for f in workspace_dir.rglob("*"):
            if files_scanned >= _MAX_SNAPSHOT_FILES:
                break
            if not f.is_file():
                continue
            rel_parts = f.relative_to(workspace_dir).parts
            if any(part in _IGNORED_WS_DIRS for part in rel_parts):
                continue
            if f.suffix.lower() in _IGNORED_WS_EXTS:
                continue

            files_scanned += 1
            rel_path = f.relative_to(workspace_dir)

            try:
                file_size = f.stat().st_size
                file_hash = None
                if file_size < _MAX_HASH_SIZE:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    file_hash = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()

                snapshot[rel_path] = {
                    "hash": file_hash,
                    "size": file_size,
                    "mtime": f.stat().st_mtime,
                    "path": f
                }
            except Exception:
                try:
                    snapshot[rel_path] = {
                        "hash": None,
                        "size": f.stat().st_size,
                        "mtime": f.stat().st_mtime,
                        "path": f
                    }
                except Exception:
                    pass
    except Exception as e:
        ASCIIColors.warning(f"[Snapshot] Workspace snapshot warning: {e}")

    return snapshot

_take_workspace_snapshot = take_workspace_snapshot


def sync_workspace_diff(files_before: Dict, files_after: Dict) -> List[Dict[str, Any]]:
    """Calculates created and modified files between two snapshots."""
    changes = []
    new_files = set(files_after.keys()) - set(files_before.keys())
    for rel_path in new_files:
        file_info = files_after[rel_path]
        changes.append({"action": "created", "path": str(rel_path), "size": file_info.get("size", 0)})

    common_files = set(files_after.keys()) & set(files_before.keys())
    for rel_path in common_files:
        before_hash = files_before[rel_path].get("hash")
        after_hash = files_after[rel_path].get("hash")
        before_mtime = files_before[rel_path].get("mtime")
        after_mtime = files_after[rel_path].get("mtime")
        if (before_hash is not None and after_hash is not None and before_hash != after_hash) or (before_mtime != after_mtime):
            changes.append({"action": "modified", "path": str(rel_path), "size": files_after[rel_path].get("size", 0)})

    return changes

_sync_workspace_diff = sync_workspace_diff


# ── Tool Execution Core ────────────────────────────────────────────────────

def execute_tool_call(
    tool_name: str,
    tool_params: Dict[str, Any],
    active_tools: Dict[str, Any],
    workspace_dir: Path,
    lollms_client: Any = None,
    discussion_instance: Any = None
) -> Dict[str, Any]:
    """
    Executes a tool call in a strictly sandboxed CWD, sanitizing path parameters
    and capturing outputs and exceptions cleanly.
    """
    old_cwd = os.getcwd()
    ws_dir = workspace_dir.resolve() if workspace_dir else Path(".").resolve()
    ws_dir.mkdir(parents=True, exist_ok=True)
    ws_dir_str = str(ws_dir)

    try:
        os.chdir(ws_dir_str)

        sanitized_params = {}
        for key, value in tool_params.items():
            if isinstance(value, str):
                sanitized_value = value
                for prefix in ["workspace/", "data_workspace/", "./workspace/", "./data_workspace/"]:
                    if sanitized_value.lower().startswith(prefix):
                        sanitized_value = sanitized_value[len(prefix):]
                        break
                disc_id = getattr(discussion_instance, 'id', None)
                if disc_id and sanitized_value.lower().startswith(disc_id.lower() + "/"):
                    sanitized_value = sanitized_value[len(disc_id) + 1:]
                sanitized_params[key] = sanitized_value
            else:
                sanitized_params[key] = value

        tool_def = active_tools.get(tool_name, {})
        lcp_binding = getattr(lollms_client, 'tools', None)

        if "callable" in tool_def:
            call_kwargs = dict(sanitized_params)
            _sig = inspect.signature(tool_def["callable"]).parameters
            if "discussion_instance" in _sig:
                call_kwargs["discussion_instance"] = discussion_instance
            if "lollms_client_instance" in _sig:
                call_kwargs["lollms_client_instance"] = lollms_client

            try:
                result = tool_def["callable"](**call_kwargs)
                if isinstance(result, dict):
                    return result
                if result is None:
                    return {"success": False, "error": f"Tool '{tool_name}' returned None."}
                return {"success": True, "output": str(result)}
            except Exception as exec_err:
                trace_exception(exec_err)
                return {
                    "success": False,
                    "error": sanitize_host_paths(f"Tool '{tool_name}' crashed: {exec_err}"),
                    "traceback": sanitize_host_paths(traceback.format_exc())
                }

        elif lcp_binding and hasattr(lcp_binding, 'execute_tool'):
            try:
                result = lcp_binding.execute_tool(
                    tool_name,
                    sanitized_params,
                    discussion_instance=discussion_instance,
                    lollms_client_instance=lollms_client
                )
                if isinstance(result, dict):
                    return result
                if result is None:
                    return {"success": False, "error": f"LCP tool '{tool_name}' returned None."}
                return {"success": True, "output": str(result)}
            except Exception as lcp_err:
                trace_exception(lcp_err)
                return {
                    "success": False,
                    "error": sanitize_host_paths(f"LCP tool '{tool_name}' crashed: {lcp_err}"),
                    "traceback": sanitize_host_paths(traceback.format_exc())
                }

        else:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' has no callable and no LCP binding available.",
                "status_code": 404
            }

    finally:
        os.chdir(old_cwd)

_execute_tool_call = execute_tool_call


# ── Context Visibility Operations ──────────────────────────────────────────

def execute_context_visibility_operation(
    tag_name: str,
    body: str,
    artefact_manager: Any,
    workspace_dir: Optional[Path] = None,
    client: Any = None,
    state_db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Standardized execution of visibility tags: <unlock_file>, <lock_file>, <hide_file>,
    <pin_file>, <unpin_file>, <collapse_folder>, <uncollapse_folder>.
    """
    if not artefact_manager:
        return {"status_str": "[SYSTEM ERROR] Artefact system not initialized.", "processed_files": []}

    ArtefactVisibility = _get_artefact_visibility()

    target_visibility = ArtefactVisibility.FULL
    action_verb = "Unlocking"
    if tag_name == "lock_file":
        target_visibility = ArtefactVisibility.TREE_LOCKED
        action_verb = "Locking"
    elif tag_name == "hide_file":
        target_visibility = ArtefactVisibility.HIDDEN
        action_verb = "Hiding"
    elif tag_name == "collapse_folder":
        target_visibility = ArtefactVisibility.FOLDER_COLLAPSED
        action_verb = "Collapsing"
    elif tag_name == "uncollapse_folder":
        target_visibility = ArtefactVisibility.TREE_UNLOCKABLE
        action_verb = "Uncollapsing"
    elif tag_name == "pin_file":
        target_visibility = ArtefactVisibility.PINNED
        action_verb = "Pinning"
    elif tag_name == "unpin_file":
        target_visibility = ArtefactVisibility.FULL
        action_verb = "Unpinning"

    clean_body = body
    if "<" in body and ">" in body:
        xml_bodies = re.findall(r'<[^>]+>(.*?)</[^>]+>', body, re.DOTALL)
        if xml_bodies:
            clean_body = "\n".join(xml_bodies)

    raw_targets = re.split(r'[\n,;]+', clean_body)
    targets = [t.strip().replace("\\", "/") for t in raw_targets if t.strip()]

    all_arts = artefact_manager._get_all_raw()
    processed_files = []
    already_in_state = []
    not_found = []
    blocked_files = []
    loaded_contents: Dict[str, str] = {}

    max_ctx = 0
    if client and hasattr(client, 'get_ctx_size'):
        try:
            max_ctx = client.get_ctx_size() or 0
        except Exception:
            max_ctx = 0

    _MAX_UNLOCK_TOKENS = int(max_ctx * 0.95) if max_ctx > 0 else 50000

    def _norm_path(p: str) -> str:
        p = p.replace("\\", "/").strip()
        for prefix in ("./", "workspace/", "data_workspace/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
        return p.lower()

    for t_target in targets:
        if tag_name in ("collapse_folder", "uncollapse_folder") and state_db_path:
            import sqlite3
            folder_normalized = t_target.rstrip("/")
            if not folder_normalized:
                continue
            try:
                conn = sqlite3.connect(str(state_db_path))
                cursor = conn.cursor()
                if tag_name == "collapse_folder":
                    cursor.execute("INSERT OR REPLACE INTO collapsed_folders (path) VALUES (?)", (folder_normalized,))
                else:
                    cursor.execute("DELETE FROM collapsed_folders WHERE path = ?", (folder_normalized,))
                conn.commit()
                conn.close()
                processed_files.append(folder_normalized)
                continue
            except Exception as e:
                ASCIIColors.warning(f"[Visibility] Folder DB error: {e}")
                continue

        art = next((a for a in all_arts if _norm_path(a.get("title", "")) == _norm_path(t_target)), None)
        if not art:
            art = next((a for a in all_arts if _norm_path(a.get("physical_path", "")) == _norm_path(t_target)), None)

        if not art and workspace_dir:
            disk_path = workspace_dir / t_target
            if disk_path.is_file():
                try:
                    imported = artefact_manager.import_file(file_path=disk_path, title=t_target, active=False)
                    if imported:
                        all_arts = artefact_manager._get_all_raw()
                        art = next((a for a in all_arts if _norm_path(a.get("title", "")) == _norm_path(t_target)), None)
                except Exception as ex:
                    ASCIIColors.warning(f"[Visibility] On-demand import failed for '{t_target}': {ex}")

        if not art:
            not_found.append(t_target)
            continue

        current_vis = art.get("visibility")
        if current_vis == target_visibility:
            already_in_state.append(art["title"])
            continue

        if target_visibility == ArtefactVisibility.FULL:
            content = ""
            if workspace_dir:
                file_path = workspace_dir / art["title"]
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        blocked_files.append((art["title"], 0))
                        continue

            if not content:
                content = art.get("content", "")

            token_count = len(content) // 4
            if token_count > _MAX_UNLOCK_TOKENS:
                blocked_files.append((art["title"], token_count))
                continue

            art["content"] = content
            art["token_count"] = token_count
            art["visibility"] = ArtefactVisibility.FULL
            art["active"] = True
            processed_files.append(art["title"])
            if content:
                loaded_contents[art["title"]] = content
        else:
            art["visibility"] = target_visibility
            art["active"] = False
            processed_files.append(art["title"])

    if processed_files or already_in_state:
        artefact_manager._save_all(all_arts)

    status_parts = []
    if processed_files:
        status_parts.append(f"✅ {action_verb}: {', '.join(processed_files)}")
    if already_in_state:
        status_parts.append(f"⚠️ Already in target state: {', '.join(already_in_state)}")
    if not_found:
        status_parts.append(f"❌ Not found: {', '.join(not_found)}")
    if blocked_files:
        blocked_desc = "; ".join(f"{bf} (~{tc:,} tokens)" if tc > 0 else f"{bf} (Read Error)" for bf, tc in blocked_files)
        status_parts.append(f"🛑 BLOCKED: {blocked_desc}. File exceeds context budget.")

    status_meta = "failure" if (not_found or blocked_files) and not processed_files else "success"
    status_str = f"{action_verb} context files...\nContext Update:\n{'; '.join(status_parts)}\n<!-- status:{status_meta} -->"

    return {
        "status_str": status_str,
        "processed_files": processed_files,
        "already_in_state": already_in_state,
        "not_found": not_found,
        "blocked_files": blocked_files,
        "loaded_contents": loaded_contents,
        "success": status_meta == "success"
    }

_execute_context_visibility = execute_context_visibility_operation


# ── Two-View Orchestrator & User Context Formatting ─────────────────────────

def format_orchestrator_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Renders the detailed model/orchestrator view.
    Preserves all plan, delegation, and verification facts verbatim so the orchestrator
    maintains complete coordination awareness.
    """
    formatted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            content = sanitize_host_paths(content)
        formatted.append({"role": role, "content": content})
    return formatted


def format_user_view(text: str, event_mode: Any = None) -> str:
    """
    Renders the clean user-facing view from raw model output.
    Strips internal orchestration tags, processing blocks, and thoughts while preserving natural conversational prose.
    """
    if not text:
        return ""
    cleaned = re.sub(r'<think\b[^>]*>.*?(?:</think>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<thought\b[^>]*>.*?(?:</thought>|$)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<(?:plan|delegate|verify)\b[^>]*>.*?</(?:plan|delegate|verify)>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'</?processing[^>]*>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<!--\s*status:[^>]*-->', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<(?:done|end)\s*/?>', '', cleaned, flags=re.IGNORECASE)
    cleaned = sanitize_host_paths(cleaned).strip()
    return cleaned


__all__ = [
    "calculate_dynamic_tool_char_limit",
    "_calculate_dynamic_tool_char_limit",
    "repair_llm_json",
    "_repair_llm_json",
    "repair_llm_tool_json",
    "_repair_llm_tool_json",
    "is_large_base64",
    "_is_large_base64",
    "sanitize_tool_result",
    "_sanitize_tool_result",
    "detect_structural_symbols",
    "_detect_structural_symbols",
    "extract_artefact_meta",
    "_extract_artefact_meta",
    "build_progressive_continuation_prompt",
    "_build_progressive_continuation_prompt",
    "inject_tool_images_for_vlm",
    "_inject_tool_images_for_vlm",
    "dump_error",
    "_dump_error",
    "take_workspace_snapshot",
    "_take_workspace_snapshot",
    "sync_workspace_diff",
    "_sync_workspace_diff",
    "execute_tool_call",
    "_execute_tool_call",
    "execute_context_visibility_operation",
    "_execute_context_visibility_operation",
    "_execute_context_visibility",
    "build_windowed_output_preview",
    "_build_windowed_output_preview",
    "sanitize_unicode",
    "_sanitize_unicode",
    "sanitize_host_paths",
    "_sanitize_host_paths",
    "_is_tool_binding",
    "is_tool_binding",
    "format_orchestrator_history",
    "format_user_view",
    "_BASE64_RE",
    "_BINARY_BLOB_KEYS",
]