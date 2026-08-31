# lollms_discussion/_mixin_chat.py
# ─────────────────────────────────────────────────────────────────────────────
# ChatMixin — High-performance single-agent conversational turn loop with 
#             dynamic in-process Spinoff Sub-Agent Tools.
#
# Resolves RAG pre-hydration, tiered memory, and direct inline tool calls,
# exposing specialized sub-agents as executable tools to preserve KV-cache.

import re
import json
import uuid
import traceback
import threading
import random
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from types import SimpleNamespace
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE, EventMode
from ._message import LollmsMessage
from lollms_client.lollms_artefact import ArtefactType, make_image_id, ArtefactVisibility, ArtefactStatus
from lollms_client.lollms_memory import FailureMemory
from lollms_client.lollms_personality.lollms_personality import _is_tool_binding
_MAX_BRACKET_BUF = 256

_HEARTBEAT_MESSAGES = [
    "✍️ Writing content...",
    "🧠 Structuring code...",
    "⏳ Building components...",
    "🏗️ Assembling sections...",
    "✨ Crafting artifact...",
    "🔧 Refining logic...",
]

# Type-specific initial messages for artifact processing blocks
_ARTEFACT_TYPE_MESSAGES = {
    "code": "💻 Writing code...",
    "python": "🐍 Writing Python script...",
    "javascript": "🟨 Writing JavaScript...",
    "html": "🌐 Building HTML structure...",
    "css": "🎨 Styling with CSS...",
    "data": "📊 Analyzing data structure...",
    "document": "📄 Drafting document...",
    "markdown": "📝 Writing Markdown...",
    "image": "🖼️ Preparing image generation...",
    "presentation": "📽️ Building presentation slide...",
    "note": "🗒️ Saving note...",
    "skill": "🧠 Compiling skill...",
    "tool": "🛠️ Forging tool...",
}

# ── Fast Artefact Replicas (Defaults) ────────────────────────────────────────
_DEFAULT_FAST_REPLICAS = [
    "* Artifact created instantly (empty body intercepted).\n",
    "* That was fast! Artifact created with an empty body.\n",
    "* Instant artifact creation detected. No content was intercepted.\n",
    "* Done in a flash! The artifact was created too quickly to capture content.\n",
]

_TAG_STARTS = [
    "<tool>",
    "</arg_key>", "<think ",
    "<artifact", "<artefact",
    "<generate_image", "<edit_image",
    "<note", "<skill", "<scratchpad",
    "<lollms_inline",
    "<lollms_form",
    "<mem_new", "<mem_update", "<mem_tag", "<mem_load", "<mem_delete", "<mem_search", "<mem_rel",
]

# CRITICAL: Memory tags that should NEVER be treated as tool calls
# These are infrastructure tags processed silently by the memory system.
# The LLM must NEVER wrap them in <tool>...</tool> blocks.
_MEMORY_TAGS = {
    "<mem_new", "<mem_update", "<mem_tag", "<mem_load", "<mem_delete", "<mem_search", "<mem_rel"
}

# Tool names that should NEVER be called (they're memory tags, not tools)
_FORBIDDEN_TOOL_NAMES = {
    "memory_search", "mem_search", "mem_new", "mem_update", "mem_tag", 
    "mem_load", "mem_delete", "mem_rel", "memory_new", "memory_update",
    "memory_tag", "memory_load", "memory_delete", "memory_rel"
}

_SECONDARY_TAG_MAP = {
    "<artifact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK, MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artifact>"),
    "<artefact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK, MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artefact>"),
    "<note":          ("note_start",          MSG_TYPE.MSG_TYPE_NOTE_CHUNK,     MSG_TYPE.MSG_TYPE_NOTE_DONE,         "</note>"),
    "<skill":         ("skill_start",         MSG_TYPE.MSG_TYPE_SKILL_CHUNK,    MSG_TYPE.MSG_TYPE_SKILL_DONE,        "</skill>"),
    "<lollms_inline": ("inline_widget_start", MSG_TYPE.MSG_TYPE_WIDGET_CHUNK,   MSG_TYPE.MSG_TYPE_WIDGET_DONE,       "</lollms_inline>"),
    "<lollms_form":   ("form_start",          MSG_TYPE.MSG_TYPE_FORM_READY,     MSG_TYPE.MSG_TYPE_FORM_READY,        "</lollms_form>"),
    "<mem_new":       ("memory_new",          MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</mem_new>"),
    "<mem_update":    ("memory_update",       MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</mem_update>"),
    "<think>":        ("thought_start",       MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK,  MSG_TYPE.MSG_TYPE_INFO,              "</think>"),
    "<think":         ("thought_start",       MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK,  MSG_TYPE.MSG_TYPE_INFO,              "</think>"),
    "<unlock_file":   ("context_unlock",      MSG_TYPE.MSG_TYPE_INFO,    MSG_TYPE.MSG_TYPE_INFO,              "</unlock_file>"),
    "lock_file":     ("context_lock",        MSG_TYPE.MSG_TYPE_INFO,    MSG_TYPE.MSG_TYPE_INFO,              "</lock_file>"),
    "hide_file":     ("context_hide",        MSG_TYPE.MSG_TYPE_INFO,    MSG_TYPE.MSG_TYPE_INFO,              "</hide_file>"),
}

_SECONDARY_TAG_MAP = {
    "<artifact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK, MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artifact>"),
    "<artefact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK, MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artefact>"),
    "<note":          ("note_start",          MSG_TYPE.MSG_TYPE_NOTE_CHUNK,     MSG_TYPE.MSG_TYPE_NOTE_DONE,         "</note>"),
    "<skill":         ("skill_start",         MSG_TYPE.MSG_TYPE_SKILL_CHUNK,    MSG_TYPE.MSG_TYPE_SKILL_DONE,        "</skill>"),
    "<lollms_inline": ("inline_widget_start", MSG_TYPE.MSG_TYPE_WIDGET_CHUNK,   MSG_TYPE.MSG_TYPE_WIDGET_DONE,       "</lollms_inline>"),
    "<lollms_form":   ("form_start",          MSG_TYPE.MSG_TYPE_FORM_READY,     MSG_TYPE.MSG_TYPE_FORM_READY,        "</lollms_form>"),
    "<mem_new":       ("memory_new",          MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</mem_new>"),
    "<mem_update":    ("memory_update",       MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</mem_update>"),
    "":        ("thought_start",       MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK,  MSG_TYPE.MSG_TYPE_INFO,              ""),
    "<think":         ("thought_start",       MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK,  MSG_TYPE.MSG_TYPE_INFO,              ""),
    "<unlock_file":   ("context_unlock",      MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</unlock_file>"),
    "<lock_file":     ("context_lock",        MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</lock_file>"),
    "<hide_file":     ("context_hide",        MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</hide_file>"),
}


def _cb(callback: Optional[Callable], text: str, msg_type: MSG_TYPE, meta: Optional[Dict] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        return result is not False
    except Exception as e:
        trace_exception(e)
    return True


_BASE64_RE = re.compile(r'^[A-Za-z0-9+/=\s]{500,}$')

_BINARY_BLOB_KEYS = {
    "plot_b64", "image_b64", "audio_b64", "video_b64", "file_b64",
    "screenshot_b64", "pdf_b64", "thumbnail_b64", "base64",
    "binary", "raw_image", "image_data", "raw_data",
}

_MAX_TOOL_RESULT_CHARS = 12000

def _calculate_dynamic_tool_char_limit(client: Optional[Any] = None) -> int:
    """
    Calculates the maximum allowed characters for a tool result based on the LLM's context size.
    Uses 25% of the context window, capped at 50,000 chars to preserve conversation space.
    Falls back to 12,000 chars if context size is unavailable.
    """
    if client and hasattr(client, 'get_ctx_size'):
        try:
            ctx_size = client.get_ctx_size() or 0
            if ctx_size > 0:
                dynamic_limit = int((ctx_size * 0.25) * 4)
                return min(max(dynamic_limit, 8000), 50000)
        except Exception:
            pass
    return 12000


import time as _time

def _detect_structural_symbols(buffer: str, language: Optional[str] = None, art_type: str = "code") -> List[Dict[str, Any]]:
    """
    Parses an artifact buffer and extracts all detected structural symbols:
    - Markdown: headings (# H1, ## H2, ### H3, #### H4)
    - Python: classes, methods (self/cls), functions, async functions, decorators
    - JS/TS: classes, interfaces, types, enums, functions, arrow functions, React components/hooks
    - Rust: structs, enums, traits, impls, functions (fn)
    - Go: structs, interfaces, functions (func), methods
    - C/C++/C#/Java: classes, structs, interfaces, methods, functions
    - HTML: major semantic elements (<section>, <article>, <main>, <nav>, <header>, <footer>, <form>, <table>)
    - CSS: rule selectors (.class, #id, @media, @keyframes)
    - SQL: statements (CREATE TABLE/VIEW, ALTER, SELECT, INSERT, etc.)
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

        # ── 1. Markdown / Documentation Headings ──
        if lang in ("markdown", "md") or art_type in ("document", "note", "skill", "scratchpad", "presentation") or not lang:
            m = re.match(r'^(#{1,6})\s+(.+)$', line_str)
            if m:
                level = len(m.group(1))
                h_type = "heading"
                if level == 1:
                    h_type = "major_section"
                elif level == 2:
                    h_type = "section"
                elif level == 3:
                    h_type = "subsection"
                else:
                    h_type = f"h{level}_heading"

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

        # ── 2. Python Constructs ──
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

        # ── 3. JavaScript / TypeScript / JSX / TSX ──
        if lang in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
            m_ts = re.match(r'^(?:export\s+)?(interface|type|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_str)
            if m_ts:
                kind = m_ts.group(1)
                name = m_ts.group(2)
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

        # ── 4. Rust ──
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

        # ── 5. Go ──
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

        # ── 6. C / C++ / C# / Java ──
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

        # ── 7. HTML ──
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

        # ── 8. CSS / SCSS ──
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

        # ── 9. SQL ──
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


def _extract_artefact_meta(buffer: str, language: Optional[str] = None, art_type: str = "code") -> Dict[str, Any]:
    """
    Extracts rich structural metadata from an artifact buffer without embedding the full raw content.
    """
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

    detected_symbols = _detect_structural_symbols(buffer, language, art_type)
    current_section = detected_symbols[-1]["detail"] if detected_symbols else None

    # Map detected symbols to sections list for backward compatibility
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


class _ArtefactStreamTracker:
    """Tracks the state of an artifact being built, emitting events upon discovering new structural symbols."""
    def __init__(self):
        self.is_inside_artefact = False
        self.current_buffer = ""
        self.last_event_detail = None
        self.last_event_time = 0.0
        self.current_title = None
        self.current_language = None
        self.current_art_type = "code"
        self.seen_symbol_keys: set = set()

        # ── 🎨 MEANINGFUL PROGRESS TRACKING ──
        # Track the last meaningful structural element reported to avoid spam
        self.last_reported_symbol = None
        self.last_progress_update_time = 0.0
        self.min_progress_interval = 2.0  # Minimum 2 seconds between progress updates

    def reset(self):
        self.is_inside_artefact = False
        self.current_buffer = ""
        self.last_event_detail = None
        self.last_event_time = 0.0
        self.current_title = None
        self.current_language = None
        self.current_art_type = "code"
        self.seen_symbol_keys.clear()
        self.last_reported_symbol = None
        self.last_progress_update_time = 0.0

    def open(self, title: str, language: Optional[str], art_type: str = "code"):
        self.is_inside_artefact = True
        self.current_title = title
        self.current_language = language
        self.current_art_type = art_type
        self.current_buffer = ""
        self.last_event_detail = None
        self.last_event_time = 0.0
        self.seen_symbol_keys.clear()
        self.last_reported_symbol = None
        self.last_progress_update_time = 0.0

    def feed(self, chunk: str) -> Optional[Dict[str, Any]]:
        """
        Feeds a chunk and returns event metadata if a new boundary is crossed,
        including any newly discovered symbols.

        CRITICAL: Only reports meaningful structural changes (functions, classes, sections).
        Does NOT report line numbers or generic progress to avoid UI spam.
        """
        if not self.is_inside_artefact:
            return None

        self.current_buffer += chunk

        now = _time.time()

        # ── 🛡️ THROTTLE: Minimum 2 seconds between updates ──
        # This prevents flooding the UI with useless progress updates
        if now - self.last_progress_update_time < self.min_progress_interval:
            return None

        symbols = _detect_structural_symbols(self.current_buffer, self.current_language, self.current_art_type)
        new_symbols = []
        for sym in symbols:
            key = f"{sym['symbol_type']}::{sym['symbol_name']}::{sym['line']}"
            if key not in self.seen_symbol_keys:
                self.seen_symbol_keys.add(key)
                new_symbols.append(sym)

        meta = _extract_artefact_meta(self.current_buffer, self.current_language, self.current_art_type)

        # ── 🎯 MEANINGFUL PROGRESS ONLY ──
        # Only report if we have a NEW structural symbol (function, class, section, etc.)
        # Do NOT report line numbers or generic progress
        detail = None

        if new_symbols:
            # We have a new structural element - this is meaningful progress
            latest_new_symbol = new_symbols[-1]
            detail = latest_new_symbol["detail"]

            # Update progress tracking
            self.last_reported_symbol = latest_new_symbol
            self.last_progress_update_time = now

            return {
                "title": self.current_title,
                "art_type": self.current_art_type,
                "language": self.current_language,
                "status": f"Writing {self.current_title}: {detail}",
                "detail": detail,
                "new_symbols": new_symbols,
                "latest_symbol": latest_new_symbol,
                **meta
            }

        # No new symbols - check if we should report periodic progress
        # Only report if enough time has passed AND we have a current section
        current_section = meta.get("current_section")
        if current_section and current_section != self.last_event_detail:
            # The current section changed (we moved to a new function/class)
            # This is meaningful progress
            self.last_event_detail = current_section
            self.last_progress_update_time = now

            return {
                "title": self.current_title,
                "art_type": self.current_art_type,
                "language": self.current_language,
                "status": f"Writing {self.current_title}: {current_section}",
                "detail": current_section,
                "new_symbols": [],
                "latest_symbol": symbols[-1] if symbols else None,
                **meta
            }

        # No meaningful progress to report
        return None

    def close(self):
        self.reset()


def _is_large_base64(v: str) -> bool:
    """Heuristic: a long string composed of base64 alphabet + whitespace."""
    sample = v.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(sample) < 500:
        return False
    return bool(_BASE64_RE.match(sample[:1000]))


def _sanitize_tool_result(
    tool_res: Any,
    max_chars: Optional[int] = None,
    client: Optional[Any] = None,
) -> str:
    if max_chars is None:
        max_chars = _calculate_dynamic_tool_char_limit(client)

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
            if _is_large_base64(obj):
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

        # Comprehensive Failure Detection
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
                error_msg = (
                    f"Tool returned success=False but did not provide an error message. "
                    f"Raw keys: {list(tool_res.keys()) if isinstance(tool_res, dict) else type(tool_res).__name__}. "
                    f"This may indicate a library initialization failure or an import error."
                )
            error_parts.append(f"**Error Details:**\n{error_msg}")

            stderr = tool_res.get("stderr") or (inner_dict.get("stderr") if inner_dict else None)
            if stderr and str(stderr).strip():
                error_parts.append(f"**Standard Error (stderr):**\n```\n{str(stderr).strip()}\n```")

            out_val = tool_res.get("output")
            if isinstance(out_val, dict):
                inner_stdout = out_val.get("output") or out_val.get("stdout")
                if inner_stdout and str(inner_stdout).strip() and str(inner_stdout).strip() != str(error_msg).strip():
                    error_parts.append(f"**Output before failure:**\n{str(inner_stdout).strip()}")
            elif out_val and str(out_val).strip() and str(out_val).strip() != str(error_msg).strip():
                error_parts.append(f"**Output before failure:**\n{str(out_val).strip()}")

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


def _resolve_handle(ref: str, branch_messages: List) -> Optional[Dict[str, str]]:
    parts = ref.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        msg_idx   = int(parts[0])
        block_idx = int(parts[1])
    except ValueError:
        return None

    if msg_idx < 0 or msg_idx >= len(branch_messages):
        return None

    msg = branch_messages[msg_idx]
    
    # Extract code blocks
    blocks = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    for m in pattern.finditer(getattr(msg, "content", "") or ""):
        blocks.append({
            "language": m.group(1).strip(),
            "content":  m.group(2),
            "raw":      m.group(0),
        })

    if block_idx < 0 or block_idx >= len(blocks):
        return None

    return blocks[block_idx]


def _build_handle_instructions(branch_messages: List) -> str:
    entries = []
    for msg_idx, msg in enumerate(branch_messages):
        blocks = []
        pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
        for m in pattern.finditer(getattr(msg, "content", "") or ""):
            blocks.append({
                "language": m.group(1).strip(),
                "content":  m.group(2)
            })
            
        for block_idx, blk in enumerate(blocks):
            lang    = blk["language"] or "text"
            preview = blk["content"].strip().splitlines()[0][:60] if blk["content"].strip() else ""
            entries.append(f"  {msg_idx}:{block_idx}  [{lang}]  {preview}")

    if not entries:
        return ""

    lines = [
        "",
        "=== AVAILABLE HANDLES ===",
        "Instead of rewriting a code block that already exists in the conversation,",
        "you can reference it by handle to create or update an artefact directly.",
        "",
        "Syntax (self-closing tag):",
        '  <use_handle ref="<msg_idx>:<block_idx>" name="filename.ext"',
        '              type="code" language="python"/>',
        "",
        "Available handles in this conversation:",
    ] + entries + [
        "",
        "Example — convert the Python block at position 1:0 into an artefact:",
        '  <use_handle ref="1:0" name="main.py" type="code" language="python"/>',
        "=== END HANDLES ===",
        "",
    ]
    return "\n".join(lines)


# ── StreamState Transactional Observer ─────────────────────────────────────

class _StreamState:
    """
    A high-performance, non-blocking transactional stream parser.
    """
    def __init__(
        self,
        discussion: 'LollmsDiscussion',
        forward_artefact_chunks: bool,
        callback: Optional[Callable],
        ai_message: Any,
        enable_notes: bool = True,
        enable_skills: bool = False,
        enable_inline_widgets: bool = True,
        enable_forms: bool = True,
        auto_activate_artefacts: bool = True,
        enable_artefacts: bool = True,
        enable_in_message_status: bool = True,
        content_offset: int = 0,
        fast_artefact_replicas: Optional[List[str]] = None,
        processed_tags: Optional[set] = None,
        event_mode: EventMode = EventMode.PROCESSING_TAG_MODE,
    ):
        self.discussion = discussion
        self.callback = callback
        self.ai_message = ai_message
        self.enable_artefacts = enable_artefacts
        self.enable_in_message_status = enable_in_message_status
        self.auto_activate = auto_activate_artefacts
        self.content_offset = content_offset
        self.event_mode = event_mode

        self.enable_notes = enable_notes if enable_artefacts else False
        self.enable_skills = enable_skills if enable_artefacts else False
        self.enable_inline_widgets = enable_inline_widgets if enable_artefacts else False
        self.enable_forms = enable_forms if enable_artefacts else False

        self.tool_trigger = False
        self.tool_json_data = ""
        self.affected_artefacts = []

        # Sparse artefact forwarding tracker
        self.forward_artefact_chunks = forward_artefact_chunks
        self.artefact_tracker = _ArtefactStreamTracker()

        # Track context unlock requests to force continuation round
        self.context_unlock_requested = False
        self.context_unlocked_files: List[str] = []

        # CRITICAL FIX: Initialize processed_tags from the persistent reference.
        self.processed_tags = processed_tags if processed_tags is not None else set()

        self._is_accumulating_tool = False
        self._tool_buffer = ""
        self._artefact_buffer = ""  # Dedicated buffer for raw artifact content
        self._artefact_open_tag = "" # Stores the exact opening tag (e.g., <artifact name="x">)
        self._pending_buffer = ""   # Shadow buffer to safely catch partial tags

        self._in_code_fence = False
        self._code_fence_buffer = ""
        self._code_fence_hold_buffer = ""  # Buffers content inside code fences to distinguish closed vs unclosed
        self._in_inline_code = False  # CRITICAL FIX: Track single backtick state across chunks

        # CRITICAL FIX: processed_tags must be passed in from ChatMixin
        # to persist across multiple reasoning rounds and prevent duplicate dispatch.
        self.processed_tags = processed_tags if processed_tags is not None else set()

        # ── ONE-ACTION-PER-TURN PROTOCOL ──
        # Ensures generation halts immediately after dispatching a single functional tag.
        self._action_dispatched = False

        # ── PATCH FAILURE TRACKING ──
        # Distinguishes a failed SEARCH/REPLACE patch (which should allow a correction round)
        # from a true duplicate artifact (which should hard-break the loop).
        self._last_dispatch_failed = False

        # ── DONE TAG DETECTION ──
        # Set to True when the LLM emits <done/> to signal explicit task termination.
        self._done_detected = False

        # ── Generic Secondary Tag Interceptor State ──
        # Handles <skill>, <note>, <lollms_inline>, <lollms_form>, <generate_image>, <edit_image>, etc.
        # These tags don't need the specialized dual-stream artifact tracker, but DO need
        # full body buffering + closing-tag detection + dispatch to _dispatch_closed_tag.
        self._is_accumulating_secondary = False
        self._secondary_buffer = ""
        self._secondary_tag_name = ""      # e.g., "skill", "note"
        self._secondary_closing_tag = ""   # e.g., "</skill>"
        self._secondary_open_tag = ""      # e.g., '<skill title="...">'

        # Heartbeat control for empty/slow artifacts
        self._artefact_heartbeat_thread: Optional[threading.Thread] = None
        self._artefact_heartbeat_stop = threading.Event()
        self._artefact_heartbeat_active = False
        self._artefact_received_content = False

        # Fast artefact replicas (user-provided or default)
        self._fast_artefact_replicas = fast_artefact_replicas if fast_artefact_replicas else _DEFAULT_FAST_REPLICAS


    @staticmethod
    def _sanitize_unicode(text: str) -> str:
        """
        Removes invisible Unicode characters that can corrupt XML parsing.

        Strips:
        - Zero-width spaces (U+200B, U+200C, U+200D)
        - Byte order marks (U+FEFF)
        - Directional formatting marks (U+200E, U+200F, U+202A-U+202E)
        - Word joiners (U+2060)
        - Other invisible formatting characters

        These characters are sometimes injected by tokenizers or model artifacts
        and can break functional tag detection, causing malformed XML output.
        """
        if not text:
            return text

        # Remove common invisible Unicode characters
        invisible_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Byte order mark / zero-width no-break space
            '\u200e',  # Left-to-right mark
            '\u200f',  # Right-to-left mark
            '\u202a',  # Left-to-right embedding
            '\u202b',  # Right-to-left embedding
            '\u202c',  # Pop directional formatting
            '\u202d',  # Left-to-right override
            '\u202e',  # Right-to-left override
            '\u2060',  # Word joiner
            '\u2061',  # Function application
            '\u2062',  # Invisible times
            '\u2063',  # Invisible separator
            '\u2064',  # Invisible plus
        ]

        for char in invisible_chars:
            text = text.replace(char, '')

        return text

    def _start_artefact_heartbeat(self):
        """Starts a background thread that emits cheering messages every 15s if no content arrives."""
        if self._artefact_heartbeat_thread is not None:
            return

        self._artefact_heartbeat_stop.clear()
        self._artefact_heartbeat_active = True
        self._artefact_received_content = False

        def _heartbeat_loop():
            interval = 15.0
            while not self._artefact_heartbeat_stop.wait(interval):
                if not self._artefact_received_content:
                    msg = random.choice(_HEARTBEAT_MESSAGES)
                    try:
                        # CRITICAL FIX: Do NOT use was_processed=True here.
                        # That flag causes _inline_relay to silently drop the message.
                        # Use a distinct meta key so the UI can style it if desired.
                        _cb(self.callback, f"\n{msg}\n", MSG_TYPE.MSG_TYPE_CHUNK, {"is_heartbeat": True})
                    except Exception:
                        pass

        self._artefact_heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._artefact_heartbeat_thread.start()

    def _stop_artefact_heartbeat(self):
        """Stops the heartbeat thread safely."""
        if self._artefact_heartbeat_thread is not None:
            self._artefact_heartbeat_stop.set()
            if threading.current_thread() != self._artefact_heartbeat_thread:
                self._artefact_heartbeat_thread.join(timeout=1.0)
            self._artefact_heartbeat_thread = None
            self._artefact_heartbeat_active = False

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str) or not chunk:
            return True

        # ── 🧹 UNICODE SANITIZATION (CRITICAL FIX) ──
        # Remove zero-width spaces, directional marks, and other invisible Unicode
        # that can break XML tag detection and cause malformed output.
        # These characters (U+200B, U+200C, U+200D, U+FEFF, etc.) are often injected
        # by tokenizers or model artifacts and can corrupt functional tags.
        chunk = self._sanitize_unicode(chunk)

        # ── ONE-ACTION-PER-TURN: If an action was already dispatched, consume and discard ──
        if self._action_dispatched:
            self._pending_buffer += chunk
            return True

        # CRITICAL FIX: Append to shadow buffer instead of directly to ai_message.content
        self._pending_buffer += chunk

        # ── 🛑 DONE TAG DETECTION (SUPPORTS ALL VARIANTS) ──
        # Detect <done/>, <done>, <end/>, <end>, </end> at the start of a line to signal explicit termination.
        # We strip it from the buffer so it never leaks into the UI or database.
        if not self._is_accumulating_tool and not self.artefact_tracker.is_inside_artefact and not self._is_accumulating_secondary and not self._in_code_fence:
            done_match = re.search(r'(?m)^\s*<(?:done|end)\s*/?>', self._pending_buffer, re.IGNORECASE)
            if done_match:
                ASCIIColors.info("[StreamState] Termination tag (<done/> or <end/>) detected. Halting generation.")
                self._done_detected = True
                self._pending_buffer = re.sub(r'(?m)^\s*<(?:done|end)\s*/?>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        # ── 🛑 ANTI-MIMICRY: Prevent LLM from generating <processing> blocks ──
        # The <processing> tag is strictly system-generated. If the LLM attempts to
        # output it, we halt generation immediately to prevent log hallucination.
        # STRICT: Only trigger if the tag starts at the beginning of a line (ignoring whitespace).
        if not self._is_accumulating_tool and not self.artefact_tracker.is_inside_artefact and not self._is_accumulating_secondary and not self._in_code_fence:
            proc_match = re.search(r'(?m)^\s*<processing', self._pending_buffer, re.IGNORECASE)
            if proc_match:
                ASCIIColors.warning("[StreamState] LLM attempted to generate a <processing> block. Halting generation.")
                self._pending_buffer = re.sub(r'(?m)^\s*<processing[^>]*>', '', self._pending_buffer, flags=re.IGNORECASE)
                return False

        # ── 🛡️ MARKDOWN CODE FENCE & INLINE CODE PROTECTION ──
        # Track ``` and ` to prevent intercepting functional tags inside documentation or tables.
        if not self._is_accumulating_tool and not self.artefact_tracker.is_inside_artefact and not self._is_accumulating_secondary:
            # Handle triple backticks (```...```)
            if "```" in self._pending_buffer:
                self._code_fence_buffer += self._pending_buffer
                self._pending_buffer = ""

                while "```" in self._code_fence_buffer:
                    idx = self._code_fence_buffer.find("```")
                    before = self._code_fence_buffer[:idx]
                    self._code_fence_buffer = self._code_fence_buffer[idx+3:]

                    if not self._in_code_fence:
                        self._in_code_fence = True
                        self.ai_message.content += before + "```"
                        _cb(self.callback, before + "```", MSG_TYPE.MSG_TYPE_CHUNK)
                    else:
                        self._in_code_fence = False
                        # Emit the hold buffer as verbatim text (it was inside a properly closed fence)
                        if self._code_fence_hold_buffer:
                            self.ai_message.content += self._code_fence_hold_buffer
                            _cb(self.callback, self._code_fence_hold_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
                            self._code_fence_hold_buffer = ""
                        # Also emit content before the closing fence (handles single-chunk case)
                        if before:
                            self.ai_message.content += before
                            _cb(self.callback, before, MSG_TYPE.MSG_TYPE_CHUNK)
                        self.ai_message.content += "```"
                        _cb(self.callback, "```", MSG_TYPE.MSG_TYPE_CHUNK)

                if self._in_code_fence:
                    # Still inside fence — buffer remaining content instead of emitting.
                    # This lets us distinguish closed fences (emit as text) from unclosed
                    # fences (re-process through tag detection at flush time).
                    self._code_fence_hold_buffer += self._code_fence_buffer
                    self._code_fence_buffer = ""
                    return True
                else:
                    self._pending_buffer = self._code_fence_buffer
                    self._code_fence_buffer = ""

            elif self._in_code_fence:
                # Buffer content while inside code fence instead of emitting immediately.
                # This allows us to properly handle functional tags:
                # - If the fence is closed (``` found), emit everything as verbatim text.
                # - If the fence is never closed (flush), re-process through tag detection.
                self._code_fence_hold_buffer += self._pending_buffer
                self._pending_buffer = ""
                return True

            # Handle single backticks (`...`) - CRITICAL FIX for streaming tables
            # We must buffer text when a backtick is opened to prevent the tag parser
            # from intercepting functional tags that appear inside inline code spans.
            elif "`" in self._pending_buffer:
                if self._in_inline_code:
                    # We are inside an inline code span from a previous chunk, looking for the closing backtick
                    idx = self._pending_buffer.find("`")
                    if idx != -1:
                        # Closing backtick found
                        self._in_inline_code = False
                        inline_content = self._pending_buffer[:idx]
                        self.ai_message.content += inline_content + "`"
                        _cb(self.callback, inline_content + "`", MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = self._pending_buffer[idx+1:]
                    else:
                        # Check for newline:
                        # If the LLM moves to a new line without closing the inline code, 
                        # it was a stray backtick (e.g., inside HTML body). Break out to avoid lockout.
                        newline_idx = self._pending_buffer.find("\n")
                        if newline_idx != -1 and self._in_inline_code:
                            self._in_inline_code = False
                            # Emit verbatim up to and including the newline to reset state cleanly
                            self.ai_message.content += self._pending_buffer
                            _cb(self.callback, self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
                            self._pending_buffer = ""
                        else:
                            # Still inside, emit verbatim
                            self.ai_message.content += self._pending_buffer
                            _cb(self.callback, self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
                            self._pending_buffer = ""
                        return True
                else:
                    # Not currently in inline code, look for an opening backtick
                    idx = self._pending_buffer.find("`")
                    before = self._pending_buffer[:idx]
                    remainder = self._pending_buffer[idx+1:]

                    # Check if the closing backtick is in the remainder of the current chunk
                    closing_idx = remainder.find("`")
                    if closing_idx != -1:
                        # Complete inline code span in a single chunk
                        inline_content = remainder[:closing_idx]
                        self.ai_message.content += before + "`" + inline_content + "`"
                        _cb(self.callback, before + "`" + inline_content + "`", MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = remainder[closing_idx+1:]
                    else:
                        # Opening backtick found, but no closing backtick in this chunk.
                        # IMPORTANT FIX: Only enter state if no newline exists between the backtick and the end of the chunk.
                        # If there's a newline, it means the backtick is a stray character (e.g., raw HTML),
                        # not an inline code span. Emit verbatim and DO NOT enter _in_inline_code state.
                        newline_idx = remainder.find("\n")
                        if newline_idx != -1:
                            # Stray backtick followed by a newline. Emit verbatim, do not enter code state.
                            self.ai_message.content += before + "`" + remainder
                            _cb(self.callback, before + "`" + remainder, MSG_TYPE.MSG_TYPE_CHUNK)
                            self._pending_buffer = ""
                        else:
                            # Genuine inline code span starting. Enter inline code mode.
                            self._in_inline_code = True
                            self.ai_message.content += before + "`"
                            _cb(self.callback, before + "`", MSG_TYPE.MSG_TYPE_CHUNK)
                            self._pending_buffer = remainder
                        return True

            elif self._in_inline_code:
                # We are inside an inline code span from a previous chunk, looking for the closing backtick
                idx = self._pending_buffer.find("`")
                if idx != -1:
                    self._in_inline_code = False
                    inline_content = self._pending_buffer[:idx]
                    self.ai_message.content += inline_content + "`"
                    _cb(self.callback, inline_content + "`", MSG_TYPE.MSG_TYPE_CHUNK)
                    self._pending_buffer = self._pending_buffer[idx+1:]
                else:
                    # Check for newline: if the LLM moves to a new line without
                    # closing the inline code, it was a stray backtick. Break out
                    # to avoid permanent lockout that bypasses functional tags.
                    newline_idx = self._pending_buffer.find("\n")
                    if newline_idx != -1:
                        self._in_inline_code = False
                        # Emit verbatim up to and including the newline, then let
                        # the rest of the buffer flow to tag detection logic.
                        self.ai_message.content += self._pending_buffer
                        _cb(self.callback, self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = ""
                    else:
                        # Still inside, emit verbatim
                        self.ai_message.content += self._pending_buffer
                        _cb(self.callback, self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = ""
                        return True

        # ── Tool Accumulation & Interception ──
        if self._is_accumulating_tool:
            # Use regex to be tolerant of whitespace or slight malformations in the closing tag (e.g., </tool >)
            close_match = re.search(r'</tool>\s*', self._pending_buffer, re.IGNORECASE)
            if close_match:
                end_idx = close_match.start()
                end_len = len(close_match.group(0))

                full_tool_call = self._tool_buffer + self._pending_buffer[:end_idx + end_len]
                # Robustly extract JSON body without relying on exact lstrip/rstrip of tags
                json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE)
                json_body = re.sub(r'</tool>\s*$', '', json_body, flags=re.IGNORECASE).strip()

                self._is_accumulating_tool = False
                self._tool_buffer = ""

                # Keep any text after the tool call in the pending buffer
                self._pending_buffer = self._pending_buffer[end_idx + end_len:]

                # ── ONE-ACTION-PER-TURN: Halt generation immediately after dispatch ──
                self._dispatch_closed_tag("tool", "", json_body, full_tool_call)
                self._action_dispatched = True
                return False
            else:
                self._tool_buffer += self._pending_buffer
                self._pending_buffer = ""
            return True

        # ── Tag Detection (Buffering) ──
        last_open_think = self._pending_buffer.rfind("<think")
        last_close_think = self._pending_buffer.rfind("```")
        is_inside_thoughts = (last_open_think != -1) and (last_open_think > last_close_think)

        # ── 🛡️ INLINE TAG QUARANTINE (CRITICAL FIX) ──
        # If a functional tag appears in the buffer but is NOT at the absolute start
        # of a line (ignoring whitespace), it is conversational prose and MUST NOT
        # be intercepted. We flush all text before it, then consume the tag and
        # emit it directly to the UI as raw text.
        if not is_inside_thoughts and not self._is_accumulating_tool and not self.artefact_tracker.is_inside_artefact and not self._is_accumulating_secondary and not self._in_code_fence:
            inline_tag_found = False
            # CRITICAL: Check for exact opening tags (e.g., "<tool>") and tag prefixes with attributes (e.g., "<artifact ")
            # REMOVED "<lollms_inline" so the host application can handle it directly.
            for tag_prefix in ("<artifact", "<artefact", "<tool", "<note", "<skill", "<scratchpad", "<lollms_form", "<generate_image", "<edit_image", "<unlock_file", "<lock_file", "<hide_file"):
                idx = self._pending_buffer.find(tag_prefix)
                if idx != -1:
                    # Check if it's at the absolute start of a line
                    is_at_line_start = True
                    i = idx - 1
                    while i >= 0 and self._pending_buffer[i] != '\n':
                        if not self._pending_buffer[i].isspace():
                            is_at_line_start = False
                            break
                        i -= 1

                    if not is_at_line_start:
                        # It's an inline tag! Flush text before it, then emit the tag raw.
                        text_before = self._pending_buffer[:idx]
                        if text_before:
                            self.ai_message.content += text_before
                            _cb(self.callback, text_before, MSG_TYPE.MSG_TYPE_CHUNK)

                        # Emit the tag itself directly to the UI
                        self.ai_message.content += tag_prefix
                        _cb(self.callback, tag_prefix, MSG_TYPE.MSG_TYPE_CHUNK)

                        # Consume the processed parts from the pending buffer
                        self._pending_buffer = self._pending_buffer[idx + len(tag_prefix):]
                        inline_tag_found = True
                        break # Restart the feed loop for the rest of the buffer

            if inline_tag_found:
                return True

        # ── Handle <artifact> Streaming (State-Driven Dual-Stream) ──
        if not is_inside_thoughts:
            # State 1: We are already inside an artifact (tracker is active)
            if self.artefact_tracker.is_inside_artefact:
                # Track if we received actual content (for heartbeat suppression)
                if self._pending_buffer.strip():
                    self._artefact_received_content = True

                self._artefact_buffer += self._pending_buffer
                self._pending_buffer = "" # Consume the buffer into the artifact

                # Check if the closing tag arrived (robust string search)
                lower_buffer = self._artefact_buffer.lower()
                close_idx = lower_buffer.find("</artifact>")
                if close_idx == -1:
                    close_idx = lower_buffer.find("</artefact>")

                if close_idx != -1:
                    self._stop_artefact_heartbeat()
                    self.artefact_tracker.close()

                    # Extract the full artifact block cleanly
                    # Find the opening tag first
                    open_idx = lower_buffer.find("<artifact")
                    if open_idx == -1:
                        open_idx = lower_buffer.find("<artefact")

                    end_of_open_tag = self._artefact_buffer.find(">", open_idx)
                    opening_tag = self._artefact_buffer[open_idx:end_of_open_tag+1]
                    body_content = self._artefact_buffer[end_of_open_tag+1:close_idx]
                    closing_tag = self._artefact_buffer[close_idx:close_idx+len("</artifact>")]
                    full_match_text = opening_tag + body_content + closing_tag

                    # Always dispatch the real body content to create the artifact.
                    if full_match_text not in self.processed_tags:
                        self.processed_tags.add(full_match_text)
                        self._dispatch_closed_tag(
                            "artifact", 
                            opening_tag, 
                            body_content.strip(), 
                            full_match_text
                        )
                    else:
                        ASCIIColors.warning("[StreamState] Duplicate artifact tag detected. Skipping dispatch.")
                        self._action_dispatched = True
                        return False

                    # Close the processing block cleanly with status metadata INSIDE the block.
                    if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                        proc_close_tag = '\n<!-- status:finished -->\n</processing>\n'
                        self.ai_message.content += proc_close_tag
                        _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # Keep any text that came after the closing tag
                    self._pending_buffer = self._artefact_buffer[close_idx+len(closing_tag):]
                    self._artefact_buffer = ""

                    # ── ONE-ACTION-PER-TURN: Halt generation immediately ──
                    self._action_dispatched = True
                    return False
                else:
                    # Still in the middle of the artifact body. Suppress raw output from main stream.
                    event_meta = self.artefact_tracker.feed(chunk)
                    if event_meta:
                        new_symbols = event_meta.get("new_symbols", [])

                        # ── EMIT TARGETED SYMBOL DETECTION EVENTS ──
                        if new_symbols:
                            for sym in new_symbols:
                                if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                    _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACT_SYMBOL_DETECTED, {
                                        "title": self.artefact_tracker.current_title,
                                        "art_type": self.artefact_tracker.current_art_type,
                                        "language": self.artefact_tracker.current_language,
                                        "symbol": sym,
                                        "symbol_type": sym.get("symbol_type"),
                                        "symbol_name": sym.get("symbol_name"),
                                        "line": sym.get("line"),
                                        "detail": sym.get("detail"),
                                        "signature": sym.get("signature"),
                                    })

                                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                    # ── 🎨 USER-FRIENDLY SYMBOL REPORTING ──
                                    # Report structural elements in a clean, readable format
                                    sym_type = sym.get("symbol_type", "element")
                                    sym_name = sym.get("symbol_name", "unknown")

                                    # Create user-friendly descriptions
                                    if sym_type == "class":
                                        sym_line = f"  📦 Class: {sym_name}\n"
                                    elif sym_type == "function":
                                        sym_line = f"  ⚙️ Function: {sym_name}()\n"
                                    elif sym_type == "async_function":
                                        sym_line = f"  ⚡ Async Function: {sym_name}()\n"
                                    elif sym_type == "method":
                                        sym_line = f"  🔧 Method: {sym_name}()\n"
                                    elif sym_type == "major_section":
                                        sym_line = f"  📑 Section: {sym_name}\n"
                                    elif sym_type == "section":
                                        sym_line = f"  📄 Subsection: {sym_name}\n"
                                    elif sym_type == "react_component":
                                        sym_line = f"  ⚛️ Component: <{sym_name} />\n"
                                    elif sym_type == "react_hook":
                                        sym_line = f"  🪝 Hook: {sym_name}()\n"
                                    else:
                                        sym_line = f"  • {sym['detail']}\n"

                                    self.ai_message.content += sym_line
                                    _cb(self.callback, sym_line, MSG_TYPE.MSG_TYPE_CHUNK, {
                                        "was_processed": True,
                                        "event_type": "symbol_detected",
                                        "symbol": sym
                                    })
                        else:
                            # No new symbols, but we have a status update (section change)
                            if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {
                                    **event_meta,
                                    "stream_complete": False
                                })

                            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                # ── 🎯 MEANINGFUL STATUS UPDATES ONLY ──
                                # Only show status if it's a meaningful structural element
                                detail = event_meta.get("detail")
                                if detail and not detail.startswith("Line "):
                                    # This is a real structural element (function, class, section)
                                    status_tag = f'{event_meta["status"]}\n'
                                    self.ai_message.content += status_tag
                                    _cb(self.callback, status_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        # If forward_artefact_chunks is True, also forward the raw chunk
                        if self.forward_artefact_chunks:
                            _cb(self.callback, chunk, MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK, event_meta)

                return True

            # State 2: We are not inside an artifact, check if we are entering one
            else:
                # Look for the start of an artifact tag (case-insensitive)
                # STRICT WHITELIST: Only match if the tag starts at the absolute beginning of a line (ignoring whitespace).
                # CRITICAL FIX: Exclude lines that start with markdown table/code characters (` or |)
                # to prevent intercepting documentation examples as live functional tags.
                lower_buffer = self._pending_buffer.lower()
                # The negative lookahead (?!`) ensures the tag is not immediately preceded by a backtick.
                # The (?!.*\|) ensures the line is not part of a markdown table (no pipe character after the tag).
                open_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)(?<![\w\[])<(?:artifact|artefact)', lower_buffer)
                open_idx = open_match.start() if open_match else -1

                if open_idx != -1:
                    tag_start_idx = open_idx

                    # Check if we have the full opening tag
                    end_of_tag_idx = self._pending_buffer.find(">", tag_start_idx)

                    if end_of_tag_idx != -1:
                        # We have the full opening tag!
                        attrs_str = self._pending_buffer[tag_start_idx:end_of_tag_idx+1]
                        title = "artifact"
                        lang = None
                        attrs = {}
                        for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                            attrs[m.group(1).lower()] = m.group(2)
                        m_title = re.search(r'(?:name|title)=["\']([^"\']*)["\']', attrs_str, re.IGNORECASE)
                        if m_title: title = m_title.group(1)
                        m_lang = re.search(r'language=["\']([^"\']*)["\']', attrs_str, re.IGNORECASE)
                        if m_lang: lang = m_lang.group(1)

                        atype = attrs.get("type", "code").lower()
                        self.artefact_tracker.open(title, lang, atype)

                        # Forward the text BEFORE the tag to the UI and save it
                        text_before_tag = self._pending_buffer[:tag_start_idx]
                        if text_before_tag:
                            self.ai_message.content += text_before_tag
                            _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                        # Start the artifact buffer with the opening tag
                        self._artefact_buffer = attrs_str

                        # Determine the type-specific opening message
                        opening_status = _ARTEFACT_TYPE_MESSAGES.get(atype, "✨ Starting artifact...")

                        # Check if the remaining content already contains the patch marker
                        remaining_content = self._pending_buffer[end_of_tag_idx+1:]
                        is_patch_start = "<<<<<<< SEARCH" in remaining_content
                        operation_type = "patch" if is_patch_start else "full_rewrite"

                        if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_START, {
                                "title": title,
                                "art_type": atype,
                                "language": lang,
                                "is_patch": is_patch_start,
                                "operation": operation_type,
                                "stream_complete": False,
                                "line_count": 0,
                                "size_chars": 0,
                                "current_section": None,
                                "sections": []
                            })

                        if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            proc_tag = f'\n<processing type="artefact" title="{title}" language="{lang or ""}" operation="{operation_type}">\n'
                            self.ai_message.content += proc_tag
                            _cb(self.callback, proc_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True, "operation": operation_type, "is_patch": is_patch_start})

                        # Start the heartbeat in case the artifact body is slow/empty
                        self._start_artefact_heartbeat()

                        if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            status_line = f'{opening_status} (operation: {operation_type})\n'
                            self.ai_message.content += status_line
                            _cb(self.callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True, "operation": operation_type, "is_patch": is_patch_start})

                        # Check if the closing tag also arrived in this same chunk
                        remaining_content = self._pending_buffer[end_of_tag_idx+1:]
                        close_idx = remaining_content.lower().find("</artifact>")
                        if close_idx == -1:
                            close_idx = remaining_content.lower().find("</artefact>")

                        if close_idx != -1:
                            self._stop_artefact_heartbeat()
                            self.artefact_tracker.close()

                            # Extract the body cleanly
                            body_content = remaining_content[:close_idx]
                            closing_tag = remaining_content[close_idx:close_idx+len("</artifact>")]
                            full_match_text = attrs_str + body_content + closing_tag

                            # Always dispatch the real body content to create the artifact.
                            if full_match_text not in self.processed_tags:
                                self.processed_tags.add(full_match_text)
                                self._dispatch_closed_tag(
                                    "artifact", 
                                    attrs_str, 
                                    body_content.strip(), 
                                    full_match_text
                                )
                            else:
                                ASCIIColors.warning("[StreamState] Duplicate artifact tag detected (inline). Skipping dispatch.")
                                self._action_dispatched = True
                                return False

                            # Close the processing block cleanly with status metadata.
                            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                proc_close_tag = f'\n</processing>\n'
                                status_comment = f'<!-- status:finished -->\n'
                                self.ai_message.content += proc_close_tag + status_comment
                                _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                _cb(self.callback, status_comment, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            self._artefact_buffer = ""
                            # Keep any text after the closing tag in the pending buffer
                            self._pending_buffer = remaining_content[close_idx+len(closing_tag):]

                            # ── ONE-ACTION-PER-TURN: Halt generation immediately ──
                            self._action_dispatched = True
                            return False
                        else:
                            # We are inside the artifact, waiting for the rest.
                            self._artefact_buffer += remaining_content
                            self._pending_buffer = ""

                        return True
                    else:
                        # Partial tag detected (e.g., "<art"). 
                        # Forward text before the partial tag to the UI and save it.
                        text_before_partial = self._pending_buffer[:tag_start_idx]
                        if text_before_partial:
                            self.ai_message.content += text_before_partial
                            _cb(self.callback, text_before_partial, MSG_TYPE.MSG_TYPE_CHUNK)

                        # Hold the partial tag in the pending buffer for the next chunk
                        self._pending_buffer = self._pending_buffer[tag_start_idx:]
                        return True

        # ── Handle <tool> Streaming ──
        if not is_inside_thoughts:
            # STRICT WHITELIST: Only match if the <tool> tag starts at the absolute beginning of a line (ignoring whitespace).
            # CRITICAL FIX: Exclude lines that start with markdown table/code characters (` or |)
            open_tool_match = re.search(r'(?m)^\s*(?!`)(?!.*\|)<tool>', self._pending_buffer, re.IGNORECASE)
            if open_tool_match:
                tag_start_idx = open_tool_match.start()

                # Forward text before the tool tag to the UI and save it
                text_before_tag = self._pending_buffer[:tag_start_idx]
                if text_before_tag:
                    self.ai_message.content += text_before_tag
                    _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                self._is_accumulating_tool = True
                self._tool_buffer = self._pending_buffer[tag_start_idx:]
                self._pending_buffer = ""

                # CRITICAL FIX: Do NOT emit a <processing> block here.
                # The ChatMixin will handle the execution UI block once the tool call is parsed.
                # Emitting it here causes a duplicate/empty processing block in the UI.

                return True

        # ── 🧠 MEMORY TAG INTERCEPTION (CRITICAL FIX) ──
        # Memory tags (<mem_new>, <mem_search>, etc.) must be intercepted HERE
        # and processed silently WITHOUT triggering tool execution or processing blocks.
        # They are infrastructure tags that operate on the memory system directly.
        if not is_inside_thoughts and not self._is_accumulating_tool and not self.artefact_tracker.is_inside_artefact and not self._is_accumulating_secondary and not self._in_code_fence:
            lower_buffer = self._pending_buffer.lower()
            memory_tag_entered = False
            memory_search_triggered = False
            memory_search_query = None
            memory_search_level = None

            for mem_tag_prefix in ("<mem_new", "<mem_update", "<mem_tag", "<mem_load", "<mem_delete", "<mem_search", "<mem_rel"):
                # Match self-closing or paired tags at line start
                pattern = r'(?m)^\s*(?!`)(?!.*\|)' + re.escape(mem_tag_prefix)
                open_match = re.search(pattern, lower_buffer)
                if open_match:
                    open_idx = open_match.start()
                    end_of_tag_idx = self._pending_buffer.find(">", open_idx)

                    if end_of_tag_idx != -1:
                        tag_start_idx = open_idx
                        opening_tag = self._pending_buffer[tag_start_idx:end_of_tag_idx+1]

                        # Check if this is a self-closing tag (ends with />)
                        is_self_closing = opening_tag.rstrip().endswith("/>")

                        if is_self_closing:
                            # Self-closing tag: process immediately
                            text_before_tag = self._pending_buffer[:tag_start_idx]
                            if text_before_tag:
                                self.ai_message.content += text_before_tag
                                _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                            # Extract search parameters if this is a mem_search tag
                            if mem_tag_prefix == "<mem_search":
                                query_match = re.search(r'query=["\']([^"\']+)["\']', opening_tag)
                                level_match = re.search(r'level=["\'](\d+)["\']', opening_tag)
                                if query_match:
                                    memory_search_query = query_match.group(1)
                                    memory_search_level = int(level_match.group(1)) if level_match else None
                                    memory_search_triggered = True
                                    ASCIIColors.info(f"[StreamState] Memory search triggered: query='{memory_search_query}', level={memory_search_level}")

                            # Process the memory tag silently (no UI feedback)
                            # The tag will be stripped from the final content by _process_memory_tags
                            self._pending_buffer = self._pending_buffer[end_of_tag_idx+1:]
                            memory_tag_entered = True
                            break
                        else:
                            # Paired tag: accumulate until closing tag
                            tag_name_match = re.match(r'<(\w+)', opening_tag)
                            if tag_name_match:
                                tag_name = tag_name_match.group(1).lower()
                                closing_tag = f"</{tag_name}>"

                                # Check if closing tag is already in buffer
                                close_idx = self._pending_buffer.lower().find(closing_tag.lower(), end_of_tag_idx)

                                if close_idx != -1:
                                    # Complete tag in buffer: process immediately
                                    text_before_tag = self._pending_buffer[:tag_start_idx]
                                    if text_before_tag:
                                        self.ai_message.content += text_before_tag
                                        _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                                    # Extract full tag including closing
                                    close_end_idx = close_idx + len(closing_tag)
                                    full_tag = self._pending_buffer[tag_start_idx:close_end_idx]

                                    # Process silently (memory tags are stripped later)
                                    self._pending_buffer = self._pending_buffer[close_end_idx:]
                                    memory_tag_entered = True
                                    break
                                else:
                                    # Incomplete tag: buffer it
                                    text_before_tag = self._pending_buffer[:tag_start_idx]
                                    if text_before_tag:
                                        self.ai_message.content += text_before_tag
                                        _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                                    # Keep the partial tag in pending buffer for next chunk
                                    self._pending_buffer = self._pending_buffer[tag_start_idx:]
                                    return True
                    else:
                        # Partial tag detected: buffer it
                        text_before_partial = self._pending_buffer[:open_idx]
                        if text_before_partial:
                            self.ai_message.content += text_before_partial
                            _cb(self.callback, text_before_partial, MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = self._pending_buffer[open_idx:]
                        return True

            if memory_tag_entered:
                # If a memory search was triggered, we need to execute it NOW and inject results
                if memory_search_triggered and memory_search_query:
                    # Store the search request for the ChatMixin to process
                    if not hasattr(self.discussion, '_pending_memory_searches'):
                        object.__setattr__(self.discussion, '_pending_memory_searches', [])
                    self.discussion._pending_memory_searches.append({
                        "query": memory_search_query,
                        "level": memory_search_level
                    })
                    # Mark that an action was dispatched to trigger a continuation round
                    self._action_dispatched = True
                    return False  # Halt generation to process the search

                return True

        # ── Handle <unlock_file>, <lock_file>, <hide_file> Streaming ──
        # These tags must be intercepted here BEFORE the generic secondary tag interceptor
        # so they can be routed to the specific context visibility handler in _dispatch_closed_tag.
        if not is_inside_thoughts and not self._is_accumulating_secondary:
            lower_buffer = self._pending_buffer.lower()
            context_tag_entered = False
            for tag_prefix in ("<unlock_file", "<lock_file", "<hide_file"):
                pattern = r'(?m)^\s*(?!`)(?!.*\|)' + re.escape(tag_prefix)
                open_match = re.search(pattern, lower_buffer)
                if open_match:
                    open_idx = open_match.start()
                    end_of_tag_idx = self._pending_buffer.find(">", open_idx)
                    if end_of_tag_idx != -1:
                        tag_start_idx = open_idx
                        opening_tag = self._pending_buffer[tag_start_idx:end_of_tag_idx+1]
                        tag_name_match = re.match(r'<(\w+)', opening_tag)
                        if tag_name_match:
                            self._secondary_tag_name = tag_name_match.group(1).lower()
                            self._secondary_closing_tag = f"</{self._secondary_tag_name}>"
                            self._secondary_open_tag = opening_tag
                            self._is_accumulating_secondary = True

                            text_before_tag = self._pending_buffer[:tag_start_idx]
                            if text_before_tag:
                                self.ai_message.content += text_before_tag
                                _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                            self._secondary_buffer = opening_tag
                            self._pending_buffer = ""
                            context_tag_entered = True
                            break
                    else:
                        text_before_partial = self._pending_buffer[:tag_start_idx]
                        if text_before_partial:
                            self.ai_message.content += text_before_partial
                            _cb(self.callback, text_before_partial, MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = self._pending_buffer[tag_start_idx:]
                        return True

            if context_tag_entered:
                return True

        # ── Handle Context Tag Body Accumulation & Closing ──
        # CRITICAL FIX: This block must run if we are accumulating a context tag.
        # It uses the same logic as the generic secondary tag accumulator but is placed
        # here to ensure it catches the closing tag immediately.
        if self._is_accumulating_secondary and self._secondary_tag_name in ("unlock_file", "lock_file", "hide_file"):
            self._secondary_buffer += self._pending_buffer
            self._pending_buffer = ""

            close_match = re.search(re.escape(self._secondary_closing_tag), self._secondary_buffer, re.IGNORECASE)
            if close_match:
                close_idx = close_match.start()
                close_len = close_match.end() - close_match.start()
                
                body_content = self._secondary_buffer[len(self._secondary_open_tag):close_idx]
                closing_tag = self._secondary_buffer[close_idx:close_idx+close_len]
                full_match_text = self._secondary_open_tag + body_content + closing_tag

                self._is_accumulating_secondary = False

                if full_match_text not in self.processed_tags:
                    self.processed_tags.add(full_match_text)
                    self._dispatch_closed_tag(
                        self._secondary_tag_name,
                        self._secondary_open_tag,
                        body_content.strip(),
                        full_match_text
                    )

                proc_close_tag = f'\n<!-- status:finished -->\n</processing>\n'
                self.ai_message.content += proc_close_tag
                _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                remaining_text = self._secondary_buffer[close_idx + close_len:]
                
                self._secondary_tag_name = ""
                self._secondary_closing_tag = ""
                self._secondary_open_tag = ""
                self._secondary_buffer = ""

                if remaining_text.strip():
                    self._pending_buffer = remaining_text
                else:
                    self._action_dispatched = True
                    return False
            else:
                pass
            return True

        # ── Generic Secondary Tag Interception (<skill>, <note>, <lollms_inline>, etc.) ──
        if not is_inside_thoughts and not self._is_accumulating_secondary:
            # Check if we are ENTERING a secondary tag
            lower_buffer = self._pending_buffer.lower()
            secondary_entered = False
            # REMOVED "<lollms_inline" so the host application can handle it directly.
            for tag_prefix in ("<skill", "<note", "<scratchpad", "<lollms_form", "<generate_image", "<edit_image", "<unlock_file", "<lock_file", "<hide_file"):
                # STRICT WHITELIST: Only match if the tag starts at the absolute beginning of a line (ignoring whitespace).
                # CRITICAL FIX: Exclude lines that start with markdown table/code characters (` or |)
                pattern = r'(?m)^\s*(?!`)(?!.*\|)' + re.escape(tag_prefix)
                open_match = re.search(pattern, lower_buffer)
                if open_match:
                    tag_start_idx = self._pending_buffer.find(tag_prefix, open_match.start())
                    if tag_start_idx == -1:
                        continue
                    end_of_tag_idx = self._pending_buffer.find(">", tag_start_idx)
                    if end_of_tag_idx != -1:
                        opening_tag = self._pending_buffer[tag_start_idx:end_of_tag_idx+1]

                        tag_name_match = re.match(r'<(\w+)', opening_tag)
                        if tag_name_match:
                            self._secondary_tag_name = tag_name_match.group(1).lower()
                            self._secondary_closing_tag = f"</{self._secondary_tag_name}>"
                            self._secondary_open_tag = opening_tag
                            self._is_accumulating_secondary = True

                            # Forward text BEFORE the tag to the UI and save it
                            text_before_tag = self._pending_buffer[:tag_start_idx]
                            if text_before_tag:
                                self.ai_message.content += text_before_tag
                                _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                            # Start the secondary buffer with the opening tag
                            self._secondary_buffer = opening_tag
                            self._pending_buffer = ""

                            # Emit a processing block opening for UI feedback
                            proc_type = self._secondary_tag_name
                            # Extract title from attributes if present
                            title_match = re.search(r'(?:title|name)=["\']([^"\']*)["\']', opening_tag, re.IGNORECASE)
                            proc_title = title_match.group(1) if title_match else self._secondary_tag_name.capitalize()
                            proc_open = f'\n<processing type="{proc_type}" title="{proc_title}">\n'
                            self.ai_message.content += proc_open
                            _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            status_msg = _ARTEFACT_TYPE_MESSAGES.get(self._secondary_tag_name, f"✨ Processing {self._secondary_tag_name}...")
                            status_line = f'{status_msg}\n'
                            self.ai_message.content += status_line
                            _cb(self.callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            secondary_entered = True
                            break
                    else:
                        text_before_partial = self._pending_buffer[:tag_start_idx]
                        if text_before_partial:
                            self.ai_message.content += text_before_partial
                            _cb(self.callback, text_before_partial, MSG_TYPE.MSG_TYPE_CHUNK)
                        self._pending_buffer = self._pending_buffer[tag_start_idx:]
                        return True

            if secondary_entered:
                return True

        # ── Handle Secondary Tag Body Accumulation & Closing ──
        if self._is_accumulating_secondary:
            self._secondary_buffer += self._pending_buffer
            self._pending_buffer = ""

            # Check if the closing tag arrived (case-insensitive regex search)
            close_match = re.search(re.escape(self._secondary_closing_tag), self._secondary_buffer, re.IGNORECASE)
            if close_match:
                # Closing tag found! Extract the full match.
                close_idx = close_match.start()
                close_len = close_match.end() - close_match.start()
                
                body_content = self._secondary_buffer[len(self._secondary_open_tag):close_idx]
                closing_tag = self._secondary_buffer[close_idx:close_idx+close_len]
                full_match_text = self._secondary_open_tag + body_content + closing_tag

                self._is_accumulating_secondary = False

                # Dispatch to _dispatch_closed_tag for processing
                if full_match_text not in self.processed_tags:
                    self.processed_tags.add(full_match_text)
                    self._dispatch_closed_tag(
                        self._secondary_tag_name,
                        self._secondary_open_tag,
                        body_content.strip(),
                        full_match_text
                    )

                # Close the processing block with status metadata
                proc_close_tag = f'\n<!-- status:finished -->\n</processing>\n'
                self.ai_message.content += proc_close_tag
                _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                # Preserve any text that came after the closing tag
                remaining_text = self._secondary_buffer[close_idx + close_len:]
                
                # Reset secondary state
                self._secondary_tag_name = ""
                self._secondary_closing_tag = ""
                self._secondary_open_tag = ""
                self._secondary_buffer = ""

                if remaining_text.strip():
                    self._pending_buffer = remaining_text
                # ── ONE-ACTION-PER-TURN: Halt generation immediately ──
                self._action_dispatched = True
                return False
            else:
                # Still accumulating body. Emit lightweight status if enabled.
                # (No structural analysis for secondary tags — just suppress raw output)
                pass
            return True

        # ── Default Forwarding ──
        # Robust partial tag detection: Check if the buffer ends with a prefix of any known tag.
        # This prevents raw XML from leaking when the LLM streams tokens with trailing spaces or partial attributes.
        def _ends_with_partial_tag(buffer: str) -> int:
            """Returns the start index of the partial tag if found, else -1."""
            # Include memory tags in the check so they don't leak as partial tags
            tags_to_check = [
                "<artifact", "<artefact", "<tool", "<think", "<note", "<skill", "<scratchpad", 
                "<generate_image", "<edit_image", "<lollms_form", "<unlock_file", "<lock_file", "<hide_file",
                "<mem_new", "<mem_update", "<mem_tag", "<mem_load", "<mem_delete", "<mem_search"
            ]

            # Helper to check if the start of the line is valid for a tag
            def _is_at_line_start(buf: str, idx: int) -> bool:
                if idx == 0:
                    return True
                # Walk backwards from idx to the previous newline. All chars must be whitespace.
                i = idx - 1
                while i >= 0 and buf[i] != '\n':
                    if not buf[i].isspace():
                        return False
                    i -= 1
                return True

            for tag in tags_to_check:
                # Check if the buffer ends with a STRICT prefix of the tag (e.g., "<art", "<to")
                for i in range(1, len(tag)):
                    if buffer.endswith(tag[:i]):
                        start_idx = len(buffer) - i
                        if _is_at_line_start(buffer, start_idx):
                            return start_idx
                        # If not at start of line, it's not a functional tag. Ignore.

            # Fallback: Check for partial tags with trailing spaces or partial attribute names
            for tag in tags_to_check:
                idx = buffer.rfind(tag)
                if idx != -1 and ">" not in buffer[idx:]:
                    if _is_at_line_start(buffer, idx):
                        return idx
                    # If not at start of line, ignore.

            return -1

        partial_idx = _ends_with_partial_tag(self._pending_buffer)
        if partial_idx != -1:
            # Forward text before the partial tag to the UI and save it
            text_before_partial = self._pending_buffer[:partial_idx]
            if text_before_partial:
                self.ai_message.content += text_before_partial
                _cb(self.callback, text_before_partial, MSG_TYPE.MSG_TYPE_CHUNK)

            # Hold the partial tag in the pending buffer for the next chunk
            self._pending_buffer = self._pending_buffer[partial_idx:]
            return True

        # No partial tags, forward everything and save it
        self.ai_message.content += self._pending_buffer
        _cb(self.callback, self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
        self._pending_buffer = ""
        return True
    def _dispatch_closed_tag(self, tag_name: str, attrs_str: str, body: str, full_match_text: str) -> bool:
        # If attrs_str starts with '<', it's the full opening tag. Extract attrs from it.
        if attrs_str.startswith('<'):
            attrs = {}
            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                attrs[m.group(1).lower()] = m.group(2)
            tag_name = re.match(r'<(\w+)', attrs_str).group(1).lower()
        else:
            attrs = {}
            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                attrs[m.group(1).lower()] = m.group(2)

        # 1. Artifact Creation & Patching
        if tag_name in ("artifact", "artefact"):
            if not self.enable_artefacts:
                return True
            title = attrs.get("name") or attrs.get("title") or f"artifact_{uuid.uuid4().hex[:8]}"

            atype = attrs.get("type", "document")
            lang = attrs.get("language")
            is_ephemeral = attrs.get("ephemeral", "false").lower() in ("true", "1", "yes")

            is_new = self.discussion.artefacts.get(title) is None
            is_patch = "<<<<<<< SEARCH" in body

            if is_patch and not is_new:
                existing = self.discussion.artefacts.get(title)
                try:
                    patched = self.discussion.artefacts.apply_aider_patch(existing["content"], body)
                    art = self.discussion.artefacts.update(
                        title=title, new_content=patched, language=lang, bump_version=True, active=self.auto_activate,
                        ephemeral=is_ephemeral
                    )
                except Exception as patch_err:
                    ASCIIColors.error(f"[StreamState] Artifact patch failed: {patch_err}")
                    self._last_dispatch_failed = True
                    proc_open = f'\n<processing type="artefact" title="{title}" language="{lang or ""}">\n'
                    proc_body = f'* ❌ Failed to apply patch to artifact: {patch_err}\n'
                    proc_close = f'<!-- status:failure -->\n</processing>\n'
                    proc_block = proc_open + proc_body + proc_close

                    self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                    _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    return True

                if art:
                    self.affected_artefacts.append(art)

                try:
                    self.discussion.artefacts._sync_to_disk_workspace(
                        title=art.get("title", title),
                        content=art.get("content", patched),
                        version=art.get("version", 1),
                        atype=atype,
                        language=lang
                    )
                except Exception as sync_ex:
                    ASCIIColors.warning(f"[StreamState] Failed to immediately materialize patched artifact '{title}' to disk: {sync_ex}")

                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                    "type": "artifact_updated",
                    "title": title,
                    "version": art.get("version", 1) if art else 1,
                    "art_type": atype
                })
                return True
            else:
                if is_new:
                    art = self.discussion.artefacts.add(
                        title=title, artefact_type=atype, content=body, language=lang, active=self.auto_activate,
                        ephemeral=is_ephemeral
                    )
                else:
                    art = self.discussion.artefacts.update(
                        title=title, new_content=body, new_type=atype, language=lang, bump_version=True, active=self.auto_activate,
                        ephemeral=is_ephemeral
                    )

            if art:
                self.affected_artefacts.append(art)

                # ── 📊 LOG ARTIFACT CREATION ACTION ──
                # Log this action in the turn progress tracker
                if hasattr(self.discussion, '_turn_actions_log'):
                    self.discussion._turn_actions_log.append({
                        "action": "artifact_created",
                        "title": title,
                        "type": atype,
                        "round": getattr(self.discussion, '_current_round', 0)
                    })

            # ── 🛑 CRITICAL FIX: IMMEDIATE PHYSICAL MATERIALIZATION ──
            # The physical twin MUST exist on disk the instant the artifact is created.
            # If the LLM emits a <tool> tag in the very next token that references this file,
            # the tool will fail with "File not found" if we rely on deferred syncing.
            # We force a synchronous write to the workspace_data directory right now.
            try:
                # Use the discussion's artefact manager to sync this specific file to disk
                self.discussion.artefacts._sync_to_disk_workspace(
                    title=art.get("title", title),
                    content=art.get("content", body),
                    version=art.get("version", 1),
                    atype=atype,
                    language=lang
                )
            except Exception as sync_ex:
                ASCIIColors.warning(f"[StreamState] Failed to immediately materialize artifact '{title}' to disk: {sync_ex}")

            # ── CRITICAL: DO NOT MUTATE ai_message.content ──
            # The raw <artifact> XML is preserved in the message content.
            # The export() method in _mixin_utils.py will handle replacing it
            # with the [🔒artefact tag called, content stripped for brievety, do not mimic:title|type] marker when building
            # history for the LLM. This prevents the marker from leaking into the live UI.

            # Fire an event update to the UI so it cleanly rebuilds and replaces the code block
            meta_info = _extract_artefact_meta(body, lang, atype)
            if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACT_BUILD_END, {
                    "title": title,
                    "art_type": atype,
                    "language": lang,
                    "version": art.get("version", 1) if art else 1,
                    "success": bool(art),
                    "error": None,
                    "stream_complete": True,
                    "operation": "patch" if is_patch else ("create" if is_new else "full_rewrite"),
                    "is_patch": is_patch,
                    "line_count": meta_info["line_count"],
                    "size_chars": meta_info["size_chars"],
                    "estimated_tokens": meta_info["estimated_tokens"],
                    "sections": meta_info["sections"],
                    "sections_count": meta_info["sections_count"],
                    "patch_stats": meta_info["patch_stats"],
                    "preview": meta_info["preview"]
                })

            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_updated" if not is_new else "artifact_created",
                "title": title,
                "version": art.get("version", 1) if art else 1,
                "art_type": atype,
                "line_count": meta_info["line_count"],
                "size_chars": meta_info["size_chars"],
                "estimated_tokens": meta_info["estimated_tokens"],
                "sections": meta_info["sections"]
            })
            return True

        # 2. Tools Execution Trigger
        elif tag_name in ("tool", "tool"):
            self.tool_trigger = True

            # ── ROBUST JSON PARSING & NORMALIZATION (CRITICAL FIX) ──
            # LLMs often hallucinate flat structures: {"name": "tool", "arg": "val"}
            # instead of nested: {"name": "tool", "parameters": {"arg": "val"}}
            # We MUST normalize this here to prevent execution failures.
            tool_name = ""

            def _sanitize_tool_json(raw_body: str) -> str:
                """
                Safely extracts the first valid JSON object from a tool body.
                Handles trailing backticks, markdown fences, and stray prose
                that cause 'Extra data' JSONDecodeError.
                """
                import json as _json
                stripped = raw_body.strip()
                if stripped.startswith("```"):
                    lines = stripped.splitlines()
                    if len(lines) >= 2:
                        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                stripped = stripped.strip("`").strip()
                try:
                    decoder = _json.JSONDecoder()
                    obj, end_idx = decoder.raw_decode(stripped)
                    return _json.dumps(obj)
                except (_json.JSONDecodeError, ValueError):
                    return stripped

            sanitized_body = _sanitize_tool_json(body)
            try:
                raw_data = json.loads(sanitized_body)
                if isinstance(raw_data, dict):
                    tool_name = raw_data.get("name", "")

                    if "parameters" in raw_data and isinstance(raw_data["parameters"], dict):
                        self.tool_json_data = sanitized_body
                    else:
                        params = {k: v for k, v in raw_data.items() if k != "name"}
                        normalized_data = {"name": tool_name, "parameters": params}
                        self.tool_json_data = json.dumps(normalized_data)
                else:
                    self.tool_json_data = sanitized_body
            except json.JSONDecodeError as je:
                self.tool_json_data = sanitized_body
                ASCIIColors.error(f"[StreamState] JSON decode failed: {je}")

            # ── 🛑 CRITICAL FIX: IMMEDIATE UI FEEDBACK ──
            # Emit the processing block to the UI INSTANTLY when the </tool> tag closes.
            # This guarantees the user sees "Calling tool..." while the tool executes,
            # rather than waiting for the synchronous execution to finish.
            import html
            try:
                parsed_for_ui = json.loads(self.tool_json_data)
                ui_tool_name = parsed_for_ui.get("name", "unknown") if isinstance(parsed_for_ui, dict) else "unknown"
                ui_params = parsed_for_ui.get("parameters", {}) if isinstance(parsed_for_ui, dict) else {}
            except Exception:
                ui_tool_name = tool_name or "unknown"
                ui_params = {}

            escaped_params = html.escape(json.dumps(ui_params, default=str))
            tool_open_tag = f'\n<processing type="tool" title="Tool Execution: {ui_tool_name}" params="{escaped_params}">\n'
            self.ai_message.content += tool_open_tag
            _cb(self.callback, tool_open_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            status_line = f"* Calling local tool system for '{ui_tool_name}'...\n"
            self.ai_message.content += status_line
            _cb(self.callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            # Halt generation instantly so the executor can take over the loop
            return False

        # 3. User Note
        elif tag_name == "note":
            if not self.enable_notes:
                return True
            title = attrs.get("title") or attrs.get("name") or f"note_{uuid.uuid4().hex[:8]}"

            is_patch = "<<<<<<< SEARCH" in body
            if is_patch:
                existing = self.discussion.artefacts.get(title)
                if existing:
                    try:
                        patched_content = self.discussion.artefacts.apply_aider_patch(existing["content"], body)
                        art = self.discussion.artefacts.update(
                            title=title, new_content=patched_content, bump_version=True, active=self.auto_activate
                        )
                    except Exception as patch_err:
                        ASCIIColors.error(f"[StreamState] Note patch failed: {patch_err}")
                        proc_open = f'\n<processing type="note" title="{title}">\n'
                        proc_body = f'* ❌ Failed to apply patch to note: {patch_err}\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close

                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        return True
                else:
                    ASCIIColors.warning(f"[StreamState] Note patch ignored (note '{title}' not found). Creating new.")
                    art = self.discussion.artefacts.add(
                        title=title, artefact_type=ArtefactType.NOTE, content=body, active=self.auto_activate
                    )
            else:
                art = self.discussion.artefacts.add(
                    title=title, artefact_type=ArtefactType.NOTE, content=body, active=self.auto_activate
                )

            if art:
                self.affected_artefacts.append(art)

            proc_open = f'\n<processing type="note" title="{title}">\n'
            proc_body = f'* 🗒️ Note captured and saved to workspace.\n'
            proc_close = f'<!-- status:finished -->\n</processing>\n'
            proc_block = proc_open + proc_body + proc_close

            self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
            _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_created",
                "title": title,
                "art_type": "note"
            })
            return True

        # 3b. Scratchpad (Intermediate Hypothesis Workspace)
        elif tag_name == "scratchpad":
            if not self.enable_artefacts:
                return True
            title = attrs.get("title") or attrs.get("name") or "scratchpad"

            is_patch = "<<<<<<< SEARCH" in body
            if is_patch:
                existing = self.discussion.artefacts.get(title)
                if existing:
                    try:
                        patched_content = self.discussion.artefacts.apply_aider_patch(existing["content"], body)
                        art = self.discussion.artefacts.update(
                            title=title, new_content=patched_content, bump_version=True, active=self.auto_activate
                        )
                    except Exception as patch_err:
                        ASCIIColors.error(f"[StreamState] Scratchpad patch failed: {patch_err}")
                        proc_open = f'\n<processing type="scratchpad" title="{title}">\n'
                        proc_body = f'* ❌ Failed to apply patch to scratchpad: {patch_err}\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close

                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        return True
                else:
                    ASCIIColors.warning(f"[StreamState] Scratchpad patch ignored (scratchpad '{title}' not found). Creating new.")
                    art = self.discussion.artefacts.add(
                        title=title, artefact_type=ArtefactType.SCRATCHPAD, content=body, active=self.auto_activate
                    )
            else:
                art = self.discussion.artefacts.add(
                    title=title, artefact_type=ArtefactType.SCRATCHPAD, content=body, active=self.auto_activate
                )

            if art:
                self.affected_artefacts.append(art)

            proc_open = f'\n<processing type="scratchpad" title="{title}">\n'
            proc_body = f'* 📝 Scratchpad updated and saved to workspace.\n'
            proc_close = f'<!-- status:finished -->\n</processing>\n'
            proc_block = proc_open + proc_body + proc_close

            self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
            _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_created",
                "title": title,
                "art_type": "scratchpad"
            })
            return True

        # 4. Long-term Skill
        elif tag_name == "skill":
            if not self.enable_skills:
                return True
            title = attrs.get("title") or attrs.get("name") or f"skill_{uuid.uuid4().hex[:8]}"
            desc = attrs.get("description", "")
            cat = attrs.get("category", "")
            is_patch = "<<<<<<< SEARCH" in body

            # ── HANDBAG / EXTERNAL SKILL ROUTING ──
            # Check if a personality with a SkillsManager is attached to this discussion.
            # The `personality` object is injected into _StreamState during ChatMixin.chat() execution.
            personality = getattr(self.discussion, '_active_personality', None)
            skills_mgr = getattr(personality, 'skills_manager', None) if personality else None

            if skills_mgr:
                # Route to the SkillsManager (Handbag or external folder)
                existing_skill = skills_mgr.skills.get(title.lower())
                if not existing_skill:
                    # Fuzzy search if exact title not found
                    matches = skills_mgr.search_skills(title)
                    if matches:
                        existing_skill = matches[0]

                if existing_skill:
                    if not existing_skill.modifiable:
                        # BLOCK: Skill is marked as read-only
                        self._last_dispatch_failed = True
                        proc_open = f'\n<processing type="skill" title="{title}">\n'
                        proc_body = f'* 🚫 BLOCKED: Skill \'{existing_skill.title}\' is marked as READ-ONLY (unmodifiable).\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close

                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        return True

                    # Skill exists and is modifiable. Attempt patch or update via SkillsManager.
                    if is_patch:
                        try:
                            patched_content = self.discussion.artefacts.apply_aider_patch(existing_skill.content, body)
                        except Exception as patch_err:
                            ASCIIColors.error(f"[StreamState] Handbag skill patch failed: {patch_err}")
                            self._last_dispatch_failed = True
                            proc_open = f'\n<processing type="skill" title="{title}">\n'
                            proc_body = f'* ❌ Failed to apply patch to skill: {patch_err}\n'
                            proc_close = f'<!-- status:failure -->\n</processing>\n'
                            proc_block = proc_open + proc_body + proc_close
                            self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                            _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                            _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                            _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                            return True
                    else:
                        patched_content = body

                    updated_skill = skills_mgr.update_skill(
                        title=existing_skill.title,
                        content=patched_content,
                        description=desc if desc else None,
                        category=cat if cat else None
                    )

                    if updated_skill:
                        proc_open = f'\n<processing type="skill" title="{existing_skill.title}">\n'
                        proc_body = f'* 🧠 Skill \'{existing_skill.title}\' updated successfully in your handbag.\n'
                        proc_close = f'<!-- status:finished -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                            "type": "artifact_updated",
                            "title": existing_skill.title,
                            "art_type": "skill"
                        })
                        return True
                    else:
                        # update_skill returned None (failed write)
                        self._last_dispatch_failed = True
                        proc_open = f'\n<processing type="skill" title="{existing_skill.title}">\n'
                        proc_body = f'* ❌ Failed to write skill updates to disk.\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        return True
                else:
                    # Skill doesn't exist in manager. Create it.
                    new_skill = skills_mgr.create_skill(
                        title=title,
                        content=body,
                        description=desc,
                        category=cat,
                        visibility="loadable"
                    )
                    if new_skill:
                        proc_open = f'\n<processing type="skill" title="{title}">\n'
                        proc_body = f'* 🧠 Skill \'{title}\' created successfully in your handbag.\n'
                        proc_close = f'<!-- status:finished -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                            "type": "artifact_created",
                            "title": title,
                            "art_type": "skill"
                        })
                        return True
                    else:
                        self._last_dispatch_failed = True
                        ASCIIColors.warning(f"[StreamState] Failed to create skill '{title}' via SkillsManager. Falling back to artifact store.")
                        # Fall through to generic artifact creation below

            # ── FALLBACK: GENERIC ARTIFACT CREATION (No SkillsManager or Hardcoded Skill) ──
            # If we reach here, either there is no skills_manager, or creation failed.
            # Hardcoded skills (passed directly to chat) should be treated as unmodifiable.
            # Since we cannot enforce mutability on generic artifacts, we just save it.
            if is_patch:
                existing = self.discussion.artefacts.get(title)
                if existing:
                    try:
                        patched_content = self.discussion.artefacts.apply_aider_patch(existing["content"], body)
                        art = self.discussion.artefacts.update(
                            title=title, new_content=patched_content, bump_version=True, active=self.auto_activate
                        )
                    except Exception as patch_err:
                        ASCIIColors.error(f"[StreamState] Skill patch failed: {patch_err}")
                        self._last_dispatch_failed = True
                        proc_open = f'\n<processing type="skill" title="{title}">\n'
                        proc_body = f'* ❌ Failed to apply patch to skill: {patch_err}\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        return True
                else:
                    ASCIIColors.warning(f"[StreamState] Skill patch ignored (skill '{title}' not found). Creating new.")
                    art = self.discussion.artefacts.add(
                        title=title, artefact_type=ArtefactType.SKILL, content=body, active=self.auto_activate, description=desc, category=cat
                    )
            else:
                art = self.discussion.artefacts.add(
                    title=title, artefact_type=ArtefactType.SKILL, content=body, active=self.auto_activate, description=desc, category=cat
                )

            if art:
                self.affected_artefacts.append(art)

            proc_open = f'\n<processing type="skill" title="{title}">\n'
            proc_body = f'* 🧠 Skill captured and saved to workspace.\n'
            proc_close = f'<!-- status:finished -->\n</processing>\n'
            proc_block = proc_open + proc_body + proc_close

            self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
            _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_created",
                "title": title,
                "art_type": "skill"
            })
            return True

        # 5. Multi-tier Context Visibility Management
        elif tag_name in ("unlock_file", "lock_file", "hide_file"):
            from lollms_client.lollms_artefact import ArtefactVisibility

            # ── CONTEXT BUDGET GUARD ──
            # Maximum tokens allowed for a single file to be unlocked into context.
            # Files exceeding this threshold are blocked from FULL visibility to
            # prevent context overflow and empty-response loops.
            _MAX_UNLOCK_TOKENS = 50000

            # Map tag name to target visibility state
            target_visibility = ArtefactVisibility.FULL
            action_verb = "Unlocking"
            if tag_name == "lock_file":
                target_visibility = ArtefactVisibility.TREE_LOCKED
                action_verb = "Locking"
            elif tag_name == "hide_file":
                target_visibility = ArtefactVisibility.HIDDEN
                action_verb = "Hiding"

            targets = [t.strip() for t in body.splitlines() if t.strip()]

            processed_files = []
            already_in_state = []
            not_found = []
            blocked_files = []

            for t_file in targets:
                art = self.discussion.artefacts.get(t_file)
                if not art:
                    not_found.append(t_file)
                elif art.get("visibility") == target_visibility:
                    already_in_state.append(t_file)
                elif target_visibility == ArtefactVisibility.FULL:
                    # ── CONTEXT BUDGET CHECK ──
                    # Check if the file is too large to safely load into context
                    token_count = art.get("token_count", 0)
                    content_len = len(art.get("content", ""))

                    # If token_count is 0 or unreliable, estimate from content length
                    if token_count == 0 and content_len > 0:
                        token_count = content_len // 4

                    if token_count > _MAX_UNLOCK_TOKENS:
                        ASCIIColors.warning(
                            f"[ContextBudgetGuard] Blocked unlock of '{t_file}': "
                            f"~{token_count:,} tokens exceeds limit of {_MAX_UNLOCK_TOKENS:,}."
                        )
                        blocked_files.append((t_file, token_count))
                    else:
                        self.discussion.artefacts.set_visibility(t_file, target_visibility)
                        processed_files.append(t_file)
                else:
                    self.discussion.artefacts.set_visibility(t_file, target_visibility)
                    processed_files.append(t_file)

            if processed_files:
                self.discussion.commit()
                # If we unlocked files, mark that we need a continuation round
                if target_visibility == ArtefactVisibility.FULL:
                    self.context_unlock_requested = True
                    self.context_unlocked_files.extend(processed_files)

            # Build UI feedback inside a processing block
            status_parts = []
            if processed_files:
                status_parts.append(f"✅ {action_verb}: {', '.join(processed_files)}")
            if already_in_state:
                status_parts.append(f"⚠️ Already in target state: {', '.join(already_in_state)}")
            if not_found:
                status_parts.append(f"❌ Not found: {', '.join(not_found)}")
            if blocked_files:
                blocked_desc = "; ".join(
                    f"{bf} (~{tc:,} tokens)" for bf, tc in blocked_files
                )
                status_parts.append(
                    f"🛑 BLOCKED (too large for context): {blocked_desc}. "
                    f"Use a tool (SQL query, grep, or Python script) to extract "
                    f"specific data from this file instead of loading it fully."
                )

            status_line = f"* {action_verb} context files...\n"
            details_block = f"Context Update:\n{'; '.join(status_parts)}\n"
            status_meta = "failure" if (not_found and not processed_files) or blocked_files else "success"

            # The generic secondary tag interceptor already emitted the <processing> opening block.
            # We just need to append the status content and close the block.
            proc_close = f'{status_line}{details_block}<!-- status:{status_meta} -->\n</processing>\n\n'
            self.ai_message.content += proc_close
            _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            # ── INJECT CONTEXT BUDGET GUIDANCE INTO VIRTUAL HISTORY ──
            # If files were blocked, inject a system message so the LLM knows
            # it must use tools to access that data, not <unlock_file>.
            if blocked_files:
                blocked_names = ", ".join(f"`{bf}`" for bf, _ in blocked_files)
                self.context_unlock_requested = True  # Force continuation so LLM sees the guidance
                self.context_unlocked_files.extend([bf for bf, _ in blocked_files])
                # Store the blocked guidance for the continuation prompt
                if not hasattr(self, '_blocked_files_guidance'):
                    object.__setattr__(self, '_blocked_files_guidance', [])
                self._blocked_files_guidance.append(
                    f"The following files are too large to load into context directly: {blocked_names}. "
                    f"You MUST use a tool (e.g., SQL query, grep, or execute_python_code) to extract "
                    f"specific data from these files. Do NOT attempt to <unlock_file> them again."
                )

            return True

        return True

    def was_action_dispatched(self) -> bool:
        """Returns True if a functional tag was fully dispatched during this generation turn."""
        return self._action_dispatched

    def was_done_detected(self) -> bool:
        """Returns True if the LLM emitted the <done/> termination tag."""
        return self._done_detected

    def was_last_dispatch_failed(self) -> bool:
        """Returns True if the last dispatched artifact tag failed (e.g., SEARCH/REPLACE mismatch)."""
        return self._last_dispatch_failed

    def passthrough(self, chunk, msg_type=None, meta=None) -> bool:
        if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
            if msg_type in (MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, MSG_TYPE.MSG_TYPE_REASONING):
                self.ai_message.thoughts = (self.ai_message.thoughts or "") + (chunk or "")
            return _cb(self.callback, chunk, msg_type, meta)
        return True

    def flush_remaining_buffer(self):
        """Flushes any safe text remaining in the shadow buffer at the end of generation."""
        # CRITICAL: Stop heartbeat if artifact was never closed
        self._stop_artefact_heartbeat()

        # ── Handle unclosed code fence ──
        # If we're still in code fence mode at flush time, the fence was never closed.
        # Re-process the hold buffer through tag detection to intercept any functional tags
        # that were trapped inside the unclosed fence.
        if self._in_code_fence:
            self._in_code_fence = False
            hold = self._code_fence_hold_buffer
            self._code_fence_hold_buffer = ""
            if hold:
                # Re-feed through the full parser to intercept any functional tags
                self.feed(hold)

        # ── CRITICAL FIX: Force-dispatch incomplete tool calls ──
        # If the LLM finishes generation while we are still accumulating a tool call 
        # (e.g., it omitted the closing </tool> tag or hit a stop token), we must 
        # synthesize the closing tag and dispatch it so tool_trigger is set to True.
        if self._is_accumulating_tool:
            # Combine buffers to capture any partial JSON that arrived in the last chunk
            full_tool_call = self._tool_buffer + self._pending_buffer
            json_body = re.sub(r'^<tool>', '', full_tool_call, flags=re.IGNORECASE)
            json_body = re.sub(r'</tool>\s*$', '', json_body, flags=re.IGNORECASE).strip()

            self._is_accumulating_tool = False
            self._pending_buffer = ""
            self._tool_buffer = ""

            # Dispatch the tool call silently. The ChatMixin will handle the UI processing block.
            self._dispatch_closed_tag("tool", "", json_body, full_tool_call)
            return  # Exit early; the tool call has been dispatched

        # ── Force-dispatch incomplete secondary tags (unclosed <skill>, <note>, etc.) ──
        if self._is_accumulating_secondary:
            # The LLM finished generation without closing the tag.
            # Synthesize a closing tag and dispatch what we have.
            full_match_text = self._secondary_buffer + self._secondary_closing_tag
            body_content = self._secondary_buffer[len(self._secondary_open_tag):]

            self._is_accumulating_secondary = False

            if full_match_text not in self.processed_tags:
                self.processed_tags.add(full_match_text)
                self._dispatch_closed_tag(
                    self._secondary_tag_name,
                    self._secondary_open_tag,
                    body_content.strip(),
                    full_match_text
                )

            # Close the processing block
            proc_close_tag = f'\n<!-- status:finished -->\n</processing>\n'
            self.ai_message.content += proc_close_tag
            _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            self._secondary_tag_name = ""
            self._secondary_closing_tag = ""
            self._secondary_open_tag = ""
            self._secondary_buffer = ""

            # ── ONE-ACTION-PER-TURN: Halt generation immediately ──
            self._action_dispatched = True
            return

        # ── 🛑 POST-STREAM <done/> / <end/> SWEEP (DEFENSE-IN-DEPTH) ──
        # The streaming interceptor in feed() can miss <done/> when the parser
        # is inside a code fence, inline code, artifact, or secondary tag state.
        # After all buffers are flushed, scan the ENTIRE accumulated content
        # for any termination tag that was missed, strip it, and set the flag.
        if not self._done_detected:
            done_pattern = re.compile(r'(?i)<(?:done|end)\s*/?>')
            if done_pattern.search(self.ai_message.content):
                ASCIIColors.info("[StreamState] Post-stream sweep detected missed <done/> or <end/> tag. Setting termination flag.")
                self._done_detected = True
                self.ai_message.content = done_pattern.sub('', self.ai_message.content).strip()

        if self._pending_buffer or self.artefact_tracker.is_inside_artefact:
            # If we are still inside an artifact for some reason (unclosed tag), dump it to the UI
            if self.artefact_tracker.is_inside_artefact:
                # ── 🛑 TRUNCATED ARTIFACT RECOVERY ──
                # The LLM finished generation without closing the <artifact> tag.
                # This often happens with SEARCH/REPLACE blocks that hit max_tokens.
                # We synthesize the closing tag and attempt a best-effort dispatch.
                self._artefact_buffer += self._pending_buffer
                self._pending_buffer = ""

                # Check if we have a valid opening tag to extract attributes from
                lower_buf = self._artefact_buffer.lower()
                open_idx = lower_buf.find("<artifact")
                if open_idx == -1:
                    open_idx = lower_buf.find("<artefact")

                if open_idx != -1:
                    end_of_open_tag = self._artefact_buffer.find(">", open_idx)
                    if end_of_open_tag != -1:
                        opening_tag = self._artefact_buffer[open_idx:end_of_open_tag+1]
                        body_content = self._artefact_buffer[end_of_open_tag+1:]
                        closing_tag = "</artifact>"
                        full_match_text = opening_tag + body_content + closing_tag

                        if full_match_text not in self.processed_tags:
                            self.processed_tags.add(full_match_text)
                            ASCIIColors.warning("[StreamState] Detected truncated artifact. Attempting best-effort dispatch.")
                            self._dispatch_closed_tag(
                                "artifact",
                                opening_tag,
                                body_content.strip(),
                                full_match_text
                            )

                        # Close the processing block
                        proc_close_tag = '\n<!-- status:finished -->\n</processing>\n'
                        self.ai_message.content += proc_close_tag
                        _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        # Mark that an action was dispatched so the loop continues correctly
                        self._action_dispatched = True
                        self.artefact_tracker.close()
                        self._artefact_buffer = ""
                        return

                # Fallback: if we couldn't parse the opening tag, just dump to UI
                self.ai_message.content += self._artefact_buffer
                _cb(self.callback, self._artefact_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
                self._artefact_buffer = ""
                self.artefact_tracker.close()
            else:
                # Otherwise, it's just trailing text or a partial tag that never completed
                self.ai_message.content += self._pending_buffer
                _cb(self.callback, self._pending_buffer, MSG_TYPE.MSG_TYPE_CHUNK)
            self._pending_buffer = ""

    def get_tool_call_json(self) -> Optional[str]:
        return self.tool_json_data if self.tool_trigger else None

    def get_clean_text_so_far(self) -> str:
        return self.ai_message.content

# ── ChatMixin Implementation ────────────────────────────────────────────────

class ChatMixin:
    """ChatMixin: orchestrates RAG, tiered memory, and alternating tool rounds."""

    def __init__(self, *args, **kwargs):
        """Initialize ChatMixin with sequential cancellation support."""
        # Simple boolean flag for sequential control
        object.__setattr__(self, '_cancel_flag', False)
        super().__init__(*args, **kwargs)

        from ..lollms_memory.lollms_memory import FailureMemory
        object.__setattr__(self, '_failure_memory', FailureMemory())

    def cancel_generation(self) -> bool:
        """
        Signals the active generation loop to stop gracefully.
        """
        object.__setattr__(self, '_cancel_flag', True)

        # Propagate to client immediately to stop low-level streaming
        if hasattr(self, 'lollmsClient') and self.lollmsClient:
            try:
                if hasattr(self.lollmsClient, 'cancel'):
                    self.lollmsClient.cancel()
                elif hasattr(self.lollmsClient, 'llm') and hasattr(self.lollmsClient.llm, 'cancel'):
                    self.lollmsClient.llm.cancel()
            except Exception:
                pass
        return True

    def is_generation_cancelled(self) -> bool:
        """
        Checks if cancellation has been requested.

        Returns:
            bool: True if cancellation is active, False otherwise.
        """
        return getattr(self, '_cancel_flag', False)

    def reset_cancel_state(self) -> None:
        """Resets the cancellation flag for a new generation turn."""
        object.__setattr__(self, '_cancel_flag', False)

    def _get_pending_forms(self) -> Dict[str, Dict]:
        if not hasattr(self, '_pending_forms_store'):
            object.__setattr__(self, '_pending_forms_store', {})
        return self._pending_forms_store

    def submit_form_response(self, form_id: str, answers: Dict[str, Any]) -> bool:
        pending = self._get_pending_forms()
        form_descriptor = pending.pop(form_id, None)
        if form_descriptor is None:
            ASCIIColors.warning(f"[Form] submit_form_response: form_id '{form_id}' not found.")
            return False

        answer_text = _format_form_answers_for_llm(form_descriptor, answers)
        self.add_message(
            sender="user",
            sender_type="user",
            content=answer_text,
            metadata={"form_id": form_id, "form_answers": answers},
        )

        cb = getattr(self, '_active_callback', None)
        _cb(cb, json.dumps({"form_id": form_id, "answers": answers}),
            MSG_TYPE.MSG_TYPE_FORM_SUBMITTED,
            {"form_id": form_id, "answers": answers, "form": form_descriptor})

        ASCIIColors.success(f"[Form] '{form_descriptor.get('title')}' answers injected.")
        return True

    def _sync_tool_artifacts(
        self,
        tool_name: str,
        files_before: Dict,
        files_after: Dict,
        callback: Optional[Callable]
    ) -> None:
        """
        Detects new and modified files by diffing before/after workspace snapshots,
        then registers them as artifacts following the Tool-Generated File Visibility Doctrine.
        This logic is shared between the direct-callable and LCP dispatch paths.
        """
        from pathlib import Path
        from lollms_client.lollms_artefact import ArtefactVisibility, ArtefactType

        # Detect NEW files
        new_files = set(files_after.keys()) - set(files_before.keys())

        for rel_path in new_files:
            file_info = files_after[rel_path]
            file_name = rel_path.name
            file_ext = rel_path.suffix.lower()
            file_path = file_info["path"]
            file_size = file_path.stat().st_size

            atype = "document"
            if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
                atype = "code"
            elif file_ext in (".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
                atype = "data"
            elif file_ext in (".md", ".txt", ".log", ".out", ".trace", ".asc", ".raw", ".json", ".yaml", ".yml", ".xml", ".ttl", ".pdf", ".docx", ".pptx", ".odt"):
                atype = "document"
            elif file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
                atype = "image"

            EXPLICIT_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet",
                                    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                                    ".zip", ".tar", ".gz"}

            should_read_content = True
            content_placeholder = None

            if file_ext in EXPLICIT_BINARY_EXTS:
                should_read_content = False
                content_placeholder = (
                    f"### Data File Generated: `{file_name}`\n\n"
                    f"This file was created by the tool `{tool_name}`.\n"
                    f"- **Type**: {file_ext.upper()} (Binary/Structured Data)\n"
                    f"- **Size**: {file_size:,} bytes\n"
                    f"- **Location**: `./{file_name}`\n\n"
                    f"> **Action**: You can download this file from the Workspace Artifacts panel or reference it in SQL/Python tools."
                )
            else:
                try:
                    with open(file_path, 'rb') as f:
                        chunk = f.read(1024)
                        if b'\x00' in chunk:
                            should_read_content = False
                            content_placeholder = (
                                f"### Binary File Detected: `{file_name}`\n\n"
                                f"This file appears to be binary (contains null bytes).\n"
                                f"- **Type**: {file_ext.upper()} (Unknown Binary)\n"
                                f"- **Size**: {file_size:,} bytes\n"
                                f"- **Location**: `./{file_name}`\n\n"
                                f"> **Action**: Download from Workspace Artifacts panel."
                            )
                        else:
                            forced_content = file_path.read_text(encoding='utf-8', errors='ignore')
                            file_info["content"] = forced_content
                            should_read_content = True
                except Exception as e:
                    should_read_content = False
                    content_placeholder = f"### File Error: `{file_name}`\n\nFailed to read or inspect file: {e}"

            if not should_read_content and content_placeholder:
                existing_art = self.artefacts.get(file_name)
                if existing_art:
                    art = self.artefacts.update(
                        title=file_name,
                        new_content=content_placeholder,
                        new_type=atype,
                        active=True,
                        visibility=ArtefactVisibility.FULL,
                        commit_message=f"Updated binary file reference by tool '{tool_name}'"
                    )
                else:
                    art = self.artefacts.add(
                        title=file_name,
                        artefact_type=atype,
                        content=content_placeholder,
                        active=True,
                        visibility=ArtefactVisibility.FULL,
                        commit_message=f"Created by tool '{tool_name}'"
                    )
                self.commit()

                if atype == "image":
                    try:
                        import base64
                        raw_img = file_path.read_bytes()
                        img_b64 = base64.b64encode(raw_img).decode('utf-8')
                        self.artefacts.update(
                            title=file_name,
                            new_images=[img_b64],
                            new_image_media_types=[f"image/{file_ext[1:]}"],
                            bump_version=False
                        )
                        self.commit()
                        self._affected_artefacts_this_turn.append(self.artefacts.get(file_name))
                    except Exception as ex:
                        trace_exception(ex)

                if self.active_branch_id:
                    ai_msg_local = self.get_message(self.active_branch_id)
                    if ai_msg_local:
                        tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                        if tag not in ai_msg_local.content:
                            ai_msg_local.content += f'\n\n{tag}\n'
                        self.commit()
                continue

            existing_art = self.artefacts.get(file_name)
            if existing_art:
                art = self.artefacts.update(
                    title=file_name,
                    new_content=file_info["content"],
                    new_type=atype,
                    active=True,
                    visibility=ArtefactVisibility.FULL,
                    commit_message=f"Restored by tool '{tool_name}'"
                )
            else:
                art = self.artefacts.add(
                    title=file_name,
                    artefact_type=atype,
                    content=file_info["content"],
                    active=True,
                    visibility=ArtefactVisibility.FULL,
                    commit_message=f"Created by tool '{tool_name}'"
                )
            self.commit()

            if self.active_branch_id:
                ai_msg_local = self.get_message(self.active_branch_id)
                if ai_msg_local:
                    tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                    if tag not in ai_msg_local.content:
                        ai_msg_local.content += f'\n\n{tag}\n'
                    self.commit()

        # Detect MODIFIED files
        common_files = set(files_after.keys()) & set(files_before.keys())
        for rel_path in common_files:
            before_info = files_before[rel_path]
            after_info = files_after[rel_path]
            file_name = rel_path.name
            file_ext = rel_path.suffix.lower()
            file_path = after_info["path"]

            mtime_changed = before_info["mtime"] != after_info["mtime"]
            content_changed = before_info.get("hash") != after_info.get("hash")

            img_b64 = None
            img_mtypes = None

            if mtime_changed or content_changed:
                atype = "document"
                if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
                    atype = "code"
                elif file_ext in (".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
                    atype = "data"
                elif file_ext in (".md", ".txt", ".log", ".out", ".trace", ".asc", ".raw", ".json", ".yaml", ".yml", ".xml", ".ttl", ".pdf", ".docx", ".pptx", ".odt"):
                    atype = "document"
                elif file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
                    atype = "image"

                EXPLICIT_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet",
                                        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                                        ".zip", ".tar", ".gz"}

                should_read_content = True
                content_placeholder = None

                if file_ext in EXPLICIT_BINARY_EXTS:
                    should_read_content = False
                    content_placeholder = (
                        f"### Data File Modified: `{file_name}`\n\n"
                        f"This file was modified by the tool `{tool_name}`.\n"
                        f"- **Type**: {file_ext.upper()} (Binary/Structured Data)\n"
                        f"- **Size**: {file_path.stat().st_size:,} bytes\n"
                        f"- **Location**: `./{file_name}`\n\n"
                        f"> **Action**: You can download this file from the Workspace Artifacts panel or reference it in SQL/Python tools."
                    )
                else:
                    try:
                        with open(file_path, 'rb') as f:
                            chunk = f.read(1024)
                            if b'\x00' in chunk:
                                should_read_content = False
                                content_placeholder = (
                                    f"### Binary File Modified: `{file_name}`\n\n"
                                    f"This file was modified by the tool `{tool_name}`.\n"
                                    f"- **Type**: {file_ext.upper()} (Unknown Binary)\n"
                                    f"- **Size**: {file_path.stat().st_size:,} bytes\n"
                                    f"- **Location**: `./{file_name}`\n\n"
                                    f"> **Action**: Download from Workspace Artifacts panel."
                                )
                            else:
                                forced_content = file_path.read_text(encoding='utf-8', errors='ignore')
                                after_info["content"] = forced_content
                                should_read_content = True
                    except Exception as e:
                        should_read_content = False
                        content_placeholder = f"### File Error: `{file_name}`\n\nFailed to read or inspect file: {e}"

                if not should_read_content and content_placeholder:
                    if atype == "image":
                        try:
                            import base64 as _b64_mod
                            raw_img = file_path.read_bytes()
                            img_b64 = _b64_mod.b64encode(raw_img).decode('utf-8')
                            img_mtypes = [f"image/{file_ext[1:]}"]
                        except Exception as ex:
                            trace_exception(ex)

                    existing_art = self.artefacts.get(file_name)
                    if existing_art:
                        art = self.artefacts.update(
                            title=file_name,
                            new_content=content_placeholder,
                            new_type=atype,
                            new_images=img_b64,
                            new_image_media_types=img_mtypes,
                            active=(atype == "image"),
                            visibility=ArtefactVisibility.FULL if atype == "image" else ArtefactVisibility.TREE_UNLOCKABLE,
                            bump_version=True,
                            commit_message=f"Updated binary file reference by tool '{tool_name}'"
                        )
                    else:
                        art = self.artefacts.add(
                            title=file_name,
                            artefact_type=atype,
                            content=content_placeholder,
                            images=img_b64,
                            image_media_types=img_mtypes,
                            active=(atype == "image"),
                            visibility=ArtefactVisibility.FULL if atype == "image" else ArtefactVisibility.TREE_UNLOCKABLE,
                            commit_message=f"Created by tool '{tool_name}'"
                        )
                    self.commit()

                    if atype == "image":
                        try:
                            import base64
                            raw_img = file_path.read_bytes()
                            img_b64 = base64.b64encode(raw_img).decode('utf-8')
                            self.artefacts.update(
                                title=file_name,
                                new_images=[img_b64],
                                new_image_media_types=[f"image/{file_ext[1:]}"],
                                bump_version=True
                            )
                            self.commit()
                            self._affected_artefacts_this_turn.append(self.artefacts.get(file_name))
                        except Exception as ex:
                            trace_exception(ex)

                    if self.active_branch_id:
                        ai_msg_local = self.get_message(self.active_branch_id)
                        if ai_msg_local:
                            tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                            if tag not in ai_msg_local.content:
                                ai_msg_local.content += f'\n\n{tag}\n'
                            self.commit()
                    continue

                file_size_kb = file_path.stat().st_size / 1024
                is_large_file = file_size_kb > 100

                existing_art = self.artefacts.get(file_name)
                if existing_art:
                    art = self.artefacts.update(
                        title=file_name,
                        new_content=after_info["content"],
                        new_type=atype,
                        active=not is_large_file,
                        visibility=ArtefactVisibility.FULL if not is_large_file else ArtefactVisibility.TREE_UNLOCKABLE,
                        commit_message=f"Modified by tool '{tool_name}'"
                    )
                else:
                    art = self.artefacts.add(
                        title=file_name,
                        artefact_type=atype,
                        content=after_info["content"],
                        active=not is_large_file,
                        visibility=ArtefactVisibility.FULL if not is_large_file else ArtefactVisibility.TREE_UNLOCKABLE,
                        commit_message=f"Created by tool '{tool_name}'"
                    )
                self.commit()

                if self.active_branch_id:
                    ai_msg_local = self.get_message(self.active_branch_id)
                    if ai_msg_local:
                        tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                        if tag not in ai_msg_local.content:
                            ai_msg_local.content += f'\n\n{tag}\n'
                        self.commit()

    def wipe_all_memories(self) -> bool:
        """
        Permanently deletes all episodic and associative memories from the database.
        This includes working, deep, and archived memory tiers.
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            ASCIIColors.warning("[ChatMixin] No memory manager attached. Cannot wipe memories.")
            return False

        try:
            import sqlite3
            db_path = self.memory_manager.db_path.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = {row[0] for row in cursor.fetchall()}

            if "memories" in existing_tables:
                cursor.execute("DELETE FROM memories")
            if "memory_embeddings" in existing_tables:
                cursor.execute("DELETE FROM memory_embeddings")
            if "memory_decay_history" in existing_tables:
                cursor.execute("DELETE FROM memory_decay_history")

            conn.commit()
            conn.close()

            ASCIIColors.success("[ChatMixin] ✅ All memories wiped successfully.")
            return True
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"[ChatMixin] Failed to wipe memories: {e}")
            return False

    def _get_spinoff_agent_tools(self, current_prompt: str, images: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically registers specialized sub-agents as executable in-process tools.
        Enables the LLM to delegate heavy cognitive, formatting, or parsing tasks on-demand
        without breaking the main stream or bloating the primary conversation context.
        """
        spinoffs = {}

        # Spinoff 1: Surgical Artifact Specialist
        def tool_spinoff_code_specialist(task_instructions: str) -> dict:
            """
            Spawns a specialized Surgical Code Specialist in a focused, low-temperature sandbox.
            Ideal for generating complete Python scripts, performing exact aider patches, or refactoring logic.

            Args:
                task_instructions (str): The specific coding or refactoring instructions for the specialist.
            """
            custom_system = (
                "You are an expert Surgical Code Specialist.\n"
                "You operate in a hyper-focused sandbox isolated from the main conversation's noise.\n"
                "Your sole task is to implement the requested code modifications perfectly.\n\n"
                "STRICT RULES:\n"
                "1. Output ONLY a valid <artifact> block containing your code or SEARCH/REPLACE patch.\n"
                "2. Do NOT use markdown fences or write introductory/concluding prose outside the tags.\n"
                "3. Ensure character-for-character accuracy in aider SEARCH/REPLACE blocks."
            )
            # Fetch active artifacts context
            art_zone = self.artefacts.build_artefacts_context_zone()
            payload = f"=== CONTEXT ARTIFACTS ===\n{art_zone}\n\n=== SPECIALIST TASK ===\n{task_instructions}"
            try:
                res = self.lollmsClient.generate_text(
                    prompt=payload,
                    system_prompt=custom_system,
                    images=images,
                    temperature=0.1,  # Low temperature for deterministic precision
                    **{k: v for k, v in kwargs.items() if k not in ("temperature", "streaming_callback")}
                )
                return {"success": True, "output": res.strip()}
            except Exception as e:
                return {"success": False, "error": str(e)}

        spinoffs["tool_spinoff_code_specialist"] = {
            "name": "tool_spinoff_code_specialist",
            "description": "Spawns a specialized Surgical Code Specialist in a focused, low-temperature sandbox to write, patch, or refactor Python/code artifacts.",
            "parameters": [{"name": "task_instructions", "type": "str", "description": "Specific code or patch instructions."}],
            "callable": tool_spinoff_code_specialist
        }

        # Spinoff 2: HTML Slide Presentation Designer
        def tool_spinoff_presentation_designer(style: str, slide_count: int, structure_hints: str) -> dict:
            """
            Spawns a specialized HTML Slide Presentation Designer in a focused sandbox.
            Converts active artifacts into a styled, structured multi-slide HTML5 presentation deck.

            Args:
                style (str): The design theme (e.g. 'dark', 'light', 'creative').
                slide_count (int): Expected number of slides.
                structure_hints (str): Specific topics or structural outlines to focus on.
            """
            custom_system = (
                "You are an expert HTML Slide Presentation Designer.\n"
                "You design beautiful, modern 16:9 slideshows using semantic HTML5 and CSS.\n\n"
                "STRICT RULES:\n"
                "1. Output ONLY a single <artifact> tag containing your complete, valid HTML document.\n"
                "2. Do NOT write conversational prose or use markdown code blocks outside the tags."
            )
            art_zone = self.artefacts.build_artefacts_context_zone()
            payload = (
                f"=== CONTEXT ARTIFACTS ===\n{art_zone}\n\n"
                f"=== DESIGN REQUIREMENTS ===\n"
                f"• Style Theme: {style}\n"
                f"• Slides Count: {slide_count}\n"
                f"• Structure Outlines: {structure_hints}"
            )
            try:
                res = self.lollmsClient.generate_text(
                    prompt=payload,
                    system_prompt=custom_system,
                    temperature=0.3,
                    **{k: v for k, v in kwargs.items() if k not in ("temperature", "streaming_callback")}
                )
                return {"success": True, "output": res.strip()}
            except Exception as e:
                return {"success": False, "error": str(e)}

        spinoffs["tool_spinoff_presentation_designer"] = {
            "name": "tool_spinoff_presentation_designer",
            "description": "Spawns a specialized HTML Slide Presentation Designer in a focused sandbox to synthesize active datasets/artifacts into a highly styled multi-slide HTML5 presentation deck.",
            "parameters": [
                {"name": "style", "type": "str", "description": "Design theme (dark, light, creative, minimal)."},
                {"name": "slide_count", "type": "int", "description": "Expected number of slides."},
                {"name": "structure_hints", "type": "str", "description": "Outlines and structural hints."}
            ],
            "callable": tool_spinoff_presentation_designer
        }

        return spinoffs

    def chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        tools=None,
        add_user_message: bool = True,
        images=None,
        remove_thinking_blocks: bool = True,
        enable_image_generation: bool = True,
        enable_image_editing:    bool = True,
        auto_activate_artefacts: bool = True,
        enable_inline_widgets:        bool = False,
        enable_notes:                 bool = True,
        enable_skills:                bool = True,
        enable_forms:                 bool = True,
        enable_books:                 bool = False,
        enable_presentations:         bool = False,
        memory_manager=None,
        enable_artefacts:             bool = True,
        enable_memory:                bool = True,
        enable_episodic_memory:       bool = True,  # 🆕 NEW: Control episodic memory saving
        enable_auto_dream:            bool = True,
        enable_deep_memory_pulling:   bool = True,
        prehydrate_rag:               bool = True,
        max_nb_rounds:                Optional[int] = None,
        max_reasoning_steps:          Optional[int] = None,
        enable_in_message_status:     bool = False,
        enable_sub_agents:            bool = False,
        forward_artefact_chunks:      bool = False,
        fast_artefact_replicas:       Optional[List[str]] = None,
        tolerance_level:              Optional[str] = "strict",
        allow_dynamic_tools:          bool = False,
        enable_data_tools:            bool = True,
        enable_code_execution:        bool = False,
        suppress_images:              bool = False,
        debug_export:                 bool = False,
        debug:                        bool = False,
        enable_vlm_query:             bool = False,
        event_mode:                   EventMode = EventMode.PROCESSING_TAG_MODE,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs the conversational loop, resolving RAG, tiered memories, and tool calls.

        Args:
            user_message (str): The user's input message.
            personality: Optional personality object with system prompt and tools.
            branch_tip_id: Optional branch tip ID to continue from.
            tools: Optional dict of additional tools or list of tool names to enable.
            add_user_message (bool): If True, adds the user message to the discussion. Default True.
            images: Optional list of image paths/base64 for multimodal input.
            remove_thinking_blocks (bool): If True, strips <think>...</think> blocks from output. Default True.
            enable_image_generation (bool): Enable image generation capabilities. Default True.
            enable_image_editing (bool): Enable image editing capabilities. Default True.
            auto_activate_artefacts (bool): Automatically activate created artifacts. Default True.
            enable_inline_widgets (bool): Enable inline widget support. Default False.
            enable_notes (bool): Enable note-taking functionality. Default True.
            enable_skills (bool): Enable skill capture functionality. Default True.
            enable_forms (bool): Enable form generation functionality. Default True.
            enable_books (bool): Enable book/document generation. Default False.
            enable_presentations (bool): Enable presentation/slides generation. Default False.
            memory_manager: Optional memory manager instance for persistent memory.
            enable_artefacts (bool): Enable artifact creation and management. Default True.
            enable_memory (bool): Enable memory system (working/deep/archived). Default True.
            enable_episodic_memory (bool): Enable episodic memory saving (conversation history). Default True.
                Set to False to prevent automatic saving of conversation turns as episodic memories.
                Useful for privacy-sensitive applications or when you want manual control over memory persistence.
            enable_auto_dream (bool): Enable automatic memory dream/consolidation cycles. Default True.
            enable_deep_memory_pulling (bool): Enable automatic pulling of relevant deep memories. Default True.
            prehydrate_rag (bool): Pre-hydrate RAG context before generation. Default True.
            max_nb_rounds (Optional[int]): Maximum number of agentic reasoning rounds. Primary parameter. Defaults to 20 if None.
            max_reasoning_steps (Optional[int]): Deprecated. Backward-compatible alias for max_nb_rounds.
            enable_in_message_status (bool): Show in-message status updates. Default False.
            enable_sub_agents (bool): Enable spinoff sub-agent tools. Default False.
            forward_artefact_chunks (bool): Forward artifact chunks to callback. Default False.
            fast_artefact_replicas (Optional[List[str]]): Custom fast replica messages for artifacts.
            tolerance_level (Optional[str]): Tolerance level for data tools ('strict', 'lenient'). Default 'strict'.
            allow_dynamic_tools (bool): Allow dynamic tool registration from artifacts. Default False.
            enable_data_tools (bool): Enable data manipulation tools (SQL, pandas). Default True.
            enable_code_execution (bool): Enable arbitrary Python code execution tool. Default False.
            suppress_images (bool): Suppress image hydration in context. Default False.
            debug_export (bool): Enable debug export of context dumps. Default False.
            debug (bool): Enable debug mode with additional logging. Default False.
            enable_vlm_query (bool): Enable VLM query tool for vision fallback. Default False.
            event_mode (EventMode): Event reporting mode. Default PROCESSING_TAG_MODE.
            **kwargs: Additional generation parameters passed to the LLM binding.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - user_message: The user message object
                - ai_message: The AI response message object
                - sources: List of RAG sources used
                - artefacts: List of artifacts created/modified
                - memory_report: Report of memory operations
                - dream_report: Report of dream cycle (if enabled)
                - was_cancelled: Boolean indicating if generation was cancelled
        """
        resolved_max_rounds = max_nb_rounds if max_nb_rounds is not None else max_reasoning_steps
        if resolved_max_rounds is None:
            resolved_max_rounds = 20

        # Store tolerance level on active discussion for downstream execution tools (like execute_python_data_query)
        if not hasattr(self, "tolerance_level") or tolerance_level:
            object.__setattr__(self, "tolerance_level", tolerance_level or "strict")

        # 🛡️ SECURITY: Store the dynamic tool execution flag.
        # If False, the ArtefactManager will NOT register type="tool" artefacts as executable LCP tools.
        object.__setattr__(self, "allow_dynamic_tools", allow_dynamic_tools)

        # 🛡️ SECURITY: Store the arbitrary code execution flag.
        object.__setattr__(self, "enable_code_execution", enable_code_execution)

        # Initialize list to collect all created/modified artifacts during this turn safely
        object.__setattr__(self, "_affected_artefacts_this_turn", [])

        # 🛡️ CRITICAL FIX: Preserve pre-turn cancellation signal.
        # If cancel_generation() was called BEFORE chat(), we must observe it.
        # We capture the state, then reset the flag. The loop will check the captured state.
        _was_pre_cancelled = self.is_generation_cancelled()
        self.reset_cancel_state()
        if _was_pre_cancelled:
            object.__setattr__(self, '_cancel_flag', True)

        self.scratchpad = ""
        callback = kwargs.get("streaming_callback")
        temperature = kwargs.get("temperature")

        # ── 1. Safe SQLite Memory Ingestion (CONDITIONAL) ──
        # Memory Scoping: If personality has its own memory_manager (Independent Life), use it.
        # Otherwise, fallback to the Discussion's memory_manager (System-Managed Life).
        # CRITICAL: Only initialize memory manager if enable_memory is True
        _mm = None
        if enable_memory:
            if personality and hasattr(personality, "memory_manager") and personality.memory_manager:
                _mm = personality.memory_manager
            else:
                _mm = self._get_memory_manager(memory_manager)

        _counter = self.lollmsClient.count_tokens if self.lollmsClient else None

        # Only perform memory operations if memory is enabled AND manager exists
        if enable_memory and _mm:
            try:
                _mm.apply_decay()
            except Exception:
                pass

            if user_message and enable_deep_memory_pulling:
                try:
                    _mm.auto_pull_deep_memories(user_message)
                except Exception as ex:
                    trace_exception(ex)

            try:
                _mm.enforce_budget(token_counter=_counter)
            except Exception as ex:
                trace_exception(ex)

        # ── 2. Add or Retrieve User Message ──
        user_msg = None
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=user_message,
                images=images,
                **kwargs,
            )
        else:
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found in index.")
            user_msg = LollmsMessage(self, self._message_index[self.active_branch_id])
            images = user_msg.get_active_images()
            user_message = user_msg.content

        # ── 3. Build Dynamic System Prompt ──
        sys_prompt = (personality.system_prompt if personality else None) or self.system_prompt or ""

        # Inject Skills Context (Progressive Enhancement)
        if personality and hasattr(personality, "skills_manager") and personality.skills_manager:
            if not getattr(personality, "_skills_context_injected", False):
                skills_ctx = personality.skills_manager.build_context()
                if skills_ctx:
                    sys_prompt += "\n" + skills_ctx
                    object.__setattr__(personality, "_skills_context_injected", True)
            else:
                skills_ctx = personality.skills_manager.build_context()
                if skills_ctx and skills_ctx not in sys_prompt:
                    sys_prompt += "\n" + skills_ctx

        # ── 🧹 CORE RULES (ALWAYS ACTIVE) ──
        # These are fundamental behavioral rules that apply regardless of feature flags
        core_rules = (
            "\n=== VERACITY & ATTRIBUTION REQUIREMENTS ===\n"
            "Cite retrieved sources as [1],[2]... "
            "Never fabricate facts. Say 'I don't know' when uncertain.\n"
            "\n=== CODE & STRUCTURED FORMATTING RULES (MANDATORY) ===\n"
            "ALWAYS wrap any code, scripts, configurations, or structured formats "
            "(such as HTML, CSS, Python, SQL, XML, JSON, YAML, etc.) inside standard "
            "markdown code blocks specifying the correct language identifier, e.g.:\n"
            "```python\n"
            "# python code here\n"
            "```\n"
            "Never output raw code or markup directly in conversational text without these code blocks.\n"
            "\n=== UNICODE & CHARACTER HYGIENE ===\n"
            "1. **NO INVISIBLE CHARACTERS**: NEVER output zero-width spaces (U+200B), directional marks, or other invisible Unicode characters in your response.\n"
            "2. **CLEAN XML TAGS**: When emitting functional tags, use only standard ASCII characters. Do NOT use special Unicode characters as delimiters or separators.\n"
            "3. **STANDARD PIPE CHARACTER**: If you need to use a pipe character, use the standard ASCII pipe `|` (U+007C), not any Unicode variant.\n"
        )

        # ── 🎨 FEATURE-SPECIFIC INSTRUCTIONS ──
        extra_instructions = ""
        user_msg_lower = user_message.lower()

        # Artifact Instructions (only if artifacts are enabled)
        if enable_artefacts:
            extra_instructions += self._build_artefact_instructions()

            # Sub-feature instructions (only if their parent feature is enabled)
            if enable_inline_widgets:
                extra_instructions += self._build_inline_widget_instructions()
            if enable_notes:
                extra_instructions += self._build_note_instructions()
            if enable_skills:
                extra_instructions += self._build_skill_instructions()
            if enable_forms and any(kw in user_msg_lower for kw in ("form", "formulaire", "survey", "questionnaire")):
                extra_instructions += self._build_form_instructions()
            if enable_books and any(kw in user_msg_lower for kw in ("book", "tome", "novel", "chapitre")):
                extra_instructions += self._build_book_instructions()
            if enable_presentations and any(kw in user_msg_lower for kw in ("presentation", "slide", "slideshow", "deck", "diaporama")):
                extra_instructions += self._build_presentation_instructions()

            # Handle instructions (only if artifacts are enabled)
            branch_msgs_now = self.get_branch(user_msg.id)
            handle_instructions = _build_handle_instructions(branch_msgs_now)
            if handle_instructions:
                extra_instructions += handle_instructions

        # Memory Instructions (only if memory is enabled AND memory manager exists)
        if enable_memory and _mm:
            extra_instructions += _mm.build_system_instructions()

        # Image Generation Instructions (only if image generation/editing is enabled AND TTI binding exists)
        if (enable_image_generation or enable_image_editing) and getattr(self.lollmsClient, 'tti', None) is not None:
            extra_instructions += self._build_image_generation_instructions()

        # Combine core sections (feature rules will be added later after active_tools is built)
        full_system_prompt = sys_prompt + "\n" + core_rules + "\n" + extra_instructions

        # ── 4. RAG Ingestion & Pre-Hydration ──
        rag_context = ""
        if prehydrate_rag and personality and hasattr(personality, "has_data") and personality.has_data:
            try:
                rag_res = personality.query_data(user_message)
                if rag_res and rag_res.get("success") and rag_res.get("sources"):
                    sources_text = []
                    _MAX_RAG_CHARS = 50000
                    current_rag_chars = 0
                    for src in rag_res.get("sources", []):
                        title = src.get("title") or src.get("source") or "Document"
                        ds_label = f" [{src.get('datasource_name')}]" if src.get('datasource_name') else ""
                        score_val = src.get("score")
                        score_str = f" (Score: {score_val:.2f})" if isinstance(score_val, (int, float)) and score_val <= 1.0 else (f" (Score: {score_val})" if score_val is not None else "")
                        chunk_text = f"--- Source [{title}]{ds_label}{score_str} ---\n{src.get('content')}"
                        if current_rag_chars + len(chunk_text) > _MAX_RAG_CHARS:
                            sources_text.append(f"... [Remaining RAG context truncated at {_MAX_RAG_CHARS} chars to prevent context bloat]")
                            break
                        sources_text.append(chunk_text)
                        current_rag_chars += len(chunk_text)
                    if sources_text:
                        rag_context = "\n=== RETRIEVED RAG CONTEXT ===\n" + "\n\n".join(sources_text) + "\n=== END RAG CONTEXT ===\n"
            except Exception as e:
                trace_exception(e)

        if rag_context:
            full_system_prompt += "\n" + rag_context

        if personality and hasattr(personality, "build_rag_system_block"):
            rag_sys_block = personality.build_rag_system_block()
            if rag_sys_block:
                full_system_prompt += "\n" + rag_sys_block

        # ── 5. Active Artifacts & Memories Injection (CONDITIONAL) ──
        # Only inject artifact context if artifacts are enabled
        if enable_artefacts:
            artefacts_zone = self.artefacts.build_artefacts_context_zone()
            if artefacts_zone:
                full_system_prompt += "\n=== ACTIVE ARTIFACTS ===\n" + artefacts_zone + "\n"

        # Only inject memory context if memory is enabled AND memory manager exists
        if enable_memory and _mm:
            mem_block = self._build_memory_context_block(_mm, token_counter=_counter)
            if mem_block:
                full_system_prompt += "\n=== ACTIVE MEMORIES ===\n" + mem_block + "\n"

        # ── 6. Data Zones Ingestion ──
        data_zones = []
        udz = (self.user_data_zone or "").strip()
        if udz:
            data_zones.append(f"=== USER DATA ===\n{udz}\n=== END USER DATA ===")
        ddz = (self.discussion_data_zone or "").strip()
        if ddz:
            data_zones.append(f"=== DISCUSSION DATA ===\n{ddz}\n=== END DISCUSSION DATA ===")
        pdz = (self.personality_data_zone or "").strip()
        if pdz:
            data_zones.append(f"=== PERSONALITY DATA ===\n{pdz}\n=== END PERSONALITY DATA ===")

        if data_zones:
            full_system_prompt += "\n" + "\n\n".join(data_zones)

        # ── 7. Tool calling registry & Dynamic Library Mounting ──
        # ── SOVEREIGN OPT-IN DOCTRINE ──
        active_tools = {}

        # 1. Personality Handbag Tools, RAG Tools & Skill Tools
        if personality and hasattr(personality, "build_rag_tools"):
            active_tools.update(personality.build_rag_tools())

        if personality and hasattr(personality, "skills_manager") and personality.skills_manager:
            active_tools.update(personality.skills_manager.build_skill_tools())

        if personality and hasattr(personality, "capabilities") and personality.capabilities and personality.capabilities.enable_skill_creation:
            if personality.skills_manager:
                skill_tools = personality.skills_manager.build_skill_tools()
                for t_name, t_spec in skill_tools.items():
                    if t_name in ("tool_create_skill", "tool_update_skill", "tool_append_to_skill", "tool_remove_skill"):
                        active_tools[t_name] = t_spec

        if personality and hasattr(personality, "tools") and _is_tool_binding(personality.tools):
            try:
                pers_tools = personality.tools.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                active_tools.update(pers_tools)
            except Exception as ex:
                trace_exception(ex)
        elif personality and hasattr(personality, "tools") and isinstance(personality.tools, dict):
            active_tools.update(personality.tools)

        # 2. Explicit User-Supplied Tools (Callables or Default Tool Names)
        if isinstance(tools, dict):
            active_tools.update(tools)
        elif isinstance(tools, list):
            lcp_binding = getattr(self.lollmsClient, "tools", None)
            if lcp_binding and hasattr(lcp_binding, "to_chat_tool_specs"):
                try:
                    lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                    for tool_name in tools:
                        if tool_name in lcp_tools:
                            active_tools[tool_name] = lcp_tools[tool_name]
                        else:
                            ASCIIColors.warning(f"[ChatMixin] Requested default tool '{tool_name}' not found in LCP registry.")
                except Exception as ex:
                    trace_exception(ex)

        # 3. Auto-Mount Data Tools (ONLY if data files exist AND enable_data_tools is True)
        lcp_binding = getattr(self.lollmsClient, "tools", None)
        enable_data_tools_flag = enable_data_tools

        from pathlib import Path
        workspace_dir = Path(self.workspace_data_path) if getattr(self, "workspace_data_path", None) else Path("./data_workspace")

        data_extensions = {".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"}
        has_data_files = any(f.suffix.lower() in data_extensions for f in workspace_dir.rglob("*") if f.is_file())

        if has_data_files and (lcp_binding is None):
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                lcp_binding = LCPBinding(
                    tools_folders=[Path(__file__).parent.parent / "tools_bindings" / "lcp" / "default_tools"]
                )
                self.lollmsClient.tools = lcp_binding
            except Exception as ex:
                trace_exception(ex)
                lcp_binding = None

        if enable_data_tools_flag and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            lcp_binding.mount_tool_library("as_is_document_tools")
            try:
                lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                for t_name, t_spec in lcp_tools.items():
                    if t_name.startswith("tool_inspect_document") or \
                       t_name.startswith("tool_read_document_content") or \
                       t_name.startswith("tool_grep_document") or \
                       t_name.startswith("tool_modify_docx") or \
                       t_name.startswith("tool_modify_excel") or \
                       t_name.startswith("tool_modify_pdf_annotation") or \
                       t_name.startswith("tool_modify_pptx_slide"):
                        active_tools[t_name] = t_spec
            except Exception as ex:
                trace_exception(ex)

        if enable_data_tools_flag and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            if has_data_files:
                lcp_binding.mount_tool_library("semantic_data_engineer")
                try:
                    lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                    for t_name, t_spec in lcp_tools.items():
                        if t_name == "tool_execute_python_data_query":
                            active_tools[t_name] = t_spec
                except Exception as ex:
                    trace_exception(ex)

        # 4. Mount Arbitrary Code Execution Tool if enabled
        if enable_code_execution and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            lcp_binding.mount_tool_library("execute_python_code")
            try:
                lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                for t_name, t_spec in lcp_tools.items():
                    if t_name == "tool_execute_python_code":
                        active_tools[t_name] = t_spec
            except Exception as ex:
                trace_exception(ex)

        if debug and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            lcp_binding.mount_tool_library("debug_toolset")
            try:
                lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                for t_name, t_spec in lcp_tools.items():
                    if t_name == "tool_dump_context":
                        active_tools[t_name] = t_spec
            except Exception as ex:
                trace_exception(ex)

        # 5. Mount VLM Query Tool (Conditional Fallback)
        if enable_vlm_query and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            def _active_llm_has_vision() -> bool:
                active_llm = getattr(self.lollmsClient, "llm", None)
                if not active_llm:
                    return False
                if getattr(active_llm, "vision_enabled", False):
                    return True
                if hasattr(active_llm, "child_bindings"):
                    for child in active_llm.child_bindings.values():
                        if getattr(child, "vision_enabled", False):
                            return True
                return False

            if not _active_llm_has_vision():
                has_vlm_fallback = any(
                    getattr(b, "vision_enabled", False) 
                    for b in getattr(self.lollmsClient, "llms", {}).values()
                )
                if has_vlm_fallback:
                    lcp_binding.mount_tool_library("vlm_query")
                    try:
                        lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                        for t_name, t_spec in lcp_tools.items():
                            if t_name == "tool_vlm_query":
                                active_tools[t_name] = t_spec
                    except Exception as ex:
                        trace_exception(ex)

        # Optionally merge spinoff agents as dynamic local tools
        if enable_sub_agents:
            spinoff_tools = self._get_spinoff_agent_tools(full_system_prompt, images or [], **kwargs)
            active_tools.update(spinoff_tools)

        # ── 🎯 CONDITIONAL FEATURE RULES (INJECTED AFTER active_tools IS BUILT) ──
        # Now that we know which features are actually enabled, inject the appropriate rules
        feature_rules = ""

        # Action Execution & Termination Protocol (only if agentic features are enabled)
        if enable_artefacts or active_tools or enable_memory:
            feature_rules += (
                "\n=== ACTION EXECUTION & TERMINATION PROTOCOL (CRITICAL) ===\n"
                "1. **INTENT ≠ EXECUTION**: Stating 'I will search...', 'Let me analyze...', or 'I will create...' in conversational text DOES NOT execute the action. "
                "Conversational declarations are completely inert. You have NO ability to perform actions unless you emit the exact functional XML tags.\n"
                "2. **MANDATORY TAG EMISSION**: To execute an action, you MUST output the corresponding functional tag (`<tool>`, `<artifact>`, `<note>`, etc.) immediately. "
                "Do not promise an action in one turn and expect the system to execute it. If you need another round to perform work, you MUST emit the tag that triggers that work.\n"
                "3. **EXPLICIT TERMINATION**: You are in control of the agentic loop. When you have finished your task and provided your final conversational answer to the user, "
                "you MUST end your generation with a termination tag on a new line. If you stop generating without emitting a termination tag, the system will assume you have more work to do and will "
                "force you to continue. If you have no further actions to take, simply write your final response and append a termination tag at the end.\n"
                "   **SUPPORTED TERMINATION TAGS** (use any of these):\n"
                "   - `<done/>` (preferred)\n"
                "   - `<end/>`\n"
                "   - `</end>`\n"
                "   **CRITICAL**: The termination tag must be on its own line, with nothing else on that line.\n"
                "   **EXAMPLE**:\n"
                "   ```\n"
                "   Here is my final answer to your question.\n"
                "   \n"
                "   <done/>\n"
                "   ```\n"
                "4. **SAME-SESSION CONTINUATION (MULTI-TURN CHAINS)**: When you are executing a sequence of actions across multiple turns (e.g., testing tools one by one), "
                "you MUST emit the next action's tag in your IMMEDIATE NEXT response. Do NOT wait for the user to prompt you again. The system preserves your exact execution path, "
                "so you have full visibility of the previous tool results. If you state 'Now testing tool_X...', the VERY NEXT token you generate MUST be `<tool>{\"name\": \"tool_X\"...}`.\n"
                "5. **ROUND 1 SHORT-CIRCUIT**: If the user's request is purely conversational and requires NO tools or artifacts, simply respond conversationally. The system will terminate after the first round. "
                "Do NOT emit a termination tag if you are not in an agentic loop.\n"
            )

        # System Notification Handling (only if context unlocking is possible)
        if enable_artefacts:
            feature_rules += (
                "\n=== SYSTEM NOTIFICATION HANDLING ===\n"
                "1. **RECOGNIZE SYSTEM NOTIFICATIONS**: Messages wrapped in `[SYSTEM NOTIFICATION - NOT A USER MESSAGE]...[END SYSTEM NOTIFICATION]` are infrastructure events, not user input.\n"
                "2. **DO NOT RESPOND TO NOTIFICATIONS**: If you receive a system notification, simply acknowledge it internally and continue with the pending task. Do NOT treat it as a user question.\n"
                "3. **CONTENT AVAILABILITY**: When files are unlocked, check the `[CONTENT AVAILABILITY]` section to see which files actually have readable content vs. which are empty.\n"
                "4. **EMPTY FILES**: If a file is listed as empty, do NOT pretend to have read content from it. Acknowledge that it's empty and move on.\n"
            )

        # Tool Calling Discipline (only if tools are available)
        if active_tools:
            feature_rules += (
                "\n=== TOOL CALLING DISCIPLINE (CRITICAL) ===\n"
                "1. **Tool Results ≠ Tool Calls**: When a tool returns JSON output (e.g., {\"success\": true, \"output\": ...}), "
                "this is a **RESULT**, NOT a new tool call. Do **NOT** re-execute or re-emit the same tool call.\n"
                "2. **One Call Per Task**: Once a tool executes successfully, the data is retrieved. Your job is to **ANALYZE** and **ANSWER**, not to call the tool again.\n"
                "3. **Loop Prevention**: Repeating a successful tool call with identical parameters is a **CRITICAL ERROR**. "
                "The system will block duplicate calls. If you see a tool result, move on to the next step.\n"
                "4. **File Outputs**: When a tool successfully returns a file (image, plot, screenshot, PDF, audio, etc.), "
                "the file is ALREADY saved to the workspace by the tool. Do NOT call the same tool "
                "again with the same parameters to regenerate it. Instead, reference the produced "
                "file URL in your final answer (e.g. <img src=\"/api/workspace_files/filename.png\" /> "
                "for images) and STOP generating.\n"
            )

        # Thinking & Reasoning Constraint (only if agentic features are enabled)
        if enable_artefacts or active_tools or enable_memory:
            feature_rules += (
                "\n=== THINKING & REASONING CONSTRAINT ===\n"
                "If you decide to output a thought process enclosed in  tags, "
                "you MUST output all functional XML tags (such as <artifact>, <tool>, or <mem_new>) "
                "on a NEW LINE strictly AFTER the closing  warn_tag tag. "
                "NEVER place functional tags inside the  warn_tag reasoning block.\n"
            )

        # Anti-Mimicry Protocol (only if agentic features are enabled)
        if enable_artefacts or active_tools:
            feature_rules += (
                "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
                "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating text patterns like `[🔒SYSTEM_ARTIFACT_ANCHOR:...`, `[SYSTEM:`, or `[content stripped...`. These are **INFRASTRUCTURE-ONLY** markers used in history to save space. If you output them, NO ACTION will occur.\n"
                "2. **USE REAL TAGS**: To create artifacts, you MUST use the actual `<artifact name=\"...\">` XML tags. To call tools, use `<tool>`. Do NOT mimic the placeholder markers from past messages.\n"
                "3. **TAG ISOLATION**: Functional tags (`<artifact>`, `<tool>`, `<tool_result>`) MUST NEVER appear inside  warn_tag blocks. They must ONLY appear in the final response body AFTER the closing  warn_tag tag.\n"
            )

        # Inject feature rules into the full system prompt
        if feature_rules:
            full_system_prompt += feature_rules

        tools_prompt = ""
        if active_tools:
            tools_prompt = "\n=== TOOLS AVAILABLE ===\n"
            tools_prompt += "To use a tool, you MUST emit a single <tool> tag on a new line with the tool parameters as a JSON object, and then stop generating. Do NOT write prose before or after the tag.\n"
            tools_prompt += (
                "\n=== TOOL CALLING DISCIPLINE (CRITICAL — READ BEFORE CALLING TOOLS) ===\n"
                "1. **EXACT CLOSING TAG**: The closing tag is  `</tool>` . You MUST NOT write  `` `` ``  or any other variation.\n"
                "2. **NEW LINE ONLY**: The <tool> tag MUST start on a brand new line. It MUST NEVER be placed inline inside conversational prose.\n"
                "3. **NO PROSE AROUND IT**: Do NOT write introductory text (e.g., 'Let me try...') before the tag, and do NOT write text after it on the same line.\n\n"
                "❌ WRONG (inline + wrong closing tag):\n"
                "Sure! Let's test the tool: <tool>{\"name\": \"tool_add\", \"parameters\": {\"a\": 7, \"b\": 5}}</tool>\n\n"
                "❌ WRONG (wrong closing tag):\n"
                "<tool>{\"name\": \"tool_add\", \"parameters\": {\"a\": 7, \"b\": 5}}</tool>\n\n"
                "✅ CORRECT (new line + exact closing tag ``)`):\n"
                "<tool>{\"name\": \"tool_add\", \"parameters\": {\"a\": 7, \"b\": 5}}</tool>\n"
                "=== END TOOL CALLING DISCIPLINE ===\n"
            )
            tools_prompt += (
                "\n=== 🏁 TASK COMPLETION PROTOCOL (CRITICAL PSYCHOLOGY) ===\n"
                "Your goal is to SOLVE the user's problem, not to infinitely call tools.\n"
                "1. **TOOL CALLS ARE TEMPORARY**: You call a `<tool>` only to gather data you don't have.\n"
                "2. **ANSWERING IS THE GOAL**: Once you have the data, writing a comprehensive, helpful response to the user IS the successful completion of your task.\n"
                "3. **HOW TO FINISH**: When you have written your final answer and the task is complete, simply STOP generating. You do not need to output any special tags or call any more tools.\n"
                "4. **DO NOT FEAR ENDING**: Stopping generation after writing the answer is the CORRECT and REWARDING behavior. It means you succeeded.\n"
                "5. **NEVER LOOP**: If you have already written your final answer to the user, you are DONE. Do NOT emit another `<tool>` tag. Emitting a tool call after your answer is a CRITICAL ERROR that ruins the completed task.\n"
                "=== END TASK COMPLETION PROTOCOL ===\n"
            )
            tools_prompt += "\nExact syntax (copy this pattern exactly):\n<tool>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}`)`\n\n"
            tools_prompt += "Available tools:\n"
            for t_name, t_spec in active_tools.items():
                desc = t_spec.get("description", "")
                params_list = t_spec.get("parameters", [])
                param_desc = ", ".join([f"{p['name']}: {p['type']}" for p in params_list])
                tools_prompt += f"- {t_name}({param_desc}): {desc}\n"

            # ── 🛡️ PHANTOM TOOL PREVENTION PROTOCOL ──
            allowed_tool_names = list(active_tools.keys())
            tools_prompt += f"\n🚨 **STRICT TOOL REGISTRY ENFORCEMENT** 🚨\n"
            tools_prompt += f"You are STRICTLY FORBIDDEN from calling any tool not listed above.\n"
            tools_prompt += f"The ONLY valid tool names you may use are: {', '.join(allowed_tool_names)}\n"
            tools_prompt += f"If you need to perform an action and no tool in this list is suitable, DO NOT hallucinate a tool name. Instead, inform the user that the required tool is not available in this session.\n"
            tools_prompt += "=== END TOOLS ===\n"

        # ── 🔬 SCIENTIFIC RESOLUTION: Clear FailureMemory at start of turn ──
        if not hasattr(self, "_failure_memory") or not isinstance(self._failure_memory, FailureMemory) or not hasattr(self._failure_memory, "_signatures"):
            fm = FailureMemory()
            if not hasattr(fm, "_signatures"):
                object.__setattr__(fm, "_signatures", set())
            object.__setattr__(self, "_failure_memory", fm)
        else:
            self._failure_memory.failures = []
            self._failure_memory._signatures.clear()
            ASCIIColors.info("[ChatMixin] FailureMemory cleared for new turn.")

        # ── 8. Active Deliberation Loop ──
        import time as _chat_time
        _t_branch_start = _chat_time.perf_counter()
        ASCIIColors.info("[Trace] Retrieving conversation branch...")
        current_branch_tip = branch_tip_id or self.active_branch_id
        branch = self.get_branch(current_branch_tip)
        _t_branch_end = _chat_time.perf_counter()
        ASCIIColors.info(f"[Trace] Branch retrieved in {(_t_branch_end - _t_branch_start)*1000:.2f} ms ({len(branch)} messages).")

        # ── 🧠 VIRTUAL HISTORY & KV-CACHE PROTOCOL ──
        # 1. `virtual_history` is managed by `export()` in `UtilsMixin`.
        # 2. During agentic rounds, we append RAW assistant text (including <tool> tags) and
        #    structured tool results to this list. This preserves the LLM's KV-cache.
        # 3. `ai_msg.content` is the UI/DB buffer. It only receives conversational text and
        #    <processing> blocks. We track `conversational_gist` separately to avoid polluting
        #    the final message with raw XML or execution logs.
        # 4. 🛑 CRITICAL: virtual_history MUST start empty. The user's prompt is already
        #    part of the real historical branch (added via add_message). If we append it
        #    here, export() produces two consecutive user messages, which breaks strict
        #    alternation rules (e.g., llama.cpp Jinja templates) and causes KV-cache
        #    poisoning. virtual_history strictly tracks the NEW assistant answers and
        #    tool results generated during the agentic loop.
        #
        # 5. 🔄 ROLLING WINDOW PROTOCOL (NEW):
        #    - The LLM must ALWAYS see its own actions and responses from recent rounds.
        #    - We maintain a rolling window of the last N rounds (default: 4) in full detail.
        #    - Older rounds are compressed into summaries to prevent context bloat.
        #    - This ensures the LLM has full situational awareness without overwhelming the context.

        virtual_history = []

        # ── 🔄 ROLLING WINDOW CONFIGURATION ──
        # Maximum number of recent rounds to keep in full detail
        ROLLING_WINDOW_SIZE = 4
        # Maximum total tokens for virtual history before compression kicks in
        VIRTUAL_HISTORY_TOKEN_BUDGET = 8000

        tool_calls_this_turn = []
        round_count = 0
        conversational_gist = ""  # Accumulates only the conversational text for the final DB message

        # Track the count of exact tool call signatures to prevent infinite loops (Success Loops)
        # We allow up to 2 identical calls per turn to permit legitimate retries after null/empty output.
        tool_signature_counts = {}

        successful_tool_signatures = set()

        # ── 📊 TURN PROGRESS TRACKER ──
        # Tracks all actions taken during this turn so the LLM can see what it has accomplished.
        # This prevents infinite loops where the LLM forgets it already performed an action.
        turn_actions_log = []

        # Make it accessible to _StreamState via the discussion object
        object.__setattr__(self, '_turn_actions_log', turn_actions_log)

        # ── 🔍 MEMORY SEARCH DEDUPLICATION TRACKER ──
        # Tracks executed memory searches to prevent duplicate searches with the same query
        executed_memory_searches = set()
        object.__setattr__(self, '_executed_memory_searches', executed_memory_searches)

        # ── 🔄 ROLLING WINDOW HELPER FUNCTION ──
        def _compress_virtual_history_if_needed():
            """
            Compresses older rounds in virtual_history if it exceeds the token budget
            or if we have more than ROLLING_WINDOW_SIZE rounds.

            Strategy:
            1. Keep the last ROLLING_WINDOW_SIZE rounds in full detail
            2. Compress older rounds into summaries
            3. Always preserve the first round (initial user context)
            """
            nonlocal virtual_history

            if not virtual_history:
                return

            # Estimate token count
            total_chars = sum(len(vh.content) for vh in virtual_history)
            estimated_tokens = total_chars // 4

            # Check if compression is needed
            needs_compression = estimated_tokens > VIRTUAL_HISTORY_TOKEN_BUDGET or len(virtual_history) > (ROLLING_WINDOW_SIZE * 2)

            if not needs_compression:
                return

            ASCIIColors.info(f"[ChatMixin] Virtual history compression triggered: {len(virtual_history)} messages, ~{estimated_tokens} tokens")

            # Separate into rounds (each round = assistant message + user response)
            rounds = []
            current_round = []
            for vh in virtual_history:
                current_round.append(vh)
                if vh.sender_type == "user":
                    rounds.append(current_round)
                    current_round = []

            # Add any remaining messages (incomplete round)
            if current_round:
                rounds.append(current_round)

            # If we have more rounds than the window size, compress older ones
            if len(rounds) > ROLLING_WINDOW_SIZE:
                # Keep the first round (initial context) and the last ROLLING_WINDOW_SIZE rounds
                rounds_to_compress = rounds[1:-ROLLING_WINDOW_SIZE]  # Skip first, compress middle
                rounds_to_keep = [rounds[0]] + rounds[-ROLLING_WINDOW_SIZE:]  # Keep first + last N

                # Compress the middle rounds into a summary
                compressed_summary = "[COMPRESSED EARLIER ROUNDS]\n"
                compressed_summary += f"The following {len(rounds_to_compress)} rounds were compressed to save context:\n\n"

                for idx, round_msgs in enumerate(rounds_to_compress, 1):
                    assistant_msg = next((m for m in round_msgs if m.sender_type == "assistant"), None)
                    user_msg = next((m for m in round_msgs if m.sender_type == "user"), None)

                    if assistant_msg:
                        # Extract key actions from the assistant message
                        actions = []
                        if "<tool>" in assistant_msg.content:
                            tool_match = re.search(r'<tool>\s*{"name":\s*"([^"]+)"', assistant_msg.content)
                            if tool_match:
                                actions.append(f"Called tool: {tool_match.group(1)}")
                        if "<artifact" in assistant_msg.content or "<artefact" in assistant_msg.content:
                            art_match = re.search(r'<(?:artifact|artefact)\s+name=["\']([^"\']+)["\']', assistant_msg.content)
                            if art_match:
                                actions.append(f"Created artifact: {art_match.group(1)}")
                        if "<mem_search" in assistant_msg.content:
                            search_match = re.search(r'<mem_search\s+query=["\']([^"\']+)["\']', assistant_msg.content)
                            if search_match:
                                actions.append(f"Searched memory: {search_match.group(1)}")

                        if actions:
                            compressed_summary += f"Round {idx}: {'; '.join(actions)}\n"
                        else:
                            # Fallback: include first 100 chars of response
                            compressed_summary += f"Round {idx}: {assistant_msg.content[:100]}...\n"

                    if user_msg and "<tool_result" in user_msg.content:
                        compressed_summary += f"  → Tool executed successfully\n"
                    elif user_msg and "[MEMORY SEARCH RESULTS" in user_msg.content:
                        compressed_summary += f"  → Memory search completed\n"

                compressed_summary += "[END COMPRESSED ROUNDS]\n"

                # Rebuild virtual_history with compression
                new_virtual_history = []

                # Add first round (initial context)
                new_virtual_history.extend(rounds[0])

                # Add compressed summary as a user message
                new_virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=compressed_summary
                ))

                # Add recent rounds in full detail
                for round_msgs in rounds_to_keep[1:]:  # Skip first (already added)
                    new_virtual_history.extend(round_msgs)

                virtual_history = new_virtual_history
                ASCIIColors.success(f"[ChatMixin] Virtual history compressed: {len(virtual_history)} messages (was {len(rounds)} rounds)")

        # Initialize the single, clean database assistant message ONCE before entering the loop
        ai_msg = self.add_message(
            sender=personality.name if personality else self.lollmsClient.ai_name,
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=getattr(self.lollmsClient.llm, "model_name", "unknown") if self.lollmsClient else "unknown",
            binding_name=getattr(self.lollmsClient.llm, "binding_name", "unknown") if self.lollmsClient else "unknown"
        )

        # CRITICAL: Expose the active personality to _StreamState so it can access the SkillsManager
        # for Handbag skill routing (modifiable/read-only enforcement) during <skill> tag dispatch.
        object.__setattr__(self, '_active_personality', personality)

        object.__setattr__(self, "_consecutive_text_only_stalls", 0)
        if callback:
            callback(ai_msg.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, {"message_id": ai_msg.id})

        # Track if we exited due to cancellation
        was_cancelled = False

        # CRITICAL FIX: Initialize ss to None to prevent UnboundLocalError
        # if the loop breaks before _StreamState is instantiated (e.g., pre-turn cancellation).
        ss = None

        # Initialize mimicry attempt counter exactly once at the start of the turn
        # CRITICAL FIX: Use a list to ensure safe mutation across reasoning rounds.
        object.__setattr__(self, "_mimicry_attempt_counts", [0])

        # CRITICAL FIX: Persistent set to track dispatched tags across all reasoning rounds.
        # This prevents the LLM from re-dispatching the same artifact in a subsequent round,
        # which causes infinite loops and unwanted version bumps.
        persistent_processed_tags = set()

        # Initialize pending memory searches list for this turn
        object.__setattr__(self, '_pending_memory_searches', [])

        while round_count < max_reasoning_steps:
            # Check cancellation at the start of each reasoning round
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            round_count += 1

            # Make round count accessible to _StreamState for logging
            object.__setattr__(self, '_current_round', round_count)

            # Guarantee a clean, un-canceled state before launching each independent generation round
            if self.lollmsClient and getattr(self.lollmsClient, "llm", None):
                try:
                    self.lollmsClient.llm.reset_cancel()
                except Exception:
                    pass

            current_system_prompt = full_system_prompt
            if tools_prompt:
                current_system_prompt += "\n" + tools_prompt
            else:
                # 🛑 CRITICAL FIX: If no tools are active, ensure tools_prompt is empty 
                # so it doesn't append an empty string with a newline.
                pass

            # ── 📊 INJECT TURN PROGRESS TRACKER ──
            # If any actions have been taken in this turn, inject a progress summary
            # so the LLM can see what it has already accomplished and avoid repeating actions.
            if turn_actions_log:
                progress_summary = "\n[TURN PROGRESS TRACKER]\n"
                progress_summary += f"You have completed {len(turn_actions_log)} action(s) in this turn:\n\n"

                for idx, action in enumerate(turn_actions_log, 1):
                    action_type = action.get("action", "unknown")
                    action_round = action.get("round", "?")

                    if action_type == "memory_search":
                        query = action.get("query", "")
                        results_count = action.get("results_count", 0)
                        progress_summary += f"{idx}. Memory Search (Round {action_round}): Searched for '{query}' → Found {results_count} result(s)\n"
                    elif action_type == "tool_call":
                        tool_name = action.get("tool_name", "unknown")
                        success = action.get("success", False)
                        status = "✅ Success" if success else "❌ Failed"
                        progress_summary += f"{idx}. Tool Call (Round {action_round}): {tool_name} → {status}\n"
                    elif action_type == "artifact_created":
                        title = action.get("title", "unknown")
                        progress_summary += f"{idx}. Artifact Created (Round {action_round}): {title}\n"
                    else:
                        progress_summary += f"{idx}. {action_type} (Round {action_round})\n"

                progress_summary += "\n💡 **IMPORTANT**: You have already performed the actions listed above. Do NOT repeat them.\n"
                progress_summary += "If you have gathered enough information to answer the user's question, provide your final answer and emit `<done/>`.\n"
                progress_summary += "[END TURN PROGRESS TRACKER]\n"

                current_system_prompt += "\n" + progress_summary

            messages_list = self.export(
                format_type="openai_chat",
                branch_tip_id=current_branch_tip,
                suppress_system_prompt=False,
                suppress_images=suppress_images,
                virtual_history=virtual_history,
                debug=debug_export,
                system_prompt_override=current_system_prompt
            )

            # ── 🎨 DYNAMIC VISION HYDRATION ──
            # Retrieve all images generated or modified during previous rounds of this turn
            # and append their base64 pixels to the active vision context so the LLM can "see" them!
            # CRITICAL FIX: Only hydrate images that are explicitly in FULL visibility context.
            # Injecting pixels for [U] (TREE_UNLOCKABLE) images crashes non-vision LLMs.
            if suppress_images:
                round_images = None
            else:
                round_images = list(images) if images else []
                affected_arts = getattr(self, "_affected_artefacts_this_turn", [])
                for art in affected_arts:
                    if art.get("type") == "image" and art.get("images") and art.get("visibility") == ArtefactVisibility.FULL:
                        for img_b64 in art["images"]:
                            if img_b64 not in round_images:
                                round_images.append(img_b64)
                                ASCIIColors.success(f"[Vision Sync] Hydrated LLM context with generated plot: '{art['title']}'")

            # ── 🔬 SCIENTIFIC DEBUG: EXPORTED PROMPT TRACE ──
            # (Logging removed per user request)

            # CRITICAL FIX: Track content offset to prevent re-parsing old tool calls
            current_content_length = len(ai_msg.content)

            ss = _StreamState(
                discussion=self,
                callback=callback,
                forward_artefact_chunks=forward_artefact_chunks,
                ai_message=ai_msg,
                enable_notes=enable_notes,
                enable_skills=enable_skills,
                enable_inline_widgets=enable_inline_widgets,
                enable_forms=enable_forms,
                auto_activate_artefacts=auto_activate_artefacts,
                enable_artefacts=enable_artefacts,
                enable_in_message_status=enable_in_message_status,
                fast_artefact_replicas=fast_artefact_replicas,
                content_offset=current_content_length,
                processed_tags=persistent_processed_tags,
                event_mode=event_mode
            )

            def _inline_relay(chunk, msg_type=None, meta=None):
                # Check cancellation on EVERY token chunk
                if self.is_generation_cancelled():
                    return False  # Signal to stop generation

                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    # ── ⏱️ TIME TO FIRST TOKEN (TTFT) ──
                    if not getattr(self, "_ttft_logged", True) and chunk:
                        ttft = _time.perf_counter() - _t_gen_start
                        ASCIIColors.info(f"[TTFT] First token received in {ttft:.3f} s.")
                        object.__setattr__(self, "_ttft_logged", True)

                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            # Sanitize kwargs to prevent duplicate argument passing
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "stream")}

            # ── 📊 CONTEXT FILL TELEMETRY ──
            try:
                total_tokens = 0
                if self.lollmsClient and hasattr(self.lollmsClient, "count_tokens"):
                    for msg in messages_list:
                        content = msg.get("content", "") if isinstance(msg, dict) else ""
                        if isinstance(content, str):
                            total_tokens += self.lollmsClient.count_tokens(content)
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    total_tokens += self.lollmsClient.count_tokens(part.get("text", ""))

                max_ctx = 4096
                if self.lollmsClient and hasattr(self.lollmsClient, "get_ctx_size"):
                    max_ctx = self.lollmsClient.get_ctx_size() or max_ctx

                if max_ctx > 1:
                    fill_pct = (total_tokens / max_ctx) * 100.0
                    ASCIIColors.info(f"[Context] Round {round_count} fill: {total_tokens}/{max_ctx} tokens ({fill_pct:.1f}%)")
            except Exception as ctx_err:
                ASCIIColors.warning(f"[Context] Failed to calculate context fill: {ctx_err}")

            # Execute generation turn (streams and appends to the existing ai_msg.content directly)
            ASCIIColors.info(f"[Trace] Starting generation for round {round_count}...")
            _t_gen_start = _time.perf_counter()
            object.__setattr__(self, "_ttft_logged", False)
            try:
                self.lollmsClient.generate_from_messages(
                    messages=messages_list,
                    images=round_images if round_images else None,
                    stream=True,
                    temperature=temperature,
                    streaming_callback=_inline_relay,
                    **gen_kwargs
                )
                _t_gen_end = _time.perf_counter()
                ASCIIColors.info(f"[Trace] Generation round {round_count} stream completed in {(_t_gen_end - _t_gen_start):.2f} s.")
            except Exception as gen_err:
                _t_gen_end = _time.perf_counter()
                ASCIIColors.warning(f"[Trace] Generation round {round_count} failed after {(_t_gen_end - _t_gen_start):.2f} s.")
                if self.is_generation_cancelled():
                    was_cancelled = True
                    break
                else:
                    raise

            # Check cancellation after generation completes
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            ss.flush_remaining_buffer()

            # ── 🏁 TERMINATION TAG PROTOCOL ──
            # If the LLM emitted <done/> or <end/>, the task is explicitly complete. Break immediately.
            if ss.was_done_detected():
                ASCIIColors.info("[ChatMixin] Termination tag detected. Terminating agentic loop.")
                break

            # ── 🔍 PROCESS PENDING MEMORY SEARCHES (HIGHEST PRIORITY) ──
            # If the LLM emitted a <mem_search> tag, execute it NOW and inject results.
            # This MUST happen BEFORE the duplicate artifact check because memory searches
            # are NOT artifacts - they're infrastructure operations that need immediate processing.
            if hasattr(self, '_pending_memory_searches') and self._pending_memory_searches:
                # ── 🧠 CRITICAL FIX: CAPTURE THE LLM'S RESPONSE BEFORE PROCESSING SEARCH ──
                # The LLM has already generated a response in this round (the one that contained
                # the <mem_search> tag). We MUST capture this response and add it to virtual_history
                # BEFORE processing the search, so the LLM can see its own answer in the next round.
                full_round_text = ss.get_clean_text_so_far()
                raw_round_text_delta = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text

                # Sanitize the response to remove processing blocks and functional tags
                clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text_delta, flags=re.DOTALL | re.IGNORECASE)
                clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
                clean_history_text = re.sub(r'<mem_[^>]*?/?>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = clean_history_text.strip()

                # Add the LLM's response to virtual history (this is the response that contained the mem_search tag)
                if clean_history_text:
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_history_text
                    ))
                    ASCIIColors.debug(f"[ChatMixin] Captured assistant response before memory search: {clean_history_text[:100]}...")

                for search_req in self._pending_memory_searches:
                    query = search_req["query"]
                    level = search_req["level"]

                    # ── 🛡️ DUPLICATE SEARCH PREVENTION ──
                    # Check if we've already executed this exact search in this turn
                    search_signature = f"{query}::{level}"
                    if search_signature in executed_memory_searches:
                        ASCIIColors.warning(f"[ChatMixin] Duplicate memory search detected: '{query}' (level={level}). Skipping.")

                        # Inject a warning into virtual history
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f"[SYSTEM: DUPLICATE SEARCH BLOCKED]\n"
                                f"You have already searched for '{query}' in this turn.\n"
                                f"The results are already in your context above.\n"
                                f"Do NOT repeat this search. Analyze the existing results and provide your final answer.\n"
                                f"If the task is complete, emit `<done/>` now.\n"
                                f"[END DUPLICATE SEARCH BLOCKED]"
                            )
                        ))
                        continue

                    # Mark this search as executed
                    executed_memory_searches.add(search_signature)

                    ASCIIColors.info(f"[ChatMixin] Processing memory search: query='{query}', level={level}")

                    # ── 🎨 EMIT UI FEEDBACK EVENT ──
                    # Emit a processing block to show the user that a memory search is happening
                    if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                        proc_open = f'\n<processing type="memory_search" title="Memory Search: {query}">\n'
                        status_line = f'* Searching memory archives for "{query}"...\n'
                        ai_msg.content += proc_open + status_line
                        _cb(callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # Emit structured event for FULL_CALLBACK_MODE
                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                        _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_START, {
                            "tool_name": "memory_search",
                            "parameters": {"query": query, "level": level}
                        })

                    # Execute the search
                    if _mm:
                        if level is not None:
                            results = _mm.query(text=query, top_k=5, level=level)
                        else:
                            # Search all levels
                            results = _mm.query(text=query, top_k=10)

                        # Build the search results context
                        if results:
                            search_context = f"\n[MEMORY SEARCH RESULTS for query: '{query}']\n"
                            if level is not None:
                                level_names = {1: "Working", 2: "Deep", 3: "Archived"}
                                search_context += f"Searched in: {level_names.get(level, f'Level {level}')} Memory\n"
                            search_context += f"Found {len(results)} matching memories:\n\n"

                            for idx, mem in enumerate(results, 1):
                                mem_id = mem.get("id", "")[:8]
                                content = mem.get("content", "")[:200]  # Truncate long content
                                importance = mem.get("importance", 0)
                                tags = mem.get("tags", "")
                                mem_level = mem.get("level", 3)

                                level_name = {1: "Working", 2: "Deep", 3: "Archived"}.get(mem_level, f"L{mem_level}")
                                search_context += f"{idx}. [{mem_id}] ({level_name}, importance: {importance:.0%}) {content}"
                                if tags:
                                    search_context += f"  #{tags.replace(',', ' #')}"
                                search_context += "\n"

                            search_context += "\n💡 You can load any of these memories into Working Memory using <mem_load id=\"ID\" />\n"
                            search_context += "[END MEMORY SEARCH RESULTS]\n"

                            # ── 🎨 EMIT SUCCESS FEEDBACK ──
                            if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                result_line = f'* ✅ Found {len(results)} matching memories.\n'
                                proc_close = f'<!-- status:success -->\n</processing>\n\n'
                                ai_msg.content += result_line + proc_close
                                _cb(callback, result_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                _cb(callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                    "tool_name": "memory_search",
                                    "success": True,
                                    "output": f"Found {len(results)} matching memories",
                                    "error": None
                                })

                            # ── 🎯 CONTEXT-AWARE TASK GUIDANCE ──
                            # Provide flexible guidance that allows the LLM to decide whether to continue or terminate
                            search_context += (
                                f"\n[MEMORY SEARCH COMPLETE - DECISION POINT]\n"
                                f"You have successfully searched your memory and found {len(results)} relevant result(s).\n"
                                f"The search results are now visible above.\n\n"
                                f"**ANALYZE THE USER'S ORIGINAL REQUEST**:\n"
                                f"- Was the memory search the ONLY thing they asked for?\n"
                                f"  → If YES: Provide your final answer and emit `<done/>`\n"
                                f"- Or is this search just ONE STEP in a larger task (e.g., create a plan, build an artifact, call a tool)?\n"
                                f"  → If YES: Use the retrieved information to continue with the next action (e.g., `<artifact>`, `<tool>`, etc.)\n\n"
                                f"**EXAMPLES**:\n\n"
                                f"Scenario 1: User asked 'Do you remember my daughter?'\n"
                                f"```\n"
                                f"Yes! I found information about your daughter. She was preparing for the Cambridge A2 Key exam...\n"
                                f"\n"
                                f"<done/>\n"
                                f"```\n\n"
                                f"Scenario 2: User asked 'Find my daughter's exam info and create a study plan'\n"
                                f"```\n"
                                f"I found your daughter's exam information. Now let me create a personalized study plan based on the Cambridge A2 Key format...\n"
                                f"\n"
                                f"<artifact name=\"study_plan.md\" type=\"document\">\n"
                                f"# Cambridge A2 Key Study Plan\n"
                                f"...\n"
                                f"</artifact>\n"
                                f"```\n\n"
                                f"**CRITICAL RULES**:\n"
                                f"1. Do NOT emit another `<mem_search>` tag - you have already retrieved the information.\n"
                                f"2. If you need to load a specific memory into Working Memory for detailed access, use `<mem_load id=\"ID\" />`.\n"
                                f"3. If the task is complete after the search, emit `<done/>`.\n"
                                f"4. If the task requires more actions (create artifacts, call tools, etc.), continue with those actions.\n"
                                f"[END DECISION POINT]\n"
                            )
                        else:
                            search_context = f"\n[MEMORY SEARCH RESULTS for query: '{query}']\n"
                            search_context += "No matching memories found in any tier.\n"
                            search_context += "[END MEMORY SEARCH RESULTS]\n"

                            # ── 🎨 EMIT NO-RESULTS FEEDBACK ──
                            if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                result_line = f'* ❌ No matching memories found.\n'
                                proc_close = f'<!-- status:success -->\n</processing>\n\n'
                                ai_msg.content += result_line + proc_close
                                _cb(callback, result_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                _cb(callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                                _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                    "tool_name": "memory_search",
                                    "success": True,
                                    "output": "No matching memories found",
                                    "error": None
                                })

                            # ── 🎯 CONTEXT-AWARE GUIDANCE (No results) ──
                            search_context += (
                                f"\n[MEMORY SEARCH COMPLETE - DECISION POINT]\n"
                                f"You have searched your memory but found no matching results.\n\n"
                                f"**ANALYZE THE USER'S ORIGINAL REQUEST**:\n"
                                f"- Was the memory search the ONLY thing they asked for?\n"
                                f"  → If YES: Acknowledge you don't have the information and emit `<done/>`\n"
                                f"- Or is this search just ONE STEP in a larger task?\n"
                                f"  → If YES: Continue with the task using alternative approaches (ask user, use tools, etc.)\n\n"
                                f"**EXAMPLES**:\n\n"
                                f"Scenario 1: User asked 'Do you remember my daughter?'\n"
                                f"```\n"
                                f"I don't have any information about your daughter in my memory. Could you tell me about her?\n"
                                f"\n"
                                f"<done/>\n"
                                f"```\n\n"
                                f"Scenario 2: User asked 'Find my daughter's exam info and create a study plan'\n"
                                f"```\n"
                                f"I don't have information about your daughter's exam in my memory. Could you share the details so I can create a personalized study plan?\n"
                                f"\n"
                                f"<done/>\n"
                                f"```\n\n"
                                f"**CRITICAL RULES**:\n"
                                f"1. Do NOT emit another `<mem_search>` tag with the same query - the search is complete.\n"
                                f"2. If the task is complete, emit `<done/>`.\n"
                                f"3. If you need more information from the user, ask them directly and emit `<done/>`.\n"
                                f"4. If you can proceed with alternative approaches, do so.\n"
                                f"[END DECISION POINT]\n"
                            )

                        # Inject the search results into virtual history
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=search_context
                        ))

                        ASCIIColors.success(f"[ChatMixin] Injected {len(results)} memory search results into virtual history")

                # Clear the pending searches
                self._pending_memory_searches = []

                # ── 📊 LOG MEMORY SEARCH ACTION ──
                # Record this action in the turn progress tracker so the LLM can see it
                turn_actions_log.append({
                    "action": "memory_search",
                    "query": search_req["query"],
                    "results_count": len(results) if results else 0,
                    "round": round_count
                })

                # ── 🔄 COMPRESS VIRTUAL HISTORY IF NEEDED ──
                # After adding the search results, check if we need to compress older rounds
                _compress_virtual_history_if_needed()

                # Force a continuation round so the LLM can see the search results
                continue

            # ── 🛑 ARTIFACT LOOP ENFORCEMENT ──
            # If we previously flagged a force-final-answer due to an artifact loop,
            # and the LLM attempts to dispatch another artifact, we instantly break the loop.
            if getattr(self, "_force_final_answer", False) and ss.was_action_dispatched() and not ss.tool_trigger:
                ASCIIColors.warning("[ChatMixin] LLM attempted artifact dispatch after force-final-answer. Breaking loop.")
                break

            # ── 🛑 CRITICAL FIX: DISTINGUISH FAILED PATCH FROM TRUE DUPLICATE ──
            # If the LLM emits an artifact tag that was ALREADY processed in a previous round,
            # _StreamState skips dispatching it (affected_artefacts remains empty).
            # HOWEVER, a failed SEARCH/REPLACE patch ALSO results in empty affected_artefacts.
            # We must only force-final-answer for TRUE duplicates, not failed patches.
            if ss.was_action_dispatched() and not ss.tool_trigger and not ss.affected_artefacts:
                if ss.was_last_dispatch_failed():
                    # ── PATCH CORRECTION PATH ──
                    # The SEARCH/REPLACE block failed. Inject the error and let the LLM correct it.
                    ASCIIColors.warning("[ChatMixin] Artifact patch failed. Injecting correction context.")
                    full_round_text = ss.get_clean_text_so_far()
                    raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                    clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)

                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_history_text.strip()
                    ))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            "[SYSTEM: Your last <artifact> SEARCH/REPLACE patch FAILED. The SEARCH block text was not found in the existing file content. "
                            "You MUST retry the patch with a corrected SEARCH block that exactly matches the current file content. "
                            "Look at the 'Fully Loaded File Contents [C]' section in your context to find the exact text to match. "
                            "Do NOT emit <done/> until the patch succeeds or you decide to do a full rewrite instead.]"
                        )
                    ))
                    continue
                else:
                    # ── TRUE DUPLICATE PATH ──
                    ASCIIColors.warning("[ChatMixin] LLM emitted a duplicate artifact tag. Forcing final answer.")
                    object.__setattr__(self, "_force_final_answer", True)

                    full_round_text = ss.get_clean_text_so_far()
                    raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                    clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)

                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_history_text.strip()
                    ))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: CRITICAL. You just attempted to recreate an artifact that already exists with the exact same content. This is a loop. You MUST NOT create or update this artifact again. You MUST now provide your final conversational answer to the user, explaining what you have done, and end with <done/>.]"
                    ))
                    break

            # ── 🛑 ONE-ACTION-PER-TURN PROTOCOL ──
            # If the StreamState dispatched an artifact, note, skill, or context update
            # (but NOT a tool), we must halt generation, hydrate virtual_history, and re-prompt.
            if ss.was_action_dispatched() and not ss.tool_trigger:
                full_round_text = ss.get_clean_text_so_far()
                raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text

                # Sanitize the raw text to remove processing blocks and HTML comments
                clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)

                if not clean_history_text.strip() and ss.affected_artefacts:
                    for art in ss.affected_artefacts:
                        title = art.get("title", "artifact")
                        atype = art.get("type", "code")
                        lang = art.get("language", "")
                        content = art.get("content", "")
                        ephemeral_attr = ' ephemeral="true"' if art.get("ephemeral") else ""
                        clean_history_text += f'<artifact name="{title}" type="{atype}" language="{lang}"{ephemeral_attr}>\n{content}\n</artifact>\n'

                # Append the sanitized assistant text (containing the raw <artifact> tag) to virtual_history
                # Ensure content is always a string, never a dict
                if isinstance(clean_history_text, dict):
                    # If somehow a dict got in, extract the content field
                    clean_history_text = clean_history_text.get("content", str(clean_history_text))
                elif not isinstance(clean_history_text, str):
                    clean_history_text = str(clean_history_text)

                virtual_history.append(SimpleNamespace(
                    sender_type="assistant",
                    content=clean_history_text.strip()
                ))

                # Determine the action type and title for the system marker
                action_type = "artifact"
                action_title = ""
                if ss.affected_artefacts:
                    last_art = ss.affected_artefacts[-1]
                    action_type = last_art.get("type", "artifact")
                    action_title = last_art.get("title", "")

                # 🧠 CONTEXTUAL ANCHORING PROTOCOL
                # If the sanitized history is empty, it means the artifact was emitted with 
                # no conversational wrapper. We inject the RAW artifact XML into virtual_history
                # so the LLM can literally "see" the code it just wrote. This enables multi-step
                # workflows (e.g., Code -> Review -> Patch) and prevents blind recreation loops.
                if not clean_history_text.strip() and ss.affected_artefacts:
                    for art in ss.affected_artefacts:
                        title = art.get("title", "artifact")
                        atype = art.get("type", "code")
                        lang = art.get("language", "")
                        content = art.get("content", "")
                        ephemeral_attr = ' ephemeral="true"' if art.get("ephemeral") else ""
                        clean_history_text += f'<artifact name="{title}" type="{atype}" language="{lang}"{ephemeral_attr}>\n{content}\n</artifact>\n'

                # Update the assistant message to contain the real content (if we injected it)
                virtual_history[-1].content = clean_history_text.strip()

                # ── 🔄 COMPRESS VIRTUAL HISTORY IF NEEDED ──
                # After adding the artifact creation, check if we need to compress older rounds
                _compress_virtual_history_if_needed()

                # 🧠 STATUS CHECK PROTOCOL (Replaces Forced Done)
                # We anchor the LLM to the fact that the artifact is saved, and ask it if it is finished.
                # This allows multi-step workflows (verification, patching) before final termination.
                title_str = f" '{action_title}'" if action_title else ""
                system_marker = (
                    f"[SYSTEM: The {action_type}{title_str} has been successfully created and saved to the workspace. "
                    f"You can see its full content in your previous message. "
                    f"Are you finished with the user's request? "
                    f"If YES, provide your final conversational answer to the user and end your generation with a `<done/>` tag. "
                    f"If NO, and you need to perform more tasks (e.g., reviewing the code, patching imports, running tests), do that now.]"
                )
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=system_marker
                ))

                # Force another reasoning round
                continue

            if ss.tool_trigger:
                tool_call_json_str = ss.get_tool_call_json()
                if tool_call_json_str:
                    try:
                        call_data = json.loads(tool_call_json_str)
                    except Exception:
                        call_data = {}

                    # ── 🛑 CRITICAL FIX: PHANTOM TOOL CALL PREVENTION ──
                    # If the LLM emits a <tool> tag but the JSON is malformed or missing
                    # the "name" key, we MUST NOT execute active_tools[""]. 
                    # Instead, we inject a correction and force a continuation.
                    if not isinstance(call_data, dict) or not call_data.get("name"):
                        ASCIIColors.warning(f"[ChatMixin] Malformed tool call detected. JSON: {tool_call_json_str[:200]}")

                        # 🛡️ CRITICAL FIX: Record this malformed call in FailureMemory
                        # to prevent infinite loops of the same malformed payload.
                        failure_memory = getattr(self, "_failure_memory", None)
                        malformed_sig = "unknown::malformed"
                        if failure_memory:
                            try:
                                if hasattr(failure_memory, "record_failure_by_signature"):
                                    failure_memory.record_failure_by_signature(malformed_sig, "Malformed tool call: missing 'name' or invalid JSON")
                                if hasattr(failure_memory, "_signatures"):
                                    failure_memory._signatures.add(malformed_sig)
                            except Exception:
                                pass

                        # Inject a correction into the virtual history so the LLM knows it failed
                        correction_msg = (
                            "=== ⚠️ TOOL CALL FORMAT ERROR ===\n"
                            "Your last tool call was malformed or missing the 'name' field. "
                            f"Raw received: `{tool_call_json_str[:150]}`\n"
                            "You MUST output a valid JSON object with a 'name' key matching an available tool, "
                            "and a 'parameters' key containing the arguments.\n"
                            "Example: <tool>{\"name\": \"tool_wikipedia_search\", \"parameters\": {\"query\": \"Einstein\"}}</tool>\n"
                            "Please output the corrected tool call now."
                        )

                        full_round_text = ss.get_clean_text_so_far()
                        raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                        # 🛑 CRITICAL FIX: Sanitize raw_round_text before appending to virtual_history.
                        # The _StreamState emits <processing> blocks into ai_msg.content when it
                        # dispatches the tool tag. If we append this unsanitized, the LLM sees the
                        # <processing> blocks in its history and mimics them, causing infinite
                        # nested <processing> generation loops.
                        clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                        # Also strip any raw <tool> tags to prevent the LLM from seeing its own failed call
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = clean_history_text.strip()
                        if not clean_history_text:
                            clean_history_text = "[Malformed tool call emitted with no conversational text]"
                        virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=clean_history_text
                        ))
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=correction_msg
                        ))

                        # 🛑 CRITICAL FIX: If the malformed call has been seen before, break immediately.
                        # We use a dedicated counter dict because _signatures is a set (no duplicates).
                        if not hasattr(self, "_malformed_call_counts"):
                            object.__setattr__(self, "_malformed_call_counts", {})
                        self._malformed_call_counts[malformed_sig] = self._malformed_call_counts.get(malformed_sig, 0) + 1
                        if self._malformed_call_counts[malformed_sig] >= 2:
                            ASCIIColors.warning("[ChatMixin] Second identical malformed tool call detected. Breaking loop to prevent infinite cycle.")
                            break

                        # Force another round to let the LLM correct itself
                        continue

                    tool_name = call_data.get("name", "")
                    tool_params = call_data.get("parameters", {})

                    # ── 🛡️ MEMORY TAG AS TOOL INTERCEPTION (CRITICAL FIX) ──
                    # If the LLM tries to call a memory tag as a tool (e.g., <tool>{"name": "memory_search"...}</tool>),
                    # we MUST block it and inject a correction. Memory tags are infrastructure tags, not tools.
                    if tool_name.lower() in _FORBIDDEN_TOOL_NAMES:
                        ASCIIColors.error(f"[ChatMixin] LLM attempted to call memory tag '{tool_name}' as a tool. Blocking and correcting.")

                        # Emit a failure processing block to the UI
                        status_err_line = f"* Tool call blocked.\n"
                        details_block = (
                            f"Error: '{tool_name}' is a MEMORY SYSTEM TAG, not a tool.\n"
                            f"Memory tags are processed silently by the memory system and must NEVER be wrapped in <tool> blocks.\n"
                            f"Use the XML tag directly instead.\n"
                        )
                        tool_close_tag = f"{status_err_line}{details_block}<!-- status:failure -->\n</processing>\n\n"
                        ai_msg.content += tool_close_tag
                        _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        # Inject a targeted correction into virtual history
                        correction_msg = (
                            f"=== ⚠️ CRITICAL: MEMORY TAG MISUSE ===\n"
                            f"You attempted to call `{tool_name}` as a tool using `<tool>{{\"name\": \"{tool_name}\", ...}}</tool>`.\n"
                            f"This is **WRONG**. `{tool_name}` is a **MEMORY SYSTEM TAG**, not a tool.\n\n"
                            f"**CORRECT USAGE**:\n"
                            f"Memory tags are used directly as XML tags, NOT wrapped in `<tool>` blocks.\n\n"
                        )

                        # Provide specific examples for common memory tags
                        if "search" in tool_name.lower():
                            correction_msg += (
                                f"To search memories, use:\n"
                                f"```\n"
                                f"<mem_search query=\"your search terms\" />\n"
                                f"```\n"
                                f"NOT:\n"
                                f"```\n"
                                f"<tool>{{\"name\": \"memory_search\", \"parameters\": {{\"query\": \"...\"}}}}</tool>\n"
                                f"```\n\n"
                            )
                        elif "new" in tool_name.lower():
                            correction_msg += (
                                f"To create a new memory, use:\n"
                                f"```\n"
                                f"<mem_new importance=\"0.8\">Memory content here</mem_new>\n"
                                f"```\n"
                                f"NOT:\n"
                                f"```\n"
                                f"<tool>{{\"name\": \"mem_new\", \"parameters\": {{...}}}}</tool>\n"
                                f"```\n\n"
                            )
                        elif "load" in tool_name.lower():
                            correction_msg += (
                                f"To load a memory from Deep Memory, use:\n"
                                f"```\n"
                                f"<mem_load id=\"abc123de\" />\n"
                                f"```\n"
                                f"NOT:\n"
                                f"```\n"
                                f"<tool>{{\"name\": \"mem_load\", \"parameters\": {{\"id\": \"...\"}}}}</tool>\n"
                                f"```\n\n"
                            )

                        correction_msg += (
                            f"**AVAILABLE MEMORY TAGS** (use directly, NOT as tools):\n"
                            f"  • `<mem_new importance=\"...\">content</mem_new>` — Create a new memory\n"
                            f"  • `<mem_update id=\"ID\">content</mem_update>` — Update an existing memory\n"
                            f"  • `<mem_tag id=\"ID\" />` — Tag a memory as used\n"
                            f"  • `<mem_load id=\"ID\" />` — Load a memory from Deep Memory\n"
                            f"  • `<mem_delete id=\"ID\" />` — Delete a memory\n"
                            f"  • `<mem_search query=\"terms\" />` — Search archived memories\n"
                            f"  • `<mem_rel source=\"ID\" target=\"ID\" type=\"TYPE\" />` — Create a relationship\n\n"
                            f"Please continue your response using the correct memory tag syntax."
                        )

                        # Sanitize and append to virtual history
                        full_round_text = ss.get_clean_text_so_far()
                        raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                        clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = clean_history_text.strip()
                        if not clean_history_text:
                            clean_history_text = f"[Attempted to call memory tag '{tool_name}' as a tool]"

                        virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=clean_history_text
                        ))
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=correction_msg
                        ))

                        # Force another reasoning round to let the LLM correct itself
                        continue

                    full_round_text = ss.get_clean_text_so_far()
                    raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text

                    # Remove <processing> blocks and HTML status comments for LLM context
                    # 🛑 CRITICAL FIX 3: Use robust regex that catches partial/malformed blocks.
                    # The previous regex required a perfect </processing> close tag, but streaming
                    # fragmentation could leave orphaned opening tags or partial content.
                    clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                    # Remove any orphaned closing tags from partial stripping
                    clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                    # Remove standalone <lollms_artifact> and <artefact_image> tags that were injected outside blocks
                    clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                    clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)

                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_history_text.strip()
                    ))

                    # ── 🛡️ PHANTOM TOOL INTERCEPTION ──
                    # If the LLM hallucinates a tool that is not in the active registry,
                    # we intercept it BEFORE execution, inject a correction, and force a retry.
                    # This prevents cascading failures where the LLM panics and tries other unregistered tools.
                    if not active_tools or tool_name not in active_tools:
                        ASCIIColors.warning(f"[ChatMixin] Phantom tool call detected: '{tool_name}' is not registered.")

                        # 🛡️ CRITICAL FIX: Record phantom tool in FailureMemory to prevent infinite loops
                        failure_memory = getattr(self, "_failure_memory", None)
                        if failure_memory:
                            try:
                                param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                            except Exception:
                                param_sig = str(tool_params)
                            phantom_sig = f"{tool_name}::{param_sig}"
                            if hasattr(failure_memory, "record_failure_by_signature"):
                                failure_memory.record_failure_by_signature(phantom_sig, f"Phantom tool '{tool_name}' not registered")
                            elif hasattr(failure_memory, "_signatures"):
                                failure_memory._signatures.add(phantom_sig)

                        # Emit a failure processing block to the UI
                        status_err_line = f"* Tool call blocked.\n"
                        details_block = f"Error Logs:\nTool '{tool_name}' is not available in this session.\n"
                        tool_close_tag = f"{status_err_line}{details_block}<!-- status:failure -->\n</processing>\n\n"
                        ai_msg.content += tool_close_tag
                        _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        # Inject a targeted correction into virtual history
                        available_tools_str = ", ".join(f"`{t}`" for t in active_tools.keys()) if active_tools else "No tools are available."
                        correction_msg = (
                            f"=== ⚠️ INVALID TOOL CALL ===\n"
                            f"You attempted to call `{tool_name}`, which is **NOT REGISTERED** in this session.\n"
                            f"You are STRICTLY FORBIDDEN from hallucinating tool names.\n\n"
                            f"The ONLY tools available to you right now are:\n"
                            f"{available_tools_str}\n\n"
                            f"If one of these tools is suitable, output the corrected `<tool>` call now.\n"
                            f"If NONE of these tools can accomplish the task, DO NOT try to call any tool. "
                            f"Instead, inform the user that the required tool is not available and complete your response."
                        )

                        # 🛑 CRITICAL FIX: Sanitize raw_round_text before appending to virtual_history.
                        # The _StreamState emits <processing> blocks into ai_msg.content when it
                        # dispatches the tool tag. If we append this unsanitized, the LLM sees the
                        # <processing> blocks in its history and mimics them, causing infinite
                        # nested <processing> generation loops.
                        full_round_text = ss.get_clean_text_so_far()
                        raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                        clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
                        clean_history_text = clean_history_text.strip()
                        if not clean_history_text:
                            clean_history_text = f"[Phantom tool call to '{tool_name}' with no conversational text]"
                        virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=clean_history_text
                        ))
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=correction_msg
                        ))

                        # 🛑 CRITICAL FIX: If the phantom call has been seen before, break immediately.
                        if not hasattr(self, "_phantom_call_counts"):
                            object.__setattr__(self, "_phantom_call_counts", {})
                        
                        try:
                            param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                        except Exception:
                            param_sig = str(tool_params)
                        phantom_sig = f"{tool_name}::{param_sig}"
                        
                        self._phantom_call_counts[phantom_sig] = self._phantom_call_counts.get(phantom_sig, 0) + 1
                        if self._phantom_call_counts[phantom_sig] >= 2:
                            ASCIIColors.warning(f"[ChatMixin] Second identical phantom tool call '{tool_name}' detected. Breaking loop to prevent infinite cycle.")
                            break

                        # Force another reasoning round to let the LLM correct itself
                        continue

                    tool_res = None
                    _lcp_executed = False

                    if active_tools and tool_name in active_tools and "callable" not in active_tools[tool_name]:
                        if lcp_binding and hasattr(lcp_binding, "execute_tool"):
                            import os as _os
                            from pathlib import Path as _Path
                            _old_cwd_lcp = _os.getcwd()

                            # Resolve workspace_data path
                            if hasattr(self, "workspace_data_path") and self.workspace_data_path:
                                _lcp_workspace_dir = _Path(self.workspace_data_path)
                            else:
                                _base_ws = _Path(self.workspace_path) if hasattr(self, "workspace_path") and self.workspace_path else _Path("./data_workspace")
                                _lcp_workspace_dir = _base_ws / self.id / "workspace_data"

                            _lcp_workspace_dir.mkdir(parents=True, exist_ok=True)
                            _lcp_workspace_str = str(_lcp_workspace_dir.resolve())

                            try:
                                _os.chdir(_lcp_workspace_str)

                                # ── Take BEFORE Snapshot (LCP Path) ──
                                _lcp_files_before = {}
                                _lcp_cwd = _Path(_lcp_workspace_str)
                                if _lcp_cwd.exists():
                                    for f in _lcp_cwd.rglob("*"):
                                        if f.is_file():
                                            try:
                                                rel_path = f.relative_to(_lcp_cwd)
                                                content = f.read_text(encoding="utf-8", errors="ignore")
                                                _lcp_files_before[rel_path] = {
                                                    "hash": hash(content),
                                                    "mtime": f.stat().st_mtime,
                                                    "path": f,
                                                    "content": content
                                                }
                                            except Exception:
                                                try:
                                                    rel_path = f.relative_to(_lcp_cwd)
                                                    _lcp_files_before[rel_path] = {
                                                        "hash": None,
                                                        "mtime": f.stat().st_mtime,
                                                        "path": f,
                                                        "content": None
                                                    }
                                                except Exception:
                                                    pass

                                try:
                                    tool_res = lcp_binding.execute_tool(
                                       tool_name, 
                                       tool_params, 
                                       lollms_client_instance=self.lollmsClient, 
                                       discussion_instance=self,
                                    )
                                except Exception as lcp_exec_err:
                                    trace_exception(lcp_exec_err)
                                    tool_res = {
                                       "success": False,
                                       "error": f"Tool '{tool_name}' crashed: {lcp_exec_err}",
                                       "traceback": traceback.format_exc()
                                    }
                                _lcp_executed = True

                                # ── Take AFTER Snapshot & Auto-Sync Artifacts (LCP Path) ──
                                _lcp_files_after = {}
                                if _lcp_cwd.exists():
                                    for f in _lcp_cwd.rglob("*"):
                                        if f.is_file():
                                            try:
                                                rel_path = f.relative_to(_lcp_cwd)
                                                content = f.read_text(encoding="utf-8", errors="ignore")
                                                _lcp_files_after[rel_path] = {
                                                    "hash": hash(content),
                                                    "mtime": f.stat().st_mtime,
                                                    "path": f,
                                                    "content": content
                                                }
                                            except Exception:
                                                try:
                                                    rel_path = f.relative_to(_lcp_cwd)
                                                    _lcp_files_after[rel_path] = {
                                                        "hash": None,
                                                        "mtime": f.stat().st_mtime,
                                                        "path": f,
                                                        "content": None
                                                    }
                                                except Exception:
                                                    pass

                                self._sync_tool_artifacts(tool_name, _lcp_files_before, _lcp_files_after, callback)
                            finally:
                                # 🛑 CRITICAL: Always restore CWD to prevent workspace corruption
                                _os.chdir(_old_cwd_lcp)
                        else:
                            tool_res = {
                                "success": False,
                                "error": f"Tool '{tool_name}' has no callable and no LCP tools binding is available on the client.",
                                "status_code": 404
                            }
                            _lcp_executed = True
                    elif active_tools and tool_name in active_tools and "callable" in active_tools[tool_name]:
                        import os as _os
                        from pathlib import Path as _Path
                        _old_cwd_direct = _os.getcwd()

                        if hasattr(self, "workspace_data_path") and self.workspace_data_path:
                            _direct_workspace_dir = _Path(self.workspace_data_path)
                        else:
                            _base_ws_direct = _Path(self.workspace_path) if hasattr(self, "workspace_path") and self.workspace_path else _Path("./data_workspace")
                            _direct_workspace_dir = _base_ws_direct / self.id / "workspace_data"

                        _direct_workspace_dir.mkdir(parents=True, exist_ok=True)
                        _direct_workspace_str = str(_direct_workspace_dir.resolve())

                        try:
                            _os.chdir(_direct_workspace_str)
                            try:
                                import inspect as _inspect
                                _direct_sig = _inspect.signature(active_tools[tool_name]["callable"]).parameters
                                _direct_call_kwargs = dict(tool_params)
                                if "discussion_instance" in _direct_sig:
                                    _direct_call_kwargs["discussion_instance"] = self
                                if "lollms_client_instance" in _direct_sig:
                                    _direct_call_kwargs["lollms_client_instance"] = self.lollmsClient

                                tool_res = active_tools[tool_name]["callable"](**_direct_call_kwargs)
                                _lcp_executed = True
                            except Exception as direct_err:
                                trace_exception(direct_err)
                                tool_res = {
                                    "success": False,
                                    "error": f"Direct callable execution failed: {direct_err}",
                                    "traceback": traceback.format_exc()
                                }
                                _lcp_executed = True
                        finally:
                            _os.chdir(_old_cwd_direct)
                    else:
                        tool_res = {
                            "success": False,
                            "error": f"Tool '{tool_name}' is not registered in the active tools dictionary for this session.",
                            "status_code": 404
                        }
                        _lcp_executed = True

                    # 2. Strip ONLY the raw <tool> JSON tag from the UI/DB buffer (ai_msg.content).
                    # 🛑 CRITICAL: Do NOT strip <processing> blocks here. They are part of the 
                    # execution log and must remain in the final saved message. The export() 
                    # method will sanitize them when building context for the LLM.
                    if tool_call_json_str in ai_msg.content:
                        ai_msg.content = ai_msg.content.replace(f"<tool>{tool_call_json_str}</tool>", "")
                        ai_msg.content = ai_msg.content.replace(tool_call_json_str, "")

                    # ── 🛑 CRITICAL FIX: PREVENT DUPLICATE UI BLOCKS ──
                    # The _StreamState parser ALREADY emitted the <processing> block and
                    # "Calling tool..." status to the UI instantly when the </tool> tag closed.
                    # We MUST NOT emit it again here, or the UI will render duplicate blocks.
                    # We simply proceed directly to tool execution.

                    # ── REFLEXIVE LOOP DETECTION (FailureMemory) ──
                    failure_memory = getattr(self, "_failure_memory", None)

                    try:
                        param_signature = json.dumps(tool_params, sort_keys=True, default=str)
                    except Exception:
                        param_signature = str(tool_params)
                    full_signature = f"{tool_name}::{param_signature}"

                    # 🛑 INSTRUMENTATION: Log the state of the signatures set
                    if failure_memory and hasattr(failure_memory, "_signatures"):
                        ASCIIColors.warning(f"[LoopTrace] Checking signature: {full_signature}. Current signatures: {failure_memory._signatures}")
                    else:
                        ASCIIColors.warning(f"[LoopTrace] FailureMemory or _signatures missing.")

                    has_prev_failure = (
                        hasattr(failure_memory, "_signatures") and full_signature in failure_memory._signatures
                    ) if failure_memory else False

                    if has_prev_failure:
                        if self.is_generation_cancelled():
                            was_cancelled = True
                            break

                        result_str = (
                            f"Error executing tool '{tool_name}': This exact parameters configuration failed on a previous round of this conversation. "
                            f"To prevent an infinite loop, execution was blocked. You must modify your parameters, inspect the data schemas, "
                            f"or try a different approach. If you cannot proceed, inform the user of the error and suggest alternatives."
                        )
                        status_err_line = f"* Tool call blocked to prevent loop.\n"
                        details_block = f"Loop Intercepted:\n{result_str}\n"
                        tool_close_tag = f"{status_err_line}{details_block}<!-- status:failure -->\n</processing>\n\n"
                        ai_msg.content += tool_close_tag
                        _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f'<tool_result name="{tool_name}" status="FAILED">\n'
                                f"{result_str}\n"
                                f"</tool_result>\n\n"
                                f"⚠️ **Tool Execution Failed & Loop Blocked.**\n"
                                f"You attempted to retry a failing tool with identical parameters. The system has blocked this to prevent an infinite loop. "
                                f"You MUST now write a final response to the user explaining that the operation could not be completed, "
                                f"detailing the error above, and suggesting possible workarounds or alternative approaches. Do NOT attempt to call the tool again."
                            )
                        ))
                        continue

                    # Execute the tool sequentially
                    try:
                        def _get_file_hashes(params: dict) -> dict:
                            """Returns a dict of {param_name: file_hash} for any param that is an existing file."""
                            hashes = {}
                            for k, v in params.items():
                                if isinstance(v, str):
                                    p = Path(v)
                                    if p.is_file():
                                        try:
                                            import hashlib
                                            content = p.read_bytes()
                                            hashes[k] = hashlib.md5(content).hexdigest()
                                        except Exception:
                                            pass
                            return hashes

                        current_file_hashes = _get_file_hashes(tool_params)
                        has_real_file_hashes = any(v is not None for v in current_file_hashes.values())

                        if has_real_file_hashes:
                            context_aware_signature = f"{full_signature}::{json.dumps(current_file_hashes, sort_keys=True)}"
                        else:
                            context_aware_signature = full_signature

                        ASCIIColors.info(f"[ChatMixin] Success-loop check: tool='{tool_name}', sig='{context_aware_signature[:120]}...', in_set={context_aware_signature in successful_tool_signatures}, set_size={len(successful_tool_signatures)}")

                        if context_aware_signature in successful_tool_signatures:
                            ASCIIColors.warning(f"[ChatMixin] Repetitive SUCCESS loop blocked for '{tool_name}'. Signature already in successful set and files unchanged.")
                            tool_res = {
                                "success": False,
                                "error": f"Repetitive tool call detected. You have already successfully called '{tool_name}' with these exact parameters, and the input files have not changed. The output is already in your context. Do not call it again.",
                                "prompt_injection": f"\n\n🛑 **STOP.** You are calling '{tool_name}' again with the exact same parameters after it already succeeded. This is a loop. The data from the previous execution is already in your context above. Analyze it and move on to answer the user."
                            }
                            virtual_history.append(SimpleNamespace(
                                sender_type="user",
                                content=(
                                    f'<tool_result name="{tool_name}" status="FAILED">\n'
                                    f"Repetitive tool call detected. The output is already in your context.\n"
                                    f"</tool_result>\n\n"
                                    f"⚠️ **Tool Execution Blocked.**\n"
                                    f"You have already successfully called '{tool_name}' with these exact parameters. The system has blocked this duplicate call. "
                                    f"You MUST now write a final response to the user using the data already retrieved. Do NOT attempt to call the tool again."
                                )
                            ))
                            # Append the processing block to UI
                            status_err_line = f"* Tool call blocked to prevent success loop.\n"
                            details_block = f"Loop Intercepted:\nRepetitive successful tool call blocked\n<!-- status:failure -->\n</processing>\n\n"
                            ai_msg.content += status_err_line + details_block
                            _cb(callback, status_err_line + details_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                            continue
                        else:
                            tool_signature_counts[full_signature] = tool_signature_counts.get(full_signature, 0) + 1
                        if self.is_generation_cancelled():
                            # Generation cancelled (logging removed)
                            tool_res = {
                                "success": False, 
                                "error": "Execution aborted by user cancellation.",
                                "prompt_injection": "\n\n⚠️ **Execution Aborted.**\nThe user cancelled the generation. Do not attempt to call tools again."
                            }
                        elif active_tools and tool_name in active_tools and "callable" in active_tools[tool_name]:
                            # Sync all active artifacts to disk BEFORE tool execution
                            try:
                                sync_ws, sync_files = self.artefacts.sync_all_active_to_disk()
                            except Exception as ex:
                                trace_exception(ex)
                                sync_ws, sync_files = None, []

                            import os
                            from pathlib import Path
                            old_cwd = os.getcwd()

                            if hasattr(self, 'workspace_path') and self.workspace_path:
                                base_workspace_dir = Path(self.workspace_path)
                            else:
                                base_workspace_dir = Path("./data_workspace")
                                # Fallback to server APP_WORKSPACE_DIR if workspace_path is not bound
                                try:
                                    from lollms_client.apps.lollms_discussions.server import APP_WORKSPACE_DIR
                                    if APP_WORKSPACE_DIR is not None:
                                        base_workspace_dir = APP_WORKSPACE_DIR
                                except ImportError:
                                    pass

                            if hasattr(self, "workspace_data_path") and self.workspace_data_path:
                                workspace_dir = Path(self.workspace_data_path)
                            else:
                                workspace_dir = base_workspace_dir / self.id / "workspace_data"

                            workspace_dir.mkdir(parents=True, exist_ok=True)
                            workspace_dir_str = str(workspace_dir.resolve())

                            try:
                                os.chdir(workspace_dir_str)

                                sanitized_params = {}
                                for key, value in tool_params.items():
                                    if isinstance(value, str):
                                        sanitized_value = value
                                        for prefix in ["workspace/", "data_workspace/", "./workspace/", "./data_workspace/"]:
                                            if sanitized_value.lower().startswith(prefix):
                                                sanitized_value = sanitized_value[len(prefix):]
                                                break
                                        if sanitized_value.lower().startswith(self.id.lower() + "/"):
                                            sanitized_value = sanitized_value[len(self.id) + 1:]
                                        sanitized_params[key] = sanitized_value
                                    else:
                                        sanitized_params[key] = value

                                ASCIIColors.info(f"[ChatMixin] Sanitized tool params: {sanitized_params}")

                                call_kwargs = dict(sanitized_params)
                                import inspect as _inspect
                                _tool_sig_params = _inspect.signature(active_tools[tool_name]["callable"]).parameters
                                if "discussion_instance" in _tool_sig_params:
                                    call_kwargs["discussion_instance"] = self
                                if "lollms_client_instance" in _tool_sig_params:
                                    call_kwargs["lollms_client_instance"] = self.lollmsClient

                                # ── Take BEFORE Snapshot ──
                                files_before = {}
                                current_cwd = Path(workspace_dir_str)
                                if current_cwd.exists():
                                    for f in current_cwd.rglob("*"):
                                        if f.is_file():
                                            try:
                                                rel_path = f.relative_to(current_cwd)
                                                content = f.read_text(encoding="utf-8", errors="ignore")
                                                files_before[rel_path] = {
                                                    "hash": hash(content),
                                                    "mtime": f.stat().st_mtime,
                                                    "path": f,
                                                    "content": content
                                                }
                                            except Exception:
                                                try:
                                                    rel_path = f.relative_to(current_cwd)
                                                    files_before[rel_path] = {
                                                        "hash": None,
                                                        "mtime": f.stat().st_mtime,
                                                        "path": f,
                                                        "content": None
                                                    }
                                                except Exception:
                                                    pass

                                # Execute directly (no thread) - LCP handles CWD internally
                                # The signature check above already safely injected
                                # 'discussion_instance' and 'lollms_client_instance' ONLY if
                                # the tool explicitly declared them in its function signature.
                                # Unconditional injection breaks agnostic tools (e.g., tool_internet_search).
                                tool_res = active_tools[tool_name]["callable"](**call_kwargs)

                                # ── Take AFTER Snapshot and Auto-Sync Artifacts ──
                                files_after = {}
                                if current_cwd.exists():
                                    for f in current_cwd.rglob("*"):
                                        if f.is_file():
                                            try:
                                                rel_path = f.relative_to(current_cwd)
                                                content = f.read_text(encoding="utf-8", errors="ignore")
                                                files_after[rel_path] = {
                                                    "hash": hash(content),
                                                    "mtime": f.stat().st_mtime,
                                                    "path": f,
                                                    "content": content
                                                }
                                            except Exception:
                                                try:
                                                    rel_path = f.relative_to(current_cwd)
                                                    files_after[rel_path] = {
                                                        "hash": None,
                                                        "mtime": f.stat().st_mtime,
                                                        "path": f,
                                                        "content": None
                                                    }
                                                except Exception:
                                                    pass

                                self._sync_tool_artifacts(tool_name, files_before, files_after, callback)
                            finally:
                                os.chdir(old_cwd)

                        if tool_res is None:
                            tool_res = {
                                "success": False,
                                "error": f"Tool '{tool_name}' execution path did not produce a result.",
                                "status_code": 500
                            }

                        if isinstance(tool_res, dict):
                            is_lcp_error = tool_res.get("status_code") and tool_res.get("status_code") != 200
                            has_error_key = "error" in tool_res and not tool_res.get("success", True)

                            if not tool_res.get("success", True) or is_lcp_error or has_error_key:
                                error_msg = tool_res.get("error", "Unknown tool error")

                                is_404 = tool_res.get("status_code") == 404

                                if failure_memory and not is_404:
                                    try:
                                        param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                                    except Exception:
                                        param_sig = str(tool_params)
                                    full_sig = f"{tool_name}::{param_sig}"
                                    if hasattr(failure_memory, "record_failure_by_signature"):
                                        failure_memory.record_failure_by_signature(full_sig, error_msg)
                                    else:
                                        if not hasattr(failure_memory, "_signatures"):
                                            object.__setattr__(failure_memory, "_signatures", set())
                                        failure_memory._signatures.add(full_sig)

                                # 🛑 ARCHITECTURAL FIX: Removed the flawed has_prev_failure check here.
                                # The previous code recorded the signature and immediately checked if it existed,
                                # which always evaluated to True and caused every failure to be mislabeled as "Loop Intercepted".
                                result_str = f"Error executing tool '{tool_name}': {error_msg}"
                                clean_result_str = result_str
                                status_done_line = f"* Completed execution with errors.\n"
                                details_block = f"Error Logs:\n{error_msg}\n"
                            else:
                                raw_output = tool_res.get("output", tool_res)

                                # Handle nested output dictionaries (common in MCP/external tools)
                                if isinstance(raw_output, dict):
                                    # If output is a dict, try to extract the most relevant field
                                    # Expanded key list to catch Wikipedia/external tool patterns
                                    extracted = None
                                    for key in ("content", "text", "result", "data", "page_content", "summary", "extract", "html", "body", "query", "pages"):
                                        if key in raw_output:
                                            extracted = raw_output[key]
                                            break
                                    
                                    if extracted is not None:
                                        raw_output = extracted
                                    else:
                                        # Fall back to JSON dump of the whole dict
                                        raw_output = json.dumps(raw_output, indent=2, default=str, ensure_ascii=False)
                                elif isinstance(raw_output, list):
                                    raw_output = json.dumps(raw_output, indent=2, default=str, ensure_ascii=False)
                                elif raw_output is None and isinstance(tool_res, dict) and len(tool_res) > 1:
                                    # CRITICAL: If 'output' was explicitly None but the tool returned
                                    # other metadata (success, error, etc.), dump the whole dict.
                                    raw_output = json.dumps(tool_res, indent=2, default=str, ensure_ascii=False)
                                else:
                                    raw_output = str(raw_output) if raw_output is not None else "No output returned."

                                full_dump = raw_output
                                result_str = full_dump
                                clean_result_str = _sanitize_tool_result(tool_res)
                                self._trigger_evolutionary_reflection(tool_name, tool_params, clean_result_str)

                                if self.lollmsClient and hasattr(self.lollmsClient, "count_tokens"):
                                    tool_output_tokens = self.lollmsClient.count_tokens(clean_result_str)
                                else:
                                    tool_output_tokens = len(clean_result_str) // 4

                                if tool_output_tokens > 1500:
                                    is_structured = (
                                        tool_name.startswith("tool_query") or 
                                        tool_name.startswith("tool_execute_python_data") or
                                        "|" in clean_result_str or 
                                        "```json" in clean_result_str
                                    )

                                    if is_structured:
                                        clean_result_str = f"[SYSTEM: Tool returned {tool_output_tokens} tokens of structured data. The data has been processed and is available in the workspace. DO NOT attempt to read the raw rows. Use aggregation/plotting tools next.]"
                                    else:
                                        log_filename = f"tool_output_{tool_name}_{round_count}.log"
                                        log_filepath = Path(self.workspace_data_path) / log_filename
                                        log_filepath.parent.mkdir(parents=True, exist_ok=True)
                                        log_filepath.write_text(clean_result_str, encoding="utf-8", errors="ignore")

                                        self.artefacts.add(
                                            title=log_filename,
                                            artefact_type="document",
                                            content=clean_result_str,
                                            active=True,
                                            visibility=ArtefactVisibility.TREE_UNLOCKABLE
                                        )
                                        self.commit()

                                        clean_result_str = f"[SYSTEM: Tool returned {tool_output_tokens} tokens of text. It has been saved to '{log_filename}'. Unlock it to read a portion, or save findings to your scratchpad.]"

                                status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                # 🛡️ CRITICAL FIX: Guard against NoneType output from tools
                                if full_dump is None:
                                    full_dump = "Tool executed successfully but returned no output content."
                                if not isinstance(full_dump, str):
                                    try:
                                        full_dump = json.dumps(full_dump, indent=2, default=str, ensure_ascii=False)
                                    except Exception:
                                        full_dump = str(full_dump)
                                safe_output = full_dump[:2000] + ("..." if len(full_dump) > 2000 else "")
                                details_block = f"Output Logs:\n{safe_output}\n"
                        else:
                            result_str = str(tool_res) if tool_res is not None else "No output returned."
                            if "error" in result_str.lower() or "fail" in result_str.lower():
                                if failure_memory:
                                    try:
                                        param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                                    except Exception:
                                        param_sig = str(tool_params)
                                    full_sig = f"{tool_name}::{param_sig}"
                                    if hasattr(failure_memory, "record_failure_by_signature"):
                                        failure_memory.record_failure_by_signature(full_sig, result_str)
                                    else:
                                        if not hasattr(failure_memory, "_signatures"):
                                            object.__setattr__(failure_memory, "_signatures", set())
                                        failure_memory._signatures.add(full_sig)
                                clean_result_str = result_str
                                status_done_line = f"* Completed execution with errors.\n"
                                details_block = f"Error Logs:\n{result_str}\n"
                            else:
                                status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                clean_result_str = _sanitize_tool_result(tool_res, client=self.lollmsClient)
                                safe_output = result_str[:2000] + ("..." if len(result_str) > 2000 else "")
                                details_block = f"Output Logs:\n{safe_output}\n"
                    except Exception as e:
                        trace_exception(e)
                        if failure_memory:
                            try:
                                param_sig = json.dumps(tool_params, sort_keys=True, default=str)
                            except Exception:
                                param_sig = str(tool_params)
                            full_sig = f"{tool_name}::{param_sig}"
                            if hasattr(failure_memory, "record_failure_by_signature"):
                                failure_memory.record_failure_by_signature(full_sig, str(e))
                            else:
                                if not hasattr(failure_memory, "_signatures"):
                                    object.__setattr__(failure_memory, "_signatures", set())
                                failure_memory._signatures.add(full_sig)
                        result_str = f"Error executing tool '{tool_name}': {e}"
                        clean_result_str = f"Error executing tool '{tool_name}': {e}"
                        status_done_line = f"* Execution crashed.\n"
                        details_block = f"Crash Details:\n{str(e)}\n"
                        tool_res = {"success": False, "error": str(e)}

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f'<tool_result name="{tool_name}" status="FAILED">\n'
                                f"{clean_result_str}\n"
                                f"</tool_result>\n\n"
                                f"⚠️ **Tool Execution Crashed.**\n"
                                f"The tool '{tool_name}' encountered an unexpected system error. "
                                f"Analyze the error and inform the user, or try a different approach."
                            )
                        ))
                    inner_res = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res

                    is_failure = (
                        (isinstance(inner_res, dict) and inner_res.get("success") is False)
                        or (isinstance(tool_res, dict) and tool_res.get("status_code", 200) not in (200, 201))
                        or (isinstance(tool_res, dict) and bool(tool_res.get("error")))
                        or (isinstance(inner_res, dict) and bool(inner_res.get("error")) and not inner_res.get("success", True))
                        or (isinstance(tool_res, dict) and tool_res.get("return_code", 0) != 0)
                        or (isinstance(inner_res, dict) and inner_res.get("return_code", 0) != 0)
                        or "crashed" in status_done_line.lower()
                        or "⚠" in clean_result_str
                    )
                    status_meta = "failure" if is_failure else "success"
                    tool_close_tag = f"{status_done_line}{details_block}<!-- status:{status_meta} -->\n</processing>\n\n"
                    ai_msg.content += tool_close_tag
                    _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    tool_success = not is_failure

                    if tool_success :
                        successful_tool_signatures.add(context_aware_signature)
                        ASCIIColors.info(f"[ChatMixin] Recorded successful signature for '{tool_name}'. Total successful: {len(successful_tool_signatures)}")

                    tool_calls_this_turn.append({
                        "name": tool_name,
                        "params": tool_params,
                        "result": {"output": clean_result_str, "success": tool_success}
                    })

                    # ── 📊 LOG TOOL CALL ACTION ──
                    turn_actions_log.append({
                        "action": "tool_call",
                        "tool_name": tool_name,
                        "success": tool_success,
                        "round": round_count
                    })

                    # ── 🔄 COMPRESS VIRTUAL HISTORY IF NEEDED ──
                    # After adding the tool result, check if we need to compress older rounds
                    _compress_virtual_history_if_needed()

                    # ── 🛑 SUCCESS LOOP DETECTION & PREVENTION ─────────────────────
                    # Check if the LAST assistant message in history was a tool call to the SAME tool
                    # This prevents the LLM from getting stuck in a "success loop"
                    last_assistant_msg = virtual_history[-3] if len(virtual_history) >= 3 else None

                    # Always append the tool result to the conversational history so the LLM can see the output
                    if tool_success:
                        # Extract explicit filename if returned in the result dictionary
                        real_filename_instr = ""
                        if isinstance(tool_res, dict) and tool_res.get("plot_filename"):
                            p_fn = tool_res["plot_filename"]
                            real_filename_instr = (
                                f"🚨 **ACTUAL GENERATED FILE NAME**: `{p_fn}`\n"
                                f"   You must reference this exact file in your final answer using:\n"
                                f"   `<artefact_image id=\"{p_fn}::0\" />` or `<img src=\"/api/workspace_files/{p_fn}\" />`\n"
                                f"   Do NOT hallucinate or guess any other file name (such as 'sales_over_time.png'). Only use `{p_fn}`.\n\n"
                            )

                        # Check if this is a data query tool and guide the LLM to the next phase
                        next_step_guidance = ""
                        if tool_name in ("tool_query_database_sql", "tool_execute_sql_query", "tool_execute_python_data_query"):
                            next_step_guidance = (
                                f"6. 📊 **DATA GATHERED → BUILD PHASE**: You now have enough data to proceed. "
                                f"If you have gathered sufficient data for the user's request, your NEXT action should be to "
                                f"either:\n"
                                f"   a) Write a Python script artifact to process/visualize the data, OR\n"
                                f"   b) Build the HTML animation artifact the user requested, OR\n"
                                f"   c) Provide your final analysis answer.\n"
                                f"   Do NOT run another SQL query unless you need genuinely different data.\n"
                            )

                        user_part = (
                            f"=== ✅ TOOL RESULT (NOT A TOOL CALL): {tool_name} ===\n"
                            f"⚠️ **WARNING**: The JSON below is the **RESULT** of your previous tool call. "
                            f"It is **NOT** a new tool call request. Do **NOT** re-execute it.\n\n"
                            f"{real_filename_instr}"
                            f"<tool_result name=\"{tool_name}\" status=\"SUCCESS\">\n"
                            f"{clean_result_str}\n"
                            f"</tool_result>\n\n"
                            f"🚨 **MANDATORY NEXT STEPS**:\n"
                            f"1. ✅ **ACKNOWLEDGE** the data above is already retrieved.\n"
                            f"2. 🧠 **ANALYZE** the result: What does it tell you?\n"
                            f"3. 💬 **RESPOND** to the user's original question using this data.\n"
                            f"4. 🚫 **FORBIDDEN**: Do **NOT** call '{tool_name}' again with these parameters.\n"
                            f"   The tool already ran successfully. Calling it again is a **LOOP ERROR**.\n"
                            f"5. 🔀 If you need MORE data, call a **DIFFERENT** tool or ask a **DIFFERENT** question.\n"
                            f"6. 🏁 **TERMINATION**: When you have finished your task and written your final answer, you MUST end your generation with a `<done/>` tag on a new line.\n"
                            f"{next_step_guidance}\n"
                            f"### Example of CORRECT behavior:\n"
                            f"❌ WRONG: <tool>{{\"name\": \"{tool_name}\", ...}}</tool>  (LOOP!)\n"
                            f"✅ RIGHT:  \"Based on the results, I can see that...\"  (ANSWER!)\n"
                        )

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=user_part
                        ))

                        # If the tool created new files, we must update the virtual_history
                        # so the LLM knows they exist. To preserve the KV-cache, we append
                        # a system marker to the LAST user message we just added.
                        new_files_this_run = [a.get("title") for a in self._affected_artefacts_this_turn if a.get("title")]
                        if new_files_this_run:
                            new_files_str = ", ".join(f"`{f}`" for f in new_files_this_run)
                            # Mutate the last user message in-place to inject the artifact update
                            virtual_history[-1].content += (
                                f"\n\n[SYSTEM: New artifacts available in workspace: {new_files_str}. "
                                f"You can read or reference these files in your next steps.]"
                            )
                            ASCIIColors.info(f"[ChatMixin] Injected {len(new_files_this_run)} new artifacts into virtual_history context.")

                        # Inject a summary of what has been accomplished so far to prevent
                        # the LLM from re-starting its analysis from scratch each round.
                        tools_so_far = [tc["name"] for tc in tool_calls_this_turn]
                        unique_tools = list(dict.fromkeys(tools_so_far))
                        progress_summary = (
                            f"\n\n[SYSTEM: PROGRESS TRACKER — You have completed {len(tool_calls_this_turn)} tool call(s) so far: "
                            f"{', '.join(unique_tools)}. "
                            f"You DO NOT need to re-explore the data. Use the results already in your context to proceed. "
                            f"If the user asked you to build something (e.g., an animation, chart, or report), your NEXT step "
                            f"should be to CREATE that artifact using the data you have already gathered. "
                            f"Do NOT re-run the same exploratory queries.]"
                        )
                        virtual_history[-1].content += progress_summary
                    else:
                        user_part = (
                            f'<tool_result name="{tool_name}" status="FAILED">\n'
                            f"{clean_result_str}\n"
                            f"</tool_result>\n\n"
                            f"⚠️ **Tool Execution Failed.**\n"
                            f"The tool '{tool_name}' encountered an error. Here is your mandatory protocol:\n"
                            f"1. **Analyze**: Read the error log above carefully to understand why it failed.\n"
                            f"2. **Explore Alternatives**: If there is another way to accomplish the task (e.g., using a different tool, modifying the parameters, or fixing the data), you MUST attempt it.\n"
                            f"3. **Inform the User**: If you cannot find an alternative approach, you MUST gracefully inform the user about the failure. "
                            f"Clearly explain what you were trying to do, why it failed (based on the error), and explicitly tell the user what they can do to help (e.g., provide a missing file, change a configuration, or grant permissions)."
                        )
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=user_part
                        ))
                    continue
                else:
                    break
            else:
                full_round_text = ss.get_clean_text_so_far()
                raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text

                clean_history_text = re.sub(r'<processing[^>]*>.*?(?:</processing>|$)', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE)
                clean_history_text = re.sub(r'<!-- status:[^>]*-->', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'</processing>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<lollms_artifact[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)
                clean_history_text = re.sub(r'<artefact_image[^/]*/>', '', clean_history_text, flags=re.IGNORECASE)

                virtual_history.append(SimpleNamespace(
                    sender_type="assistant",
                    content=clean_history_text.strip()
                ))

                text_only_stall_count = getattr(self, "_consecutive_text_only_stalls", 0) + 1
                object.__setattr__(self, "_consecutive_text_only_stalls", text_only_stall_count)

                if text_only_stall_count >= 3:
                    ASCIIColors.warning(f"[ChatMixin] Terminating after {text_only_stall_count} consecutive text-only stalls without <done/> or actions. The LLM is stuck.")
                    break

                ASCIIColors.warning(f"[ChatMixin] Text-only stall detected (#{text_only_stall_count}). LLM stopped without <done/> or actions. Forcing continuation.")

                if ss.context_unlock_requested and not was_cancelled:
                    unlock_files_str = ', '.join(ss.context_unlocked_files)
                    continuation_prompt = f"[SYSTEM: The following files have been processed: {unlock_files_str}. You can now read their full content and use them. Please continue your task.]"
                    ss.context_unlock_requested = False
                else:
                    continuation_prompt = (
                        "[SYSTEM: CRITICAL. You stopped generation without emitting a <done/> tag and without executing any tool or artifact. "
                        "If your task is complete, output your final conversational summary and end with a <done/> tag on a new line. "
                        "If you need to continue working, emit the next `<tool>` or `<artifact>` tag NOW. "
                        "Do NOT write prose preambles like 'Je vais...' or 'Let me...' without following through with the actual action tag.]"
                    )

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=continuation_prompt
                ))
                continue

        # ── 11. Final Post-Processing & Database Commit ──

        # Handle cancellation cleanup
        if was_cancelled:
            if ai_msg.content.strip():
                ai_msg.content += "\n\n[Generation cancelled by user]"
            else:
                ai_msg.content = "[Generation cancelled by user]"
            ai_msg.metadata = {
                "mode": "cancelled",
                "tool_calls": tool_calls_this_turn,
                "artefacts_modified": [a.get("title") for a in (ss.affected_artefacts if ss else [])],
                "cancelled": True
            }
        else:
            # ── 🧠 DUAL-COPY PERSISTENCE PROTOCOL ──
            # If this turn involved multiple agentic steps (tool calls or artifact dispatches),
            # we persist the FULL virtual_history into the message metadata.
            # This allows the next turn's export() to reconstruct the exact KV-cache state
            # so the LLM can continue multi-turn sequences without losing the path.
            has_virtual_history = len(virtual_history) > 0 and (
                any(vh.sender_type == "user" and "<tool_result" in (vh.content or "") for vh in virtual_history)
                or any(vh.sender_type == "assistant" and "<tool" in (vh.content or "") for vh in virtual_history)
                or any("SYSTEM MARKER MIMICRY DETECTED" in (vh.content or "") for vh in virtual_history)
            )

            ai_msg.metadata = {
                "mode": "agentic" if tool_calls_this_turn else "direct",
                "tool_calls": tool_calls_this_turn,
                "artefacts_modified": [a.get("title") for a in (ss.affected_artefacts if ss else [])],
            }

            if has_virtual_history:
                # Store the virtual history as a list of serializable dicts
                ai_msg.metadata["virtual_history"] = [
                    {"sender_type": vh.sender_type, "content": vh.content}
                    for vh in virtual_history
                ]

        if remove_thinking_blocks:
            ai_msg.content = self.lollmsClient.remove_thinking_blocks(ai_msg.content)

        # The Dual-Stream Buffer architecture now ensures raw <artifact> XML 
        # never enters ai_msg.content in the first place, so no post-generation
        # regex cleanup is required.

        # ── 🛡️ AUTO-CORRECT HALLUCINATED FILENAMES ──
        # Scan through the tool executions of this turn and fix any mismatched filenames
        for tc in tool_calls_this_turn:
            if tc.get("result") and tc["result"].get("success"):
                out_str = str(tc["result"].get("output", ""))
                # Locate real plot filename inside the output logs
                match_fn = re.search(r'plot_filename":\s*"([^"]+)"', out_str) or re.search(r'plot_filename:\s*(\S+)', out_str)
                if match_fn:
                    real_fn = match_fn.group(1).strip().strip("'\"")
                    # Dynamically replace hallucinated filenames (like sales_over_time, plot.png) inside image/artifact tags
                    ai_msg.content = re.sub(
                        r'(src|id)=["\'](?:[^"\']*/)?(?:sales_over_time|plot|chart|visualization)\.(?:png|jpg|jpeg)(?:::\d+)?["\']',
                        f'\\1="{real_fn}::0"',
                        ai_msg.content,
                        flags=re.IGNORECASE
                    )
                    # Also replace plain markdown/HTML source references if outputted as plain text
                    ai_msg.content = re.sub(
                        r'src=["\'](?:/api/workspace_files/)?(?:sales_over_time|plot|chart|visualization)\.png["\']',
                        f'src="/api/workspace_files/{real_fn}"',
                        ai_msg.content,
                        flags=re.IGNORECASE
                    )
                    ai_msg.content = ai_msg.content.replace("sales_over_time.png", real_fn)

        # Process memories (only if memory is enabled)
        mem_cleaned, mem_report = ai_msg.content, {}
        if enable_memory and _mm:
            mem_cleaned, mem_report = self._process_memory_tags(ai_msg.content, _mm, callback)
            if mem_cleaned != ai_msg.content:
                ai_msg.content = mem_cleaned

        # ── 🔍 INJECT SEARCH RESULTS ──
        # If the LLM searched archived memories, inject the results into the context
        # so it can see what was found and potentially load relevant memories
        if mem_report.get("searches"):
            for search_result in mem_report["searches"]:
                query = search_result.get("query", "")
                level = search_result.get("level")
                results = search_result.get("results", [])

                if results:
                    # Build a context block with the search results
                    search_context = f"\n[MEMORY SEARCH RESULTS for query: '{query}']\n"
                    if level is not None:
                        level_names = {1: "Working", 2: "Deep", 3: "Archived"}
                        search_context += f"Searched in: {level_names.get(level, f'Level {level}')} Memory\n"
                    search_context += f"Found {len(results)} matching memories:\n\n"

                    for idx, mem in enumerate(results, 1):
                        # Handle both dict and object access patterns safely
                        if isinstance(mem, dict):
                            mem_id = mem.get("id", "")[:8]
                            content = mem.get("content", "")[:200]  # Truncate long content
                            importance = mem.get("importance", 0)
                            tags = mem.get("tags", "")
                        else:
                            # If it's an object, use attribute access
                            mem_id = getattr(mem, "id", "")[:8]
                            content = getattr(mem, "content", "")[:200]
                            importance = getattr(mem, "importance", 0)
                            tags = getattr(mem, "tags", "")

                        search_context += f"{idx}. [{mem_id}] (importance: {importance:.0%}) {content}"
                        if tags:
                            search_context += f"  #{tags.replace(',', ' #')}"
                        search_context += "\n"

                    search_context += "\nYou can load any of these memories into Working Memory using <mem_load id=\"ID\" />\n"
                    search_context += "[END MEMORY SEARCH RESULTS]\n"

                    # Append to the AI message content so it's visible in the next round
                    ai_msg.content += search_context

                    ASCIIColors.info(f"[ChatMixin] Injected {len(results)} memory search results for query: '{query}'")

        # ── 🧠 SELECTIVE EPISODIC MEMORY SAVING (CONDITIONAL) ──
        # Only save episodic memory if:
        # 1. Memory system is enabled (enable_memory=True), AND
        # 2. Episodic memory is explicitly enabled (enable_episodic_memory=True), AND
        # 3. Memory manager exists (_mm is not None), AND
        # 4. The conversation is substantial enough (not trivial exchanges)
        if enable_memory and _mm and enable_episodic_memory:
            try:
                # Calculate conversation significance
                user_msg_length = len(user_message.strip())
                ai_msg_length = len(ai_msg.content.strip())
                total_length = user_msg_length + ai_msg_length

                # Only save if:
                # 1. The conversation is substantial (>200 chars total), OR
                # 2. Tools were used (indicating a task was performed), OR
                # 3. Artifacts were created (indicating work was done), OR
                # 4. The conversation contains meaningful content (not just greetings)

                should_save_episodic = False

                # Check for substantial content
                if total_length > 200:
                    should_save_episodic = True

                # Check for tool usage
                if tool_calls_this_turn:
                    should_save_episodic = True

                # Check for artifact creation
                if ss and ss.affected_artefacts:
                    should_save_episodic = True

                # Check for meaningful keywords (not just greetings)
                trivial_patterns = [
                    r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)\s*[.!?]*$',
                    r'^(thanks|thank you|thx|ty)\s*[.!?]*$',
                    r'^(ok|okay|k|alright|sure)\s*[.!?]*$',
                    r'^(yes|no|yeah|nope)\s*[.!?]*$',
                ]
                is_trivial = any(re.match(pattern, user_message.strip().lower()) for pattern in trivial_patterns)

                if is_trivial and not tool_calls_this_turn and not (ss and ss.affected_artefacts):
                    should_save_episodic = False

                if should_save_episodic:
                    self._save_episodic_memory_turn(user_message, ai_msg.content, _mm)
                    ASCIIColors.info(f"[ChatMixin] Saved episodic memory (length: {total_length} chars, tools: {len(tool_calls_this_turn)}, artifacts: {len(ss.affected_artefacts) if ss else 0})")
                else:
                    ASCIIColors.debug(f"[ChatMixin] Skipped episodic memory (trivial exchange, length: {total_length} chars)")

            except Exception as ex:
                trace_exception(ex)
        elif _mm and not enable_episodic_memory:
            ASCIIColors.debug(f"[ChatMixin] Episodic memory saving disabled via enable_episodic_memory=False")

        # Update metadata for alternating exports
        # CRITICAL: Preserve virtual_history if it was set in the cancellation/non-cancellation block above.
        # We only update the mode and counts here to avoid overwriting the persisted virtual history.
        existing_virtual_history = ai_msg.metadata.get("virtual_history")
        ai_msg.metadata = {
            "mode": "agentic" if tool_calls_this_turn else "direct",
            "tool_calls": tool_calls_this_turn,
            "artefacts_modified": [a.get("title") for a in (ss.affected_artefacts if ss else [])]
        }
        if existing_virtual_history:
            ai_msg.metadata["virtual_history"] = existing_virtual_history

        # Auto dream (only if memory is enabled)
        dream_report = None
        if enable_memory and enable_auto_dream and _mm is not None:
            try:
                dream_report = _mm.dream(self.lollmsClient)
            except Exception as ex:
                trace_exception(ex)

        if self._is_db_backed and self.autosave:
            self.commit()

        self.scratchpad = ""
        object.__setattr__(self, '_active_callback', None)

        # 🛡️ CRITICAL FIX: Always reset the cancellation flag at the end of the turn.
        # This ensures that pre-turn and mid-turn cancellation signals are consumed
        # and do not bleed into subsequent turns.
        self.reset_cancel_state()

        # ── 🔬 SCIENTIFIC DEBUG: EXPORT CONTEXT DUMP ──
        # Dumps the exact virtual_history (LLM context) and ai_msg.content (UI context)
        # to a JSON file in the discussion workspace to verify context separation.
        if debug_export:
            try:
                import os as _os
                from pathlib import Path as _Path
                import json as _json
                from datetime import datetime as _dt

                debug_dir = _Path(self.workspace_data_path) / "_debug_dumps"
                debug_dir.mkdir(parents=True, exist_ok=True)

                timestamp = _dt.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                dump_file = debug_dir / f"turn_dump_{timestamp}.json"

                # Safely serialize SimpleNamespace objects in virtual_history
                vh_serializable = []
                for m in virtual_history:
                    if hasattr(m, '__dict__'):
                        vh_serializable.append({
                            "sender_type": getattr(m, "sender_type", "unknown"),
                            "content": getattr(m, "content", "")
                        })
                    elif isinstance(m, dict):
                        vh_serializable.append(m)

                dump_payload = {
                    "timestamp": timestamp,
                    "discussion_id": self.id,
                    "round_count": round_count,
                    "was_cancelled": was_cancelled,
                    "virtual_history_length": len(vh_serializable),
                    "virtual_history": vh_serializable,
                    "ai_message_content": ai_msg.content,
                    "ai_message_metadata": ai_msg.metadata
                }

                with open(dump_file, "w", encoding="utf-8") as f:
                    _json.dump(dump_payload, f, indent=2, default=str, ensure_ascii=False)

                ASCIIColors.info(f"[ChatMixin] 🔬 Debug context dump saved to: {dump_file}")
            except Exception as dump_err:
                ASCIIColors.warning(f"[ChatMixin] Failed to write debug context dump: {dump_err}")

        return {
            "user_message": user_msg,
            "ai_message": ai_msg,
            "sources": [],
            "artefacts": ss.affected_artefacts if ss else [],
            "memory_report": mem_report,
            "dream_report": dream_report,
            "was_cancelled": was_cancelled
        }

            
            
            
# ── Internal parsing helpers ──

def _format_form_answers_for_llm(form_descriptor: Dict, answers: Dict[str, Any]) -> str:
    lines = [
        f"### 📋 Form Submission: {form_descriptor.get('title', 'User Form')}",
        "",
    ]
    fields = form_descriptor.get("fields", [])
    field_map = {f["name"]: f for f in fields if f.get("type") != "section"}

    for name, value in answers.items():
        label = field_map.get(name, {}).get("label", name)
        lines.append(f"* **{label}**: {value}")

    lines.append("\n*Form submitted successfully.*")
    return "\n".join(lines)


def _parse_form_xml(tag_attrs_str: str, body: str) -> Optional[Dict[str, Any]]:
    def _parse_attrs(s: str) -> Dict[str, str]:
        return {m.group(1): m.group(2)
                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', s)}

    top_attrs = _parse_attrs(tag_attrs_str)

    form: Dict[str, Any] = {
        "id":           str(uuid.uuid4()),
        "title":        top_attrs.get("title", "Please fill in the form"),
        "description":  top_attrs.get("description", ""),
        "submit_label": top_attrs.get("submit_label", "Submit"),
        "fields":       [],
    }

    body_stripped = body.strip()

    if body_stripped.startswith("{") or body_stripped.startswith("["):
        try:
            parsed = json.loads(body_stripped)
            if isinstance(parsed, dict):
                form.update({k: v for k, v in parsed.items() if k != "id"})
                if "fields" not in form:
                    form["fields"] = []
                return form
        except json.JSONDecodeError as ex:
            trace_exception(ex)

    field_pattern = re.compile(
        r'<(?:field|section)\s([^/]*?)(?:/\s*>|>.*?</(?:field|section)>)',
        re.DOTALL | re.IGNORECASE,
    )
    fields_found = []
    for m in field_pattern.finditer(body_stripped):
        attrs = _parse_attrs(m.group(1))
        field: Dict[str, Any] = {
            "name":    attrs.get("name", f"field_{len(fields_found)}"),
            "label":   attrs.get("label", attrs.get("name", f"Field {len(fields_found)+1}")),
            "type":    attrs.get("type", "text"),
            "required": attrs.get("required", "true").lower() not in ("false", "0", "no"),
        }
        for num_key in ("min", "max", "step", "rows", "min_rating", "max_rating"):
            if num_key in attrs:
                try:
                    field[num_key] = float(attrs[num_key]) if '.' in attrs[num_key] \
                                     else int(attrs[num_key])
                except ValueError:
                    field[num_key] = attrs[num_key]
        for str_key in ("default", "placeholder", "hint", "accept", "language",
                        "category", "options"):
            if str_key in attrs:
                field[str_key] = attrs[str_key]
        if "options" in field and isinstance(field["options"], str):
            field["options"] = [o.strip() for o in field["options"].split(",") if o.strip()]
        if "multiple" in attrs:
            field["multiple"] = attrs["multiple"].lower() not in ("false", "0", "no")
        fields_found.append(field)

    if fields_found:
        form["fields"] = fields_found
        return form

    question_re = re.compile(r'^[-*\d.]+\s+(.+)', re.MULTILINE)
    questions = question_re.findall(body_stripped)
    if questions:
        form["fields"] = [
            {
                "name":     f"q{i+1}",
                "label":    q.strip().rstrip("?:"),
                "type":     "textarea",
                "required": True,
                "rows":     3,
            }
            for i, q in enumerate(questions)
        ]
        return form

    ASCIIColors.warning(f"[Form] Could not parse form body. Returning empty form.")
    return form