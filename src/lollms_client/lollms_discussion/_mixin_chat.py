# lollms_discussion/_mixin_chat.py
# ─────────────────────────────────────────────────────────────────────────────
# ChatMixin — High-performance single-agent conversational turn loop with 
#             dynamic in-process Spinoff Sub-Agent Tools.
#
# Resolves RAG pre-hydration, tiered memory, and direct inline tool calls,
# exposing specialized sub-agents as executable tools to preserve KV-cache.

import re
import json
import html
import uuid
import random
import base64
import sqlite3
import hashlib
import inspect
import os
import time
import traceback
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from types import SimpleNamespace
from datetime import datetime
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE, EventMode
from ._message import LollmsMessage
from lollms_client.lollms_artefact import ArtefactType, make_image_id, ArtefactVisibility, ArtefactStatus
from lollms_client.lollms_memory import FailureMemory
from lollms_client.lollms_chat_core import _is_tool_binding
from lollms_client.lollms_artefact import ArtefactVisibility, ArtefactType
from ._context_sanitizer import scrub_processing_and_status_blocks

_MAX_BRACKET_BUF = 256

from lollms_client.lollms_chat_core import (
    sanitize_unicode as _sanitize_unicode,
    sanitize_host_paths as _sanitize_host_paths,
    is_large_base64 as _is_large_base64,
    repair_llm_json as _repair_llm_json,
    calculate_dynamic_tool_char_limit as _calculate_dynamic_tool_char_limit,
    sanitize_tool_result as _sanitize_tool_result,
    build_windowed_output_preview as _build_windowed_output_preview,
    detect_structural_symbols as _detect_structural_symbols,
    extract_artefact_meta as _extract_artefact_meta,
    dump_error as _core_dump_error,
)

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
    "<delegate":      ("delegation_start",    MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</delegate>"),
    "<agent":         ("agent_spawn",         MSG_TYPE.MSG_TYPE_INFO,           MSG_TYPE.MSG_TYPE_INFO,              "</agent>"),
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


def _resolve_tool_workspace_root(discussion) -> "Path":
    if getattr(discussion, "workspace_data_path", None):
        return Path(discussion.workspace_data_path).resolve()
    base_ws = Path(discussion.workspace_path) if getattr(discussion, "workspace_path", None) else Path("./data_workspace")
    return (base_ws / discussion.id / "workspace_data").resolve()


def _hash_workspace_file_refs(discussion, params: Dict[str, Any]) -> Dict[str, str]:
    """
    Builds a context fingerprint from tool parameters that reference existing
    workspace files. A tool call is only a repetition when the workspace state
    it depends on is unchanged.
    """
    ws_root = _resolve_tool_workspace_root(discussion)

    def _sanitize_candidate(value: str) -> str:
        cleaned = value.replace("\\", "/").strip()
        for prefix in ("./workspace/", "./data_workspace/", "workspace/", "data_workspace/"):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        disc_id = getattr(discussion, "id", "")
        if disc_id and cleaned.lower().startswith(disc_id.lower() + "/"):
            cleaned = cleaned[len(disc_id) + 1:]
        return cleaned

    def _resolve_candidate(value: str) -> Optional[Path]:
        sanitized = _sanitize_candidate(value)
        candidates = []
        raw_path = Path(value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(ws_root / sanitized)
            candidates.append(Path.cwd() / sanitized)
        for cand in candidates:
            try:
                resolved = cand.resolve()
                if not str(resolved).startswith(str(ws_root)) and not raw_path.is_absolute():
                    continue
                if resolved.is_file():
                    return resolved
            except Exception:
                continue
        return None

    hashes: Dict[str, str] = {}
    for key, value in params.items():
        items = value if isinstance(value, list) else [value]
        found = []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            resolved = _resolve_candidate(item)
            if resolved is None:
                continue
            try:
                found.append(hashlib.md5(resolved.read_bytes()).hexdigest())
            except Exception:
                continue
        if found:
            hashes[key] = ",".join(found)
    return hashes

_BINARY_BLOB_KEYS = {
    "plot_b64", "image_b64", "audio_b64", "video_b64", "file_b64",
    "screenshot_b64", "pdf_b64", "thumbnail_b64", "base64",
    "binary", "raw_image", "image_data", "raw_data",
}

_EXPLICIT_BINARY_EXTS = {
    ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".xz", ".bz2", ".zst",
    ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx",
    ".h5", ".hdf5", ".gguf", ".pkl", ".pickle", ".joblib",
    ".npy", ".npz", ".msgpack", ".pb", ".tflite", ".mlmodel",
}

_ML_WEIGHT_EXTS = {
    ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx",
    ".h5", ".hdf5", ".gguf", ".pkl", ".pickle", ".joblib",
    ".npy", ".npz", ".msgpack", ".pb", ".tflite", ".mlmodel",
}

_MAX_TOOL_RESULT_CHARS = 24000

def _calculate_dynamic_tool_char_limit(client: Optional[Any] = None) -> int:
    """
    Calculates the maximum allowed characters for a tool result based on the LLM's context size.
    Uses 25% of the context window in characters, clamped between 16,000 and
    90,000 chars. Falls back to 24,000 chars if context size is unavailable.
    """
    if client and hasattr(client, 'get_ctx_size'):
        try:
            ctx_size = client.get_ctx_size() or 0
            if ctx_size > 0:
                dynamic_limit = int((ctx_size * 0.25) * 4)
                return min(max(dynamic_limit, 16000), 90000)
        except Exception:
            pass
    return 24000


import time as _time

from ._context_sanitizer import scrub_processing_and_status_blocks

def _scrub_for_llm_context(text: str) -> str:
    if not text:
        return ""
    cleaned = scrub_processing_and_status_blocks(text)
    cleaned = re.sub(r'<lollms_artifact[^/]*/>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<artefact_image[^/]*/>', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


_INTENT_ANNOUNCEMENT_RE = re.compile(
    r'(?im)^\s*(?:'
    r'i\s+will\b'
    r'|i\s+am\s+going\s+to\b'
    r'|i\'?m\s+going\s+to\b'
    r'|i\'?ll\b'
    r'|let\s+me\b'
    r'|let\'?s\b'
    r'|allow\s+me\b'
    r'|first[,.]?\s+(?:i\s+will|let\s+me|allow\s+me)\b'
    r'|now\s+(?:i\s+will|let\s+me|allow\s+me)\b'
    r'|next[,.]?\s+(?:i\s+will|let\s+me|allow\s+me)\b'
    r'|je\s+vais\b'
    r'|permettez[- ]moi\b'
    r'|laissez[- ]moi\b'
    r'|laisse[- ]moi\b'
    r'|je\s+commence\b'
    r')'
)


def _detect_intent_only_round(round_text: str) -> bool:
    """
    Deterministic detector for the announced-intent-without-action pathology.

    Returns True when a TEXT-ONLY round announces an upcoming action (using
    the exact finite intent-verb vocabulary taught by the system prompt
    doctrine) but the round performed no action at all. This is NOT an open
    NLP heuristic: it matches the closed set of first-person intent openers
    the SAME-RESPONSE EXECUTION MANDATE forbids ending a round on.
    """
    if not round_text:
        return False
    return bool(_INTENT_ANNOUNCEMENT_RE.search(round_text))

# Structural symbol and metadata extraction delegated to lollms_chat_core

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


# Tool sanitization and preview logic delegated to lollms_chat_core

_TOOL_UI_PREVIEW_WINDOW = 6000
_TOOL_UI_PREVIEW_HALF = _TOOL_UI_PREVIEW_WINDOW // 2


def _build_windowed_output_preview(text: str) -> str:
    """
    Builds a head/tail windowed preview of a tool output for UI display.

    Short outputs pass through unchanged. Long outputs show the head half-window,
    a stripping marker, then the tail half-window — so both the beginning and the
    end of the execution log remain visible without flooding the chat bubble.
    """
    if not isinstance(text, str) or len(text) <= _TOOL_UI_PREVIEW_WINDOW:
        return text if isinstance(text, str) else str(text)
    head = text[:_TOOL_UI_PREVIEW_HALF]
    tail = text[-_TOOL_UI_PREVIEW_HALF:]
    stripped_chars = len(text) - _TOOL_UI_PREVIEW_WINDOW
    return (
        f"{head}\n"
        f"... [stripped for brevity — {stripped_chars} middle characters omitted "
        f"(use read file tools to inspect the full output)] ...\n"
        f"{tail}"
    )


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
        enable_tools: bool = True,
        content_offset: int = 0,
        fast_artefact_replicas: Optional[List[str]] = None,
        processed_tags: Optional[set] = None,
        event_mode: EventMode = EventMode.PROCESSING_TAG_MODE,
        remove_thinking_blocks: bool = True,
    ):
        self.discussion = discussion
        self.callback = callback
        self.ai_message = ai_message
        self.enable_artefacts = enable_artefacts
        self.enable_in_message_status = enable_in_message_status
        self.auto_activate = auto_activate_artefacts
        self.content_offset = content_offset
        self.event_mode = event_mode
        self.remove_thinking_blocks = remove_thinking_blocks
        self.enable_tools = enable_tools

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
        self.processed_tags = processed_tags if processed_tags is not None else set()

        # Orchestrator→Worker delegation payload captured this round.
        self.delegation_payload: Optional[Dict[str, Any]] = None

        # ── ONE-ACTION-PER-TURN PROTOCOL ──
        # Ensures generation halts immediately after dispatching a single functional tag.
        self._action_dispatched = False

        # ── PATCH FAILURE TRACKING ──
        # Distinguishes a failed SEARCH/REPLACE patch (which should allow a correction round)
        # from a true duplicate artifact (which should hard-break the loop).
        self._last_dispatch_failed = False
        self._last_failure_kind: Optional[str] = None

        # ── DONE TAG DETECTION ──
        # Set to True when the LLM emits <done/> to signal explicit task termination.
        self._done_detected = False

        # ── TOOL-LESS PERSONA REFUSAL STATE ──
        # Set when a <tool> dispatch is refused because this agent tier has
        # no execution capability. The chat loop converts it into a
        # delegation-correction envelope instead of executing anything.
        self._tool_refusal_detected = False

        # Orchestrator→SubAgent <agent> tag payload captured this round.
        self.sub_agent_payload: Optional[Dict[str, Any]] = None

        # ── Generic Secondary Tag Interceptor State ──
        # Handles <skill>, <note>, <lollms_inline>, <lollms_form>, <generate_image>, <edit_image>, etc.
        # These tags don't need the specialized dual-stream artifact tracker, but DO need
        # full body buffering + closing-tag detection + dispatch to _dispatch_closed_tag.
        self._is_accumulating_secondary = False
        self._secondary_buffer = ""
        self._secondary_tag_name = ""      # e.g., "skill", "note"
        self._secondary_closing_tag = ""   # e.g., "</skill>"
        self._secondary_open_tag = ""      # e.g., '<skill title="...">'
        self._secondary_attrs = {}

        # ── PROCESSING BLOCK PARITY LATCH ──
        # True while a <processing> block is open in ai_message.content.
        # Guarantees every </processing> close has exactly one matching open:
        # the entry block sets it, and exactly one owner (dispatcher for
        # context tags, accumulator otherwise) clears it.
        self._processing_block_open = False
        # Heartbeat control for empty/slow artifacts
        self._artefact_heartbeat_thread: Optional[threading.Thread] = None
        self._artefact_heartbeat_stop = threading.Event()
        self._artefact_heartbeat_active = False
        self._artefact_received_content = False

        # Fast artefact replicas (user-provided or default)
        self._fast_artefact_replicas = fast_artefact_replicas if fast_artefact_replicas else _DEFAULT_FAST_REPLICAS

        # ── STREAM PARSER CORE STATE ──
        # Every attribute consumed by feed() / flush_remaining_buffer() must be
        # initialized here. A stream parser that reaches its first chunk with
        # missing state crashes the generation thread (AttributeError) and
        # wedges the turn with no ROUND_END ever emitted.
        self._pending_buffer = ""
        self._in_code_fence = False
        self._code_fence_buffer = ""
        self._code_fence_hold_buffer = ""
        self._in_inline_code = False
        self._is_accumulating_tool = False
        self._tool_buffer = ""
        self._artefact_buffer = ""

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

    @staticmethod
    def _opening_tag_is_malformed(buffer: str, tag_start_idx: int) -> bool:
        """
        A functional opening tag must be entirely contained on a single line.
        Returns True when a newline appears before the closing '>', meaning
        the tag can never complete (truncated/multi-line tag). Such content
        must be suppressed as prose, never intercepted.
        """
        end_of_tag_idx = buffer.find(">", tag_start_idx)
        newline_idx = buffer.find("\n", tag_start_idx)
        if end_of_tag_idx != -1 and newline_idx != -1 and newline_idx < end_of_tag_idx:
            return True
        if end_of_tag_idx == -1 and newline_idx != -1:
            return True
        return False

    def _suppress_malformed_tag(self, tag_start_idx: int) -> bool:
        """
        Handles an opening tag that violates the single-line protocol.

        The dead tag fragment (from tag_start_idx up to and including the
        first newline) is a known functional tag start that can never
        complete: it is suppressed instead of leaked into the content
        stream. Any prose after the newline is requeued into the pending
        buffer so normal tag detection resumes on the next chunk.
        """
        text_before = self._pending_buffer[:tag_start_idx]
        if text_before:
            self.ai_message.content += text_before
            _cb(self.callback, text_before, MSG_TYPE.MSG_TYPE_CHUNK)

        newline_idx = self._pending_buffer.find("\n", tag_start_idx)
        if newline_idx == -1:
            self._pending_buffer = ""
        else:
            self._pending_buffer = self._pending_buffer[newline_idx + 1:]
        return True

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
            return False

        # CRITICAL FIX: Append to shadow buffer instead of directly to ai_message.content
        self._pending_buffer += chunk

        # ── 🧹 THINKING BLOCK SUPPRESSION ──
        # If remove_thinking_blocks is True, we strip  ...  blocks from the live stream.
        if self.remove_thinking_blocks and not self._is_accumulating_tool and not self.artefact_tracker.is_inside_artefact and not self._is_accumulating_secondary and not self._in_code_fence:
            if "</think>" in self._pending_buffer:
                end_idx = self._pending_buffer.find(" ")
                if end_idx != -1:
                    # Discard the thought block entirely
                    self._pending_buffer = self._pending_buffer[end_idx + 8:]
                else:
                    # Wait for the closing tag in the next chunk
                    self._pending_buffer = ""
                    return True

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
                    self._in_code_fence = False
                    self._code_fence_buffer = ""
                    self._code_fence_hold_buffer = ""
                    self._in_inline_code = False

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
                        if "<<<<<<< SEARCH" in self._artefact_buffer:
                            self._last_dispatch_failed = True
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
                                detail = event_meta.get("detail")
                                if detail and not detail.startswith("Line "):
                                    status_tag = f'{event_meta["status"]}\n'
                                    self.ai_message.content += status_tag
                                    _cb(self.callback, status_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

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

                    if self._opening_tag_is_malformed(self._pending_buffer, tag_start_idx):
                        return self._suppress_malformed_tag(tag_start_idx)

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
                            self._in_code_fence = False
                            self._code_fence_buffer = ""
                            self._code_fence_hold_buffer = ""
                            self._in_inline_code = False

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
                                if "<<<<<<< SEARCH" in body_content:
                                    self._last_dispatch_failed = True
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
                    if self._opening_tag_is_malformed(self._pending_buffer, open_idx):
                        return self._suppress_malformed_tag(open_idx)
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
                    tag_start_idx = lower_buffer.find(tag_tag_prefix, open_idx)
                    if tag_start_idx == -1:
                        tag_start_idx = open_idx
                    if self._opening_tag_is_malformed(self._pending_buffer, tag_start_idx):
                        return self._suppress_malformed_tag(tag_start_idx)
                    end_of_tag_idx = self._pending_buffer.find(">", tag_start_idx)
                    if end_of_tag_idx != -1:
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
                            self._pending_buffer = self._pending_buffer[end_of_tag_idx+1:]
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

                if self._secondary_tag_name not in ("unlock_file", "lock_file", "hide_file", "agent", "generate_image", "edit_image"):
                        if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            proc_close_tag = f'\n<!-- status:finished -->\n</processing>\n'
                            self.ai_message.content += proc_close_tag
                            _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        self._processing_block_open = False

                remaining_text = self._secondary_buffer[close_idx + close_len:]

                self._secondary_tag_name = ""
                self._secondary_closing_tag = ""
                self._secondary_open_tag = ""
                self._secondary_buffer = ""
                self._secondary_attrs = {}

                if remaining_text.strip():
                    self._pending_buffer = remaining_text

                self._action_dispatched = True
                return False
            else:
                pass
            return True

        # ── Generic Secondary Tag Interception (<skill>, <note>, <lollms_inline>, etc.) ──
        if not is_inside_thoughts and not self._is_accumulating_secondary:
            lower_buffer = self._pending_buffer.lower()
            secondary_entered = False
            for tag_prefix in ("<agent", "<delegate", "<skill", "<note", "<scratchpad", "<lollms_form", "<generate_image", "<edit_image"):
                pattern = r'(?m)^\s*(?!`)(?!.*\|)' + re.escape(tag_prefix)
                open_match = re.search(pattern, lower_buffer)
                if open_match:
                    tag_start_idx = lower_buffer.find(tag_prefix, open_match.start())
                    if tag_start_idx == -1:
                        tag_start_idx = open_match.start()
                    if self._opening_tag_is_malformed(self._pending_buffer, tag_start_idx):
                        return self._suppress_malformed_tag(tag_start_idx)
                    end_of_tag_idx = self._pending_buffer.find(">", tag_start_idx)
                    if end_of_tag_idx != -1:
                        opening_tag = self._pending_buffer[tag_start_idx:end_of_tag_idx+1]

                        tag_name_match = re.match(r'<(\w+)', opening_tag)
                        if tag_name_match:
                            self._secondary_tag_name = tag_name_match.group(1).lower()
                            self._secondary_closing_tag = f"</{self._secondary_tag_name}>"
                            self._secondary_open_tag = opening_tag
                            self._is_accumulating_secondary = True

                            # Parse attributes for streaming metadata
                            attrs = {}
                            for m_attr in re.finditer(r'(\w+)=["\']([^"\']*)["\']', opening_tag):
                                attrs[m_attr.group(1).lower()] = m_attr.group(2)
                            self._secondary_attrs = attrs

                            text_before_tag = self._pending_buffer[:tag_start_idx]
                            if text_before_tag:
                                self.ai_message.content += text_before_tag
                                _cb(self.callback, text_before_tag, MSG_TYPE.MSG_TYPE_CHUNK)

                            self._secondary_buffer = opening_tag
                            self._pending_buffer = ""

                            tag_name = self._secondary_tag_name
                            if tag_name != "agent":
                                proc_type = self._secondary_tag_name
                                title_val = attrs.get("title") or attrs.get("name") or self._secondary_tag_name.capitalize()

                                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                    proc_open = f'\n<processing type="{proc_type}" title="{title_val}">\n'
                                    self.ai_message.content += proc_open
                                    self._processing_block_open = True
                                    _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                                    status_msg = _ARTEFACT_TYPE_MESSAGES.get(self._secondary_tag_name, f"✨ Processing {self._secondary_tag_name}...")
                                    status_line = f'{status_msg}\n'
                                    self.ai_message.content += status_line
                                    _cb(self.callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                                tag_info = _SECONDARY_TAG_MAP.get(f"<{self._secondary_tag_name}")
                                if tag_info:
                                    open_evt = tag_info[0]
                                    _cb(self.callback, "", MSG_TYPE.MSG_TYPE_INFO, {
                                        "type": open_evt,
                                        "title": title_val,
                                        "category": attrs.get("category", ""),
                                        "description": attrs.get("description", "")
                                    })

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
            chunk_delta = self._pending_buffer
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

                # Emit final *_DONE stream event
                sec_attrs = getattr(self, '_secondary_attrs', {}) or {}
                title_val = sec_attrs.get("title") or sec_attrs.get("name") or self._secondary_tag_name
                tag_info = _SECONDARY_TAG_MAP.get(f"<{self._secondary_tag_name}")
                if tag_info:
                    done_msg_type = tag_info[2]
                    _cb(self.callback, body_content.strip(), done_msg_type, {
                        "title": title_val,
                        "content": body_content.strip(),
                        "category": sec_attrs.get("category", ""),
                        "description": sec_attrs.get("description", ""),
                        "attrs": sec_attrs
                    })

                # Dispatch to _dispatch_closed_tag for processing
                if full_match_text not in self.processed_tags:
                    self.processed_tags.add(full_match_text)
                    self._dispatch_closed_tag(
                        self._secondary_tag_name,
                        self._secondary_open_tag,
                        body_content.strip(),
                        full_match_text
                    )

                if self._secondary_tag_name not in ("agent", "generate_image", "edit_image"):
                    if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                        proc_close_tag = f'\n<!-- status:finished -->\n</processing>\n'
                        self.ai_message.content += proc_close_tag
                        _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    self._processing_block_open = False

                remaining_text = self._secondary_buffer[close_idx + close_len:]

                self._secondary_tag_name = ""
                self._secondary_closing_tag = ""
                self._secondary_open_tag = ""
                self._secondary_buffer = ""
                self._secondary_attrs = {}

                if remaining_text.strip():
                    self._pending_buffer = remaining_text

                self._action_dispatched = True
                return False
            else:
                # Emit live *_CHUNK stream events as content arrives
                sec_attrs = getattr(self, '_secondary_attrs', {}) or {}
                title_val = sec_attrs.get("title") or sec_attrs.get("name") or self._secondary_tag_name
                tag_info = _SECONDARY_TAG_MAP.get(f"<{self._secondary_tag_name}")
                if tag_info and chunk_delta:
                    chunk_msg_type = tag_info[1]
                    _cb(self.callback, chunk_delta, chunk_msg_type, {
                        "title": title_val,
                        "chunk": chunk_delta,
                        "category": sec_attrs.get("category", ""),
                        "description": sec_attrs.get("description", "")
                    })
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

            if is_new is False and is_patch is False:
                existing_art = self.discussion.artefacts.get(title)
                if existing_art is not None:
                    existing_raw = existing_art.get("content") if isinstance(existing_art, dict) else None
                    if isinstance(existing_raw, (bytes, bytearray)):
                        existing_content = existing_raw.decode("utf-8", errors="ignore")
                    elif isinstance(existing_raw, str):
                        existing_content = existing_raw
                    else:
                        existing_content = ""
                    new_body_hash = hashlib.sha256(body.strip().encode("utf-8", errors="ignore")).hexdigest()
                    existing_hash = hashlib.sha256(
                        existing_content.strip().encode("utf-8", errors="ignore")
                    ).hexdigest()
                    if new_body_hash == existing_hash:
                        self._last_dispatch_failed = True
                        ASCIIColors.warning(
                            f"[StreamState] Redundant full-rewrite of '{title}' rejected "
                            f"(content identical to active version)."
                        )
                        if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            proc_close = (
                                f"\n* ⚠️ Redundant rewrite rejected: '{title}' already contains exactly this content.\n"
                                f"<!-- status:failure -->\n</processing>\n"
                            )
                            self.ai_message.content += proc_close
                            _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        return True

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
                    if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                        proc_open = f'\n<processing type="artefact" title="{title}" language="{lang or ""}">\n'
                        proc_body = f'* ❌ Failed to apply patch to artifact: {patch_err}\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close

                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    else:
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, "")
                    return True

                if art:
                    self.affected_artefacts.append(art)

                try:
                    _versions_root_patched = Path(
                        getattr(self.discussion, "workspace_path", "") or "."
                    ) / ".versions" / str(getattr(self.discussion, "id", ""))
                    if not _versions_root_patched.exists():
                        _versions_root_patched.mkdir(parents=True, exist_ok=True)
                    self.discussion.artefacts._sync_to_disk_workspace(
                        title=art.get("title", title),
                        content=art.get("content", patched),
                        version=art.get("version", 1),
                        atype=atype,
                        language=lang
                    )
                except Exception as sync_ex:
                    ASCIIColors.warning(
                        "[StreamState] Failed to immediately materialize patched artifact "
                        f"'{title}' to disk: {_sanitize_host_paths(str(sync_ex))}"
                    )

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
                # Log this action in the turn progress tracker. Action-window
                # recollection registration is performed by the chat loop at
                # the dispatch-hydration site, which owns round_count.
                if hasattr(self.discussion, '_turn_actions_log'):
                    self.discussion._turn_actions_log.append({
                        "action": "artifact_created",
                        "title": title,
                        "type": atype,
                        "round": getattr(self.discussion, '_current_round', 0)
                    })

            # ── 🛑 CRITICAL FIX: IMMEDIATE PHYSICAL MATERIALIZATION ──
            # The physical twin MUST exist on disk the instant the artifact is created.
            # Defense-in-depth: ensure the versioned storage directory exists before
            # the write, preventing FileNotFoundError on first-time nested paths.
            try:
                _versions_root = Path(
                    getattr(self.discussion, "workspace_path", "") or "."
                ) / ".versions" / str(getattr(self.discussion, "id", ""))
                if not _versions_root.exists():
                    _versions_root.mkdir(parents=True, exist_ok=True)
                self.discussion.artefacts._sync_to_disk_workspace(
                    title=art.get("title", title),
                    content=art.get("content", body),
                    version=art.get("version", 1),
                    atype=atype,
                    language=lang
                )
            except Exception as sync_ex:
                ASCIIColors.warning(
                    "[StreamState] Failed to immediately materialize artifact "
                    f"'{title}' to disk: {_sanitize_host_paths(str(sync_ex))}"
                )

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
            if not self.enable_tools:
                ASCIIColors.warning(
                    "[StreamState] Tool dispatch refused: this agent tier has "
                    "no execution capability. Only workers may execute tools."
                )
                self._tool_refusal_detected = True
                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    refused_block = (
                        '\n<processing type="tool" title="Tool Execution Refused">\n'
                        "* 🚫 Tool calls are not available to you. You coordinate; "
                        "workers execute. Delegate the work instead.\n"
                        '<!-- status:failure -->\n</processing>\n'
                    )
                    self.ai_message.content += refused_block
                    _cb(self.callback, refused_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                    _cb(self.callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                        "tool_name": "tool_refused",
                        "success": False,
                        "output": "",
                        "error": "Tool calls are not available to this agent tier. Delegate the work instead.",
                    })
                return True
            self.tool_trigger = True

            # ── ROBUST JSON PARSING & NORMALIZATION (CRITICAL FIX) ──
            # LLMs often hallucinate flat structures: {"name": "tool", "arg": "val"}
            # instead of nested: {"name": "tool", "parameters": {"arg": "val"}}
            # We MUST normalize this here to prevent execution failures.
            tool_name = ""

            def _sanitize_tool_json(raw_body: str) -> str:
                """
                Safely extracts the first valid JSON object from a tool body.
                Handles trailing backticks, markdown fences, stray prose, and
                multi-line payloads (SPARQL/code) emitted with literal
                newlines inside JSON string values.
                """
                stripped = raw_body.strip()
                if stripped.startswith("```"):
                    lines = stripped.splitlines()
                    if len(lines) >= 2:
                        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                stripped = stripped.strip("`").strip()

                try:
                    decoder = json.JSONDecoder()
                    obj, end_idx = decoder.raw_decode(stripped)
                    return json.dumps(obj, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    pass

                repaired = _repair_llm_json(stripped)
                try:
                    decoder = json.JSONDecoder()
                    obj, end_idx = decoder.raw_decode(repaired)
                    return json.dumps(obj, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    return repaired

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
                self.tool_json_data = _repair_llm_json(sanitized_body)
                ASCIIColors.warning(f"[StreamState] JSON decode failed: {je}. Applied LLM JSON repair as fallback.")

            # ── 🛑 CRITICAL FIX: IMMEDIATE UI FEEDBACK ──
            # Emit the processing block to the UI INSTANTLY when the </tool> tag closes.
            # This guarantees the user sees "Calling tool..." while the tool executes,
            # rather than waiting for the synchronous execution to finish.
            try:
                parsed_for_ui = json.loads(self.tool_json_data)
                ui_tool_name = parsed_for_ui.get("name", "unknown") if isinstance(parsed_for_ui, dict) else "unknown"
                ui_params = parsed_for_ui.get("parameters", {}) if isinstance(parsed_for_ui, dict) else {}
            except Exception:
                ui_tool_name = tool_name or "unknown"
                ui_params = {}

            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                escaped_params = html.escape(json.dumps(ui_params, default=str))
                tool_open_tag = f'\n<processing type="tool" title="Tool Execution: {ui_tool_name}" params="{escaped_params}">\n'
                self.ai_message.content += tool_open_tag
                _cb(self.callback, tool_open_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                status_line = f"* Calling local tool system for '{ui_tool_name}'...\n"
                self.ai_message.content += status_line
                _cb(self.callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_TOOL_START, {
                    "tool_name": ui_tool_name,
                    "parameters": ui_params,
                })

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

            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                proc_open = f'\n<processing type="note" title="{title}">\n'
                proc_body = f'* 🗒️ Note captured and saved to workspace.\n'
                proc_close = f'<!-- status:finished -->\n</processing>\n'
                proc_block = proc_open + proc_body + proc_close

                self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            else:
                self.ai_message.content = self.ai_message.content.replace(full_match_text, "")

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

            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                proc_open = f'\n<processing type="scratchpad" title="{title}">\n'
                proc_body = f'* 📝 Scratchpad updated and saved to workspace.\n'
                proc_close = f'<!-- status:finished -->\n</processing>\n'
                proc_block = proc_open + proc_body + proc_close

                self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            else:
                self.ai_message.content = self.ai_message.content.replace(full_match_text, "")

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
            tags_attr = attrs.get("tags", "")
            tags_list = [t.strip() for t in tags_attr.split(",") if t.strip()] if tags_attr else []
            is_patch = "<<<<<<< SEARCH" in body

            personality = getattr(self.discussion, '_active_personality', None)
            is_handbag = bool(
                personality 
                and getattr(personality, 'handbag_path', None) 
                and getattr(personality, 'skills_manager', None)
            )

            if is_handbag:
                # ── HANDBAG MODE: Save directly into handbag's skills/ directory ──
                skills_mgr = personality.skills_manager
                existing_skill = skills_mgr.skills.get(title.lower()) or (skills_mgr.search_skills(title)[0] if skills_mgr.search_skills(title) else None)

                if existing_skill and not existing_skill.modifiable:
                    self._last_dispatch_failed = True
                    proc_open = f'\n<processing type="skill" title="{title}">\n'
                    proc_body = f'* 🚫 BLOCKED: Skill \'{existing_skill.title}\' in handbag is marked as READ-ONLY (unmodifiable).\n'
                    proc_close = f'<!-- status:failure -->\n</processing>\n'
                    proc_block = proc_open + proc_body + proc_close
                    if full_match_text in self.ai_message.content:
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                    else:
                        self.ai_message.content += proc_block
                    return True

                if is_patch and existing_skill:
                    try:
                        patched_content = self.discussion.artefacts.apply_aider_patch(existing_skill.content, body)
                    except Exception as patch_err:
                        ASCIIColors.error(f"[StreamState] Handbag skill patch failed: {patch_err}")
                        self._last_dispatch_failed = True
                        proc_open = f'\n<processing type="skill" title="{title}">\n'
                        proc_body = f'* ❌ Failed to apply patch to handbag skill: {patch_err}\n'
                        proc_close = f'<!-- status:failure -->\n</processing>\n'
                        proc_block = proc_open + proc_body + proc_close
                        if full_match_text in self.ai_message.content:
                            self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                        else:
                            self.ai_message.content += proc_block
                        return True
                    skills_mgr.update_skill(
                        title=existing_skill.title,
                        content=patched_content,
                        description=desc or None,
                        category=cat or None,
                        tags=tags_list
                    )
                else:
                    if existing_skill:
                        skills_mgr.update_skill(
                            title=existing_skill.title,
                            content=body,
                            description=desc or None,
                            category=cat or None,
                            tags=tags_list
                        )
                    else:
                        skills_mgr.create_skill(
                            title=title,
                            content=body,
                            description=desc,
                            category=cat,
                            tags=tags_list,
                            visibility="loadable"
                        )

                skill_entry = {
                    "title": title,
                    "type": "skill",
                    "content": body,
                    "category": cat,
                    "description": desc,
                    "destination": "handbag"
                }
                self.affected_artefacts.append(skill_entry)

                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    proc_open = f'\n<processing type="skill" title="{title}">\n'
                    proc_body = f'* 🧠 Skill \'{title}\' created/updated in handbag \'{personality.handbag_path.name}\'.\n'
                    proc_close = f'<!-- status:finished -->\n</processing>\n'
                    proc_block = proc_open + proc_body + proc_close

                    if full_match_text in self.ai_message.content:
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                    else:
                        self.ai_message.content += proc_block

                    _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                else:
                    if full_match_text in self.ai_message.content:
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, "")

                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                    "type": "artifact_created",
                    "title": title,
                    "art_type": "skill"
                })
                return True

            else:
                # ── ARTEFACT MODE (Manual / Stateless Personality): Save as Discussion Artefact ──
                art_title = f"{title}.md" if not title.lower().endswith(".md") else title
                if is_patch:
                    existing = self.discussion.artefacts.get(art_title) or self.discussion.artefacts.get(title)
                    if existing:
                        try:
                            patched_content = self.discussion.artefacts.apply_aider_patch(existing["content"], body)
                            art = self.discussion.artefacts.update(
                                title=existing["title"],
                                new_content=patched_content,
                                bump_version=True,
                                active=self.auto_activate,
                                description=desc,
                                category=cat
                            )
                        except Exception as patch_err:
                            ASCIIColors.error(f"[StreamState] Skill artifact patch failed: {patch_err}")
                            art = self.discussion.artefacts.add(
                                title=art_title,
                                artefact_type=ArtefactType.SKILL,
                                content=body,
                                active=self.auto_activate,
                                description=desc,
                                category=cat
                            )
                    else:
                        art = self.discussion.artefacts.add(
                            title=art_title,
                            artefact_type=ArtefactType.SKILL,
                            content=body,
                            active=self.auto_activate,
                            description=desc,
                            category=cat
                        )
                else:
                    art = self.discussion.artefacts.add(
                        title=art_title,
                        artefact_type=ArtefactType.SKILL,
                        content=body,
                        active=self.auto_activate,
                        description=desc,
                        category=cat
                    )

                if art:
                    self.affected_artefacts.append(art)

                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    proc_open = f'\n<processing type="skill" title="{title}">\n'
                    proc_body = f'* 🧠 Skill \'{title}\' captured and saved as discussion artefact.\n'
                    proc_close = f'<!-- status:finished -->\n</processing>\n'
                    proc_block = proc_open + proc_body + proc_close

                    if full_match_text in self.ai_message.content:
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, proc_block)
                    else:
                        self.ai_message.content += proc_block

                    _cb(self.callback, proc_open, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    _cb(self.callback, proc_body, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                else:
                    if full_match_text in self.ai_message.content:
                        self.ai_message.content = self.ai_message.content.replace(full_match_text, "")

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

            # The entry interceptor opened the <processing> block; this dispatcher
            # is its single owner for context tags. Close it exactly once.
            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                proc_close = f'{status_line}{details_block}<!-- status:{status_meta} -->\n</processing>\n\n'
                if self._processing_block_open:
                    self.ai_message.content += proc_close
                    _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                    self._processing_block_open = False
                else:
                    proc_close_tag = f'\n<processing type="context_update" title="{action_verb} context files">\n'
                    self.ai_message.content += proc_close_tag
                    self.ai_message.content += proc_close
                    _cb(self.callback, proc_close_tag + proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            if self.event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_CONTEXT_UPDATE, {
                    "action": tag_name.replace("_file", ""),
                    "files": targets,
                    "status": status_meta,
                    "error": None,
                })

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
                    f"you MUST use a tool (SQL query, grep, or execute_python_code) to extract "
                    f"specific data from these files. Do NOT attempt to <unlock_file> them again."
                )

            return True

        # 6. Image Generation / Editing (Intercepted during streaming)
        elif tag_name in ("generate_image", "edit_image"):
            if not self.enable_artefacts:
                return True
            title = attrs.get("name") or attrs.get("title") or f"generated_image_{uuid.uuid4().hex[:6]}"
            prompt = (body or "").strip()
            tti = getattr(self.discussion.lollmsClient, "tti", None)
            if tti is None and not bool(getattr(self.discussion.lollmsClient, "tti_model_profiles_registry", None)):
                ASCIIColors.warning(
                    "[StreamState] <" + tag_name + "> tag received but no TTI binding is available."
                )
                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    failure_block = (
                        f"\n* ⚠️ Image generation requested ('{title}'), but no image engine is "
                        f"available in this session. Nothing was generated.\n"
                        f"<!-- status:failure -->\n</processing>\n"
                    )
                    self.ai_message.content += failure_block
                    _cb(self.callback, failure_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                return True

            width = 1024
            height = 1024
            try:
                width = max(64, min(int(attrs.get("width", 1024)), 4096))
            except (TypeError, ValueError):
                width = 1024
            try:
                height = max(64, min(int(attrs.get("height", 1024)), 4096))
            except (TypeError, ValueError):
                height = 1024

            if tag_name == "edit_image" and not prompt:
                failure_block = (
                    f"\n* ⚠️ Image edit requested ('{title}'), but no editing instructions were "
                    f"provided inside the tag. Nothing was modified.\n"
                    f"<!-- status:failure -->\n</processing>\n"
                )
                self.ai_message.content += failure_block
                _cb(self.callback, failure_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                return True
            if tag_name == "generate_image" and not prompt:
                prompt = title

            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                status_line = (
                    f"* 🎨 Generating image '{title}' ({width}×{height}) via the image engine...\n"
                    if tag_name == "generate_image"
                    else f"* 🎨 Editing image '{title}' with the image engine...\n"
                )
                self.ai_message.content += status_line
                _cb(self.callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            img_b64 = None
            generation_error = None
            try:
                if tag_name == "generate_image":
                    img_bytes = tti.generate_image(prompt=prompt, width=width, height=height)
                else:
                    source_b64 = None
                    source_art = self.discussion.artefacts.get(title)
                    if source_art and source_art.get("images"):
                        source_b64 = source_art["images"][-1]
                    if source_b64 is None:
                        source_b64 = self.ai_message.get_active_images()[-1] if self.ai_message.get_active_images() else None
                    if source_b64 is None:
                        raise ValueError(f"No source image available to edit for '{title}'.")
                    img_bytes = tti.edit_image(image=source_b64, prompt=prompt)
                if img_bytes:
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            except Exception as gen_ex:
                trace_exception(gen_ex)
                generation_error = _sanitize_host_paths(str(gen_ex))

            if img_b64:
                art = self.discussion.artefacts.add(
                    title=title,
                    artefact_type="image",
                    content=f"### Image: '{prompt[:200]}'\n\n<artefact_image id=\"{title}::0\" />",
                    images=[img_b64],
                    image_media_types=["image/png"],
                    active=self.auto_activate
                )
                if art:
                    self.affected_artefacts.append(art)
                    self.discussion._turn_actions_log.append({
                        "action": "artifact_created",
                        "title": title,
                        "type": "image",
                        "round": getattr(self.discussion, "_current_round", 0)
                    })

                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    result_block = (
                        f"* ✅ Image '{title}' generated and saved to the workspace.\n"
                        f"<!-- status:success -->\n</processing>\n"
                    )
                    self.ai_message.content += result_block
                    _cb(self.callback, result_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                anchor = f'<artefact_image id="{title}::0" />'
                self.ai_message.content += f"\n\n{anchor}\n"
                _cb(self.callback, f"\n\n{anchor}\n", MSG_TYPE.MSG_TYPE_CHUNK)

                _cb(self.callback, "", MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                    "type": "artifact_created",
                    "title": title,
                    "art_type": "image"
                })
            else:
                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    reason = generation_error or "The image engine returned no data."
                    failure_block = (
                        f"\n* ❌ Image '{title}' could not be generated. Reason: {reason}\n"
                        f"<!-- status:failure -->\n</processing>\n"
                    )
                    self.ai_message.content += failure_block
                    _cb(self.callback, failure_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

            self._action_dispatched = True
            return True

        # 7. Sub-Agent Spawning via <agent> tag (functional tag, NOT a tool call)
        elif tag_name == "agent":
            from lollms_client.lollms_agentic.sub_agent_spawner import (
                parse_agent_tag as _parse_agent_tag,
                repair_agent_tag_body as _repair_agent_tag_body,
            )

            full_opening_tag = attrs_str if attrs_str.startswith("<") else f"<{attrs_str}>"
            agent_config = _parse_agent_tag(full_opening_tag, body)
            if agent_config is None:
                repaired_task, repair_notes = _repair_agent_tag_body(full_opening_tag, body)
                if repaired_task:
                    agent_config = _parse_agent_tag(full_opening_tag, repaired_task)
                    if agent_config is not None and repair_notes:
                        ASCIIColors.info(
                            "[StreamState] Agent tag auto-repaired ("
                            + "; ".join(repair_notes)
                            + "); proceeding with delegated spawn."
                        )

            if agent_config is None:
                self._last_dispatch_failed = True
                self._last_failure_kind = "agent_tag"
                failure_block = (
                    '\n<processing type="agent_spawn" title="Invalid agent tag">\n'
                    "* ❌ The `<agent>` block was malformed: no `<task>` XML body found.\n"
                    "* Expected syntax (must contain a `<task>...</task>` XML block):\n"
                    "<agent name=\"worker\" system_prompt=\"...\" max_rounds=\"8\">\n"
                    "<task>\n"
                    "Self-contained instructions for the specialist.\n"
                    "</task>\n"
                    "<context_files>\n"
                    "filename.ext\n"
                    "</context_files>\n"
                    "</agent>\n"
                    "* Plain-text `=== TASK ===` wrappers are NOT valid inside `<agent>`; "
                    "the task MUST be wrapped in `<task>` XML tags.\n"
                    '<!-- status:failure -->\n</processing>\n'
                )
                self.ai_message.content += failure_block
                _cb(self.callback, failure_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                return True

            self.sub_agent_payload = {
                "config": agent_config,
                "worker_index": int(getattr(self.discussion, "_worker_counter", 0)) + 1,
            }
            return True

        # 7b. Orchestrator→Worker Delegation (legacy <delegate>)
        elif tag_name in ("delegate", "task_block"):
            self.delegation_payload = {
                "opening_tag": attrs_str,
                "body": body,
            }
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

    def was_tool_refusal_detected(self) -> bool:
        """Returns True if a <tool> dispatch was refused for a tool-less agent tier."""
        return self._tool_refusal_detected

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
            # Merge any body text still parked in the pending buffer (single-chunk
            # feeds return from the entry block before the close-check runs, so
            # the body lives in _pending_buffer until the next feed or this flush).
            self._secondary_buffer += self._pending_buffer
            self._pending_buffer = ""

            # Extract the body and closing tag precisely if the closing tag
            # already arrived inside the merged buffer.
            close_match = re.search(re.escape(self._secondary_closing_tag), self._secondary_buffer, re.IGNORECASE)
            if close_match:
                close_idx = close_match.start()
                close_len = close_match.end() - close_match.start()
                body_content = self._secondary_buffer[len(self._secondary_open_tag):close_idx]
                full_match_text = self._secondary_buffer[:close_idx + close_len]
                remaining_text = self._secondary_buffer[close_idx + close_len:]
            else:
                body_content = self._secondary_buffer[len(self._secondary_open_tag):]
                full_match_text = self._secondary_buffer + self._secondary_closing_tag
                remaining_text = ""

            self._is_accumulating_secondary = False

            if full_match_text not in self.processed_tags:
                self.processed_tags.add(full_match_text)
                self._dispatch_closed_tag(
                    self._secondary_tag_name,
                    self._secondary_open_tag,
                    body_content.strip(),
                    full_match_text
                )

            # Context visibility tags and image tags close their own
            # <processing> block with a status meta inside the dispatcher;
            # do not emit a duplicate close.
            if self._secondary_tag_name not in ("unlock_file", "lock_file", "hide_file", "agent", "generate_image", "edit_image"):
                if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                    proc_close_tag = f'\n<!-- status:finished -->\n</processing>\n'
                    self.ai_message.content += proc_close_tag
                    _cb(self.callback, proc_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                self._processing_block_open = False

            self._secondary_tag_name = ""
            self._secondary_closing_tag = ""
            self._secondary_open_tag = ""
            self._secondary_buffer = ""
            self._secondary_attrs = {}

            if remaining_text.strip():
                self._pending_buffer = remaining_text

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

        if self._action_dispatched:
            self._pending_buffer = ""
            return

        if self._pending_buffer and not self.artefact_tracker.is_inside_artefact \
                and not self._is_accumulating_tool and not self._is_accumulating_secondary:
            if not self._done_detected:
                pending_done_re = re.compile(r'(?i)<(?:done|end)\s*/?>')
                if pending_done_re.search(self._pending_buffer):
                    ASCIIColors.info("[StreamState] Post-flush sweep detected <done/> in pending buffer. Setting termination flag.")
                    self._done_detected = True
                    self._pending_buffer = pending_done_re.sub('', self._pending_buffer)

            complete_tag_re = re.compile(
                r'(?ms)^[ \t]*<(artifact|artefact|skill|note)\s([^>]*)>(.*?)</\1>',
                re.IGNORECASE,
            )
            parked_matches = list(complete_tag_re.finditer(self._pending_buffer))
            for parked_match in parked_matches:
                full_match_text = parked_match.group(0)
                if full_match_text in self.processed_tags:
                    continue
                self.processed_tags.add(full_match_text)
                opening_tag_end = full_match_text.index('>') + 1
                self._dispatch_closed_tag(
                    parked_match.group(1).lower(),
                    full_match_text[:opening_tag_end],
                    parked_match.group(3).strip(),
                    full_match_text,
                )
                self._action_dispatched = True
            if parked_matches:
                self._pending_buffer = complete_tag_re.sub('', self._pending_buffer)

            self._pending_buffer = re.sub(
                r'(?ms)^[ \t]*<(?:artifact|artefact|skill|note|scratchpad|lollms_inline|lollms_form|generate_image|edit_image|tool)\b[^>]*$',
                '',
                self._pending_buffer,
                flags=re.IGNORECASE,
            )
            self._pending_buffer = re.sub(
                r'(?ms)^[ \t]*<processing\b[^>]*$',
                '',
                self._pending_buffer,
                flags=re.IGNORECASE,
            )

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

                        is_patch = "<<<<<<< SEARCH" in body_content
                        is_truncated_patch = is_patch and not re.search(
                            r'^>{6,8}(?:\s*\w+)?\s*$', body_content, re.MULTILINE
                        )
                        is_truncated_full = not is_patch

                        if is_truncated_patch or is_truncated_full:
                            # ── TRUNCATED ARTIFACT REJECTION ──
                            # Generation stopped before the artifact body was
                            # complete (or before the final >>>>>>> REPLACE
                            # sentinel arrived). Registering this as a success
                            # teaches the model that prose-claims equal
                            # completed files, which is the root cause of
                            # phantom completion hallucinations. We flag the
                            # dispatch as failed and let ChatMixin inject a
                            # corrective round.
                            self._last_dispatch_failed = True
                            self._action_dispatched = True
                            ASCIIColors.warning(
                                "[StreamState] Truncated artifact intercepted "
                                f"(patch={is_patch}). Rejecting dispatch and flagging failure."
                            )

                            if self.event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                                proc_close = (
                                    f"\n* ⚠️ Artifact generation was INTERRUPTED before completion "
                                    f"(received {len(body_content)} chars). The file was NOT saved.\n"
                                    f"<!-- status:failure -->\n</processing>\n"
                                )
                                self.ai_message.content += proc_close
                                _cb(self.callback, proc_close, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                            self.artefact_tracker.close()
                            self._artefact_buffer = ""
                            self._pending_buffer = ""
                            return

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
                clean_leftover = self._pending_buffer.strip()
                if clean_leftover:
                    self.ai_message.content += clean_leftover
                    _cb(self.callback, clean_leftover, MSG_TYPE.MSG_TYPE_CHUNK)
            self._pending_buffer = ""

    def get_tool_call_json(self) -> Optional[str]:
        return self.tool_json_data if self.tool_trigger else None

    def get_clean_text_so_far(self) -> str:
        return self.ai_message.content

# ── ChatMixin Implementation ────────────────────────────────────────────────

class ChatMixin:
    """ChatMixin: orchestrates RAG, tiered memory, delegation, and alternating tool rounds."""

    _WORKSPACE_SCAN_CACHE_KEY = "_workspace_scan_cache"

    def _cached_workspace_file_scan(self, workspace_dir: Path):
        """
        Cached workspace file classification.

        The full `rglob('*')` walk plus suffix classification is expensive and
        was previously recomputed on every chat() call — including for every
        spawned Worker inside a delegation. The cache is invalidated by the
        workspace write revision (`_workspace_write_revision`), which is
        bumped whenever an artifact write, tool file mutation, or context
        visibility change occurs.

        Returns:
            (all_files, has_data_files, has_doc_files) — identical semantics
            to the previous inline scan.
        """
        data_extensions = {".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"}
        doc_extensions = {".pdf", ".docx", ".pptx", ".odt", ".epub", ".txt", ".md", ".json", ".yaml", ".xml"}

        revision = int(getattr(self, "_workspace_write_revision", 0))
        cache = getattr(self, "_workspace_scan_cache", None)
        if cache is not None and cache.get("revision") == revision:
            return cache["all_files"], cache["has_data_files"], cache["has_doc_files"]

        if not workspace_dir.exists():
            all_files: List[Path] = []
            has_data_files = False
            has_doc_files = False
        else:
            all_files = list(workspace_dir.rglob("*"))
            has_data_files = any(f.suffix.lower() in data_extensions for f in all_files if f.is_file())
            has_doc_files = any(f.suffix.lower() in doc_extensions for f in all_files if f.is_file())

        object.__setattr__(
            self, "_workspace_scan_cache",
            {"revision": revision, "all_files": all_files, "has_data_files": has_data_files, "has_doc_files": has_doc_files},
        )
        return all_files, has_data_files, has_doc_files

    def __init__(self, *args, **kwargs):
        """Initialize ChatMixin with sequential cancellation support."""
        # Simple boolean flag for sequential control
        object.__setattr__(self, '_cancel_flag', False)
        super().__init__(*args, **kwargs)
        object.__setattr__(self, '_delegation_depth', 0)
        object.__setattr__(self, '_worker_counter', 0)
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

    def _dump_error(
        self,
        error: Exception,
        context_desc: str,
        round_count: int,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Writes a detailed, host-path-sanitized error log to the discussion debug dumps directory."""
        ws_path = getattr(self, "workspace_data_path", None)
        _core_dump_error(
            error=error,
            context_desc=context_desc,
            round_count=round_count,
            workspace_dir=Path(ws_path) if ws_path else None,
            extra_data=extra_data,
            debug_mode=bool(getattr(self, "_debug_mode", False)),
        )

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
        # Detect NEW files
        new_files = set(files_after.keys()) - set(files_before.keys())

        for rel_path in new_files:
            file_info = files_after[rel_path]
            rel_str = str(rel_path).replace("\\", "/")
            file_name = rel_str
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

            rich_doc_bytes = None
            rich_doc_content = None
            if file_ext in (".docx", ".pptx", ".odt"):
                rich_doc_content = self._extract_rich_document_content(file_path, file_ext, tool_name, file_size)
                try:
                    rich_doc_bytes = file_path.read_bytes()
                except Exception as read_err:
                    ASCIIColors.warning(f"[ChatMixin] Failed to read physical bytes for '{file_name}': {read_err}")

            should_read_content = True
            content_placeholder = None
            physical_bytes = rich_doc_bytes

            if rich_doc_content is not None:
                should_read_content = False
                content_placeholder = rich_doc_content
            elif file_ext in _EXPLICIT_BINARY_EXTS:
                should_read_content = False
                content_placeholder = (
                    f"### Data File Generated: `{file_name}`\n\n"
                    f"This file was created by the tool `{tool_name}`.\n"
                    f"- **Type**: {file_ext.upper()} (Binary/Structured Data)\n"
                    f"- **Size**: {file_size:,} bytes\n"
                    f"- **Location**: `./{file_name}`\n\n"
                    f"> **Action**: You can download this file from the Workspace Artifacts panel or reference it in SQL/Python tools."
                )
                try:
                    physical_bytes = file_path.read_bytes()
                except Exception as read_err:
                    ASCIIColors.warning(f"[ChatMixin] Failed to read physical bytes for '{file_name}': {read_err}")
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
                            try:
                                physical_bytes = file_path.read_bytes()
                            except Exception as read_err:
                                ASCIIColors.warning(f"[ChatMixin] Failed to read physical bytes for '{file_name}': {read_err}")
                        else:
                            forced_content = file_path.read_text(encoding='utf-8', errors='ignore')
                            file_info["content"] = forced_content
                            should_read_content = True
                except Exception as e:
                    should_read_content = False
                    content_placeholder = f"### File Error: `{file_name}`\n\nFailed to read or inspect file: {e}"

            if not should_read_content and content_placeholder:
                if file_ext in _ML_WEIGHT_EXTS:
                    atype = "data"
                existing_art = self.artefacts.get(file_name)
                _agentic_mode = bool(getattr(self, "disable_artefact_versioning", False))
                if existing_art:
                    art = self.artefacts.update(
                        title=file_name,
                        new_content=content_placeholder,
                        new_type=atype,
                        active=False,
                        visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                        physical_data=physical_bytes,
                        logical_content=None if _agentic_mode else content_placeholder,
                        commit_message=f"Updated binary file by tool '{tool_name}'"
                    )
                else:
                    art = self.artefacts.add(
                        title=file_name,
                        artefact_type=atype,
                        content=content_placeholder,
                        active=False,
                        visibility=ArtefactVisibility.TREE_UNLOCKABLE,
                        physical_data=physical_bytes,
                        logical_content=None if _agentic_mode else content_placeholder,
                        commit_message=f"Created by tool '{tool_name}'"
                    )
                self.commit()

                self._affected_artefacts_this_turn.append(art)

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
            rel_str = str(rel_path).replace("\\", "/")
            file_name = rel_str
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

                rich_doc_bytes = None
                rich_doc_content = None
                if file_ext in (".docx", ".pptx", ".odt"):
                    rich_doc_content = self._extract_rich_document_content(file_path, file_ext, tool_name, file_path.stat().st_size)
                    try:
                        rich_doc_bytes = file_path.read_bytes()
                    except Exception as read_err:
                        ASCIIColors.warning(f"[ChatMixin] Failed to read physical bytes for '{file_name}': {read_err}")

                should_read_content = True
                content_placeholder = None
                physical_bytes = rich_doc_bytes

                if rich_doc_content is not None:
                    should_read_content = False
                    content_placeholder = rich_doc_content
                elif file_ext in _EXPLICIT_BINARY_EXTS:
                    should_read_content = False
                    content_placeholder = (
                        f"### Data File Modified: `{file_name}`\n\n"
                        f"This file was modified by the tool `{tool_name}`.\n"
                        f"- **Type**: {file_ext.upper()} (Binary/Structured Data)\n"
                        f"- **Size**: {file_path.stat().st_size:,} bytes\n"
                        f"- **Location**: `./{file_name}`\n\n"
                        f"> **Action**: You can download this file from the Workspace Artifacts panel or reference it in SQL/Python tools."
                    )
                    try:
                        physical_bytes = file_path.read_bytes()
                    except Exception as read_err:
                        ASCIIColors.warning(f"[ChatMixin] Failed to read physical bytes for '{file_name}': {read_err}")
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
                                try:
                                    physical_bytes = file_path.read_bytes()
                                except Exception as read_err:
                                    ASCIIColors.warning(f"[ChatMixin] Failed to read physical bytes for '{file_name}': {read_err}")
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
                            raw_img = file_path.read_bytes()
                            img_b64 = base64.b64encode(raw_img).decode('utf-8')
                            img_mtypes = [f"image/{file_ext[1:]}"]
                        except Exception as ex:
                            trace_exception(ex)
                    elif file_ext in _ML_WEIGHT_EXTS:
                        atype = "data"

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
                            physical_data=physical_bytes,
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
                            physical_data=physical_bytes,
                            commit_message=f"Created by tool '{tool_name}'"
                        )
                    self.commit()

                    if atype == "image":
                        try:
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

    def _extract_rich_document_content(
        self,
        file_path,
        file_ext: str,
        tool_name: str,
        file_size: int,
    ) -> str:
        """
        Builds the logical twin for OOXML rich documents generated by tools.

        DOCX/PPTX/ODT are ZIP containers, so the generic null-byte binary sniffer
        misclassifies them and would overwrite the logical content with a metadata
        card. This helper extracts real text instead, falling back to an
        information card only when extraction is impossible.
        """
        header = (
            f"### Rich Document: `{file_path.name}`\n\n"
            f"- **Type**: {file_ext.upper()} (Extracted Document Text)\n"
            f"- **Size**: {file_size:,} bytes\n"
            f"- **Location**: `./{file_path.name}`\n\n"
        )
        try:
            if file_ext == ".docx":
                from lollms_client.lollms_artefact.file_import import _extract_docx_text
                extracted = _extract_docx_text(file_path).strip()
            elif file_ext == ".pptx":
                from lollms_client.lollms_artefact.file_import import _extract_pptx_text
                extracted = _extract_pptx_text(file_path).strip()
            else:
                extracted = ""

            if not extracted:
                return (
                    header
                    + f"No extractable text found. The physical file `./{file_path.name}` "
                    f"is preserved on disk and can be inspected with the document tools."
                )

            return header + "#### Extracted Content:\n\n" + extracted
        except Exception as extract_err:
            ASCIIColors.warning(
                f"[ChatMixin] Rich document extraction failed for '{file_path.name}': {extract_err}"
            )
            return (
                header
                + f"Text extraction failed ({extract_err}). The physical file "
                f"`./{file_path.name}` is preserved on disk and can be inspected "
                f"with the document tools."
            )

    def wipe_all_memories(self) -> bool:
        """
        Permanently deletes all episodic and associative memories from the database.
        This includes working, deep, and archived memory tiers.
        """
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            ASCIIColors.warning("[ChatMixin] No memory manager attached. Cannot wipe memories.")
            return False

        try:
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

    def _resolve_active_tools(
        self,
        personality,
        tools,
        enable_data_tools: bool,
        enable_code_execution: bool,
        debug: bool,
        user_message: str,
        suppress_images: bool,
        images: Optional[List[str]] = None,
        orchestrator_mode: bool = False,
        orchestrator_persona: bool = False,
        shell_autonomy_level: Optional[str] = "safe",
        python_autonomy_level: Optional[str] = "safe",
        auto_approve_python: bool = False,
        confirm_handler: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        if confirm_handler is None and "confirm_handler" in kwargs:
            confirm_handler = kwargs.get("confirm_handler")
        """
        Single source of truth for tool-registry resolution (Sovereign Opt-In
        Doctrine). Used by chat() and by the AgenticRunner so the orchestrator
        inherits the same registry without ever seeing its grammar.

        When the orchestrator persona is active, personality-owned tool
        sources (RAG tools, skill tools) are withheld: they carry live
        callables, and the persona tier must never hold executable grammar.
        The plain-language catalogue for delegation is derived from the
        worker-grade registry (LCP execution tools + spinoffs), which is
        exactly the set the spawned Workers will receive.
        """
        active_tools: Dict[str, Dict[str, Any]] = {}
        _persona_active = orchestrator_mode or orchestrator_persona

        if personality and hasattr(personality, "build_rag_tools") and not _persona_active:
            active_tools.update(personality.build_rag_tools())

        if personality and hasattr(personality, "skills_manager") and personality.skills_manager and not _persona_active:
            active_tools.update(personality.skills_manager.build_skill_tools())

        if isinstance(tools, dict):
            active_tools.update(tools)
        elif isinstance(tools, list):
            lcp_binding = getattr(self.lollmsClient, "tools", None)
            if lcp_binding and hasattr(lcp_binding, "to_chat_tool_specs"):
                try:
                    lcp_tools = lcp_binding.to_chat_tool_specs(
                        discussion_instance=self,
                        lollms_client_instance=self.lollmsClient,
                    )
                    for tool_name in tools:
                        if tool_name in lcp_tools:
                            active_tools[tool_name] = lcp_tools[tool_name]
                        else:
                            ASCIIColors.warning(
                                f"[ChatMixin] Requested default tool '{tool_name}' not found in LCP registry."
                            )
                except Exception as ex:
                    trace_exception(ex)

        lcp_binding = getattr(self.lollmsClient, "tools", None)

        workspace_dir = Path(self.workspace_data_path) if getattr(self, "workspace_data_path", None) else Path("./data_workspace")

        data_extensions = {".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"}
        doc_extensions = {".pdf", ".docx", ".pptx", ".odt", ".epub", ".txt", ".md", ".json", ".yaml", ".xml"}

        all_files, has_data_files, has_doc_files = self._cached_workspace_file_scan(workspace_dir)
        needs_lcp_binding = enable_data_tools or enable_code_execution or has_data_files or has_doc_files

        if needs_lcp_binding and lcp_binding is None:
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding
                lcp_binding = LCPBinding(tools_folders=[])
                if not hasattr(self.lollmsClient, "tools") or self.lollmsClient.tools is None:
                    self.lollmsClient.tools = lcp_binding
                ASCIIColors.success("[ChatMixin] Auto-provisioned shared LCPBinding for context-aware tools.")
            except Exception as ex:
                trace_exception(ex)
                lcp_binding = None

        if lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            if not hasattr(lcp_binding, "host_tool_configs") or lcp_binding.host_tool_configs is None:
                lcp_binding.host_tool_configs = {}

            if confirm_handler:
                if hasattr(lcp_binding, "set_confirm_handler"):
                    lcp_binding.set_confirm_handler(confirm_handler)
                lcp_binding.host_tool_configs.setdefault("execute_python", {})["confirm_handler"] = confirm_handler
                lcp_binding.host_tool_configs.setdefault("system_shell", {})["confirm_handler"] = confirm_handler

            if shell_autonomy_level:
                lcp_binding.host_tool_configs.setdefault("system_shell", {})["autonomy_level"] = shell_autonomy_level
            if python_autonomy_level or auto_approve_python is not None:
                py_cfg = lcp_binding.host_tool_configs.setdefault("execute_python", {})
                if python_autonomy_level:
                    py_cfg["autonomy_level"] = python_autonomy_level
                if auto_approve_python is not None:
                    py_cfg["auto_approve"] = auto_approve_python

            if enable_data_tools and has_data_files:
                lcp_binding.mount_tool_library("semantic_data_engineer")
                ASCIIColors.info("[ChatMixin] Mounted 'semantic_data_engineer' (data files detected).")

            if enable_data_tools and has_doc_files:
                lcp_binding.mount_tool_library("as_is_document_tools")
                lcp_binding.mount_tool_library_if_absent("document_editor")
                ASCIIColors.info("[ChatMixin] Mounted 'document_editor' and 'as_is_document_tools' (document files detected).")

            if enable_code_execution:
                lcp_binding.mount_tool_library_if_absent("execute_python")
                ASCIIColors.info("[ChatMixin] Mounted 'execute_python' (inline + file execution enabled).")
                lcp_binding.mount_tool_library_if_absent("inspect_text")
                ASCIIColors.info("[ChatMixin] Mounted 'inspect_text' (targeted log inspection enabled).")

            try:
                lcp_tools = lcp_binding.to_chat_tool_specs(
                    discussion_instance=self,
                    lollms_client_instance=self.lollmsClient,
                )
                for t_name, t_spec in lcp_tools.items():
                    if t_name == "tool_execute_python_data_query" and enable_data_tools and has_data_files:
                        active_tools[t_name] = t_spec
                    elif t_name in ("tool_execute_python_code", "tool_execute_python_file") and enable_code_execution:
                        active_tools[t_name] = t_spec
                    elif t_name in ("tool_read_lines", "tool_read_chars", "tool_grep_file") and enable_code_execution:
                        active_tools[t_name] = t_spec
                    elif t_name.startswith(("tool_inspect_document", "tool_read_document_content", "tool_grep_document", "tool_modify_docx", "tool_modify_excel", "tool_edit_document_text", "tool_annotate_document", "tool_modify_pdf_annotation", "tool_modify_pptx_slide")) and enable_data_tools and has_doc_files:
                        active_tools[t_name] = t_spec
            except Exception as ex:
                trace_exception(ex)
                ASCIIColors.error(f"[ChatMixin] Failed to extract tool specs from LCP binding: {ex}")

            for td in lcp_binding.discovered_tools:
                t_name = td.get("name", "")
                if t_name not in active_tools:
                    if (t_name == "tool_execute_python_data_query" and enable_data_tools and has_data_files) or \
                       (t_name in ("tool_execute_python_code", "tool_execute_python_file") and enable_code_execution) or \
                       (t_name in ("tool_read_lines", "tool_read_chars", "tool_grep_file") and enable_code_execution) or \
                       (t_name.startswith(("tool_inspect_document", "tool_read_document_content", "tool_grep_document", "tool_modify_docx", "tool_modify_excel", "tool_edit_document_text", "tool_annotate_document", "tool_modify_pdf_annotation", "tool_modify_pptx_slide")) and enable_data_tools and has_doc_files):
                        params_list = []
                        input_schema = td.get("input_schema", {})
                        for prop_name, prop_info in input_schema.get("properties", {}).items():
                            params_list.append({
                                "name": prop_name,
                                "type": prop_info.get("type", "string"),
                                "description": prop_info.get("description", ""),
                            })
                        active_tools[t_name] = {
                            "name": t_name,
                            "description": td.get("description", "Executes tool operation."),
                            "parameters": params_list,
                        }
                        ASCIIColors.success(f"[ChatMixin] Registered {t_name} via direct discovered_tools fallback.")

        if debug:
            if active_tools:
                ASCIIColors.info(
                    f"[ChatMixin] Final active tool registry ({len(active_tools)} tool(s)): "
                    f"{sorted(active_tools.keys())}"
                )
            else:
                ASCIIColors.warning("[ChatMixin] Final active tool registry is EMPTY — no tools available to the LLM this turn.")

        if debug and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            lcp_binding.mount_tool_library("debug_toolset")
            try:
                lcp_tools = lcp_binding.to_chat_tool_specs(
                    discussion_instance=self,
                    lollms_client_instance=self.lollmsClient,
                )
                for t_name, t_spec in lcp_tools.items():
                    if t_name == "tool_dump_context":
                        active_tools[t_name] = t_spec
            except Exception as ex:
                trace_exception(ex)

        return active_tools

    def chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        tools=None,
        add_user_message: bool = True,
        images=None,
        streaming_callback: Callable[[Any,MSG_TYPE,dict],bool] = None,
        remove_thinking_blocks: bool = True,
        enable_image_generation: bool = True,
        enable_image_editing:    bool = True,
        auto_activate_artefacts: bool = True,
        enable_inline_widgets:        bool = False,
        enable_forms:                 bool = False,
        enable_notes:                 bool = True,
        enable_skills:                bool = True,
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
        forward_artefact_chunks:      bool = False,
        fast_artefact_replicas:       Optional[List[str]] = None,
        tolerance_level:              Optional[str] = "strict",
        allow_dynamic_tools:          bool = False,
        enable_data_tools:            bool = True,
        enable_code_execution:        bool = False,
        suppress_images:              bool = False,
        orchestrator_mode:            bool = False,
        orchestrator_persona:         bool = False,
        debug_export:                 bool = False,
        debug:                        bool = False,
        enable_vlm_query:             bool = False,
        enable_computer_use:          bool = False,
        event_mode:                   EventMode = EventMode.PROCESSING_TAG_MODE,
        think:                        Optional[bool] = None,
        reasoning_effort:             Optional[str] = None,
        reasoning_summary:            Optional[str] = None,
        shell_autonomy_level:         Optional[str] = "safe",
        python_autonomy_level:        Optional[str] = "safe",
        auto_approve_python:          bool = False,
        confirm_handler:              Optional[Callable] = None,
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
            streaming_callback: Optional callable to receive events like streamed chunks.
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
            forward_artefact_chunks (bool): Forward artifact chunks to callback. Default False.
            fast_artefact_replicas (Optional[List[str]]): Custom fast replica messages for artifacts.
            tolerance_level (Optional[str]): Tolerance level for data tools ('strict', 'lenient'). Default 'strict'.
            allow_dynamic_tools (bool): Allow dynamic tool registration from artifacts. Default False.
            enable_data_tools (bool): Enable data manipulation tools (SQL, pandas). Default True.
            enable_code_execution (bool): Enable arbitrary Python code execution tool. Default False.
            suppress_images (bool): Suppress image hydration in context. Default False.
            orchestrator_persona (bool): Run this turn as the delegation-only
                Orchestrator persona: the prompt shows a plain-language worker
                capability catalogue (no tool-call grammar), and artifact writes
                plus tool execution are structurally refused at the dispatcher.
            debug_export (bool): Enable debug export of context dumps for this turn. Default False.
            debug (bool): Enable debug mode: mounts the debug toolset with additional logging. Default False.
                Persistent per-discussion debug dumps can also be enabled by setting
                discussion._debug_mode = True externally (mirrors personality.debug_mode).
            enable_vlm_query (bool): Enable VLM query tool for vision fallback. Default False.
            enable_computer_use (bool): Mount the computer use desktop automation toolset
                (screenshots, click, type, key, scroll). Requires the active model to
                support vision; silently skipped when no vision capability is detected.
                Default False.
            event_mode (EventMode): Event reporting mode. Default PROCESSING_TAG_MODE.
            think (Optional[bool]): Legacy flag to toggle reasoning/thinking output. If True, maps to reasoning_effort='high'. If False, disables reasoning. Default None.
            reasoning_effort (Optional[str]): Level of reasoning effort for reasoning/thinking models ('low', 'medium', 'high', 'max'). Overrides `think`. Default None.
            reasoning_summary (Optional[str]): Format of reasoning summary for thinking models ('auto', 'concise', 'detailed'). Default None.
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

        callback = streaming_callback

        if orchestrator_mode:
            from ..lollms_agentic.runner import AgenticRunner
            active_tools = self._resolve_active_tools(
                personality=personality,
                tools=tools,
                enable_data_tools=enable_data_tools,
                enable_code_execution=enable_code_execution,
                debug=debug,
                user_message=user_message,
                suppress_images=suppress_images,
                images=images,
                orchestrator_mode=orchestrator_mode,
                shell_autonomy_level=shell_autonomy_level,
                python_autonomy_level=python_autonomy_level,
                auto_approve_python=auto_approve_python,
                confirm_handler=confirm_handler,
                **kwargs,
            )
            runner = AgenticRunner(
                discussion=self,
                tools_registry=active_tools,
                callback=callback,
                event_mode=event_mode,
                max_orchestrator_rounds=resolved_max_rounds,
                max_worker_rounds=max(2, resolved_max_rounds // 2),
            )
            return runner.run(user_message=user_message)

        debug_enabled = bool(debug_export) or bool(getattr(self, "_debug_mode", False))

        # Store tolerance level on active discussion for downstream execution tools (like execute_python_data_query)
        if not hasattr(self, "tolerance_level") or tolerance_level:
            object.__setattr__(self, "tolerance_level", tolerance_level or "strict")

        # 🛡️ SECURITY: Store the dynamic tool execution flag.
        # If False, the ArtefactManager will NOT register type="tool" artefacts as executable LCP tools.
        object.__setattr__(self, "allow_dynamic_tools", allow_dynamic_tools)
        object.__setattr__(self, "remove_thinking_blocks", remove_thinking_blocks)
        object.__setattr__(self, "_orchestrator_mode", orchestrator_mode)

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
            "\n=== SANDBOX OPACITY REQUIREMENT (CRITICAL) ===\n"
            "1. NEVER reference, print, or embed absolute host paths (C:\\..., /home/..., /Users/...) in code, "
            "tool parameters, artifacts, or answers. The workspace root is '.' — always use relative paths.\n"
            "2. NEVER call sys.path.insert with a physical disk location. Import workspace siblings directly "
            "(e.g. 'from ontology import server').\n"
            "3. When executing scripts, assume the CWD IS the workspace. Reference files as './filename.ext'.\n"
            "4. If an error trace reveals a host path, treat it as redacted ('<host-path>') — do not attempt to "
            "reconstruct or print it.\n\n"
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
        if enable_artefacts and not orchestrator_persona:
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

        # Image Generation Instructions (only if image generation/editing is enabled AND TTI capability exists)
        _has_tti = getattr(self.lollmsClient, 'tti', None) is not None or bool(getattr(self.lollmsClient, 'tti_model_profiles_registry', None))
        if (enable_image_generation or enable_image_editing) and _has_tti and not orchestrator_persona:
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

        active_tools = self._resolve_active_tools(
            personality=personality,
            tools=tools,
            enable_data_tools=enable_data_tools,
            enable_code_execution=enable_code_execution,
            debug=debug,
            user_message=user_message,
            suppress_images=suppress_images,
            images=images,
            orchestrator_mode=orchestrator_mode,
            orchestrator_persona=orchestrator_persona,
            shell_autonomy_level=shell_autonomy_level,
            python_autonomy_level=python_autonomy_level,
            auto_approve_python=auto_approve_python,
            confirm_handler=confirm_handler,
            **kwargs,
        )

        if personality and hasattr(personality, "tools") and _is_tool_binding(personality.tools) and not orchestrator_persona:
            try:
                pers_tools = personality.tools.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                active_tools.update(pers_tools)
                if pers_tools:
                    ASCIIColors.success(
                        f"[ChatMixin] Personality binding '{getattr(personality.tools, 'binding_name', '?')}' "
                        f"registered {len(pers_tools)} tool(s): {list(pers_tools.keys())}"
                    )
                else:
                    raw_disc = personality.tools.discover_tools() if hasattr(personality.tools, "discover_tools") else []
                    ASCIIColors.warning(
                        f"[ChatMixin] Personality binding '{getattr(personality.tools, 'binding_name', '?')}' "
                        f"yielded 0 chat tool specs (discovered_tools count: {len(raw_disc)})."
                    )
            except Exception as ex:
                ASCIIColors.error(f"[ChatMixin] Personality tool binding failed to produce specs: {ex}")
                trace_exception(ex)
        elif personality and hasattr(personality, "tools") and isinstance(personality.tools, dict) and not orchestrator_persona:
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

        lcp_binding = getattr(self.lollmsClient, "tools", None)

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

        # ── 6. Mount Computer Use Toolset (Vision-Gated) ──
        # Desktop automation (screenshot → act → verify) is only coherent when
        # the ACTIVE model can see the screenshots it takes: the visual
        # grounding loop depends on a vision-capable binding. The user flag is
        # an explicit opt-in; the vision check is a hard structural gate.
        _computer_use_tool_names = (
            "tool_computer_desktop_info",
            "tool_computer_screenshot",
            "tool_computer_click",
            "tool_computer_move_cursor",
            "tool_computer_type",
            "tool_computer_key",
            "tool_computer_scroll",
        )
        if enable_computer_use:
            _computer_use_vision_ready = False
            if self.lollmsClient and hasattr(self.lollmsClient, "has_vision_capability"):
                try:
                    _computer_use_vision_ready = bool(self.lollmsClient.has_vision_capability())
                except Exception:
                    _computer_use_vision_ready = False
            if not _computer_use_vision_ready and self.lollmsClient:
                active_llm = getattr(self.lollmsClient, "llm", None)
                _computer_use_vision_ready = bool(getattr(active_llm, "vision_enabled", False))
                if not _computer_use_vision_ready and active_llm and hasattr(active_llm, "child_bindings"):
                    _computer_use_vision_ready = any(
                        getattr(child, "vision_enabled", False)
                        for child in active_llm.child_bindings.values()
                    )

            if _computer_use_vision_ready and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
                lcp_binding.mount_tool_library("computer_use")
                try:
                    lcp_tools = lcp_binding.to_chat_tool_specs(discussion_instance=self, lollms_client_instance=self.lollmsClient)
                    for t_name, t_spec in lcp_tools.items():
                        if t_name in _computer_use_tool_names:
                            active_tools[t_name] = t_spec
                    ASCIIColors.success(
                        f"[ChatMixin] Mounted 'computer_use' toolset "
                        f"({len([n for n in _computer_use_tool_names if n in active_tools])} tools, vision model active)."
                    )
                except Exception as ex:
                    trace_exception(ex)
            else:
                ASCIIColors.warning(
                    "[ChatMixin] enable_computer_use=True but no vision-capable "
                    "model is active or no LCP binding is available — computer use "
                    "toolset NOT mounted."
                )


        # ── 🎯 WORKER-TIER EXECUTION DOCTRINE ──
        # chat() always runs the worker persona here. The orchestrator persona
        # lives in lollms_agentic.OrchestratorAgent and never reaches this code.
        feature_rules = ""

        if (enable_artefacts or active_tools or enable_memory) and not orchestrator_persona:
            feature_rules += (
                "\n=== ACTION EXECUTION & TERMINATION PROTOCOL (CRITICAL) ===\n"
                "1. **INTENT ≠ EXECUTION (SAME-RESPONSE EXECUTION MANDATE)**: Stating 'I will search...', 'Let me analyze...', 'I am now writing...', or any equivalent declaration in ANY language (English, Arabic, Chinese, French, Spanish, etc.) DOES NOT execute the action. Conversational text is completely inert.\n"
                "   - You MUST emit the corresponding functional XML tag (`<tool>`, `<artifact>`, `<skill>`, `<note>`, `<unlock_file>`, `<generate_image>`, etc.) IN THE EXACT SAME RESPONSE immediately following your brief statement of intent.\n"
                "   - **NEVER SPLIT INTENT AND TAGS**: Never announce what you are going to do and then stop without emitting the tag. If you state intent without outputting the XML tag in the same response, the turn will end with nothing done.\n"
                "   - **DESTRUCTIVE VS CONSTRUCTIVE OPERATIONS**:\n"
                "     • If an action is risky, destructive, or irreversible (e.g. deleting files, force-pushing git branches, dropping database tables), explicitly ask the user for confirmation and wait for their reply before emitting destructive tags.\n"
                "     • For ALL normal, constructive tasks (creating/editing files, querying data, reading files, searching memories), output the functional tag IMMEDIATELY in the same turn without asking or waiting.\n"
                "2. **MANDATORY TAG EMISSION**: If your response states that you are writing code, searching, or loading files, the functional tag MUST appear in that same response.\n"
                "3. **THE ONLY WAY TO FINISH IS `<done/>`**: The loop never ends on prose alone. Every response you produce must contain EITHER at least one functional action tag OR, when your work is complete, your final answer followed by the `<done/>` tag on a new line. A response containing neither a tag nor `<done/>` is invalid and will be rejected.\n"
                "   **EXAMPLE**:\n"
                "   ```\n"
                "   Here is my complete answer and solution.\n"
                "   \n"
                "   <done/>\n"
                "   ```\n"
                "4. **SAME-SESSION CONTINUATION**: In multi-step workflows, emit the next action tag immediately in your next response upon receiving previous tool/action results.\n"
            )

        # System Notification Handling (only if context unlocking is possible)
        if enable_artefacts and not orchestrator_persona:
            feature_rules += (
                "\n=== SYSTEM NOTIFICATION HANDLING ===\n"
                "1. **RECOGNIZE SYSTEM NOTIFICATIONS**: Messages wrapped in `[SYSTEM NOTIFICATION - NOT A USER MESSAGE]...[END SYSTEM NOTIFICATION]` are infrastructure events, not user input.\n"
                "2. **DO NOT RESPOND TO NOTIFICATIONS**: If you receive a system notification, simply acknowledge it internally and continue with the pending task. Do NOT treat it as a user question.\n"
                "3. **CONTENT AVAILABILITY**: When files are unlocked, check the `[CONTENT AVAILABILITY]` section to see which files actually have readable content vs. which are empty.\n"
                "4. **EMPTY FILES**: If a file is listed as empty, do NOT pretend to have read content from it. Acknowledge that it's empty and move on.\n"
            )

        # Tool Calling Discipline (only if tools are available)
        if active_tools and not orchestrator_persona:
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

        # Sub-Agent Delegation Grammar (single canonical teacher)
        if enable_artefacts and not orchestrator_persona:
            feature_rules += (
                "\n=== SUB-AGENT DELEGATION (OPTIONAL) ===\n"
                "To delegate a self-contained task to a specialist agent, emit this EXACT structure "
                "(starting on a new line, never inside code fences):\n"
                "<agent name=\"worker\" system_prompt=\"You are an expert in X\" max_rounds=\"8\">\n"
                "<task>\n"
                "Self-contained instructions for the specialist.\n"
                "</task>\n"
                "</agent>\n"
                "MANDATORY RULES:\n"
                "1. The instructions MUST be wrapped in literal <task> and </task> XML tags — "
                "nothing else is accepted.\n"
                "2. NEVER use plain-text markers like '=== TASK ===' — they are not parsed.\n"
                "3. Optionally add <context_files>filename.ext</context_files> INSIDE the agent tag "
                "to give the specialist workspace files.\n"
                "4. After the delegation, STOP generating. The agent's report arrives in your next turn; "
                "then answer the user or delegate again.\n"
                "=== END SUB-AGENT DELEGATION ===\n"
            )

        # Thinking & Reasoning Constraint (only if agentic features are enabled)
        if (enable_artefacts or active_tools or enable_memory) and not orchestrator_persona:
            feature_rules += (
                "\n=== THINKING & REASONING CONSTRAINT ===\n"
                "If you decide to output a thought process enclosed in  tags, "
                "you MUST output all functional XML tags (such as <artifact>, <tool>, or <mem_new>) "
                "on a NEW LINE strictly AFTER the closing  warn_tag tag. "
                "NEVER place functional tags inside the  warn_tag reasoning block.\n"
            )

        # Anti-Mimicry Protocol (only if agentic features are enabled)
        if (enable_artefacts or active_tools) and not orchestrator_persona:
            feature_rules += (
                "\n=== ACTION INTEGRITY PROTOCOL (CRITICAL) ===\n"
                "1. **USE REAL TAGS**: To create artifacts, you MUST use `<artifact name=\"...\">` XML tags. To call tools, use `<tool>`. Saying you will do something in text without emitting tags produces NO changes.\n"
                "2. **TAG ISOLATION**: Functional tags (`<artifact>`, `<tool>`) MUST NEVER appear inside <think> blocks. They must ONLY appear in the final response body AFTER the closing </think> tag.\n"
            )

        # Inject feature rules into the full system prompt
        if feature_rules:
            full_system_prompt += feature_rules

        # Orchestrator persona: replace all worker doctrine with the
        # grammar-free delegation protocol (zero tool syntax, zero artifact XML).
        if orchestrator_persona:
            full_system_prompt += self._build_orchestrator_instructions()

        tools_prompt = ""
        if orchestrator_persona and active_tools:
            tools_prompt = (
                "\n=== WORKER SPECIALTIES (delegation catalogue — NOT callable by you) ===\n"
                "These are the capabilities held by the specialist workers you spawn.\n"
                "You cannot invoke them yourself. Never emit tool-call tags or tool "
                "syntax of any kind. When a specialist needs one of these "
                "capabilities, describe it in plain words inside your <task> text.\n"
            )
            for t_name, t_spec in active_tools.items():
                tools_prompt += f"- {t_name}: {t_spec.get('description', '')}\n"
            tools_prompt += "=== END WORKER SPECIALTIES ===\n"
        elif active_tools:
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
                "\n=== 🏁 TASK COMPLETION PROTOCOL (CRITICAL) ===\n"
                "Your goal is to SOLVE the user's problem, not to infinitely call tools.\n"
                "1. **TOOL CALLS ARE TEMPORARY**: You call a `<tool>` only to gather data you don't have.\n"
                "2. **ANSWERING IS THE GOAL**: Once you have the data, writing a comprehensive, helpful response to the user IS the successful completion of your task.\n"
                "3. **HOW TO FINISH**: When your work is complete, write your final answer to the user and then emit `<done/>` on a new line. This is the ONLY way the turn ends.\n"
                "4. **NEVER LOOP**: If you have already written your final answer, do NOT emit another `<tool>` tag. Emitting a tool call after your answer is a CRITICAL ERROR that ruins the completed task. Terminate with `<done/>`.\n"
                "=== END TASK COMPLETION PROTOCOL ===\n"
            )
            tools_prompt += "\nExact syntax (copy this pattern exactly):\n<tool>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}</tool>\n\n"
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
        _t_branch_start = time.perf_counter()
        ASCIIColors.info("[Trace] Retrieving conversation branch...")
        current_branch_tip = branch_tip_id or self.active_branch_id
        branch = self.get_branch(current_branch_tip)
        _t_branch_end = time.perf_counter()
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
        object.__setattr__(self, "_vh_append", virtual_history.append)

        # ── 🔄 ACTION-WINDOW RECOLLECTION PROTOCOL (Placeholder-Free) ──
        # The compression window counts ACTIONS (successful tool calls and
        # artifact dispatches), not raw message count. When the window
        # overflows, the oldest action round is removed from virtual_history
        # entirely and a deterministic narrative digest entry is recorded in
        # the system-zone digest log instead. No stub tokens, no
        # status="superseded" anchors, no synthetic history entries.
        action_window = max(1, int(getattr(self, "history_compression_window", 4) or 4))
        turn_digest_log: List[str] = []
        object.__setattr__(self, "_turn_digest_log", turn_digest_log)
        completed_actions: List[Dict[str, Any]] = []
        digested_action_count = 0
        total_actions_this_turn = 0

        tool_calls_this_turn = []
        round_count = 0
        conversational_gist = ""  # Accumulates only the conversational text for the final DB message
        worker_turn = not (orchestrator_mode or orchestrator_persona)

        # ── ENVIRONMENT EPOCH: TRUE REPETITION GATING ────────────────────────────
        # A repetition exists ONLY when the LLM re-issues the exact same call
        # with the exact same parameters AND the observable environment
        # (workspace files, artifacts, context) is unchanged since the last
        # identical call. ANY intervening action — a different tool call, an
        # artifact build/patch, a context unlock, a tool that wrote files —
        # increments the epoch and legitimately re-enables the call.
        environment_epoch = 0
        _last_failure_epoch = -1
        successful_tool_signatures: set = set()

        def _current_workspace_revision() -> int:
            return int(getattr(self, "_workspace_write_revision", 0))

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

        # ── 🔄 ACTION-WINDOW RECOLLECTION COMPRESSION (Placeholder-Free) ──
        def _digest_params(params: Dict[str, Any]) -> str:
            try:
                rendered = json.dumps(params, ensure_ascii=False, default=str)
            except Exception:
                rendered = str(params)
            if len(rendered) > 160:
                rendered = rendered[:157] + "..."
            return rendered

        def _split_rounds() -> List[List[Any]]:
            rounds: List[List[Any]] = []
            current: List[Any] = []
            for vh in virtual_history:
                current.append(vh)
                if vh.sender_type == "user":
                    rounds.append(current)
                    current = []
            if current:
                rounds.append(current)
            return rounds

        def _register_completed_action(action: Dict[str, Any]) -> None:
            nonlocal total_actions_this_turn, digested_action_count
            total_actions_this_turn += 1
            completed_actions.append(action)
            if len(completed_actions) > action_window:
                expired = completed_actions.pop(0)
                expired_round = expired["round"]
                digest_line = _build_digest_line(expired)
                if digest_line:
                    turn_digest_log.append(digest_line)
                _drop_round_from_virtual_history(expired_round)
                digested_action_count += 1

        def _build_digest_line(action: Dict[str, Any]) -> str:
            kind = action.get("kind", "tool")
            if kind == "tool":
                name = action.get("name", "unknown")
                params = _digest_params(action.get("params", {}))
                verdict = "succeeded" if action.get("success") else "failed"
                return (
                    f"- (round {action.get('round', '?')}) Executed tool '{name}' "
                    f"with parameters {params} — the call {verdict}."
                )
            if kind == "artifact":
                title = action.get("title", "untitled")
                op = action.get("op", "updated")
                version = action.get("version")
                version_str = f" (now v{version})" if version else ""
                return (
                    f"- (round {action.get('round', '?')}) {op} artifact '{title}'{version_str} "
                    f"in the workspace; its current content is loaded in the Active Artifacts zone."
                )
            return f"- (round {action.get('round', '?')}) Performed action: {action.get('detail', 'unknown action')}."

        def _drop_round_from_virtual_history(round_id: int) -> None:
            rounds = _split_rounds()
            kept: List[Any] = []
            for i, msgs in enumerate(rounds, start=1):
                if i == round_id:
                    continue
                kept.extend(msgs)
            virtual_history[:] = kept

        def _compress_virtual_history_if_needed():
            """
            Enforces the action-window doctrine on virtual_history.

            Called after every completed action. Any action round that has
            aged out of the window is removed from virtual_history and its
            narrative digest line is appended to the system-zone digest log
            (injected into the per-round system prompt). The workspace itself
            remains the source of truth for artifact content, so dropping the
            raw round loses nothing that cannot be re-derived.
            """
            nonlocal digested_action_count
            if not completed_actions:
                return
            while len(completed_actions) > action_window:
                expired = completed_actions.pop(0)
                digest_line = _build_digest_line(expired)
                if digest_line:
                    turn_digest_log.append(digest_line)
                _drop_round_from_virtual_history(expired["round"])
                digested_action_count += 1
            if digested_action_count:
                ASCIIColors.info(
                    f"[ChatMixin] Action-window recollection: {digested_action_count} "
                    f"action round(s) digested into the system-zone narrative; "
                    f"{len(virtual_history)} message(s) remain verbatim in window."
                )

        # Initialize the single, clean database assistant message ONCE before entering the loop
        ai_msg = self.add_message(
            sender=personality.name if personality else self.lollmsClient.ai_name,
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=getattr(self.lollmsClient.llm, "model_name", "unknown") if self.lollmsClient else "unknown",
            binding_name=getattr(self.lollmsClient.llm, "binding_name", "unknown") if self.lollmsClient else "unknown"
        )

        def _persist_round_state():
            """Commits active message content and discussion state immediately to the database."""
            if self._is_db_backed:
                try:
                    self.touch()
                    self.commit()
                except Exception as commit_err:
                    ASCIIColors.warning(f"[ChatMixin] Mid-turn round commit warning: {commit_err}")

        # Commit initial user message and assistant message anchor immediately so message is never lost
        _persist_round_state()

        # CRITICAL: Expose the active personality to _StreamState so it can access the SkillsManager
        # for Handbag skill routing (modifiable/read-only enforcement) during <skill> tag dispatch.
        object.__setattr__(self, '_active_personality', personality)

        if callback:
            callback(ai_msg.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, {"message_id": ai_msg.id})

        # Track if we exited due to cancellation
        was_cancelled = False
        failed_tools_pending_fix = False

        raw_llm_output_buffer = [""]
        raw_llm_output_buffer = [""]

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

        def _bump_environment_epoch() -> None:
            nonlocal environment_epoch
            environment_epoch += 1
            object.__setattr__(
                self, "_workspace_write_revision", _current_workspace_revision() + 1
            )

        object.__setattr__(self, "_bump_environment_epoch", _bump_environment_epoch)

        round_event_state = {"last_status": None}

        def _emit_round_event(msg_type: MSG_TYPE, status: Optional[str] = None, round_id: Optional[int] = None) -> None:
            if event_mode == EventMode.SILENT_MODE:
                return
            effective_round_id = round_id if round_id is not None else round_count
            if msg_type == MSG_TYPE.MSG_TYPE_ROUND_START:
                _cb(callback, "", msg_type, {"round_id": effective_round_id, "max_rounds": resolved_max_rounds})
                return
            round_event_state["last_status"] = status or "action"
            _cb(callback, "", msg_type, {"round_id": effective_round_id, "status": status or "action"})

        while round_count < resolved_max_rounds:
            round_count += 1

            # Check cancellation at the start of each reasoning round
            if self.is_generation_cancelled():
                was_cancelled = True
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="cancelled")
                break

            _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_START)

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

            # ── 📜 COMPLETED ACTIONS DIGEST (System Zone, Placeholder-Free) ──
            # Action rounds that aged out of the window are recalled here as a
            # plain-language narrative. This lives in the system zone only: it
            # is never rendered as a history message, so the model cannot learn
            # to reproduce it as conversational output.
            if turn_digest_log:
                digest_block = "\n[COMPLETED ACTIONS DIGEST — earlier rounds of this turn]\n"
                digest_block += "The following actions were completed in earlier rounds. Their raw transcripts were released to reclaim context; the facts and re-execution recipes are preserved here, and current file contents remain visible in the Active Artifacts zone:\n\n"
                digest_block += "\n".join(turn_digest_log)
                digest_block += "\n[END COMPLETED ACTIONS DIGEST]\n"
                current_system_prompt += "\n" + digest_block

            messages_list = self.export(
                format_type="openai_chat",
                branch_tip_id=current_branch_tip,
                suppress_system_prompt=False,
                suppress_images=suppress_images,
                virtual_history=virtual_history,
                debug=debug_enabled,
                system_prompt_override=current_system_prompt
            )

            if debug_enabled:
                try:
                    debug_dir = Path(self.workspace_data_path) / "_debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)

                    with open(debug_dir / f"full_prompt_round_{round_count}.log", "w", encoding="utf-8") as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"🐛 [DEBUG] ROUND {round_count} - FULL PROMPT\n")
                        f.write("=" * 80 + "\n")
                        for i, msg in enumerate(messages_list):
                            role = msg.get("role", "unknown").upper() if isinstance(msg, dict) else "UNKNOWN"
                            content = msg.get("content", "") if isinstance(msg, dict) else ""
                            f.write(f"\n--- MSG [{i}] ROLE: {role} ---\n")
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        f.write(item.get("text", "") + "\n")
                                    elif isinstance(item, dict) and item.get("type") == "image_url":
                                        f.write("[IMAGE ATTACHED]\n")
                                    else:
                                        f.write(str(item) + "\n")
                            else:
                                f.write(str(content) + "\n")
                        f.write("\n" + "=" * 80 + "\n")

                    with open(debug_dir / f"prompt_dump_round_{round_count}_shortened.md", "w", encoding="utf-8") as f:
                        f.write(f"# 🐛 Round {round_count} - Shortened Prompt Dump\n\n")
                        for i, msg in enumerate(messages_list):
                            role = msg.get("role", "unknown").upper() if isinstance(msg, dict) else "UNKNOWN"
                            content = msg.get("content", "") if isinstance(msg, dict) else ""
                            if isinstance(content, list):
                                content = "\n".join(
                                    item.get("text", "") for item in content
                                    if isinstance(item, dict) and item.get("type") == "text"
                                )
                            if not isinstance(content, str):
                                content = str(content)
                            short_content = (
                                content[:500] + "\n\n[... truncated ...]\n\n" + content[-500:]
                                if len(content) > 1000
                                else content
                            )
                            f.write(f"## MSG [{i}] - {role}\n\n```\n{short_content}\n```\n\n")
                except Exception as debug_err:
                    ASCIIColors.warning(f"[ChatMixin] Failed to write prompt debug logs: {debug_err}")

            # ── 🎨 DYNAMIC VISION HYDRATION ──
            # Retrieve all images generated or modified during previous rounds of this turn
            # and append their base64 pixels to the active vision context so the LLM can "see" them!
            # CRITICAL FIX: Only hydrate images that are explicitly in FULL visibility context.
            # Injecting pixels for [U] (TREE_UNLOCKABLE) images crashes non-vision LLMs.
            # Auto-suppress raw images if active LLM profile lacks vision capabilities
            has_vision = True
            if self.lollmsClient and hasattr(self.lollmsClient, "has_vision_capability"):
                has_vision = self.lollmsClient.has_vision_capability()
            elif hasattr(self, "lollmsClient") and self.lollmsClient and hasattr(self.lollmsClient, "llm"):
                has_vision = getattr(self.lollmsClient.llm, "vision_enabled", True)

            if suppress_images or not has_vision:
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
                enable_artefacts=enable_artefacts and not orchestrator_persona,
                enable_in_message_status=enable_in_message_status,
                enable_tools=not orchestrator_persona,
                fast_artefact_replicas=fast_artefact_replicas,
                content_offset=current_content_length,
                processed_tags=persistent_processed_tags,
                event_mode=event_mode,
                remove_thinking_blocks=remove_thinking_blocks
            )

            def _inline_relay(chunk, msg_type=None, meta=None):
                # Check cancellation on EVERY token chunk
                if self.is_generation_cancelled():
                    return False  # Signal to stop generation

                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    if debug_enabled:
                        raw_llm_output_buffer[0] += chunk
                    # ── ⏱️ TIME TO FIRST TOKEN (TTFT) ──
                    if not getattr(self, "_ttft_logged", True) and chunk:
                        ttft = time.perf_counter() - _t_gen_start
                        ASCIIColors.info(f"[TTFT] First token received in {ttft:.3f} s.")
                        object.__setattr__(self, "_ttft_logged", True)

                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            # Sanitize kwargs to prevent duplicate argument passing
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "stream", "think", "reasoning_effort", "reasoning_summary")}
            if think is not None:
                gen_kwargs["think"] = think
            if reasoning_effort is not None:
                gen_kwargs["reasoning_effort"] = reasoning_effort
            if reasoning_summary is not None:
                gen_kwargs["reasoning_summary"] = reasoning_summary

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
            _t_gen_start = time.perf_counter()
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
                _t_gen_end = time.perf_counter()
                ASCIIColors.info(f"[Trace] Generation round {round_count} stream completed in {(_t_gen_end - _t_gen_start):.2f} s.")
            except Exception as gen_err:
                _t_gen_end = time.perf_counter()
                ASCIIColors.warning(f"[Trace] Generation round {round_count} failed after {(_t_gen_end - _t_gen_start):.2f} s.")
                if debug_enabled:
                    self._dump_error(
                        error=gen_err,
                        context_desc="LLM Generation Error",
                        round_count=round_count,
                        extra_data={"raw_llm_output": _sanitize_host_paths(raw_llm_output_buffer[0][-4000:])}
                    )
                if self.is_generation_cancelled():
                    was_cancelled = True
                    _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="cancelled")
                    break
                else:
                    raise

            # Check cancellation after generation completes
            if self.is_generation_cancelled():
                was_cancelled = True
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="cancelled")
                break

            if debug_enabled and raw_llm_output_buffer[0]:
                try:
                    debug_dir = Path(self.workspace_data_path) / "_debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    raw_output_log_path = debug_dir / f"raw_llm_output_round_{round_count}.log"
                    with open(raw_output_log_path, "w", encoding="utf-8") as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"🐛 [DEBUG] ROUND {round_count} - RAW LLM STREAM OUTPUT\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(raw_llm_output_buffer[0])
                        f.write("\n\n" + "=" * 80 + "\n")
                except Exception as debug_err:
                    ASCIIColors.warning(f"[ChatMixin] Failed to write raw LLM output log: {debug_err}")

            ss.flush_remaining_buffer()

            # ── 🏁 TERMINATION TAG PROTOCOL ──
            # The <done/> tag is sovereign: any round ending in <done/> terminates
            # the loop immediately. The former analysis-gate and phantom-done
            # rejection guards are removed by the Orchestrator/Worker split:
            # the Orchestrator never executes tools, and the Worker's own
            # self-verification is enforced by its task prompt, not by
            # rejecting its termination signal.
            if ss.was_done_detected():
                ASCIIColors.info("[ChatMixin] Termination tag detected. Terminating agentic loop.")
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="done")
                break

            # ── 🛑 TOOL-LESS PERSONA REFUSAL HANDLING ──
            # A tool-less tier (orchestrator persona) attempted a <tool> call.
            # The dispatcher already refused execution; convert the refusal into
            # a delegation-correction envelope. The loop keeps running: the
            # sovereign <done/> tag is the only termination signal.
            if ss.was_tool_refusal_detected():
                refusal_text = _scrub_for_llm_context(
                    ss.get_clean_text_so_far()[current_content_length:]
                ).strip()
                if refusal_text:
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=refusal_text
                    ))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: TOOL CALL REFUSED — You are the orchestrator. "
                        "You have no tools and cannot execute anything yourself. "
                        "Delegate the work with a <delegate> block containing "
                        "<task>...</task>, or, if the user's request needs no "
                        "workspace action, answer them directly and emit <done/>.]"
                    )
                ))
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                _persist_round_state()
                continue

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

                _persist_round_state()

                # Force a continuation round so the LLM can see the search results
                continue

            # ── 🤖 SUB-AGENT TAG ROUTING (MUST PRECEDE ALL ARTIFACT/LOOP GATES) ──
            # The <agent> tag is a delegation, not an artifact dispatch: routing it
            # through the duplicate-artifact gate below would misread the empty
            # affected_artefacts list as a true duplicate and break the loop with
            # the worker never spawned. Execution happens HERE, after the stream
            # has closed — never inside the provider's streaming callback.
            if getattr(ss, "sub_agent_payload", None) is not None:
                payload = ss.sub_agent_payload
                ss.sub_agent_payload = None

                from lollms_client.lollms_agentic.sub_agent_spawner import (
                    run_sub_agent as _run_sub_agent,
                )

                sealed_run = None
                spawn_error = None
                try:
                    sealed_run = _run_sub_agent(
                        discussion=self,
                        config=payload["config"],
                        callback=callback,
                        worker_index=payload["worker_index"],
                        event_mode=event_mode,
                        parent_personality=getattr(self, "_active_personality", None),
                    )
                except Exception as spawn_ex:
                    trace_exception(spawn_ex)
                    spawn_error = _sanitize_host_paths(str(spawn_ex))

                object.__setattr__(self, "_worker_counter", payload["worker_index"])
                _bump_environment_epoch()

                if sealed_run is not None:
                    runs = getattr(ai_msg, "metadata", None)
                    if runs is None or not isinstance(runs, dict):
                        try:
                            runs = dict(ai_msg.metadata or {})
                        except Exception:
                            runs = {}
                    run_list = runs.get("sub_agent_runs")
                    if not isinstance(run_list, list):
                        run_list = []
                    run_list.append(sealed_run)
                    runs["sub_agent_runs"] = run_list
                    ai_msg.metadata = runs
                    report_envelope = sealed_run["report_envelope"]
                else:
                    report_envelope = (
                        f"[AGENT {payload['worker_index']} — FAILURE]\n"
                        f"The specialist agent crashed before producing a report."
                        f"{(' Error: ' + spawn_error) if spawn_error else ''}\n"
                    )

                gist = _scrub_for_llm_context(
                    ss.get_clean_text_so_far()[current_content_length:]
                ).strip()
                if gist:
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=gist,
                    ))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=report_envelope + (
                        "Choose your next move: ANSWER the user (then `<done/>`), "
                        "or spawn another agent with `<agent>` if this report is insufficient."
                    ),
                ))

                _register_completed_action({
                    "kind": "tool",
                    "name": f"agent_{payload['worker_index']}",
                    "params": {},
                    "success": sealed_run is not None,
                    "round": round_count,
                })
                _compress_virtual_history_if_needed()
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                _persist_round_state()
                continue

            # (Artifact loop enforcement removed per the single-signal termination
            # contract: <done/> is the only way the model ends the turn.)

            # ── 🛑 CRITICAL FIX: DISTINGUISH FAILED PATCH FROM TRUE DUPLICATE ──
            # If the LLM emits an artifact tag that was ALREADY processed in a previous round,
            # _StreamState skips dispatching it (affected_artefacts remains empty).
            # HOWEVER, a failed SEARCH/REPLACE patch ALSO results in empty affected_artefacts.
            # We must only force-final-answer for TRUE duplicates, not failed patches.
            if ss.was_action_dispatched() and not ss.tool_trigger and not ss.affected_artefacts:
                if ss.was_last_dispatch_failed():
                    correction_body = ""
                    if getattr(ss, "_last_failure_kind", None) == "agent_tag":
                        ASCIIColors.warning("[ChatMixin] <agent> tag malformed. Injecting agent-syntax correction.")
                        correction_body = (
                            "[SYSTEM: Your last <agent> block was REJECTED — no <task> body was found.\n"
                            "Re-emit the delegation using this EXACT structure:\n"
                            "<agent name=\"worker\" system_prompt=\"You are an expert ...\" max_rounds=\"8\">\n"
                            "<task>\n"
                            "Self-contained instructions for the specialist.\n"
                            "</task>\n"
                            "</agent>\n"
                            "MANDATORY: the instructions MUST be wrapped in literal <task> and </task> tags. "
                            "Plain-text markers like '=== TASK ===' are NOT parsed. "
                            "Do not wrap the delegation in code fences or any other tags.]"
                        )
                    else:
                        ASCIIColors.warning("[ChatMixin] Artifact patch failed. Injecting correction context.")
                        correction_body = (
                            "[SYSTEM: Your last <artifact> SEARCH/REPLACE patch FAILED. The SEARCH block text was not found in the existing file content. "
                            "You MUST retry the patch with a corrected SEARCH block that exactly matches the current file content. "
                            "Look at the 'Fully Loaded File Contents [C]' section in your context to find the exact text to match. "
                            "Do NOT emit <done/> until the patch succeeds or you decide to do a full rewrite instead.]"
                        )
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
                        content=correction_body
                    ))
                    _persist_round_state()
                    continue
                else:
                    # ── TRUE DUPLICATE PATH ──
                    ASCIIColors.warning("[ChatMixin] LLM emitted a duplicate artifact tag. Injecting duplicate warning.")
                    duplicate_history_text = _scrub_for_llm_context(
                        ss.get_clean_text_so_far()[current_content_length:]
                    ).strip()
                    if duplicate_history_text:
                        virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=duplicate_history_text
                        ))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content="[SYSTEM: CRITICAL. You just attempted to recreate an artifact that already exists with the exact same content. This is a loop. You MUST NOT create or update this artifact again. You MUST now provide your final conversational answer to the user, explaining what you have done, and end with <done/>.]"
                    ))
                    _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                    _persist_round_state()
                    continue

            # ── 🤖 SUB-AGENT TAG ROUTING ──
            # The orchestrator emitted an <agent> tag; the spawner already ran
            # the worker (user saw everything live). Feed ONLY the compact
            # report envelope back so the orchestrator can answer or continue.
            if getattr(ss, "sub_agent_payload", None) is not None:
                payload = ss.sub_agent_payload
                ss.sub_agent_payload = None

                gist = _scrub_for_llm_context(
                    ss.get_clean_text_so_far()[current_content_length:]
                ).strip()
                if gist:
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=gist,
                    ))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=payload["report_envelope"] + (
                        "Choose your next move: ANSWER the user (then `<done/>`), "
                        "or spawn another agent with `<agent>` if this report is insufficient."
                    ),
                ))

                _register_completed_action({
                    "kind": "tool",
                    "name": f"agent_{payload['worker_index']}",
                    "params": {},
                    "success": True,
                    "round": round_count,
                })
                _compress_virtual_history_if_needed()
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                continue

            # ── 🧠 DELEGATION ROUTING (PERSONA-AWARE) ──
            # Worker persona: <delegate> is never taught; neutralize drift.
            # Orchestrator persona: route the captured payload to a bounded
            # Worker via DelegationMixin and feed back ONE plain-data envelope.
            if ss.delegation_payload is not None:
                payload = ss.delegation_payload
                ss.delegation_payload = None
                if worker_turn:
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            "[SYSTEM: <delegate> is not available in this context. "
                            "Execute the work yourself with tools or artifacts.]"
                        ),
                    ))
                    continue

                parsed = self._parse_delegation_tag(
                    payload.get("opening_tag", ""), payload.get("body", "")
                )
                if parsed is None:
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            "[SYSTEM: The <delegate> block was malformed (missing or "
                            "empty <task>). Re-emit a valid delegation block with "
                            "<task>...</task> and <context_files>...</context_files>.]"
                        ),
                    ))
                    continue

                task, context_files = parsed
                worker_index = self._worker_counter + 1
                worker_budget = max(2, (resolved_max_rounds - round_count) // 2)
                worker_result = self._run_worker(
                    task=task,
                    context_files=context_files,
                    worker_tools=active_tools,
                    max_worker_rounds=worker_budget,
                    callback=callback,
                    event_meta={
                        "round_id": round_count,
                        "worker_index": worker_index,
                        "event_mode": event_mode,
                    },
                    think=think,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                )
                object.__setattr__(self, "_worker_counter", worker_index)
                _bump_environment_epoch()

                full_round_text = ss.get_clean_text_so_far()
                raw_delegate_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                delegate_gist = _scrub_for_llm_context(raw_delegate_text).strip()
                if delegate_gist:
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=delegate_gist,
                    ))

                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=self._build_worker_report_envelope(worker_result, worker_index),
                ))

                _register_completed_action({
                    "kind": "tool",
                    "name": f"delegate_worker_{worker_index}",
                    "params": {"task": task[:200]},
                    "success": worker_result.get("success", False),
                    "round": round_count,
                })
                _compress_virtual_history_if_needed()
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                _persist_round_state()
                continue

            # ── 🛑 ONE-ACTION-PER-TURN PROTOCOL ──
            # If the StreamState dispatched an artifact, note, skill, or context update
            # (but NOT a tool), we must halt generation, hydrate virtual_history, and re-prompt.
            if ss.was_action_dispatched() and not ss.tool_trigger:
                if ss.affected_artefacts:
                    _bump_environment_epoch()
                    ASCIIColors.info(
                        f"[ChatMixin] Workspace mutated by LLM dispatch "
                        f"(epoch {environment_epoch}). Repetition gates reset."
                    )
                full_round_text = ss.get_clean_text_so_far()
                raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text

                # Sanitize the raw text to remove processing blocks and HTML comments
                clean_history_text = scrub_processing_and_status_blocks(raw_round_text)

                # ── 🧠 VERBATIM TAG PRESERVATION (ANTI-PHANTOM DEMONSTRATION) ──
                # The most recent assistant turn MUST retain the exact raw functional
                # tag it emitted. Stripping it teaches the model (by in-context
                # demonstration) that "prose claim = completed action", which is the
                # root cause of phantom completions on follow-up turns.
                # The rolling-window compressor will fold OLDER entries; only the
                # newest dispatch keeps its verbatim body.
                if not clean_history_text.strip() and ss.affected_artefacts:
                    reconstructed_tags = []
                    for art in ss.affected_artefacts:
                        title = art.get("title", "untitled")
                        atype = art.get("type", "document")
                        content = art.get("content", "")
                        if atype == "skill":
                            desc_attr = f' description="{art.get("description", "")}"' if art.get("description") else ""
                            cat_attr = f' category="{art.get("category", "")}"' if art.get("category") else ""
                            reconstructed_tags.append(f'<skill title="{title}"{desc_attr}{cat_attr}>\n{content}\n</skill>')
                        elif atype == "note":
                            reconstructed_tags.append(f'<note title="{title}">\n{content}\n</note>')
                        else:
                            lang = art.get("language", "")
                            ephemeral_attr = ' ephemeral="true"' if art.get("ephemeral") else ""
                            reconstructed_tags.append(f'<artifact name="{title}" type="{atype}" language="{lang}"{ephemeral_attr}>\n{content}\n</artifact>')
                    clean_history_text = "\n\n".join(reconstructed_tags)
                elif clean_history_text.strip() and ss.affected_artefacts:
                    # Prose existed around the tag. Append the verbatim tag so the
                    # demonstration pattern includes BOTH prose AND the raw tag.
                    appended_tags = []
                    for art in ss.affected_artefacts:
                        title = art.get("title", "untitled")
                        atype = art.get("type", "document")
                        content = art.get("content", "")
                        if atype == "skill":
                            desc_attr = f' description="{art.get("description", "")}"' if art.get("description") else ""
                            cat_attr = f' category="{art.get("category", "")}"' if art.get("category") else ""
                            appended_tags.append(f'<skill title="{title}"{desc_attr}{cat_attr}>\n{content}\n</skill>')
                        elif atype == "note":
                            appended_tags.append(f'<note title="{title}">\n{content}\n</note>')
                        else:
                            lang = art.get("language", "")
                            ephemeral_attr = ' ephemeral="true"' if art.get("ephemeral") else ""
                            appended_tags.append(f'<artifact name="{title}" type="{atype}" language="{lang}"{ephemeral_attr}>\n{content}\n</artifact>')
                    clean_history_text = (clean_history_text.strip() + "\n\n" + "\n\n".join(appended_tags)).strip()

                # ── 🧠 VERBATIM TAG RECONSTRUCTION (ZERO AMNESIA) ──
                # If conversational wrapper was stripped, reconstruct the exact functional XML tag
                # (<skill>, <artifact>, <note>, etc.) so the assistant message retains its exact content.
                if not clean_history_text.strip() and ss.affected_artefacts:
                    reconstructed_tags = []
                    for art in ss.affected_artefacts:
                        title = art.get("title", "untitled")
                        atype = art.get("type", "document")
                        content = art.get("content", "")
                        if atype == "skill":
                            desc_attr = f' description="{art.get("description", "")}"' if art.get("description") else ""
                            cat_attr = f' category="{art.get("category", "")}"' if art.get("category") else ""
                            reconstructed_tags.append(f'<skill title="{title}"{desc_attr}{cat_attr}>\n{content}\n</skill>')
                        elif atype == "note":
                            reconstructed_tags.append(f'<note title="{title}">\n{content}\n</note>')
                        else:
                            lang = art.get("language", "")
                            ephemeral_attr = ' ephemeral="true"' if art.get("ephemeral") else ""
                            reconstructed_tags.append(f'<artifact name="{title}" type="{atype}" language="{lang}"{ephemeral_attr}>\n{content}\n</artifact>')
                    clean_history_text = "\n\n".join(reconstructed_tags)

                virtual_history.append(SimpleNamespace(
                    sender_type="assistant",
                    content=clean_history_text.strip()
                ))

                # Determine the action type and title
                action_type = "artifact"
                action_title = ""
                action_dest = "workspace"
                if ss.affected_artefacts:
                    last_art = ss.affected_artefacts[-1]
                    action_type = last_art.get("type", "artifact")
                    action_title = last_art.get("title", "")
                    if last_art.get("destination") == "handbag":
                        action_dest = f"handbag '{last_art.get('handbag_name', 'personality')}'"

                for dispatched_art in ss.affected_artefacts:
                    _register_completed_action({
                        "kind": "artifact",
                        "title": dispatched_art.get("title", "untitled"),
                        "op": "created" if dispatched_art.get("version", 1) == 1 else "updated",
                        "version": dispatched_art.get("version"),
                        "round": round_count,
                    })

                _compress_virtual_history_if_needed()

                # ── 🛡️ CLEAN ENVELOPE NOTIFICATION (ZERO MIMICRY) ──
                # Use structured <action_result> envelopes rather than conversational prompts
                # that invite the LLM to mimic system reasoning or round-count commentary.
                system_envelope = (
                    f"[SYSTEM: Artifact '{action_title}' created successfully.]\n"
                    f'<action_result type="{action_type}" name="{action_title}" status="SUCCESS" destination="{action_dest}">\n'
                    f"The {action_type} '{action_title}' has been successfully created and saved to {action_dest}.\n"
                    f"Its complete content is recorded in your previous message above.\n"
                    f"Provide your final answer to the user and append `<done/>` on a new line when complete.\n"
                    f"</action_result>"
                )
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=system_envelope
                ))

                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                _persist_round_state()
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
                        clean_history_text = _scrub_for_llm_context(raw_round_text)
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE).strip()
                        if clean_history_text:
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
                            _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="loop_break")
                            break

                        # Force another round to let the LLM correct itself
                        _persist_round_state()
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
                        if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            tool_close_tag = f"{status_err_line}{details_block}<!-- status:failure -->\n</processing>\n\n"
                            ai_msg.content += tool_close_tag
                            _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                            _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                "tool_name": tool_name,
                                "success": False,
                                "output": "",
                                "error": f"'{tool_name}' is a memory system tag, not a tool. Use the XML tag directly.",
                            })

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
                        clean_history_text = _scrub_for_llm_context(raw_round_text)
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
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
                        _persist_round_state()
                        continue

                    full_round_text = ss.get_clean_text_so_far()
                    raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text
                    clean_history_text = _scrub_for_llm_context(raw_round_text).strip()
                    if clean_history_text:
                        virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=f"{clean_history_text}\n\n<tool>{tool_call_json_str}</tool>"
                        ))
                    else:
                        virtual_history.append(SimpleNamespace(
                            sender_type="assistant",
                            content=f"<tool>{tool_call_json_str}</tool>"
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
                        if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            tool_close_tag = f"{status_err_line}{details_block}<!-- status:failure -->\n</processing>\n\n"
                            ai_msg.content += tool_close_tag
                            _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                            _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                "tool_name": tool_name,
                                "success": False,
                                "output": "",
                                "error": f"Tool '{tool_name}' is not available in this session.",
                            })

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
                        clean_history_text = _scrub_for_llm_context(raw_round_text)
                        clean_history_text = re.sub(r'<tool>.*?</tool>', '', clean_history_text, flags=re.DOTALL | re.IGNORECASE)
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
                            _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="loop_break")
                            break

                        # Force another reasoning round to let the LLM correct itself
                        _persist_round_state()
                        continue

                    # ── CONTEXT-AWARE LOOP DETECTION (BEFORE EXECUTION) ──
                    failure_memory = getattr(self, "_failure_memory", None)

                    try:
                        param_signature = json.dumps(tool_params, sort_keys=True, default=str)
                    except Exception:
                        param_signature = str(tool_params)
                    full_signature = f"{tool_name}::{param_signature}"

                    try:
                        current_file_hashes = _hash_workspace_file_refs(self, tool_params)
                    except Exception:
                        current_file_hashes = {}

                    if current_file_hashes:
                        context_token = (
                            f"epoch:{environment_epoch}:"
                            f"rev:{_current_workspace_revision()}:"
                            f"files:" + json.dumps(current_file_hashes, sort_keys=True)
                        )
                    else:
                        context_token = f"epoch:{environment_epoch}:rev:{_current_workspace_revision()}"
                    context_aware_signature = f"{full_signature}::{context_token}"

                    ASCIIColors.info(
                        f"[ChatMixin] Loop check: tool='{tool_name}', "
                        f"sig='{context_aware_signature[:120]}...', "
                        f"in_success_set={context_aware_signature in successful_tool_signatures}, "
                        f"success_set_size={len(successful_tool_signatures)}"
                    )

                    if failure_memory and hasattr(failure_memory, "_signatures"):
                        has_prev_failure = context_aware_signature in failure_memory._signatures
                    else:
                        has_prev_failure = False

                    if has_prev_failure:
                        if self.is_generation_cancelled():
                            was_cancelled = True
                            _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="cancelled")
                            break

                        # ── EPOCH-GATED FAILURE LOOP INTERCEPT ──────────────
                        # We only hard-block when the agent re-issues the byte-
                        # identical failing call AND nothing observable changed
                        # since that failure (same epoch, same file hashes).
                        # Any intervening action (different tool, artifact
                        # build, context unlock, tool that wrote files) bumps
                        # the epoch and re-enables the retry.
                        if environment_epoch == _last_failure_epoch:
                            result_str = (
                                f"Error executing tool '{tool_name}': this exact call (identical parameters "
                                f"AND no environment change since the last failure) already failed on the "
                                f"immediately preceding round. To prevent an infinite loop, execution was "
                                f"blocked. Perform ANY intervening action first (modify an artifact, call a "
                                f"different tool, fix the referenced file), or change your parameters."
                            )
                        else:
                            result_str = ""
                        if not result_str:
                            has_prev_failure = False
                            failure_memory._signatures.discard(context_aware_signature)
                            
                        status_err_line = f"* Tool call blocked to prevent loop.\n"
                        details_block = f"Loop Intercepted:\n{result_str}\n"
                        if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            tool_close_tag = f"{status_err_line}{details_block}<!-- status:failure -->\n</processing>\n\n"
                            ai_msg.content += tool_close_tag
                            _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                            _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                "tool_name": tool_name,
                                "success": False,
                                "output": "",
                                "error": result_str,
                            })

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f'<tool_result name="{tool_name}" status="FAILED">\n'
                                f"{result_str}\n"
                                f"</tool_result>\n\n"
                                f"⚠️ **Tool Execution Failed & Loop Blocked.**\n"
                                f"You attempted to retry a failing tool with identical parameters and unchanged inputs. "
                                f"The system has blocked this to prevent an infinite loop. "
                                f"You MUST now write a final response to the user explaining that the operation could not "
                                f"be completed, detailing the error above, and suggesting possible workarounds or "
                                f"alternative approaches. Do NOT attempt to call the tool again."
                            )
                        ))
                        continue

                    if context_aware_signature in successful_tool_signatures:
                        ASCIIColors.warning(
                            f"[ChatMixin] Repetitive SUCCESS loop blocked for '{tool_name}'. "
                            f"Signature recorded and workspace state unchanged since last success."
                        )
                        status_err_line = f"* Tool call blocked to prevent success loop.\n"
                        details_block = f"Loop Intercepted:\nRepetitive successful tool call blocked (workspace state unchanged)\n<!-- status:failure -->\n</processing>\n\n"
                        if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                            ai_msg.content += status_err_line + details_block
                            _cb(callback, status_err_line + details_block, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                            _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                                "tool_name": tool_name,
                                "success": False,
                                "output": "",
                                "error": "Repetitive successful tool call blocked (workspace state unchanged).",
                            })

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=(
                                f'<tool_result name="{tool_name}" status="FAILED">\n'
                                f"Repetitive tool call detected. The output is already in your context.\n"
                                f"</tool_result>\n\n"
                                f"⚠️ **Tool Execution Blocked.**\n"
                                f"You have already successfully called '{tool_name}' with these exact parameters, "
                                f"and the workspace files it depends on have not changed since. The system has blocked "
                                f"this duplicate call. If you intentionally modified the target artifact, re-emit the "
                                f"<artifact> change and try again; otherwise analyze the data already in your context "
                                f"and write your final answer. Do NOT attempt to call the tool again."
                            )
                        ))
                        continue

                    # 2. Strip ONLY the raw <tool> JSON tag from the UI/DB buffer (ai_msg.content).
                    if tool_call_json_str in ai_msg.content:
                        ai_msg.content = ai_msg.content.replace(f"<tool>{tool_call_json_str}</tool>", "")
                        ai_msg.content = ai_msg.content.replace(tool_call_json_str, "")

                    tool_res = None
                    _lcp_executed = False

                    if active_tools and tool_name in active_tools and "callable" not in active_tools[tool_name]:
                        if lcp_binding and hasattr(lcp_binding, "execute_tool"):
                            _old_cwd_lcp = os.getcwd()

                            # Resolve workspace_data path
                            if hasattr(self, "workspace_data_path") and self.workspace_data_path:
                                _lcp_workspace_dir = Path(self.workspace_data_path)
                            else:
                                _base_ws = Path(self.workspace_path) if hasattr(self, "workspace_path") and self.workspace_path else Path("./data_workspace")
                                _lcp_workspace_dir = _base_ws / self.id / "workspace_data"

                            _lcp_workspace_dir.mkdir(parents=True, exist_ok=True)
                            _lcp_workspace_str = str(_lcp_workspace_dir.resolve())

                            try:
                                os.chdir(_lcp_workspace_str)

                                try:
                                    self.artefacts.sync_all_active_to_disk()
                                except Exception as sync_ex:
                                    trace_exception(sync_ex)

                                # ── Take BEFORE Snapshot (LCP Path) ──
                                _lcp_files_before = {}
                                _lcp_cwd = Path(_lcp_workspace_str)
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
                                except Exception as e:
                                    trace_exception(e)
                                    tool_res = {
                                        "success": False,
                                        "error": _sanitize_host_paths(f"Tool '{tool_name}' crashed: {e}"),
                                        "traceback": _sanitize_host_paths(traceback.format_exc())
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
                                os.chdir(_old_cwd_lcp)
                        else:
                            tool_res = {
                                "success": False,
                                "error": f"Tool '{tool_name}' has no callable and no LCP tools binding is available on the client.",
                                "status_code": 404
                            }
                            _lcp_executed = True
                    elif active_tools and tool_name in active_tools and "callable" in active_tools[tool_name]:
                        _lcp_executed = False
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

                    # Execute the tool sequentially
                    try:
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
                                _tool_sig_params = inspect.signature(active_tools[tool_name]["callable"]).parameters
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
                                    _last_failure_epoch = environment_epoch
                                    if hasattr(failure_memory, "record_failure_by_signature"):
                                        failure_memory.record_failure_by_signature(context_aware_signature, error_msg)
                                    else:
                                        if not hasattr(failure_memory, "_signatures"):
                                            object.__setattr__(failure_memory, "_signatures", set())
                                        failure_memory._signatures.add(context_aware_signature)

                                # 🛑 ARCHITECTURAL FIX: Removed the flawed has_prev_failure check here.
                                # The previous code recorded the signature and immediately checked if it existed,
                                # which always evaluated to True and caused every failure to be mislabeled as "Loop Intercepted".
                                result_str = f"Error executing tool '{tool_name}': {error_msg}"
                                clean_result_str = result_str
                                status_done_line = f"* Completed execution with errors.\n"
                                details_block = f"Error Logs:\n{error_msg}\n"
                            else:
                                raw_output = tool_res.get("output", tool_res)

                                if isinstance(raw_output, dict) and isinstance(raw_output.get("output"), (str, int, float, bool)):
                                    raw_output = raw_output["output"]
                                elif isinstance(raw_output, dict):
                                    nested = raw_output.get("output")
                                    if isinstance(nested, dict):
                                        raw_output = nested

                                if isinstance(raw_output, dict):
                                    extracted = None
                                    for key in ("output", "content", "text", "result", "data", "page_content", "summary", "extract", "html", "body", "query", "pages"):
                                        if key in raw_output:
                                            extracted = raw_output[key]
                                            break

                                    if extracted is not None:
                                        raw_output = extracted
                                    else:
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

                                        clean_result_str = (
                                            f"[SYSTEM: Tool returned {tool_output_tokens} tokens of text. "
                                            f"It has been saved to '{log_filename}'. Inspect precisely with "
                                            f"tool_read_lines / tool_read_chars / tool_grep_file on that file "
                                            f"(line window, char offset, or grep with context), or <unlock_file> it "
                                            f"to load a portion. Do not re-run the producing tool.]"
                                        )

                                status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                # 🛡️ CRITICAL FIX: Guard against NoneType output from tools
                                if full_dump is None:
                                    full_dump = "Tool executed successfully but returned no output content."
                                if not isinstance(full_dump, str):
                                    try:
                                        full_dump = json.dumps(full_dump, indent=2, default=str, ensure_ascii=False)
                                    except Exception:
                                        full_dump = str(full_dump)
                                # Format user-facing log display cleanly (epurated of raw XML artifacts)
                                if isinstance(tool_res, dict) and (tool_res.get("artefacts") or tool_res.get("artifacts_created")):
                                    art_list = tool_res.get("artefacts") or []
                                    art_names = [a.get("title") for a in art_list if isinstance(a, dict)] or tool_res.get("artifacts_created", [])
                                    details_block = f"Output Logs:\n✅ Saved artifact(s): {', '.join(art_names)} to workspace and disk.\n"
                                elif "<artifact" in full_dump or "<artefact" in full_dump or "<<<<<<< SEARCH" in full_dump:
                                    art_title_match = re.search(r'(?:name|title)=["\']([^"\']+)["\']', full_dump)
                                    target_name = art_title_match.group(1) if art_title_match else "artifact file"
                                    if "SEARCH" in full_dump:
                                        ui_log = f"Applied SEARCH/REPLACE modifications to '{target_name}'."
                                    else:
                                        ui_log = f"Created/updated '{target_name}' in workspace."
                                    _cb(callback, ui_log, MSG_TYPE.MSG_TYPE_INFO, {"type": "artifact_status", "target": target_name})
                                    safe_dump = _build_windowed_output_preview(full_dump)
                                    details_block = f"Output Logs:\n{safe_dump}\n"
                                else:
                                    safe_output = _build_windowed_output_preview(full_dump)
                                    details_block = f"Output Logs:\n{safe_output}\n"
                        else:
                            result_str = str(tool_res) if tool_res is not None else "No output returned."
                            if "error" in result_str.lower() or "fail" in result_str.lower():
                                if failure_memory:
                                    _last_failure_epoch = environment_epoch
                                    if hasattr(failure_memory, "record_failure_by_signature"):
                                        failure_memory.record_failure_by_signature(context_aware_signature, result_str)
                                    else:
                                        if not hasattr(failure_memory, "_signatures"):
                                            object.__setattr__(failure_memory, "_signatures", set())
                                        failure_memory._signatures.add(context_aware_signature)
                                clean_result_str = result_str
                                status_done_line = f"* Completed execution with errors.\n"
                                details_block = f"Error Logs:\n{result_str}\n"
                            else:
                                status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                clean_result_str = _sanitize_tool_result(tool_res, client=self.lollmsClient)
                                safe_output = _build_windowed_output_preview(result_str)
                                details_block = f"Output Logs:\n{safe_output}\n"
                    except Exception as e:
                        trace_exception(e)
                        if debug_enabled:
                            self._dump_error(
                                error=e,
                                context_desc="Tool Execution Error",
                                round_count=round_count,
                                extra_data={"tool_name": tool_name, "parameters": tool_params}
                            )
                        if failure_memory:
                            _last_failure_epoch = environment_epoch
                            if hasattr(failure_memory, "record_failure_by_signature"):
                                failure_memory.record_failure_by_signature(context_aware_signature, str(e))
                            else:
                                if not hasattr(failure_memory, "_signatures"):
                                    object.__setattr__(failure_memory, "_signatures", set())
                                failure_memory._signatures.add(context_aware_signature)
                        result_str = _sanitize_host_paths(f"Error executing tool '{tool_name}': {e}")
                        clean_result_str = _sanitize_host_paths(f"Error executing tool '{tool_name}': {e}")
                        status_done_line = f"* Execution crashed.\n"
                        details_block = f"Crash Details:\n{_sanitize_host_paths(str(e))}\n"
                        tool_res = {"success": False, "error": _sanitize_host_paths(str(e))}
                    inner_res = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res

                    is_failure = (
                        (isinstance(inner_res, dict) and inner_res.get("success") is False)
                        or (isinstance(tool_res, dict) and tool_res.get("status_code", 200) not in (200, 201))
                        or (isinstance(tool_res, dict) and bool(tool_res.get("error")))
                        or (isinstance(inner_res, dict) and bool(inner_res.get("error")) and not inner_res.get("success", True))
                        or (isinstance(tool_res, dict) and tool_res.get("return_code", 0) != 0)
                        or (isinstance(inner_res, dict) and inner_res.get("return_code", 0) != 0)
                    )
                    if tool_res is None:
                        is_failure = True
                    elif "crashed" in status_done_line.lower():
                        is_failure = True
                    if isinstance(tool_res, dict):
                        res_success_flag = tool_res.get("success")
                        if res_success_flag is True:
                            is_failure = bool(tool_res.get("error")) and not tool_res.get("success", True)
                    status_meta = "failure" if is_failure else "success"
                    if event_mode in (EventMode.PROCESSING_TAG_MODE, EventMode.MIXED_MODE):
                        tool_close_tag = f"{status_done_line}{details_block}<!-- status:{status_meta} -->\n</processing>\n\n"
                        ai_msg.content += tool_close_tag
                        _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    if event_mode in (EventMode.FULL_CALLBACK_MODE, EventMode.MIXED_MODE):
                        cb_error = (tool_res.get("error") if isinstance(tool_res, dict) else None) if is_failure else None
                        _cb(callback, "", MSG_TYPE.MSG_TYPE_TOOL_END, {
                            "tool_name": tool_name,
                            "parameters": tool_params,
                            "success": not is_failure,
                            "output": details_block,
                            "error": cb_error,
                        })

                    tool_success = not is_failure
                    if not tool_success:
                        clean_result_str = re.sub(
                            r'<processing[^>]*>.*?(?:</processing>|$)', '',
                            clean_result_str, flags=re.DOTALL | re.IGNORECASE
                        )
                        clean_result_str = re.sub(r'<!-- status:[^>]*-->', '', clean_result_str, flags=re.IGNORECASE)
                        clean_result_str = re.sub(r'</processing>', '', clean_result_str, flags=re.IGNORECASE)
                        clean_result_str = re.sub(r'<tool_result[^>]*>.*?(?:</tool_result>|$)', '', clean_result_str, flags=re.DOTALL | re.IGNORECASE)
                        clean_result_str = clean_result_str.strip()

                    if tool_success:
                        successful_tool_signatures.add(context_aware_signature)
                        ASCIIColors.info(f"[ChatMixin] Recorded successful signature for '{tool_name}'. Total successful: {len(successful_tool_signatures)}")
                        
                        # Ingest any artifacts produced by tools (e.g. spinoff sub-agents)
                        if isinstance(tool_res, dict) and (tool_res.get("artefacts") or tool_res.get("artifacts_created")):
                            arts_to_ingest = tool_res.get("artefacts") or []
                            for a in arts_to_ingest:
                                if isinstance(a, dict):
                                    if a not in ss.affected_artefacts:
                                        ss.affected_artefacts.append(a)
                                    if a not in getattr(self, "_affected_artefacts_this_turn", []):
                                        self._affected_artefacts_this_turn.append(a)
                                    art_title = a.get("title", "")
                                    art_type = a.get("type", "code")
                                    art_ver = a.get("version", 1)
                                    card_tag = f'<lollms_artifact id="{art_title}" type="{art_type}" version="{art_ver}" />'
                                    if card_tag not in ai_msg.content:
                                        ai_msg.content += f"\n\n{card_tag}\n"

                            self.touch()
                            self.commit()
                    else:
                        _bump_environment_epoch()

                    tool_calls_this_turn.append({
                        "name": tool_name,
                        "params": tool_params,
                        "result": {"output": _sanitize_host_paths(clean_result_str), "success": tool_success}
                    })

                    # ── 📊 LOG TOOL CALL ACTION ──
                    turn_actions_log.append({
                        "action": "tool_call",
                        "tool_name": tool_name,
                        "success": tool_success,
                        "round": round_count
                    })

                    _register_completed_action({
                        "kind": "tool",
                        "name": tool_name,
                        "params": tool_params,
                        "success": tool_success,
                        "round": round_count,
                    })

                    # ── 🔄 ACTION-WINDOW RECOLLECTION COMPRESSION ──
                    # Expired action rounds are digested into the system-zone
                    # narrative; in-window rounds stay verbatim.
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
                                f"   a) Run follow-up Python immediately with 'tool_execute_python_code', passing the "
                                f"script inline through the 'code' parameter, OR\n"
                                f"   b) Build the HTML animation artifact the user requested, OR\n"
                                f"   c) Provide your final analysis answer.\n"
                                f"   Do NOT run another SQL query unless you need genuinely different data.\n"
                            )
                        elif tool_name in ("tool_execute_python_code", "tool_execute_python_file"):
                            next_step_guidance = (
                                f"6. 🐍 **CODE EXECUTION & FILE ORIENTATION**: 'tool_execute_python_code' runs inline "
                                f"Python (the 'code' parameter) and never saves files. 'tool_execute_python_file' runs "
                                f"an EXISTING workspace .py file ('file_name' parameter) and is read-only. The sandbox "
                                f"CWD IS the workspace root: files created via <artifact> tags sit as siblings of your "
                                f"code, so import them directly ('from rlc_filter import RLCFilter') with NO 'workspace/' "
                                f"prefix and NO sys.path manipulation. To persist a new script for later reuse, emit an "
                                f"<artifact type=\"code\" name=\"...\"> tag first, then run it with "
                                f"'tool_execute_python_file'.\n"
                                f"   🚨 **SIZE DOCTRINE**: 'tool_execute_python_code' accepts a MAXIMUM of 2000 "
                                f"characters. Larger programs are hard-rejected at the gate. For ANY substantial "
                                f"program (optimization loops, simulations, multi-function scripts, plotting "
                                f"pipelines), ALWAYS use the artifact + 'tool_execute_python_file' path. Never "
                                f"attempt to squeeze a large program inline.\n"
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
                            f"⚠️ **Tool Execution Failed — SELF-CORRECTION MANDATE (LONG-HORIZON MODE).**\n"
                            f"The tool '{tool_name}' failed. The turn is NOT over. You MUST fix the problem yourself:\n"
                            f"1. **DIAGNOSE**: Read the error/traceback above and identify the EXACT offending line or parameter.\n"
                            f"2. **FIX**: If the error is in a workspace file you previously created, emit a corrected `<artifact>` tag (full rewrite or SEARCH/REPLACE patch) with the fix applied. If the parameters were wrong, re-emit a corrected `<tool>` call.\n"
                            f"3. **RETRY**: Re-run the tool to verify the fix.\n"
                            f"4. **ITERATE**: Repeat diagnose→fix→retry until success or until you determine the approach is truly impossible.\n"
                            f"5. Only after the task genuinely succeeds, or you have exhausted reasonable fixes, write your final analysis to the user and emit `<done/>` on a new line.\n"
                            f"Do NOT end your response with prose only. Do NOT claim the task is complete while errors remain unverified."
                        )
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=user_part
                        ))
                    _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                    _persist_round_state()
                    continue
                else:
                    _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                    _persist_round_state()
                    break
            else:
                full_round_text = ss.get_clean_text_so_far()
                raw_round_text = full_round_text[current_content_length:] if current_content_length < len(full_round_text) else full_round_text

                # ── CONTEXT UNLOCK CONTINUATION ──
                # Unlocking a file is a state change, not an answer: the model
                # must get a continuation round to actually use the content.
                if ss.context_unlock_requested and not was_cancelled:
                    unlock_files_str = ', '.join(ss.context_unlocked_files)
                    clean_unlock_text = scrub_processing_and_status_blocks(raw_round_text)
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_unlock_text
                    ))
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=(
                            f'<action_result type="context_unlock" status="SUCCESS">\n'
                            f"The following files are now fully loaded in your context: {unlock_files_str}.\n"
                            f"Proceed with your task using the loaded content.\n"
                            f"</action_result>"
                        )
                    ))
                    ss.context_unlock_requested = False
                    _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                    _persist_round_state()
                    continue

                # ── 🛑 SOVEREIGN <done/> TERMINATION CONTRACT ──
                # A text-only round is NOT a terminal state. Per the single-
                # signal doctrine, the loop runs until the model explicitly
                # emits <done/> (or <end/>). Persist the round and prompt the
                # model to either take a real action or terminate.
                ASCIIColors.info("[ChatMixin] Text-only round without <done/>. Continuing loop until explicit termination.")
                clean_history_text = scrub_processing_and_status_blocks(raw_round_text)
                if clean_history_text.strip():
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=clean_history_text.strip()
                    ))
                virtual_history.append(SimpleNamespace(
                    sender_type="user",
                    content=(
                        "[SYSTEM: TERMINATION PROTOCOL]\n"
                        "Your last message contained no action and no `<done/>` tag.\n"
                        "You have exactly two valid moves:\n"
                        "1. EMIT a functional tag now (`<tool>{...}</tool>`, "
                        "`<artifact name=\"...\">...</artifact>`, `<agent>...</agent>`, "
                        "`<unlock_file>...</unlock_file>`, etc.) to perform the work.\n"
                        "2. FINISH by writing your final answer to the user and "
                        "emitting `<done/>` on a new line.\n"
                        "Announcing an action in prose performs nothing. Choose now."
                    )
                ))
                _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="action")
                _persist_round_state()
                continue

        # ── 11. Final Post-Processing & Database Commit ──

        if ss is not None and round_event_state["last_status"] is None:
            _emit_round_event(MSG_TYPE.MSG_TYPE_ROUND_END, status="max_rounds")

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
            _persist_round_state()
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
            failed_tools_pending_fix = any(
                vh.sender_type == "user"
                and "SELF-CORRECTION MANDATE" in (vh.content or "")
                for vh in virtual_history
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

        # Ensure any mimicked markers are purged from the saved content
        ai_msg.content = re.sub(r'\[🔒[^\]]*\]', '', ai_msg.content).strip()

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
        existing_sub_agent_runs = ai_msg.metadata.get("sub_agent_runs")
        existing_virtual_history = ai_msg.metadata.get("virtual_history")
        ai_msg.metadata = {
            "mode": "agentic" if tool_calls_this_turn else "direct",
            "tool_calls": tool_calls_this_turn,
            "artefacts_modified": [a.get("title") for a in (ss.affected_artefacts if ss else [])]
        }
        if existing_sub_agent_runs:
            ai_msg.metadata["sub_agent_runs"] = existing_sub_agent_runs
        if existing_virtual_history:
            ai_msg.metadata["virtual_history"] = existing_virtual_history
        if failed_tools_pending_fix and round_count >= resolved_max_rounds:
            ai_msg.metadata["ended_on_unresolved_failure"] = True

        # Auto dream (only if memory is enabled)
        dream_report = None
        if enable_memory and enable_auto_dream and _mm is not None:
            try:
                dream_report = _mm.dream(self.lollmsClient)
            except Exception as ex:
                trace_exception(ex)

        # Unconditionally commit the final message and discussion state to the database
        _persist_round_state()

        self.scratchpad = ""
        object.__setattr__(self, '_active_callback', None)

        # 🛡️ CRITICAL FIX: Always reset the cancellation flag at the end of the turn.
        # This ensures that pre-turn and mid-turn cancellation signals are consumed
        # and do not bleed into subsequent turns.
        self.reset_cancel_state()

        # ── 🔬 SCIENTIFIC DEBUG: EXPORT CONTEXT DUMP ──
        # Dumps the exact virtual_history (LLM context) and ai_msg.content (UI context)
        # to a JSON file in the discussion workspace to verify context separation.
        if debug_enabled:
            try:
                debug_dir = Path(self.workspace_data_path) / "_debug_dumps"
                debug_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
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
                    json.dump(dump_payload, f, indent=2, default=str, ensure_ascii=False)

                ASCIIColors.info(f"[ChatMixin] 🔬 Debug context dump saved to: {dump_file}")
            except Exception as dump_err:
                ASCIIColors.warning(f"[ChatMixin] Failed to write debug context dump: {dump_err}")

        _has_tti = getattr(self.lollmsClient, 'tti', None) is not None or bool(getattr(self.lollmsClient, 'tti_model_profiles_registry', None))

        all_turn_artefacts = []
        seen_art_titles = set()
        for a in (getattr(self, "_affected_artefacts_this_turn", []) or []) + (ss.affected_artefacts if ss else []):
            if isinstance(a, dict) and a.get("title") and a["title"] not in seen_art_titles:
                seen_art_titles.add(a["title"])
                all_turn_artefacts.append(a)

        return {
            "user_message": user_msg,
            "ai_message": ai_msg,
            "sources": [],
            "artefacts": all_turn_artefacts,
            "memory_report": mem_report,
            "dream_report": dream_report,
            "was_cancelled": was_cancelled,
            "tti_available": _has_tti
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