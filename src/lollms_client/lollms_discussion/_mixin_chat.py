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
import os
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from types import SimpleNamespace
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE
from ._message import LollmsMessage
from lollms_client.lollms_artefact import ArtefactType, make_image_id, ArtefactVisibility, ArtefactStatus
from lollms_client.lollms_memory import FailureMemory
# ── Cancellation state & limits ──────────────────────────────────────────────
_MAX_BRACKET_BUF = 256

_TAG_STARTS = [
    "<tool>",
    "<think>", "<think ",
    "<artifact", "<artefact",
    "<generate_image", "<edit_image",
    "<note", "<skill",
    "<lollms_inline",
    "<lollms_form",
    "<mem_new", "<mem_update", "<mem_tag", "<mem_load", "<mem_delete",
]

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
    "<add_files_to_context": ("context_unlock",      MSG_TYPE.MSG_TYPE_INFO,    MSG_TYPE.MSG_TYPE_INFO,              "</add_files_to_context>"),
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


# ── Tool Result Sanitizer ───────────────────────────────────────────────────

_BASE64_RE = re.compile(r'^[A-Za-z0-9+/=\s]{500,}$')

# Field names whose values are typically large base64 binaries that should NOT
# be fed back to the LLM (they bloat context, confuse the model, and trigger
# tool-stutter loops where the LLM re-invokes the same tool on the data it
# just produced).
_BINARY_BLOB_KEYS = {
    "plot_b64", "image_b64", "audio_b64", "video_b64", "file_b64",
    "screenshot_b64", "pdf_b64", "thumbnail_b64", "base64",
    "binary", "raw_image", "image_data", "raw_data",
}

_MAX_TOOL_RESULT_CHARS = 4000


def _is_large_base64(v: str) -> bool:
    """Heuristic: a long string composed of base64 alphabet + whitespace."""
    sample = v.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(sample) < 500:
        return False
    return bool(_BASE64_RE.match(sample[:1000]))


def _sanitize_tool_result(
    tool_res: Any,
    max_chars: int = _MAX_TOOL_RESULT_CHARS,
) -> str:
    """
    Converts an arbitrary tool execution result into a clean, LLM-friendly
    text representation.
    Rules
    -----
    1.  If ``tool_res`` (or its LCP-wrapped ``output`` inner dict) exposes a
        ``prompt_injection`` key anywhere in its tree, that string is used
        verbatim. Tool authors craft ``prompt_injection`` to tell the LLM
        *exactly* what to do next (e.g. reference the produced file with an
        <img /> tag). Using it as the LLM-facing message prevents the LLM
        from re-running the same tool on the freshly-produced artifact.
    2.  Known large-binary fields (``plot_b64``, ``image_b64``, ...) are
        stripped and replaced with a tiny "[base64 blob stripped: 24.3KB]"
        note so the LLM knows a file was produced without ingesting the data.
    3.  Any standalone long base64-looking string is replaced with the same
        note (defence in depth).
    4.  Any string longer than ``max_chars`` is truncated with an ellipsis.
    5.  Lists are capped at 50 entries and walked recursively.
    6.  The result is always returned as a plain ``str`` (JSON-serialised
        when the input was structured).
    """

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

    pinj = _find_prompt_injection(tool_res)
    if pinj:
        success = True
        inner = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
        if isinstance(inner, dict):
            success = inner.get("success", True)
        success_status = "✓ Success" if success else "⚠ Partial Success"
        return f"{success_status}\n{pinj}"

    sanitized = _walk(tool_res)
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
    ):
        self.discussion = discussion
        self.callback = callback
        self.ai_message = ai_message
        self.enable_artefacts = enable_artefacts
        self.enable_in_message_status = enable_in_message_status
        self.auto_activate = auto_activate_artefacts
        self.content_offset = content_offset

        self.enable_notes = enable_notes if enable_artefacts else False
        self.enable_skills = enable_skills if enable_artefacts else False
        self.enable_inline_widgets = enable_inline_widgets if enable_artefacts else False
        self.enable_forms = enable_forms if enable_artefacts else False

        self.tool_trigger = False
        self.tool_json_data = ""
        self.affected_artefacts = []
        self.processed_tags = set()

        # Track context unlock requests to force continuation round
        self.context_unlock_requested = False
        self.context_unlocked_files: List[str] = []

        self._is_accumulating_tool = False
        self._tool_buffer = ""

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str) or not chunk:
            return True

        self.ai_message.content += chunk
        _cb(self.callback, chunk, MSG_TYPE.MSG_TYPE_CHUNK)

        if self._is_accumulating_tool:
            self._tool_buffer += chunk

            if '</tool>' in self._tool_buffer:
                end_idx = self._tool_buffer.find('</tool>')
                full_tool_call = self._tool_buffer[:end_idx + len('</tool>')]
                json_body = full_tool_call.strip().lstrip('<tool>').rstrip('</tool>').strip()

                self._is_accumulating_tool = False
                self._tool_buffer = ""

                keep_generating = self._dispatch_closed_tag("tool", "", json_body, full_tool_call)
                if not keep_generating:
                    return False

                self.content_offset = len(self.ai_message.content)
                return True
            else:
                return True

        content = self.ai_message.content[self.content_offset:]

        if not content.strip():
            return True

        last_open_think = content.rfind("<think>")
        last_close_think = content.rfind("</think>")
        is_inside_thoughts = (last_open_think != -1) and (last_open_think > last_close_think)

        if is_inside_thoughts:
            return True

        last_open_art = max(content.rfind("<artifact"), content.rfind("<artefact"))
        last_close_art = max(content.rfind("</artifact>"), content.rfind("</artefact>"))
        is_inside_artifact = (last_open_art != -1) and (last_open_art > last_close_art)

        if is_inside_artifact:
            if not content.endswith("</artifact>") and not content.endswith("</artefact>"):
                return True

        safe_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        safe_content = re.sub(r'<think>.*$', '', safe_content, flags=re.DOTALL | re.IGNORECASE)

        xml_pattern = re.compile(
            r'<(artifact|artefact|note|skill|tool)\b([^>]*?)>(.*?)</\1>',
            re.DOTALL | re.IGNORECASE
        )

        xml_matches = list(xml_pattern.finditer(safe_content))

        for match in xml_matches:
            full_match_text = match.group(0)
            if full_match_text in self.processed_tags:
                continue

            self.processed_tags.add(full_match_text)
            tag_name = match.group(1).lower()
            attrs_str = match.group(2)
            body = match.group(3).strip()

            try:
                keep_generating = self._dispatch_closed_tag(tag_name, attrs_str, body, full_match_text)
                if not keep_generating:
                    return False
            except Exception as dispatch_err:
                trace_exception(dispatch_err)
                ASCIIColors.error(f"[StreamState] XML dispatch failed for tag '{tag_name}': {dispatch_err}")

        if '<tool>' in safe_content:
            start_idx = safe_content.find('<tool>')
            end_idx = safe_content.find('</tool>', start_idx)

            if end_idx != -1:
                full_tool_call = safe_content[start_idx:end_idx + len('</tool>')]
                json_body = full_tool_call.strip().lstrip('<tool>').rstrip('</tool>').strip()

                self.processed_tags.add(full_tool_call)
                keep_generating = self._dispatch_closed_tag("tool", "", json_body, full_tool_call)
                if not keep_generating:
                    return False
            else:
                self._is_accumulating_tool = True
                self._tool_buffer = safe_content[start_idx:]

        return True

    def _dispatch_closed_tag(self, tag_name: str, attrs_str: str, body: str, full_match_text: str) -> bool:
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
                except Exception as e:
                    ASCIIColors.error(f"Failed to apply patch: {e}")
                    art = None
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

            # ── SWIFT REPLACEMENT PROTOCOL ──
            # Strips the full raw XML from the message body and replaces it with an unmistakable SYSTEM marker
            # This prevents the LLM from mimicking the placeholder in future turns
            # NOTE: The triple-bracket format is NEVER used in valid XML, making it impossible to confuse with functional tags
            placeholder = f"\n[🔒SYSTEM_ARTIFACT_CREATED:{title}|{atype}]\n"
            self.ai_message.content = self.ai_message.content.replace(full_match_text, placeholder)

            # Fire an event update to the UI so it cleanly rebuilds and replaces the code block
            _cb(self.callback, placeholder, MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_updated" if not is_new else "artifact_created",
                "title": title,
                "version": art.get("version", 1) if art else 1,
                "art_type": atype
            })
            return True

        # 2. Tools Execution Trigger
        elif tag_name in ("tool", "tool"):
            self.tool_trigger = True

            # ── ROBUST JSON PARSING & NORMALIZATION (CRITICAL FIX) ──
            # LLMs often hallucinate flat structures: {"name": "tool", "arg": "val"}
            # instead of nested: {"name": "tool", "parameters": {"arg": "val"}}
            # We MUST normalize this here to prevent execution failures.
            try:
                raw_data = json.loads(body)
                if isinstance(raw_data, dict):
                    # ALWAYS normalize to nested structure for consistency
                    tool_name = raw_data.get("name", "")

                    # Check if already nested
                    if "parameters" in raw_data and isinstance(raw_data["parameters"], dict):
                        self.tool_json_data = body
                    else:
                        params = {k: v for k, v in raw_data.items() if k != "name"}
                        normalized_data = {"name": tool_name, "parameters": params}
                        self.tool_json_data = json.dumps(normalized_data)
                else:
                    self.tool_json_data = body
            except json.JSONDecodeError as je:
                self.tool_json_data = body
                ASCIIColors.error(f"[StreamState] JSON decode failed: {je}")

            # Halt generation instantly so the executor can take over the loop
            return False

        # 3. User Note
        elif tag_name == "note":
            if not self.enable_notes:
                return True
            title = attrs.get("title") or attrs.get("name") or f"note_{uuid.uuid4().hex[:8]}"
            art = self.discussion.artefacts.add(
                title=title, artefact_type=ArtefactType.NOTE, content=body, active=self.auto_activate
            )
            if art:
                self.affected_artefacts.append(art)

            # Use distinctive SYSTEM marker that cannot be mimicked
            placeholder = f"\n[🔒SYSTEM_NOTE_CREATED:{title}]\n"
            self.ai_message.content = self.ai_message.content.replace(full_match_text, placeholder)
            _cb(self.callback, placeholder, MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_created",
                "title": title,
                "art_type": "note"
            })
            return True

        # 4. Long-term Skill
        elif tag_name == "skill":
            if not self.enable_skills:
                return True
            title = attrs.get("title") or attrs.get("name") or f"skill_{uuid.uuid4().hex[:8]}"
            desc = attrs.get("description", "")
            cat = attrs.get("category", "")
            art = self.discussion.artefacts.add(
                title=title, artefact_type=ArtefactType.SKILL, content=body, active=self.auto_activate, description=desc, category=cat
            )
            if art:
                self.affected_artefacts.append(art)

            # Use distinctive SYSTEM marker that cannot be mimicked
            placeholder = f"\n[🔒SYSTEM_SKILL_CREATED:{title}]\n"
            self.ai_message.content = self.ai_message.content.replace(full_match_text, placeholder)
            _cb(self.callback, placeholder, MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_created",
                "title": title,
                "art_type": "skill"
            })
            return True

        # 5. Multi-tier Context Unlocks
        # 5. Multi-tier Context Unlocks
        elif tag_name == "add_files_to_context":
            from ._artefacts import ArtefactVisibility
            targets = [t.strip() for t in body.splitlines() if t.strip()]
            unlocked_files = []
            already_visible = []
            not_found = []

            for t_file in targets:
                art = self.discussion.artefacts.get(t_file)
                if not art:
                    not_found.append(t_file)
                elif art.get("visibility") == ArtefactVisibility.FULL:
                    # Already fully loaded - skip redundant operation
                    already_visible.append(t_file)
                else:
                    self.discussion.artefacts.set_visibility(t_file, ArtefactVisibility.FULL)
                    unlocked_files.append(t_file)

            if unlocked_files:
                self.discussion.commit()
                # Mark that we need a continuation round after this generation
                self.context_unlock_requested = True
                self.context_unlocked_files.extend(unlocked_files)

            # Build informative feedback - distinguish between newly unlocked and already visible
            feedback_parts = []
            if unlocked_files:
                feedback_parts.append(f"✅ Loaded: {', '.join(unlocked_files)}")
            if already_visible:
                feedback_parts.append(f"⚠️ Already in context: {', '.join(already_visible)}")
            if not_found:
                feedback_parts.append(f"❌ Not found: {', '.join(not_found)}")
            
            # Use a distinctive SYSTEM marker that cannot be mimicked
            placeholder = f"\n\n[SYSTEM: Context updated — {'; '.join(feedback_parts)}]\n\n"
            self.ai_message.content = self.ai_message.content.replace(full_match_text, placeholder)
            _cb(self.callback, placeholder, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
            return True

        return True

    def passthrough(self, chunk, msg_type=None, meta=None) -> bool:
        if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
            if msg_type in (MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, MSG_TYPE.MSG_TYPE_REASONING):
                self.ai_message.thoughts = (self.ai_message.thoughts or "") + (chunk or "")
            return _cb(self.callback, chunk, msg_type, meta)
        return True

    def flush_remaining_buffer(self):
        """No buffered tokens to flush because we have zero-latency passthrough."""
        pass

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

        # 🛑 CRITICAL FIX: Initialize FailureMemory IMMEDIATELY on discussion object
        # This ensures it exists BEFORE the first chat() call
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
        debug: bool = False,
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
        enable_silent_artefact_explanation: bool = True,
        memory_manager=None,
        enable_artefacts:             bool = True,
        enable_memory:                bool = True,
        enable_auto_dream:            bool = True,
        enable_deep_memory_pulling:   bool = True,
        prehydrate_rag:               bool = True,
        max_reasoning_steps:          int = 20,
        enable_in_message_status:     bool = False,
        enable_sub_agents:            bool = False,  # Enable spinoff agents as executable tools
        verbose_traceback:            bool = False,  # Disclose full sanitised traceback to the LLM
        sandbox_cwd:                  Optional[str] = None, # Custom relative sandbox run directory
        tolerance_level:              Optional[str] = "strict",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs the conversational loop, resolving RAG, tiered memories, and tool calls.
        """
        # Store tolerance level on active discussion for downstream execution tools (like execute_python_data_query)
        if not hasattr(self, "tolerance_level") or tolerance_level:
            object.__setattr__(self, "tolerance_level", tolerance_level or "strict")

        # Initialize list to collect all created/modified artifacts during this turn safely
        object.__setattr__(self, "_affected_artefacts_this_turn", [])

        # ── 🔬 SCIENTIFIC RESOLUTION: Clear FailureMemory at start of turn ──
        if not hasattr(self, "_failure_memory") or not self._failure_memory:
            object.__setattr__(self, "_failure_memory", FailureMemory())
        else:
            self._failure_memory.failures = []

        # Reset cancellation flag at the start of each chat turn
        self.reset_cancel_state()

        self.scratchpad = ""
        callback = kwargs.get("streaming_callback")
        temperature = kwargs.get("temperature")

        # ── 1. Safe SQLite Memory Ingestion ──
        _mm = self._get_memory_manager(memory_manager) if enable_memory else None
        _counter = self.lollmsClient.count_tokens if self.lollmsClient else None

        if _mm:
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
        
        # Veracity and Formatting Rules
        rules = (
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
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you decide to output a thought process enclosed in <think>...</think> tags, "
            "you MUST output all functional XML tags (such as <artifact>, <tool>, or <mem_new>) "
            "on a NEW LINE strictly AFTER the closing </think> tag. "
            "NEVER place functional tags inside the <think> reasoning block.\n"
            "\n=== ANTI-MIMICRY PROTOCOL (CRITICAL) ===\n"
            "1. **NEVER OUTPUT SYSTEM MARKERS**: You are STRICTLY FORBIDDEN from generating text patterns like `[🔒SYSTEM_ARTIFACT_ANCHOR:...`, `[SYSTEM:`, or `[content stripped...`. These are **INFRASTRUCTURE-ONLY** markers used in history to save space. If you output them, NO ACTION will occur.\n"
            "2. **USE REAL TAGS**: To create artifacts, you MUST use the actual `<artifact name=\"...\">` XML tags. To call tools, use `<tool>`. Do NOT mimic the placeholder markers from past messages.\n"
            "3. **TAG ISOLATION**: Functional tags (`<artifact>`, `<tool>`, `<tool_result>`) MUST NEVER appear inside <think> blocks. They must ONLY appear in the final response body AFTER the closing </think> tag.\n"
        )

        extra_instructions = ""
        user_msg_lower = user_message.lower()

        if enable_artefacts:
            extra_instructions += self._build_artefact_instructions()
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

            branch_msgs_now = self.get_branch(user_msg.id)
            handle_instructions = _build_handle_instructions(branch_msgs_now)
            if handle_instructions:
                extra_instructions += handle_instructions

        if _mm:
            extra_instructions += _mm.build_system_instructions()
        if (enable_image_generation or enable_image_editing) and getattr(self.lollmsClient, 'tti', None) is not None:
            extra_instructions += self._build_image_generation_instructions()

        full_system_prompt = sys_prompt + "\n" + rules + "\n" + extra_instructions

        # ── 4. RAG Ingestion & Pre-Hydration ──
        rag_context = ""
        if prehydrate_rag and personality and hasattr(personality, "has_data") and personality.has_data:
            try:
                rag_res = personality.query_data(user_message)
                if rag_res and rag_res.get("success") and rag_res.get("sources"):
                    sources_text = []
                    for src in rag_res.get("sources", []):
                        sources_text.append(f"Source [{src.get('source')}]: {src.get('content')}")
                    if sources_text:
                        rag_context = "\n=== RETRIEVED RAG CONTEXT ===\n" + "\n\n".join(sources_text[:3]) + "\n=== END RAG CONTEXT ===\n"
            except Exception as e:
                trace_exception(e)

        if rag_context:
            full_system_prompt += "\n" + rag_context

        # ── 5. Active Artifacts & Memories Injection ──
        if enable_artefacts:
            artefacts_zone = self.artefacts.build_artefacts_context_zone()
            if artefacts_zone:
                full_system_prompt += "\n=== ACTIVE ARTIFACTS ===\n" + artefacts_zone + "\n"

        if _mm:
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
        active_tools = dict(tools or {})

        # ── DYNAMIC TOOL MOUNTING PROTOCOL ──
        # Automatically mount specialized tool libraries based on workspace context
        lcp_binding = getattr(self.lollmsClient, "tools", None)

        # 1. Data Tools Auto-Load
        # If enable_data_tools is True (or not explicitly disabled) AND data files exist in workspace
        enable_data_tools = kwargs.get("enable_data_tools", True)

        # Check workspace for data files FIRST (before checking if LCP binding exists)
        from pathlib import Path
        workspace_dir = Path("./data_workspace")
        try:
            from lollms_client.app.server import APP_WORKSPACE_DIR
            if APP_WORKSPACE_DIR is not None:
                workspace_dir = APP_WORKSPACE_DIR
        except ImportError:
            pass

        # Check discussion-specific workspace first
        disc_ws = workspace_dir / "discussions" / self.id
        if disc_ws.exists():
            workspace_dir = disc_ws

        data_extensions = {".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"}
        has_data_files = any(f.suffix.lower() in data_extensions for f in workspace_dir.rglob("*") if f.is_file())

        # CRITICAL FIX: If data files exist but LCP binding is None, INSTANTIATE it on-the-fly!
        if has_data_files and (lcp_binding is None):
            try:
                from lollms_client.tools_bindings.lcp import LCPBinding

                # Create LCP binding instance with default tools folder
                lcp_binding = LCPBinding(
                    tools_folders=[Path(__file__).parent.parent / "tools_bindings" / "lcp" / "default_tools"]
                )

                # Attach it to the client for future use
                self.lollmsClient.tools = lcp_binding
            except Exception as ex:
                trace_exception(ex)
                lcp_binding = None

        # Now proceed with tool mounting if LCP binding exists
        if enable_data_tools and lcp_binding and hasattr(lcp_binding, "mount_tool_library"):
            if has_data_files:
                lcp_binding.mount_tool_library("semantic_data_engineer")

        # 2. Merge LCP Tools into Active Toolset
        # If LCP binding exists, merge all discovered tools (including newly mounted ones)
        if lcp_binding and hasattr(lcp_binding, "to_chat_tool_specs"):
            try:
                lcp_tools = lcp_binding.to_chat_tool_specs()
                active_tools.update(lcp_tools)
            except Exception as ex:
                trace_exception(ex)

        # Optionally merge spinoff agents as dynamic local tools
        if enable_sub_agents:
            spinoff_tools = self._get_spinoff_agent_tools(full_system_prompt, images or [], **kwargs)
            active_tools.update(spinoff_tools)

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
            tools_prompt += "\nExact syntax (copy this pattern exactly):\n<tool>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}`)`\n\n"
            tools_prompt += "Available tools:\n"
            for t_name, t_spec in active_tools.items():
                desc = t_spec.get("description", "")
                params_list = t_spec.get("parameters", [])
                param_desc = ", ".join([f"{p['name']}: {p['type']}" for p in params_list])
                tools_prompt += f"- {t_name}({param_desc}): {desc}\n"
            tools_prompt += "=== END TOOLS ===\n"

        # ── 8. Active Deliberation Loop ──
        ASCIIColors.info("[Trace] Retrieving conversation branch...")
        current_branch_tip = branch_tip_id or self.active_branch_id
        branch = self.get_branch(current_branch_tip)

        # Build virtual history list for generate_from_messages
        virtual_history = [m for m in branch if m.id != user_msg.id]
        virtual_history.append(user_msg)

        tool_calls_this_turn = []
        round_count = 0

        # Track the last executed tool to prevent immediate repetition loops (Success Loops)
        last_executed_tool_signature = None

        if not hasattr(self, "_failure_memory") or not self._failure_memory:
            object.__setattr__(self, "_failure_memory", FailureMemory())

        # Initialize the single, clean database assistant message ONCE before entering the loop
        ai_msg = self.add_message(
            sender=personality.name if personality else self.lollmsClient.ai_name,
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=getattr(self.lollmsClient.llm, "model_name", "unknown") if self.lollmsClient else "unknown",
            binding_name=getattr(self.lollmsClient.llm, "binding_name", "unknown") if self.lollmsClient else "unknown"
        )
        if callback:
            callback(ai_msg.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE, {"message_id": ai_msg.id})

        # Track if we exited due to cancellation
        was_cancelled = False

        while round_count < max_reasoning_steps:
            # Check cancellation at the start of each reasoning round
            if self.is_generation_cancelled():
                was_cancelled = True
                break

            round_count += 1

            # Guarantee a clean, un-canceled state before launching each independent generation round
            if self.lollmsClient and getattr(self.lollmsClient, "llm", None):
                try:
                    self.lollmsClient.llm.reset_cancel()
                except Exception:
                    pass

            current_system_prompt = full_system_prompt
            if tools_prompt:
                current_system_prompt += "\n" + tools_prompt

            # Build messages list for generation context
            messages_list = [{"role": "system", "content": current_system_prompt}]
            for m in virtual_history:
                messages_list.append({
                    "role": "user" if m.sender_type == "user" else "assistant",
                    "content": m.content
                })

            # ── 🎨 DYNAMIC VISION HYDRATION ──
            # Retrieve all images generated or modified during previous rounds of this turn
            # and append their base64 pixels to the active vision context so the LLM can "see" them!
            round_images = list(images) if images else []
            affected_arts = getattr(self, "_affected_artefacts_this_turn", [])
            for art in affected_arts:
                if art.get("type") == "image" and art.get("images"):
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
                ai_message=ai_msg,
                enable_notes=enable_notes,
                enable_skills=enable_skills,
                enable_inline_widgets=enable_inline_widgets,
                enable_forms=enable_forms,
                auto_activate_artefacts=auto_activate_artefacts,
                enable_artefacts=enable_artefacts,
                enable_in_message_status=enable_in_message_status,
                content_offset=current_content_length  # Start parsing from current position
            )

            def _inline_relay(chunk, msg_type=None, meta=None):
                # Check cancellation on EVERY token chunk
                if self.is_generation_cancelled():
                    return False  # Signal to stop generation

                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            # Sanitize kwargs to prevent duplicate argument passing
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "stream")}

            # Execute generation turn (streams and appends to the existing ai_msg.content directly)
            try:
                self.lollmsClient.generate_from_messages(
                    messages=messages_list,
                    images=round_images if round_images else None,
                    stream=True,
                    temperature=temperature,
                    streaming_callback=_inline_relay,
                    **gen_kwargs
                )
            except Exception as gen_err:
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

            if ss.tool_trigger:
                tool_call_json_str = ss.get_tool_call_json()
                if tool_call_json_str:
                    try:
                        call_data = json.loads(tool_call_json_str)
                    except Exception:
                        call_data = {}

                    tool_name = call_data.get("name", "")
                    tool_params = call_data.get("parameters", {})

                    # ── Live UI Tool Call Feedback Injection ──
                    if tool_call_json_str in ai_msg.content:
                        ai_msg.content = ai_msg.content.replace(f"<tool>{tool_call_json_str}</tool>", "")
                        ai_msg.content = ai_msg.content.replace(tool_call_json_str, "")

                    import html
                    escaped_params = html.escape(json.dumps(tool_params))
                    tool_open_tag = f'\n<processing type="tool" title="Tool Execution: {tool_name}" params="{escaped_params}">\n'
                    ai_msg.content += tool_open_tag
                    _cb(callback, tool_open_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    status_line = f"* Calling local tool system for '{tool_name}'...\n"
                    ai_msg.content += status_line
                    _cb(callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # ── REFLEXIVE LOOP DETECTION (FailureMemory) ──
                    failure_memory = getattr(self, "_failure_memory", None)

                    if failure_memory and failure_memory.has_previous_failure(tool_name, tool_params):
                        # Intercepted repetitive execution loop (logging removed)

                        if self.is_generation_cancelled():
                            was_cancelled = True
                            break


                        result_str = (
                            f"Error executing tool '{tool_name}': This exact parameters configuration failed on a previous round of this conversation. "
                            f"To prevent an infinite loop, execution was blocked. You must modify your parameters, inspect the data schemas, "
                            f"or try a different approach instead of repeating the failing call."
                        )
                        status_err_line = f"* Tool call blocked to prevent loop.\n"
                        details_block = f"Loop Intercepted:\n{result_str}\n"
                        tool_close_tag = f"{status_err_line}{details_block}</processing>\n\n"
                        ai_msg.content += tool_close_tag
                        _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        break

                    # Execute the tool sequentially
                    try:
                        # ── 🛑 REPETITIVE SUCCESS LOOP GUARD ────────────────────────
                        # Check if we just executed this exact tool in the previous round
                        # If so, force break to prevent infinite success loops
                        if tool_calls_this_turn:
                            last_call = tool_calls_this_turn[-1]
                            if last_call["name"] == tool_name and last_call["params"] == tool_params:
                                # LOOP DETECTED (logging removed)
                                tool_res = {
                                    "success": False,
                                    "error": f"Repetitive tool call detected. You already called '{tool_name}' with these parameters in the previous turn. The output was already provided to you. Do not call it again.",
                                    "prompt_injection": f"\n\n🛑 **STOP.** You are calling '{tool_name}' again with the exact same parameters. This is a loop. The data from the previous execution is already in your context. Analyze it and move on."
                                }
                                # Force break after this fake failure
                                force_break_after_tool = True
                            else:
                                force_break_after_tool = False
                        else:
                            force_break_after_tool = False

                        if force_break_after_tool:
                            # Use the fake error result prepared above
                            pass
                        # Check cancellation BEFORE executing tool
                        elif self.is_generation_cancelled():
                            # Generation cancelled (logging removed)
                            tool_res = {
                                "success": False, 
                                "error": "Execution aborted by user cancellation.",
                                "prompt_injection": "\n\n⚠️ **Execution Aborted.**\nThe user cancelled the generation. Do not attempt to call tools again."
                            }
                        elif active_tools and tool_name in active_tools:
                            # Sync all active artifacts to disk BEFORE tool execution
                            try:
                                sync_ws, sync_files = self.artefacts.sync_all_active_to_disk()
                            except Exception as ex:
                                trace_exception(ex)

                            import os
                            from pathlib import Path
                            old_cwd = os.getcwd()

                            # 🛑 CRITICAL RESOLUTION: Dynamically resolve active workspace from discussion instance path
                            if hasattr(self, 'workspace_path') and self.workspace_path:
                                base_workspace_dir = Path(self.workspace_path)
                            else:
                                base_workspace_dir = Path("./data_workspace")
                                # Fallback to server APP_WORKSPACE_DIR if workspace_path is not bound
                                try:
                                    from lollms_client.app.server import APP_WORKSPACE_DIR
                                    if APP_WORKSPACE_DIR is not None:
                                        base_workspace_dir = APP_WORKSPACE_DIR
                                except ImportError:
                                    pass

                            # 🛑 CRITICAL RESOLUTION: Direct tool CWD to the isolated workspace_data/ subfolder
                            if hasattr(self, "workspace_data_path") and self.workspace_data_path:
                                workspace_dir = Path(self.workspace_data_path)
                            else:
                                workspace_dir = base_workspace_dir / self.id / "workspace_data"

                            workspace_dir.mkdir(parents=True, exist_ok=True)
                            workspace_dir_str = str(workspace_dir.resolve())

                            try:
                                os.chdir(workspace_dir_str)

                                # 🛑 CRITICAL FIX: Strip path prefixes from tool parameters
                                # LLMs often hallucinate full paths like "workspace/file.csv" or "data_workspace/file.csv"
                                # We must normalize these to simple filenames for the tool to find them
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

                                # 🛑 SEQUENTIAL EXECUTION: Run tool directly in main thread
                                # CRITICAL: CWD management is delegated to LCP Binding execute_tool()
                                # The LCP binding will set CWD to the discussion-isolated workspace internally.
                                # We do NOT change CWD here to avoid double-changedirectory conflicts.
                                call_kwargs = dict(sanitized_params)

                                # Inject active session instances into the tool execution call
                                call_kwargs["discussion_instance"] = self
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

                                # Detect NEW files
                                new_files = set(files_after.keys()) - set(files_before.keys())

                                for rel_path in new_files:
                                    file_info = files_after[rel_path]
                                    file_name = rel_path.name
                                    file_ext = rel_path.suffix.lower()
                                    file_path = file_info["path"]
                                    file_size = file_path.stat().st_size

                                    # Determine type
                                    atype = "document"
                                    if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
                                        atype = "code"
                                    elif file_ext in (".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
                                        atype = "data"
                                    elif file_ext in (".md", ".txt", ".log", ".out", ".trace", ".asc", ".raw", ".json", ".yaml", ".yml", ".xml", ".ttl"):
                                        atype = "document"
                                    elif file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
                                        atype = "image"

                                    # Check if Binary
                                    EXPLICIT_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", 
                                                            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", 
                                                            ".zip", ".tar", ".gz", ".pdf", ".docx"}

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

                                    # Register as tree_unlockable or fully active based on type
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
                                        # Registered file (placeholder) (logging removed)
                                        self.commit()
                                        
                                        # Hydrate active turn list so we can pass base64 pixels to vision
                                        if atype == "image":
                                            try:
                                                import base64
                                                raw_img = file_path.read_bytes()
                                                img_b64 = base64.b64encode(raw_img).decode('utf-8')
                                                self.artefacts.update(
                                                    title=file_name,
                                                    new_images=[img_b64],
                                                    new_image_media_types=[f"image/{file_ext[1:]}"],
                                                    bump_version=False # In-place update so version remains v1
                                                )
                                                self.commit()
                                                self._affected_artefacts_this_turn.append(self.artefacts.get(file_name))
                                            except Exception as ex:
                                                trace_exception(ex)

                                            if self.active_branch_id:
                                                ai_msg = self.get_message(self.active_branch_id)
                                            if ai_msg:
                                                tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                                                if tag not in ai_msg.content:
                                                    ai_msg.content += f'\n\n{tag}\n'
                                                self.commit()
                                        continue

                                    # Handle Text/Readable Files (Default to Fully Active & Visible)
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
                                    # Created NEW artifact from file (logging removed)
                                    self.commit()

                                    if self.active_branch_id:
                                        ai_msg = self.get_message(self.active_branch_id)
                                        if ai_msg:
                                            tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                                            if tag not in ai_msg.content:
                                                ai_msg.content += f'\n\n{tag}\n'
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

                                    if mtime_changed or content_changed:
                                        # File was modified!
                                        atype = "document"
                                        if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
                                            atype = "code"
                                        elif file_ext in (".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
                                            atype = "data"
                                        elif file_ext in (".md", ".txt", ".log", ".out", ".trace", ".asc", ".raw", ".json", ".yaml", ".yml", ".xml", ".ttl"):
                                            atype = "document"
                                        elif file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
                                            atype = "image"

                                        # Check if Binary
                                        EXPLICIT_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", 
                                                                ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", 
                                                                ".zip", ".tar", ".gz", ".pdf", ".docx"}

                                        should_read_content = True
                                        content_placeholder = None

                                        if file_ext in EXPLICIT_BINARY_EXTS:
                                            should_read_content = False
                                            content_placeholder = (
                                                f"### Data File Modified: `{file_name}`\n\n"
                                                f"This file was modified by the tool `{tool_name}`.\n"
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
                                                            f"### Binary File Modified: `{file_name}`\n\n"
                                                            f"This file was modified by the tool `{tool_name}`.\n"
                                                            f"- **Type**: {file_ext.upper()} (Unknown Binary)\n"
                                                            f"- **Size**: {file_size:,} bytes\n"
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

                                        # Register/Update
                                        if not should_read_content and content_placeholder:
                                            existing_art = self.artefacts.get(file_name)
                                            if existing_art:
                                                art = self.artefacts.update(
                                                    title=file_name,
                                                    new_content=content_placeholder,
                                                    new_type=atype,
                                                    new_images=img_b64 if img_b64 else None,
                                                    new_image_media_types=img_mtypes if img_mtypes else None,
                                                    active=True,
                                                    visibility=ArtefactVisibility.FULL,
                                                    bump_version=True,
                                                    commit_message=f"Updated binary file reference by tool '{tool_name}'"
                                                )
                                            else:
                                                art = self.artefacts.add(
                                                    title=file_name,
                                                    artefact_type=atype,
                                                    content=content_placeholder,
                                                    images=img_b64 if img_b64 else None,
                                                    image_media_types=img_mtypes if img_mtypes else None,
                                                    active=True,
                                                    visibility=ArtefactVisibility.FULL,
                                                    commit_message=f"Created by tool '{tool_name}'"
                                                )
                                            # Updated file reference (placeholder) (logging removed)
                                            self.commit()
                                            
                                            # Hydrate active turn list so we can pass base64 pixels to vision
                                            if atype == "image":
                                                try:
                                                    import base64
                                                    raw_img = file_path.read_bytes()
                                                    img_b64 = base64.b64encode(raw_img).decode('utf-8')
                                                    self.artefacts.update(
                                                        title=file_name,
                                                        new_images=[img_b64],
                                                        new_image_media_types=[f"image/{file_ext[1:]}"],
                                                        bump_version=True # Increment version on modification
                                                        )
                                                    self.commit()
                                                    self._affected_artefacts_this_turn.append(self.artefacts.get(file_name))
                                                except Exception as ex:
                                                    trace_exception(ex)

                                            if self.active_branch_id:
                                                ai_msg = self.get_message(self.active_branch_id)
                                                if ai_msg:
                                                    tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                                                    if tag not in ai_msg.content:
                                                        ai_msg.content += f'\n\n{tag}\n'
                                                    self.commit()
                                            continue

                                        # Handle Text/Readable Files (Default to Fully Active & Visible)
                                        existing_art = self.artefacts.get(file_name)
                                        if existing_art:
                                            art = self.artefacts.update(
                                                title=file_name,
                                                new_content=after_info["content"],
                                                new_type=atype,
                                                active=True,
                                                visibility=ArtefactVisibility.FULL,
                                                commit_message=f"Modified by tool '{tool_name}'"
                                            )
                                        else:
                                            art = self.artefacts.add(
                                                title=file_name,
                                                artefact_type=atype,
                                                content=after_info["content"],
                                                active=True,
                                                visibility=ArtefactVisibility.FULL,
                                                commit_message=f"Created by tool '{tool_name}'"
                                            )
                                        # Updated artifact from modified file (logging removed)
                                        self.commit()

                                        if self.active_branch_id:
                                            ai_msg = self.get_message(self.active_branch_id)
                                            if ai_msg:
                                                tag = f'<artefact_image id="{file_name}::0" />' if atype == "image" else f'<lollms_artifact id="{file_name}" type="{atype}" version="{art.get("version", 1)}" />'
                                                if tag not in ai_msg.content:
                                                    ai_msg.content += f'\n\n{tag}\n'
                                                self.commit()
                            finally:
                                # No CWD restoration needed - ChatMixin never changed it
                                # LCP Binding handles its own CWD cleanup internally
                                pass
                        else:
                            tool_res = self.lollmsClient.tools.execute_tool(
                                tool_name, 
                                tool_params, 
                                lollms_client_instance=self.lollmsClient, 
                                discussion_instance=self,
                                discussion=self
                            )

                        if isinstance(tool_res, dict):
                            if not tool_res.get("success", True):
                                error_msg = tool_res.get("error", "Unknown tool error")
                                if failure_memory:
                                    failure_memory.record_failure(tool_name, tool_params, error_msg)

                                if failure_memory and failure_memory.has_previous_failure(tool_name, tool_params):
                                    result_str = f"Error executing tool '{tool_name}': This exact parameters configuration failed on a previous round. Execution blocked to prevent infinite loop."
                                    clean_result_str = result_str
                                    status_done_line = f"* Tool call blocked to prevent loop.\n"
                                    details_block = f"Loop Intercepted:\n{result_str}\n"
                                    tool_close_tag = f"{status_done_line}{details_block}</processing>\n\n"
                                    ai_msg.content += tool_close_tag
                                    _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                                    break
                                else:
                                    result_str = f"Error executing tool '{tool_name}': {error_msg}"
                                    clean_result_str = result_str
                                    status_done_line = f"* Completed execution with errors.\n"
                                    details_block = f"Error Logs:\n{error_msg}\n"
                            else:
                                raw_output = tool_res.get("output", tool_res)
                                if isinstance(raw_output, (dict, list)):
                                    full_dump = json.dumps(raw_output, indent=2, default=str)
                                else:
                                    full_dump = str(raw_output)
                                result_str = full_dump
                                clean_result_str = _sanitize_tool_result(tool_res)
                                self._trigger_evolutionary_reflection(tool_name, tool_params, clean_result_str)
                                status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                safe_output = full_dump[:2000] + ("..." if len(full_dump) > 2000 else "")
                                details_block = f"Output Logs:\n{safe_output}\n"
                        else:
                            result_str = str(tool_res)
                            if "error" in result_str.lower() or "fail" in result_str.lower():
                                self.failure_memory.record_failure(tool_name, tool_params, result_str)
                                clean_result_str = result_str
                                status_done_line = f"* Completed execution with errors.\n"
                                details_block = f"Error Logs:\n{result_str}\n"
                            else:
                                status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                clean_result_str = _sanitize_tool_result(tool_res)
                                safe_output = result_str[:2000] + ("..." if len(result_str) > 2000 else "")
                                details_block = f"Output Logs:\n{safe_output}\n"
                    except Exception as e:
                        trace_exception(e)
                        if failure_memory:
                            failure_memory.record_failure(tool_name, tool_params, str(e))
                        result_str = f"Error executing tool '{tool_name}': {e}"
                        clean_result_str = f"Error executing tool '{tool_name}': {e}"
                        status_done_line = f"* Execution crashed.\n"
                        details_block = f"Crash Details:\n{str(e)}\n"

                    tool_close_tag = f"{status_done_line}{details_block}</processing>\n\n"
                    ai_msg.content += tool_close_tag
                    _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    tool_success = "Error" not in clean_result_str and "failed" not in clean_result_str.lower()
                    tool_calls_this_turn.append({
                        "name": tool_name,
                        "params": tool_params,
                        "result": {"output": clean_result_str, "success": tool_success}
                    })

                    raw_round_text = ss.get_clean_text_so_far()
                    if virtual_history:
                        for prev_m in virtual_history:
                            if prev_m.sender_type == "assistant":
                                raw_round_text = raw_round_text.replace(prev_m.content, "")

                    raw_round_clean = re.sub(r'<processing.*?>.*?</processing>', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE).strip()

                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=raw_round_clean
                    ))

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

                        # 🛑 CRITICAL FIX: EXPLICITLY LABEL OUTPUT AS "NOT A TOOL CALL"
                        # We must prevent the LLM from misinterpreting JSON results as new tool calls
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
                            f"5. 🔀 If you need MORE data, call a **DIFFERENT** tool or ask a **DIFFERENT** question.\n\n"
                            f"### Example of CORRECT behavior:\n"
                            f"❌ WRONG: <tool>{{\"name\": \"{tool_name}\", ...}}</tool>  (LOOP!)\n"
                            f"✅ RIGHT:  \"Based on the results, I can see that...\"  (ANSWER!)\n"
                        )

                        # Log for debugging (removed per user request)

                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=user_part
                        ))
                        ai_msg.content += "\n\n"
                    else:
                        user_part = (
                            f'<tool_result name="{tool_name}">\n'
                            f"{clean_result_str}\n"
                            f"</tool_result>\n\n"
                            f"⚠️ **Tool Execution Failed.**\n"
                            f"The tool '{tool_name}' encountered an error. Please politely inform the user that the operation could not be completed, "
                            f"briefly explain the likely cause based on the error log above, and suggest a possible workaround or alternative approach. "
                            f"Do not attempt to call the tool again with the same parameters."
                        )
                        virtual_history.append(SimpleNamespace(
                            sender_type="user",
                            content=user_part
                        ))
                        ai_msg.content += "\n\n"
                        # If this was a forced break due to loop detection, ensure we exit
                        if 'force_break_after_tool' in locals() and force_break_after_tool:
                            ASCIIColors.error("[ChatMixin] Breaking loop due to loop protection.")
                            break
                else:
                    break
            else:
                # ── 🔄 FORCE CONTINUATION AFTER CONTEXT UNLOCK ─────────────────
                # If the model requested files to be loaded, force at least one more round
                # so it can immediately use the newly available context
                if ss.context_unlock_requested and not was_cancelled:
                    ASCIIColors.info(f"[ChatMixin] Context unlock detected ({ss.context_unlocked_files}), forcing continuation round...")

                    # Inject a system prompt confirming the unlock and inviting continuation
                    unlock_files_str = ', '.join(ss.context_unlocked_files)
                    continuation_prompt = (
                        f"\n\n[SYSTEM: The following files have been loaded into context: {unlock_files_str}. "
                        f"You can now read their full content and use them. Please continue your task.]\n\n"
                    )

                    # Append to AI message content (will be visible in next round's history)
                    ai_msg.content += continuation_prompt
                    _cb(callback, continuation_prompt, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # Add a user message to force the loop to continue
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=f"[SYSTEM: Files {unlock_files_str} are now available. Continue.]"
                    ))

                    # Reset the flag and continue the loop
                    ss.context_unlock_requested = False
                    continue  # Jump to next reasoning round immediately

                break

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
                "artefacts_modified": [a.get("title") for a in ss.affected_artefacts],
                "cancelled": True
            }
        else:
            ai_msg.metadata = {
                "mode": "agentic" if tool_calls_this_turn else "direct",
                "tool_calls": tool_calls_this_turn,
                "artefacts_modified": [a.get("title") for a in ss.affected_artefacts]
            }

        if remove_thinking_blocks:
            ai_msg.content = self.lollmsClient.remove_thinking_blocks(ai_msg.content)

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

        # Process memories
        mem_cleaned, mem_report = self._process_memory_tags(ai_msg.content, _mm, callback)
        if mem_cleaned != ai_msg.content:
            ai_msg.content = mem_cleaned

        if _mm:
            try:
                self._save_episodic_memory_turn(user_message, ai_msg.content, _mm)
            except Exception as ex:
                trace_exception(ex)

        # Update metadata for alternating exports
        ai_msg.metadata = {
            "mode": "agentic" if tool_calls_this_turn else "direct",
            "tool_calls": tool_calls_this_turn,
            "artefacts_modified": [a.get("title") for a in ss.affected_artefacts]
        }

        # Auto dream
        dream_report = None
        if enable_auto_dream and _mm is not None:
            try:
                dream_report = _mm.dream(self.lollmsClient)
            except Exception as ex:
                trace_exception(ex)

        if self._is_db_backed and self.autosave:
            self.commit()

        self.scratchpad = ""
        object.__setattr__(self, '_active_callback', None)

        return {
            "user_message": user_msg,
            "ai_message": ai_msg,
            "sources": [],
            "artefacts": ss.affected_artefacts,
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
