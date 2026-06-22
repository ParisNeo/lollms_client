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
from ._artefacts import ArtefactType, make_image_id

# ── Cancellation state & limits ──────────────────────────────────────────────
_MAX_BRACKET_BUF = 256

_TAG_STARTS = [
    "<tool_call>", "<tool>",
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
        """Recursively hunt for a non-empty prompt_injection string anywhere in the tree."""
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

    # 1. Prefer the tool's hand-crafted prompt_injection when available (searches the whole tree)
    pinj = _find_prompt_injection(tool_res)
    if pinj:
        # Look for success status somewhere in the result
        success = True
        inner = tool_res.get("output", tool_res) if isinstance(tool_res, dict) else tool_res
        if isinstance(inner, dict):
            success = inner.get("success", True)
        success_status = "✓ Success" if success else "⚠ Partial Success"
        return f"{success_status}\n{pinj}"

    # 2. Otherwise, walk the result and strip blobs
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
    Streams all code, text, and XML tags directly to the client with ZERO latency.
    The exact second an XML tag is closed, it intercepts the accumulated block,
    builds the artifact/resource on disk, and rewrites the database message 
    content to cleanly strip the raw block and replace it with a success card.
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
    ):
        self.discussion = discussion
        self.callback = callback
        self.ai_message = ai_message
        self.enable_artefacts = enable_artefacts
        self.enable_in_message_status = enable_in_message_status
        self.auto_activate = auto_activate_artefacts

        self.enable_notes = enable_notes if enable_artefacts else False
        self.enable_skills = enable_skills if enable_artefacts else False
        self.enable_inline_widgets = enable_inline_widgets if enable_artefacts else False
        self.enable_forms = enable_forms if enable_artefacts else False

        self.tool_trigger = False
        self.tool_json_data = ""
        self.affected_artefacts = []
        self.processed_tags = set()  # prevent double-processing same matched block

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str) or not chunk:
            return True

        # 1. Zero-latency stream passthrough: user sees everything in real-time
        self.ai_message.content += chunk
        _cb(self.callback, chunk, MSG_TYPE.MSG_TYPE_CHUNK)

        # 2. Stateless Thoughts Detection
        content = self.ai_message.content
        last_open_think = content.rfind("<think>")
        last_close_think = content.rfind("</think>")
        is_inside_thoughts = (last_open_think != -1) and (last_open_think > last_close_think)

        if is_inside_thoughts:
            return True

        # 3. Protect Active Unclosed Artifact Blocks from Tag Interception
        # If the LLM is actively outputting inside an open <artifact> or <note> block,
        # we must STRICTLY disable secondary tag parsing. This prevents aider symbols
        # like "<<<<<<< SEARCH" from triggering false-positive tag parsing!
        last_open_art = max(content.rfind("<artifact"), content.rfind("<artefact"))
        last_close_art = max(content.rfind("</artifact>"), content.rfind("</artefact>"))
        is_inside_artifact = (last_open_art != -1) and (last_open_art > last_close_art)

        if is_inside_artifact:
            # We are currently streaming inside an open artifact block — do not trigger secondary parses
            # until the closing tag is typed
            if not content.endswith("</artifact>") and not content.endswith("</artefact>"):
                return True

        # 4. Create a safe content copy by removing all closed and unclosed thinking blocks
        safe_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        safe_content = re.sub(r'<think>.*$', '', safe_content, flags=re.DOTALL | re.IGNORECASE)

        # Scan for closed tags of interest only within the safe non-thinking content
        pattern = re.compile(
            r'<(artifact|artefact|note|skill|tool_call|tool|add_files_to_context)\b([^>]*?)>(.*?)</\1>',
            re.DOTALL | re.IGNORECASE
        )
        
        for match in pattern.finditer(safe_content):
            full_match_text = match.group(0)
            if full_match_text in self.processed_tags:
                continue

            # Tag is fully closed outside of thoughts! Process immediately
            self.processed_tags.add(full_match_text)
            
            tag_name = match.group(1).lower()
            attrs_str = match.group(2)
            body = match.group(3).strip()

            keep_generating = self._dispatch_closed_tag(tag_name, attrs_str, body, full_match_text)
            if not keep_generating:
                return False

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
                _cb(self.callback, f"Artifact '{title}' successfully updated.", MSG_TYPE.MSG_TYPE_INFO)

            # ── SWIFT REPLACEMENT PROTOCOL ──
            # Strips the full raw XML from the message body and replaces it with the placeholder
            placeholder = f"\n[content stripped, refer to the '{title}' artefact for details]\n"
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
        elif tag_name in ("tool_call", "tool"):
            self.tool_trigger = True
            self.tool_json_data = body
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

            placeholder = f"\n[content stripped, refer to the '{title}' note for details]\n"
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

            placeholder = f"\n[content stripped, refer to the '{title}' skill for details]\n"
            self.ai_message.content = self.ai_message.content.replace(full_match_text, placeholder)
            _cb(self.callback, placeholder, MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {
                "type": "artifact_created",
                "title": title,
                "art_type": "skill"
            })
            return True

        # 5. Multi-tier Context Unlocks
        elif tag_name == "add_files_to_context":
            from ._artefacts import ArtefactVisibility
            targets = [t.strip() for t in body.splitlines() if t.strip()]
            unlocked_files = []

            for t_file in targets:
                art = self.discussion.artefacts.get(t_file)
                if art:
                    self.discussion.artefacts.set_visibility(t_file, ArtefactVisibility.FULL)
                    unlocked_files.append(t_file)

            if unlocked_files:
                self.discussion.commit()
                status_text = f"Unlocked and fully loaded {len(unlocked_files)} file(s) into context: " + ", ".join([f"'{f}'" for f in unlocked_files])
                ASCIIColors.success(f"[Context Unlock] {status_text}")

            placeholder = f"\n[unlocked and loaded context files: {', '.join(unlocked_files)}]\n"
            self.ai_message.content = self.ai_message.content.replace(full_match_text, placeholder)
            _cb(self.callback, placeholder, MSG_TYPE.MSG_TYPE_CHUNK)
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
        self.scratchpad = ""
        callback = kwargs.get("streaming_callback")
        temperature = kwargs.get("temperature")

        # ── 1. Safe SQLite Memory Ingestion ──
        _mm = self._get_memory_manager(memory_manager) if enable_memory else None
        _counter = self.lollmsClient.count_tokens if self.lollmsClient else None

        if _mm:
            try:
                ASCIIColors.info("[Trace] Applying memory decay...")
                _mm.apply_decay()
            except Exception as e:
                ASCIIColors.warning(f"[Memory] Decay update deferred due to database lock: {e}")

            if user_message and enable_deep_memory_pulling:
                try:
                    ASCIIColors.info("[Trace] Executing auto_pull_deep_memories...")
                    _mm.auto_pull_deep_memories(user_message)
                except Exception as e:
                    ASCIIColors.warning(f"[Memory] Associative pull deferred due to database lock: {e}")

            try:
                ASCIIColors.info("[Trace] Enforcing memory budget...")
                _mm.enforce_budget(token_counter=_counter)
            except Exception as e:
                ASCIIColors.warning(f"[Memory] Budget enforcement deferred due to database lock: {e}")

        # ── 2. Add or Retrieve User Message ──
        ASCIIColors.info("[Trace] Registering/Retrieving user message...")
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
            "(such as HTML, CSS, Python, SQL, XML, JSON, YAML, Turtle, etc.) inside standard "
            "markdown code blocks specifying the correct language identifier, e.g.:\n"
            "```python\n"
            "# python code here\n"
            "```\n"
            "Never output raw code or markup directly in conversational text without these code blocks.\n"
            "\n=== TOOL CALLING DISCIPLINE (CRITICAL) ===\n"
            "When a tool successfully returns a file (image, plot, screenshot, PDF, audio, etc.), "
            "the file is ALREADY saved to the workspace by the tool. Do NOT call the same tool "
            "again with the same parameters to regenerate it. Instead, reference the produced "
            "file URL in your final answer (e.g. <img src=\"/api/workspace_files/filename.png\" /> "
            "for images) and STOP generating. Repeating a successful tool call wastes tokens and "
            "may trigger anti-loop protection. The tool's response already contains the exact "
            "filename and URL you need to reference it.\n"
            "\n=== THINKING & REASONING CONSTRAINT ===\n"
            "If you decide to output a thought process enclosed in <think>...</think> tags, "
            "you MUST output all functional XML tags (such as <artifact>, <tool_call>, or <mem_new>) "
            "on a NEW LINE strictly AFTER the closing </think> tag. "
            "NEVER place functional tags inside the <think> reasoning block.\n"
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

        # ── 7. Tool calling registry ──
        active_tools = dict(tools or {})
        
        # Optionally merge spinoff agents as dynamic local tools
        if enable_sub_agents:
            spinoff_tools = self._get_spinoff_agent_tools(full_system_prompt, images or [], **kwargs)
            active_tools.update(spinoff_tools)

        tools_prompt = ""
        if active_tools:
            tools_prompt = "\n=== TOOLS AVAILABLE ===\n"
            tools_prompt += "To use a tool, you MUST emit a single <tool_call> tag on a new line with the tool parameters as a JSON object, and then stop generating. Do NOT write prose before or after the tag.\n"
            tools_prompt += "Exact syntax:\n<tool_call>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}</tool_call>\n\n"
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

        tool_calls_this_turn = []
        round_count = 0

        # Initialize the transient in-process FailureMemory tracker
        if not hasattr(self, "failure_memory") or not self.failure_memory:
            object.__setattr__(self, "failure_memory", FailureMemory())

        # Initialize the single, clean database assistant message ONCE before entering the loop
        ASCIIColors.info("[Trace] Initializing database assistant message stub...")
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

        while round_count < max_reasoning_steps:
            round_count += 1
            ASCIIColors.info(f"[Trace] Loop round {round_count}/{max_reasoning_steps} starting...")

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
            messages_list.append({
                "role": "user",
                "content": user_message
            })

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
                enable_in_message_status=enable_in_message_status
            )

            def _inline_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    if meta and meta.get("was_processed"):
                        return True
                    return ss.feed(chunk)
                return True

            # Sanitize kwargs to prevent duplicate argument passing
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ("streaming_callback", "temperature", "stream")}

            ASCIIColors.info(f"[Trace] Forwarding payload to LLM generate_from_messages (thinking={kwargs.get('think', False)})...")
            # Execute generation turn (streams and appends to the existing ai_msg.content directly)
            self.lollmsClient.generate_from_messages(
                messages=messages_list,
                images=images,
                stream=True,
                temperature=temperature,
                streaming_callback=_inline_relay,
                **gen_kwargs
            )

            ASCIIColors.info("[Trace] Generation complete. Flushing remaining buffers...")
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
                    # Open a processing block directly in the message stream to notify the user immediately
                    import html
                    escaped_params = html.escape(json.dumps(tool_params))
                    tool_open_tag = f'\n<processing type="tool_call" title="Tool Execution: {tool_name}" params="{escaped_params}">\n'
                    ai_msg.content += tool_open_tag
                    _cb(callback, tool_open_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    status_line = f"* Calling local tool system for '{tool_name}'...\n"
                    ai_msg.content += status_line
                    _cb(callback, status_line, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # ── REFLEXIVE LOOP DETECTION (FailureMemory) ──
                    if self.failure_memory.has_previous_failure(tool_name, tool_params):
                        ASCIIColors.error(f"[FailureMemory] Intercepted repetitive execution loop for tool '{tool_name}'!")
                        result_str = (
                            f"Error executing tool '{tool_name}': This exact parameters configuration failed on a previous round of this conversation. "
                            f"To prevent an infinite loop, execution was blocked. You must modify your parameters, inspect the data schemas, "
                            f"or try a different approach instead of repeating the failing call."
                        )
                        # Write the error notice to the UI before terminating
                        status_err_line = f"* Tool call blocked to prevent loop.\n"
                        details_block = f'<details class="proc-error-details"><summary>Loop Intercepted</summary><pre>{result_str}</pre></details>\n'
                        tool_close_tag = f"{status_err_line}{details_block}</processing>\n\n"
                        ai_msg.content += tool_close_tag
                        _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})
                        
                        # Instantly terminate the generation loop to stop empty accordion spams!
                        break
                    else:
                        # Execute the tool safely
                        try:
                            # Always pass the active discussion and client instances as context keywords
                            if active_tools and tool_name in active_tools:
                                tool_res = active_tools[tool_name]["callable"](
                                    lollms_client_instance=self.lollmsClient,
                                    discussion_instance=self,
                                    **tool_params
                                )
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
                                    self.failure_memory.record_failure(tool_name, tool_params, error_msg)
                                    result_str = f"Error executing tool '{tool_name}': {error_msg}"
                                    clean_result_str = result_str  # Errors are already concise and safe to feed back

                                    # Write error block
                                    status_done_line = f"* Completed execution with errors.\n"
                                    details_block = f'<details class="proc-error-details"><summary>Error Logs</summary><pre>{html.escape(error_msg)}</pre></details>\n'
                                else:
                                    # ── Stash the full raw output for the UI; build a sanitized
                                    #    version (no base64 blobs, prefers prompt_injection) to
                                    #    feed back to the LLM. This prevents the tool-stutter
                                    #    loop where the model re-invokes the same tool on the
                                    #    raw base64 data it just received.
                                    raw_output = tool_res.get("output", tool_res)
                                    if isinstance(raw_output, (dict, list)):
                                        full_dump = json.dumps(raw_output, indent=2, default=str)
                                    else:
                                        full_dump = str(raw_output)
                                    result_str = full_dump
                                    clean_result_str = _sanitize_tool_result(tool_res)
                                    # Trigger evolutionary reflection on successful recovery
                                    self._trigger_evolutionary_reflection(tool_name, tool_params, clean_result_str)

                                    # Write success block
                                    status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                    safe_output = html.escape(full_dump[:2000] + ("..." if len(full_dump) > 2000 else ""))
                                    details_block = f'<details class="proc-success-details"><summary>Output Logs</summary><pre>{safe_output}</pre></details>\n'
                            else:
                                result_str = str(tool_res)
                                if "error" in result_str.lower() or "fail" in result_str.lower():
                                    self.failure_memory.record_failure(tool_name, tool_params, result_str)
                                    clean_result_str = result_str  # Errors are already concise and safe
                                    status_done_line = f"* Completed execution with errors.\n"
                                    details_block = f'<details class="proc-error-details"><summary>Error Logs</summary><pre>{html.escape(result_str)}</pre></details>\n'
                                else:
                                    status_done_line = f"* Completed execution of '{tool_name}' successfully.\n"
                                    clean_result_str = _sanitize_tool_result(tool_res)
                                    safe_output = html.escape(result_str[:2000] + ("..." if len(result_str) > 2000 else ""))
                                    details_block = f'<details class="proc-success-details"><summary>Output Logs</summary><pre>{safe_output}</pre></details>\n'
                        except Exception as e:
                            self.failure_memory.record_failure(tool_name, tool_params, str(e))
                            result_str = f"Error executing tool '{tool_name}': {e}"
                            clean_result_str = f"Error executing tool '{tool_name}': {e}"
                            status_done_line = f"* Execution crashed.\n"
                            details_block = f'<details class="proc-error-details"><summary>Crash Details</summary><pre>{html.escape(str(e))}</pre></details>\n'

                    tool_close_tag = f"{status_done_line}{details_block}</processing>\n\n"
                    ai_msg.content += tool_close_tag
                    _cb(callback, tool_close_tag, MSG_TYPE.MSG_TYPE_CHUNK, {"was_processed": True})

                    # Track the tool call and result in this turn's metadata
                    # (use clean_result_str to keep metadata bloat-free of base64)
                    tool_calls_this_turn.append({
                        "name": tool_name,
                        "params": tool_params,
                        "result": {"output": clean_result_str, "success": "Error" not in clean_result_str}
                    })

                    # ── HIGH-FIDELITY CONTEXT PRESERVATION ──
                    # To prevent the model from falling into "dumb next-word prediction" auto-complete loops,
                    # we must feed it the exact, raw sequence of its own previous actions.
                    raw_round_text = ss.get_clean_text_so_far()

                    # Find the newly added text segment of this round
                    if virtual_history:
                        # Strip previous virtual history content to get only this round's raw output
                        for prev_m in virtual_history:
                            if prev_m.sender_type == "assistant":
                                raw_round_text = raw_round_text.replace(prev_m.content, "")

                    raw_round_clean = re.sub(r'<processing.*?>.*?</processing>', '', raw_round_text, flags=re.DOTALL | re.IGNORECASE).strip()

                    # 1. Append the raw assistant turn (with unstripped <tool_call> tags) to virtual history
                    virtual_history.append(SimpleNamespace(
                        sender_type="assistant",
                        content=raw_round_clean
                    ))

                    # 2. Append the SANITIZED tool result to virtual history.
                    #    We deliberately do NOT use `result_str` here, which may
                    #    contain multi-kilobyte base64 blobs. Feeding those blobs
                    #    back to the LLM causes tool-stutter loops where the model
                    #    re-invokes the same tool on the data it just produced.
                    user_part = (
                        f'<tool_result name="{tool_name}">\n'
                        f"{clean_result_str}\n"
                        f"</tool_result>\n\n"
                        f"Please analyze the tool output above and proceed with your response. "
                        f"If the tool already produced a file (image, plot, document, etc.), "
                        f"reference it in your final answer and STOP — do not call the same tool again."
                    )
                    virtual_history.append(SimpleNamespace(
                        sender_type="user",
                        content=user_part
                    ))

                    # Append spacing so the next turn's stream flows continuously in the same bubble
                    ai_msg.content += "\n\n"
                else:
                    break
            else:
                break

        # ── 11. Final Post-Processing & Database Commit ──
        if remove_thinking_blocks:
            ai_msg.content = self.lollmsClient.remove_thinking_blocks(ai_msg.content)

        # Process memories
        mem_cleaned, mem_report = self._process_memory_tags(ai_msg.content, _mm, callback)
        if mem_cleaned != ai_msg.content:
            ai_msg.content = mem_cleaned

        if _mm:
            try:
                self._save_episodic_memory_turn(user_message, ai_msg.content, _mm)
            except Exception as e:
                ASCIIColors.warning(f"[Memory] Episodic save deferred: {e}")

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
            except Exception as e:
                ASCIIColors.warning(f"[Memory] Auto-dream deferred: {e}")

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
            "dream_report": dream_report
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
        except json.JSONDecodeError:
            pass

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
