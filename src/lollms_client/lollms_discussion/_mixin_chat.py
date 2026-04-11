# lollms_discussion/_mixin_chat.py
# ChatMixin: simplified_chat() and chat() — the two high-level conversation methods.
#
# ARTIFACT STREAMING CONTRACT
# ---------------------------
# Content inside <artifact>, <note>, <skill>, <lollms_inline>, <lollms_form> tags is
# NEVER forwarded to the main chat bubble via MSG_TYPE_CHUNK.
#
# ARTEFACT IMAGE CONTRACT
# -----------------------
# When active artefacts carry images (e.g. PDF pages), those images are collected by
# _collect_artefact_images() and merged with any user-supplied images before the LLM
# call.  The system prompt (built by PromptMixin._build_artefact_instructions) already
# informs the LLM about <artefact_image id="TITLE::N" /> anchors.
#
# Image ordering sent to the LLM:
#   [discussion-level images] + [user message images] + [artefact images in order]
#
# Each artefact image is keyed by its id ("TITLE::N") which matches the anchor tag
# in the artefact text so the model can correlate text and pixel data.
#
# STREAMING STATE MACHINE
# -----------------------
# The streaming interceptor uses an explicit _StreamState object.
# States: NORMAL | BUFFERING_TAG | TOOL_CALL | SECONDARY
#
# BRACKET BUFFERING CONTRACT
# --------------------------
# '<' always starts buffering. Buffer flushed as text only when conclusively not a
# known tag. Whitespace or length alone NEVER trigger flush.

import json
import re
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ascii_colors import ASCIIColors, trace_exception

from ._artefacts import _find_best_title_match, ArtefactType, make_image_id, parse_image_id
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_personality import NullPersonality
from ._message import LollmsMessage

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Hard cap on bracket buffer size
# ---------------------------------------------------------------------------
_MAX_BRACKET_BUF = 4096

# ---------------------------------------------------------------------------
# All tag prefixes that must NEVER leak to the chat bubble
# ---------------------------------------------------------------------------
_TAG_STARTS = [
    "<tool_call>",
    "<think>", "<think ",
    "<artifact", "<artefact",
    "<generate_image", "<edit_image",
    "<revert_artifact", "<revert_artefact",
    "<generate_slides", "<street_view", "<schedule_task",
    "<note", "<skill",
    "<lollms_event", "<lollms_inline", "<lollms_form",
    "<use_handle",
]

# Secondary-stream tag type mapping
_SECONDARY_TAG_MAP = {
    "<artifact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK,
                       MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artifact>"),
    "<artefact":      ("artifact_update",     MSG_TYPE.MSG_TYPE_ARTEFACT_CHUNK,
                       MSG_TYPE.MSG_TYPE_ARTEFACT_DONE,    "</artefact>"),
    "<note":          ("note_start",          MSG_TYPE.MSG_TYPE_NOTE_CHUNK,
                       MSG_TYPE.MSG_TYPE_NOTE_DONE,         "</note>"),
    "<skill":         ("skill_start",         MSG_TYPE.MSG_TYPE_SKILL_CHUNK,
                       MSG_TYPE.MSG_TYPE_SKILL_DONE,        "</skill>"),
    "<lollms_inline": ("inline_widget_start", MSG_TYPE.MSG_TYPE_WIDGET_CHUNK,
                       MSG_TYPE.MSG_TYPE_WIDGET_DONE,       "</lollms_inline>"),
    "<lollms_form":   ("form_start",          MSG_TYPE.MSG_TYPE_WIDGET_CHUNK,
                       MSG_TYPE.MSG_TYPE_WIDGET_DONE,       "</lollms_form>"),
}


# ---------------------------------------------------------------------------
# Internal callback helpers
# ---------------------------------------------------------------------------

def _cb(callback, text: str, msg_type: MSG_TYPE, meta: Optional[Dict] = None) -> bool:
    if callback is None:
        return True
    try:
        result = callback(text, msg_type, meta or {})
        if result is False:
            return False
    except Exception as e:
        trace_exception(e)
    return True


def _step_start(callback, text: str, meta: Optional[Dict] = None) -> Optional[str]:
    event_id = str(uuid.uuid4())
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_START, {"id": event_id, **(meta or {})})
    return event_id


def _step_end(callback, text: str, event_id: Optional[str] = None,
              meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_STEP_END, {"id": event_id, **(meta or {})})


def _info(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_INFO, meta)


def _warning(callback, text: str, meta: Optional[Dict] = None):
    _cb(callback, text, MSG_TYPE.MSG_TYPE_WARNING, meta)


def _extract_content_title(content: str, max_len: int = 80) -> Optional[str]:
    if not content or not content.strip():
        return None
    lines     = content.splitlines()
    non_blank = [l for l in lines if l.strip()]
    if not non_blank:
        return None

    def _clean(s: str) -> str:
        s = re.sub(r"[*_`#~]+", "", s).strip(" \t\r\n|>")
        if len(s) > max_len:
            s = s[:max_len].rsplit(" ", 1)[0] + "…"
        return s or ""

    for line in non_blank[:30]:
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            t = _clean(m.group(1))
            if t:
                return t

    for i, line in enumerate(non_blank[:-1]):
        nxt = non_blank[i + 1]
        if re.match(r"^[=\-~^\"'`#*+]{4,}$", nxt.strip()) and line.strip():
            t = _clean(line)
            if t:
                return t

    fm = re.search(
        r'(?:^|[\n,{])\s*["\']?title["\']?\s*[:=]\s*["\']?([^\n"\'},]+)',
        content[:1000], re.IGNORECASE,
    )
    if fm:
        t = _clean(fm.group(1))
        if t:
            return t

    for pat in [r"<title[^>]*>([^<]+)</title>", r"<h1[^>]*>([^<]+)</h1>"]:
        m = re.search(pat, content[:2000], re.IGNORECASE)
        if m:
            t = _clean(m.group(1))
            if t:
                return t

    m = re.match(r"^\s*[*_]{1,2}([^*_\n]{4,60})[*_]{1,2}", content.lstrip())
    if m:
        t = _clean(m.group(1))
        if t:
            return t

    candidate = non_blank[0].strip()
    stripped  = _clean(candidate).rstrip(".")
    if stripped and len(stripped) <= max_len and not re.search(r"[.!?;]", stripped):
        return stripped

    return _clean(non_blank[0])


# ---------------------------------------------------------------------------
# Handle system helpers
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str) -> List[Dict[str, str]]:
    blocks = []
    pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    for m in pattern.finditer(text):
        blocks.append({
            "language": m.group(1).strip(),
            "content":  m.group(2),
            "raw":      m.group(0),
        })
    return blocks


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

    msg    = branch_messages[msg_idx]
    blocks = _extract_code_blocks(getattr(msg, "content", "") or "")

    if block_idx < 0 or block_idx >= len(blocks):
        return None

    return blocks[block_idx]


def _build_handle_instructions(branch_messages: List) -> str:
    entries = []
    for msg_idx, msg in enumerate(branch_messages):
        blocks = _extract_code_blocks(getattr(msg, "content", "") or "")
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


# ---------------------------------------------------------------------------
# Tool-catalogue helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_HEADER = """\

╔══════════════════════════════════════════════════════════════════╗
║  TOOL USE — READ CAREFULLY BEFORE GENERATING                     ║
╠══════════════════════════════════════════════════════════════════╣
║  You have external tools. To use one you MUST emit a tool_call   ║
║  tag containing a JSON object — NO markdown, NO prose calls.     ║
║                                                                  ║
║  EXACT FORMAT (copy this pattern):                               ║
║    <tool_call>{"name": "tool_name",                              ║
║                "parameters": {"key": "value"}}</tool_call>       ║
║                                                                  ║
║  Rules:                                                          ║
║    • One tool call per response turn.                            ║
║    • Do NOT call a tool you already called this turn (see STATE) ║
║    • After calling ALL needed tools, write your final answer.    ║
║    • If the user explicitly asks you to use a tool, USE IT.      ║
╚══════════════════════════════════════════════════════════════════╝

TOOLS AVAILABLE:
"""

_TOOL_CALL_REMINDER = """\
[AGENT REMINDER — YOU HAVE TOOLS]
Call syntax:  <tool_call>{"name": "NAME", "parameters": {…}}</tool_call>
One call per turn. Check AGENT STATE above before calling.
If the user asked you to use a specific tool, call it NOW.
"""

_TOOL_CALL_CORRECTION = """\
You were expected to call a tool but did not emit a <tool_call> tag.
Please re-read the available tools and call the most appropriate one now.
Do not explain why you didn't call it — just call it.
"""


def _build_tool_system_prompt(
    base_system_prompt: str,
    tool_descriptions: List[str],
) -> str:
    cleaned = re.split(
        r'\n*╔══+╗.*?╚══+╝.*?\nTOOLS AVAILABLE:',
        base_system_prompt,
        flags=re.DOTALL,
    )[0].rstrip()
    cleaned = re.split(
        r'\n*## Available Tools.*',
        cleaned,
        flags=re.DOTALL,
    )[0].rstrip()
    cleaned = re.split(
        r'\n*### FUNCTION CALLING INSTRUCTIONS.*',
        cleaned,
        flags=re.DOTALL,
    )[0].rstrip()

    tool_block = _TOOL_CALL_HEADER + "\n".join(tool_descriptions)
    return cleaned + "\n\n" + tool_block


# ---------------------------------------------------------------------------
# Widget content validation
# ---------------------------------------------------------------------------

_NON_WEB_FENCE_RE = re.compile(
    r'```(?:python|mermaid|java|c\+\+|cpp|c#|csharp|rust|go|ruby|php|r|'
    r'swift|kotlin|scala|haskell|erlang|elixir|clojure|lua|perl|bash|sh|'
    r'zsh|powershell|sql|graphql|yaml|toml|ini|json|xml|latex|tex|'
    r'dockerfile|makefile|cmake)[\s\S]*?```',
    re.IGNORECASE,
)

_HTML_TAG_RE = re.compile(r'<(?:html|head|body|div|span|script|style|p|h[1-6]|'
                           r'canvas|svg|button|input|form|table|ul|ol|li|a|'
                           r'section|article|main|header|footer|nav)[^>]*>',
                           re.IGNORECASE)


def _validate_widget_content(raw: str, title: str) -> Optional[str]:
    cleaned = _NON_WEB_FENCE_RE.sub('', raw).strip()
    sole_fence = re.match(r'^```(?:html)?\s*\n([\s\S]+?)\n```\s*$', cleaned, re.IGNORECASE)
    if sole_fence:
        cleaned = sole_fence.group(1).strip()
    if not cleaned:
        ASCIIColors.warning(
            f"[Widget '{title}'] Content is empty after stripping non-web fences. "
            "Widget discarded."
        )
        return None
    if not _HTML_TAG_RE.search(cleaned):
        ASCIIColors.warning(
            f"[Widget '{title}'] No HTML tags found after cleaning. "
            f"Content preview: {cleaned[:120]!r}. Widget discarded."
        )
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Form parsing helpers
# ---------------------------------------------------------------------------

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


def _format_form_answers_for_llm(form_descriptor: Dict, answers: Dict[str, Any]) -> str:
    lines = [
        f"=== FORM RESPONSE: {form_descriptor.get('title', 'User Form')} ===",
    ]
    fields = form_descriptor.get("fields", [])
    field_map = {f["name"]: f for f in fields if f.get("type") != "section"}

    for name, value in answers.items():
        label = field_map.get(name, {}).get("label", name)
        if isinstance(value, str) and len(value) > 2000:
            display = value[:2000] + f"… [+{len(value)-2000} chars truncated]"
        else:
            display = value
        lines.append(f"  {label}: {display}")

    lines.append("=== END FORM RESPONSE ===")
    return "\n".join(lines)


def _parse_tag_attrs(attr_str: str) -> Dict[str, str]:
    return {m.group(1): m.group(2)
            for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}


# ---------------------------------------------------------------------------
# Artefact image collection
# ---------------------------------------------------------------------------

def _collect_artefact_images(discussion: Any) -> List[str]:
    """
    Collect all images from active artefacts and return them as a list of
    base64 strings in the order: artefact images sorted by
    (activation order, image index).

    The LLM will receive these images appended after any user-supplied
    images, in the same order they appear as <artefact_image id="..."/>
    anchors in the active artefact text.
    """
    context_imgs = discussion.artefacts.get_context_images()
    return [img["data"] for img in context_imgs]


def _build_artefact_image_index(discussion: Any) -> Dict[str, int]:
    """
    Returns a mapping from image_id → position in the combined image list
    (0-based, after user images).

    Used to tell the LLM which numbered image slot corresponds to which
    artefact_image id when the binding doesn't support named images.
    This mapping is injected into the scratchpad context if needed.
    """
    context_imgs = discussion.artefacts.get_context_images()
    return {img["id"]: idx for idx, img in enumerate(context_imgs)}


def _build_artefact_image_map_note(
    discussion: Any,
    user_image_count: int,
) -> str:
    """
    Builds a compact system note that maps artefact_image IDs to the
    actual image slot numbers the model will receive.

    Only generated when there are artefact images to map.

    Example output:
        [Artefact image slots — images are 0-indexed in the vision input]
        <artefact_image id="my_doc::0" /> → image slot 2
        <artefact_image id="my_doc::1" /> → image slot 3
    """
    context_imgs = discussion.artefacts.get_context_images()
    if not context_imgs:
        return ""
    lines = [
        "[Artefact image map — images are appended after user images in the vision input]"
    ]
    for idx, img in enumerate(context_imgs):
        slot = user_image_count + idx
        lines.append(
            f'  <artefact_image id="{img["id"]}" /> → vision input image #{slot}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# StreamState — explicit state machine for the inline relay
# ---------------------------------------------------------------------------

class _StreamState:
    """
    Encapsulates all mutable state for the streaming interceptor.
    """

    STATE_NORMAL    = "normal"
    STATE_BUFFERING = "buffering"
    STATE_TOOL_CALL = "tool_call"
    STATE_SECONDARY = "secondary"

    def __init__(
        self,
        discussion: 'LollmsDiscussion',
        callback,
        ai_message,
        enable_notes: bool,
        enable_skills: bool,
        enable_inline_widgets: bool,
        enable_forms: bool,
        auto_activate_artefacts: bool = True,
    ):
        self.discussion            = discussion
        self.callback              = callback
        self.ai_message            = ai_message
        self.enable_notes          = enable_notes
        self.enable_skills         = enable_skills
        self.enable_inline_widgets = enable_inline_widgets
        self.enable_forms          = enable_forms
        self.auto_activate         = auto_activate_artefacts

        self.state: str            = self.STATE_NORMAL
        self.bracket_buf: List[str] = []

        self.tool_buf: List[str]    = []
        self.tool_trigger: bool     = False

        self.sec_prefix: str        = ""
        self.sec_chunk_mt: Any      = None
        self.sec_done_mt: Any       = None
        self.sec_close_tag: str     = ""
        self.sec_open_attrs: Dict   = {}
        self.sec_content: List[str] = []
        self.sec_close_scan: str    = ""

        self.affected_artefacts: List[Dict] = []
        self.stream_buf: List[str]  = []

    # ---------------------------------------------------------------- public entry point

    def feed(self, chunk: str) -> bool:
        if not isinstance(chunk, str):
            return True

        self.stream_buf.append(chunk)

        pos = 0
        while pos < len(chunk):
            ch = chunk[pos]

            if self.state == self.STATE_NORMAL:
                pos = self._feed_normal(chunk, pos)

            elif self.state == self.STATE_BUFFERING:
                pos = self._feed_buffering(chunk, pos)
                if self.tool_trigger:
                    return False

            elif self.state == self.STATE_TOOL_CALL:
                pos = self._feed_tool_call(chunk, pos)
                if self.tool_trigger:
                    return False

            elif self.state == self.STATE_SECONDARY:
                pos = self._feed_secondary(chunk, pos)

            else:
                pos += 1

        return True

    # ---------------------------------------------------------------- passthrough relay

    def passthrough(self, chunk, msg_type=None, meta=None) -> bool:
        if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
            if msg_type in (MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, MSG_TYPE.MSG_TYPE_REASONING):
                self.ai_message.thoughts = (self.ai_message.thoughts or "") + (chunk or "")
            return _cb(self.callback, chunk, msg_type, meta)
        return True

    # ---------------------------------------------------------------- STATE_NORMAL

    def _feed_normal(self, chunk: str, pos: int) -> int:
        lt_idx = chunk.find("<", pos)

        if lt_idx == -1:
            text = chunk[pos:]
            if text:
                self.ai_message.content += text
                _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK)
            return len(chunk)

        if lt_idx > pos:
            text = chunk[pos:lt_idx]
            self.ai_message.content += text
            _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK)

        self.state = self.STATE_BUFFERING
        self.bracket_buf = ["<"]
        return lt_idx + 1

    # ---------------------------------------------------------------- STATE_BUFFERING

    def _feed_buffering(self, chunk: str, pos: int) -> int:
        gt_idx = chunk.find(">", pos)

        if gt_idx != -1:
            self.bracket_buf.append(chunk[pos:gt_idx + 1])
            b_str = "".join(self.bracket_buf)
            new_pos = gt_idx + 1

            if "<tool_call>" in b_str:
                self.state = self.STATE_TOOL_CALL
                self.tool_buf = [b_str]
                return new_pos

            matched_prefix = self._match_secondary_prefix(b_str)
            if matched_prefix:
                self.bracket_buf.clear()
                self.state = self.STATE_SECONDARY
                self._enter_secondary(b_str, matched_prefix)
                return new_pos

            self._flush_bracket_buf_as_text()
            self.state = self.STATE_NORMAL
            return new_pos

        else:
            self.bracket_buf.append(chunk[pos:])
            b_str = "".join(self.bracket_buf)

            if len(b_str) >= _MAX_BRACKET_BUF:
                ASCIIColors.warning(
                    f"[StreamState] Bracket buffer hard cap ({_MAX_BRACKET_BUF}) reached. "
                    "Flushing as plain text."
                )
                self._flush_bracket_buf_as_text()
                self.state = self.STATE_NORMAL
                return len(chunk)

            b_str_lower = b_str.lower()
            can_still_match = any(
                s.lower().startswith(b_str_lower) or b_str_lower.startswith(s[:len(b_str_lower)].lower())
                for s in _TAG_STARTS
            )
            if not can_still_match and len(b_str) > 1:
                self._flush_bracket_buf_as_text()
                self.state = self.STATE_NORMAL

            return len(chunk)

    def _flush_bracket_buf_as_text(self):
        text = "".join(self.bracket_buf)
        self.bracket_buf.clear()
        if text:
            self.ai_message.content += text
            _cb(self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK)

    def _match_secondary_prefix(self, b_str: str) -> Optional[str]:
        b_str_lower = b_str.lower()
        for prefix in _SECONDARY_TAG_MAP:
            if b_str_lower.startswith(prefix.lower()):
                if prefix in ("<note",)          and not self.enable_notes:           continue
                if prefix in ("<skill",)         and not self.enable_skills:          continue
                if prefix in ("<lollms_inline",) and not self.enable_inline_widgets:  continue
                if prefix in ("<lollms_form",)   and not self.enable_forms:           continue
                return prefix
        return None

    # ---------------------------------------------------------------- STATE_TOOL_CALL

    def _feed_tool_call(self, chunk: str, pos: int) -> int:
        self.tool_buf.append(chunk[pos:])
        accumulated = "".join(self.tool_buf)

        if "</tool_call>" in accumulated:
            self.tool_trigger = True
            return len(chunk)

        return len(chunk)

    # ---------------------------------------------------------------- STATE_SECONDARY

    def _enter_secondary(self, opening_tag: str, prefix: str):
        ann_type, chunk_mt, done_mt, close_tag = _SECONDARY_TAG_MAP[prefix]

        self.sec_prefix    = prefix
        self.sec_chunk_mt  = chunk_mt
        self.sec_done_mt   = done_mt
        self.sec_close_tag = close_tag
        self.sec_open_attrs = _parse_tag_attrs(opening_tag)
        self.sec_content   = []
        self.sec_close_scan = ""

        if prefix in ("<artifact", "<artefact"):
            title = self.sec_open_attrs.get('name') or self.sec_open_attrs.get('title', '')
            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type":    "artifact_update",
                "content": {"title": title},
            })
        elif prefix == "<note":
            title = self.sec_open_attrs.get('title') or self.sec_open_attrs.get('name', 'Note')
            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type":    "note_start",
                "content": {"title": title},
            })
        elif prefix == "<skill":
            title    = self.sec_open_attrs.get('title') or self.sec_open_attrs.get('name', 'Skill')
            category = self.sec_open_attrs.get('category', '')
            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type":    "skill_start",
                "content": {"title": title, "category": category},
            })
        elif prefix == "<lollms_inline":
            title       = self.sec_open_attrs.get('title', 'Interactive Widget')
            widget_type = self.sec_open_attrs.get('type', 'html')
            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type":    "inline_widget_start",
                "content": {"title": title, "widget_type": widget_type},
            })
        elif prefix == "<lollms_form":
            title = self.sec_open_attrs.get('title', 'Please fill in the form')
            _cb(self.callback, "", MSG_TYPE.MSG_TYPE_CHUNK, {
                "type":    "form_start",
                "content": {"title": title},
            })

    def _feed_secondary(self, chunk: str, pos: int) -> int:
        close_tag = self.sec_close_tag
        close_len = len(close_tag)
        incoming  = chunk[pos:]

        self.sec_close_scan += incoming

        close_idx = self.sec_close_scan.find(close_tag)

        if close_idx != -1:
            pre_close  = self.sec_close_scan[:close_idx]
            post_close = self.sec_close_scan[close_idx + close_len:]

            if pre_close:
                self.sec_content.append(pre_close)
                self._fire_secondary_chunk(pre_close)

            self._fire_secondary_done()

            self.sec_prefix     = ""
            self.sec_chunk_mt   = None
            self.sec_done_mt    = None
            self.sec_close_tag  = ""
            self.sec_open_attrs = {}
            self.sec_close_scan = ""
            self.state          = self.STATE_NORMAL

            if post_close:
                self._feed_post_close(post_close)

        else:
            safe_len = max(0, len(self.sec_close_scan) - close_len + 1)
            safe_content = self.sec_close_scan[:safe_len]
            if safe_content:
                self.sec_content.append(safe_content)
                self._fire_secondary_chunk(safe_content)
                self.sec_close_scan = self.sec_close_scan[safe_len:]

        return len(chunk)

    def _feed_post_close(self, text: str):
        pos = 0
        while pos < len(text):
            if self.state == self.STATE_NORMAL:
                pos = self._feed_normal(text, pos)
            elif self.state == self.STATE_BUFFERING:
                pos = self._feed_buffering(text, pos)
                if self.tool_trigger:
                    break
            elif self.state == self.STATE_SECONDARY:
                pos = self._feed_secondary(text, pos)
            else:
                pos += 1

    # ---------------------------------------------------------------- secondary event helpers

    def _fire_secondary_chunk(self, content: str):
        if not self.sec_chunk_mt or not content:
            return
        attrs = self.sec_open_attrs
        prefix = self.sec_prefix

        if prefix in ("<artifact", "<artefact"):
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title":    attrs.get('name') or attrs.get('title', ''),
                "chunk":    content,
                "art_type": attrs.get('type', 'document'),
                "language": attrs.get('language'),
            })
        elif prefix == "<note":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title": attrs.get('title') or attrs.get('name', 'Note'),
                "chunk": content,
            })
        elif prefix == "<skill":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title":       attrs.get('title') or attrs.get('name', 'Skill'),
                "chunk":       content,
                "category":    attrs.get('category', ''),
                "description": attrs.get('description', ''),
            })
        elif prefix == "<lollms_inline":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title":       attrs.get('title', 'Interactive Widget'),
                "chunk":       content,
                "widget_type": attrs.get('type', 'html'),
            })
        elif prefix == "<lollms_form":
            _cb(self.callback, content, self.sec_chunk_mt, {
                "title": attrs.get('title', 'Form'),
                "chunk": content,
            })

    def _fire_secondary_done(self):
        if not self.sec_done_mt:
            return

        full_content = "".join(self.sec_content)
        attrs        = dict(self.sec_open_attrs)
        prefix       = self.sec_prefix

        def _fire_state_change(art, is_new):
            if not self.callback: return
            ev_type = "artifact_created" if is_new else "artifact_updated"
            _cb(self.callback, json.dumps({
                "type": ev_type, "title": art.get("title"),
                "version": art.get("version"), "art_type": art.get("type")
            }), MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefact": art, "is_new": is_new})

        if prefix in ("<artifact", "<artefact"):
            tag_title = attrs.pop('name', attrs.pop('title', 'untitled'))
            new_name  = attrs.pop('rename', None)
            atype     = attrs.pop('type', 'code')
            lang      = attrs.pop('language', None)
            attrs.pop('images', None)
            attrs.pop('image_media_types', None)

            existing_titles = self.discussion.artefacts._all_latest_titles()
            resolved_title = tag_title if tag_title in existing_titles else (
                _find_best_title_match(tag_title, existing_titles) or tag_title
            )
            is_new = resolved_title not in existing_titles

            is_patch = bool(re.search(r'<{6,8}\s*SEARCH', full_content, re.I))

            result_art = None
            if is_patch:
                existing = self.discussion.artefacts.get(resolved_title)
                if existing:
                    try:
                        patched = self.discussion.artefacts.apply_aider_patch(existing.get('content', ''), full_content)
                        result_art = self.discussion.artefacts.update(
                            resolved_title, new_content=patched, new_title=new_name,
                            language=lang, active=self.auto_activate, **attrs)
                    except ValueError: result_art = existing
                else:
                    result_art = self.discussion.artefacts.add(resolved_title, atype, full_content, language=lang, active=self.auto_activate, **attrs)
            else:
                if is_new:
                    result_art = self.discussion.artefacts.add(resolved_title, atype, full_content.strip(), language=lang, active=self.auto_activate, **attrs)
                else:
                    result_art = self.discussion.artefacts.update(resolved_title, new_content=full_content.strip(), new_title=new_name, new_type=atype, language=lang, active=self.auto_activate, **attrs)

            if result_art:
                self.affected_artefacts.append(result_art)
                _fire_state_change(result_art, is_new)

            _cb(self.callback, full_content, self.sec_done_mt, {
                "title": resolved_title, "content": full_content, "art_type": atype,
                "language": lang, "is_patch": is_patch, "attrs": attrs
            })

        elif prefix == "<note":
            title = attrs.get('title') or attrs.get('name', f'note_{uuid.uuid4().hex[:8]}')
            art = self.discussion.artefacts.add(title=title, artefact_type=ArtefactType.NOTE, content=full_content.strip(), active=self.auto_activate)
            self.affected_artefacts.append(art)
            _fire_state_change(art, True)
            _cb(self.callback, full_content, self.sec_done_mt, {"title": title, "content": full_content})

        elif prefix == "<skill":
            title = attrs.get('title') or attrs.get('name', f'skill_{uuid.uuid4().hex[:8]}')
            desc = attrs.get('description', '')
            cat = attrs.get('category', '')
            art = self.discussion.artefacts.add(title=title, artefact_type=ArtefactType.SKILL, content=full_content.strip(), active=self.auto_activate, description=desc, category=cat)
            self.affected_artefacts.append(art)
            _fire_state_change(art, True)
            _cb(self.callback, full_content, self.sec_done_mt, {"title": title, "content": full_content, "category": cat, "description": desc})

        elif prefix == "<lollms_inline":
            title = attrs.get('title', 'Interactive Widget')
            wtype = attrs.get('type', 'html').lower().strip()
            validated = _validate_widget_content(full_content, title) or ""

            meta = dict(self.ai_message.metadata or {})
            if "inline_widgets" not in meta: meta["inline_widgets"] = []
            widget_id = str(uuid.uuid4())
            meta["inline_widgets"].append({"id": widget_id, "type": wtype, "title": title, "source": validated})
            self.ai_message.metadata = meta

            anchor = f'\n<lollms_widget id="{widget_id}" />\n'
            self.ai_message.content += anchor
            _cb(self.callback, anchor, MSG_TYPE.MSG_TYPE_CHUNK)

            _cb(self.callback, validated, self.sec_done_mt, {"title": title, "content": validated, "widget_type": wtype})

        elif prefix == "<lollms_form":
            form_descriptor = _parse_form_xml(" ".join(f'{k}="{v}"' for k, v in attrs.items()), full_content)
            if form_descriptor:
                self.discussion._get_pending_forms()[form_descriptor["id"]] = form_descriptor
                meta = dict(self.ai_message.metadata or {})
                if "forms" not in meta: meta["forms"] = []
                meta["forms"].append({"id": form_descriptor["id"], "title": form_descriptor["title"]})
                self.ai_message.metadata = meta

                anchor = f'\n<lollms_form_anchor id="{form_descriptor["id"]}" />\n'
                self.ai_message.content += anchor
                _cb(self.callback, anchor, MSG_TYPE.MSG_TYPE_CHUNK)
                _cb(self.callback, json.dumps(form_descriptor), MSG_TYPE.MSG_TYPE_FORM_READY, {"form": form_descriptor, "form_id": form_descriptor["id"]})

    # ---------------------------------------------------------------- accessors

    def get_accumulated_stream(self) -> str:
        return "".join(self.stream_buf)

    def get_tool_call_json(self) -> Optional[str]:
        if not self.tool_trigger:
            return None
        raw = "".join(self.tool_buf)
        open_idx  = raw.find("<tool_call>")
        close_idx = raw.find("</tool_call>")
        if open_idx == -1 or close_idx == -1:
            return None
        return raw[open_idx + len("<tool_call>"):close_idx].strip()

    def get_clean_text_so_far(self) -> str:
        return self.ai_message.content

    def flush_remaining_buffer(self):
        if self.state == self.STATE_BUFFERING and self.bracket_buf:
            self._flush_bracket_buf_as_text()
            self.state = self.STATE_NORMAL


# ---------------------------------------------------------------------------
# ChatMixin
# ---------------------------------------------------------------------------

class ChatMixin:
    """
    Provides simplified_chat() and chat().

    Artefact images
    ---------------
    When active artefacts carry images (e.g. PDF pages rendered to PNG/JPEG),
    those images are automatically collected and appended to the LLM call
    alongside any user-supplied images.  The artefact text content uses
    <artefact_image id="TITLE::N" /> anchors so the model can correlate
    each image to its position in the document.

    Image ordering sent to the LLM:
        [discussion-level images] + [user message images] + [artefact images]

    The system prompt injection always includes a map of id → slot number so
    that even vision models that only see positional image slots can resolve
    the anchors correctly.
    """

    # ------------------------------------------------------------------ pending forms

    def _get_pending_forms(self) -> Dict[str, Dict]:
        if not hasattr(self, '_pending_forms_store'):
            object.__setattr__(self, '_pending_forms_store', {})
        return self._pending_forms_store  # type: ignore[attr-defined]

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

    # ------------------------------------------------------------------ helpers

    def _merge_artefact_images(self, user_images: Optional[List[str]]) -> List[str]:
        """
        Merge user-supplied images with active-artefact images.

        Returns the combined list: [user_images...] + [artefact_images...]
        Also injects an image-map note into self.scratchpad so the LLM
        knows which slot corresponds to which artefact_image id.
        """
        base = list(user_images or [])
        art_images = _collect_artefact_images(self)

        if not art_images:
            return base

        combined = base + art_images

        # Build the image-map note and append to scratchpad so it is visible
        # to the LLM immediately before/after the last user message.
        map_note = _build_artefact_image_map_note(self, len(base))
        if map_note:
            existing_scratch = getattr(self, 'scratchpad', '') or ''
            object.__setattr__(
                self, 'scratchpad',
                (existing_scratch + "\n\n" + map_note).strip()
            )

        return combined

    def _stream_final_answer(self, callback, images, branch_tip_id, temperature, **kwargs):
        caller_stream = kwargs.pop("stream", None)
        kwargs.pop("callback", None)
        kwargs.pop("streaming_callback", None)

        do_stream = (callback is not None) and (caller_stream is not False)
        collected = []

        # Merge artefact images into the image list before calling the LLM
        merged_images = self._merge_artefact_images(images)

        ss = _StreamState(
            discussion            = self,
            callback              = callback,
            ai_message            = LollmsMessage(self, SimpleNamespace(content="", metadata={})),
            enable_notes          = kwargs.get("enable_notes", True),
            enable_skills         = kwargs.get("enable_skills", False),
            enable_inline_widgets = kwargs.get("enable_inline_widgets", True),
            enable_forms          = kwargs.get("enable_forms", True),
            auto_activate_artefacts = kwargs.get("auto_activate_artefacts", True),
        )

        def _streaming_relay(chunk, msg_type=None, meta=None):
            if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                return ss.passthrough(chunk, msg_type, meta)
            if isinstance(chunk, str):
                collected.append(chunk)
                return ss.feed(chunk)
            return True

        result = self.lollmsClient.chat(
            self,
            images=merged_images,
            branch_tip_id=branch_tip_id,
            stream=do_stream,
            streaming_callback=_streaming_relay if do_stream else None,
            temperature=temperature,
            **kwargs,
        )

        if do_stream:
            if isinstance(result, str) and result and not collected:
                _cb(callback, result, MSG_TYPE.MSG_TYPE_CHUNK)
                return result
            return "".join(collected) if collected else (result or "")
        return result if isinstance(result, str) else (result or "")

    # ------------------------------------------------------------------ context compression
    # (unchanged from original — reproduced in full for completeness)

    def _compress_context(
        self,
        callback,
        max_context_size: int,
        answer_reserve_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        import hashlib

        budget        = int(max_context_size * (1.0 - answer_reserve_ratio))
        status        = self.get_context_status()
        tokens_before = status.get("current_tokens", 0)

        if tokens_before <= budget:
            return {
                "needed": False, "artefact_pressure": False,
                "tokens_before": tokens_before, "tokens_after": tokens_before,
                "budget": budget, "cache_hit": False, "summary_generated": False,
            }

        zones           = status.get("zones", {})
        history_tokens  = zones.get("message_history", {}).get("tokens", 0)
        artefact_tokens = (
            zones.get("system_context", {})
                 .get("breakdown", {})
                 .get("artefacts_zone", {})
                 .get("tokens", 0)
        )
        artefact_pressure = (
            artefact_tokens > 0
            and artefact_tokens > history_tokens
            and tokens_before > budget
        )

        _info(callback,
              f"Context at {tokens_before:,} / {budget:,} tokens "
              f"({'artefact-heavy' if artefact_pressure else 'history-heavy'}) -- compressing...")

        active_ids = sorted(a.get("id", "") for a in self.artefacts.list(active_only=True))
        key_src    = (self.active_branch_id or "") + "|" + ",".join(active_ids)
        cache_key  = __import__('hashlib').sha1(key_src.encode()).hexdigest()

        meta   = dict(self.metadata or {})
        cache  = meta.get("_compression_cache", {})
        cached = cache.get(cache_key)

        if cached:
            self.pruning_summary  = cached.get("summary", "")
            self.pruning_point_id = cached.get("pruning_point_id", "")
            self.touch()
            tokens_after = self.lollmsClient.count_tokens(
                self.export("lollms_text", self.active_branch_id, 999999)
            )
            _cb(callback,
                json.dumps({"type": "cache_hit", "tokens_before": tokens_before,
                            "tokens_after": tokens_after, "budget": budget,
                            "cache_key": cache_key,
                            "artefact_pressure": artefact_pressure}),
                MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION,
                {"tokens_before": tokens_before, "tokens_after": tokens_after,
                 "budget": budget, "cache_hit": True,
                 "artefact_pressure": artefact_pressure})
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_after,
                    "budget": budget, "cache_hit": True, "summary_generated": False}

        branch          = self.get_branch(self.active_branch_id)
        preserve_last_n = max(4, len(branch) // 4)

        if len(branch) <= preserve_last_n:
            _info(callback, "  History too short to prune -- reporting artefact pressure")
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_before,
                    "budget": budget, "cache_hit": False, "summary_generated": False}

        to_prune      = branch[:-preserve_last_n]
        pruning_point = branch[-preserve_last_n]
        text_to_sum   = "\n\n".join(f"{m.sender}: {m.content}" for m in to_prune)

        try:
            summary = self.lollmsClient.generate_text(
                "Produce a concise but complete summary of the following conversation "
                "segment, preserving all technical decisions, code snippets, file names, "
                "variable names, and key facts. The summary will replace the original "
                "messages in the context window.\n\n"
                f"--- BEGIN SEGMENT ---\n{text_to_sum}\n--- END SEGMENT ---\n\nSUMMARY:",
                n_predict=1024, temperature=0.1,
            )
        except Exception as e:
            _warning(callback, f"  Compression failed: {e}")
            return {"needed": True, "artefact_pressure": artefact_pressure,
                    "tokens_before": tokens_before, "tokens_after": tokens_before,
                    "budget": budget, "cache_hit": False, "summary_generated": False}

        self.pruning_summary = (
            ((self.pruning_summary or "").rstrip()
             + "\n\n--- Summary ---\n" + summary.strip()).strip()
        )
        self.pruning_point_id = pruning_point.id
        self.touch()

        cache[cache_key] = {
            "summary":          self.pruning_summary,
            "pruning_point_id": self.pruning_point_id,
            "tokens_before":    tokens_before,
            "created_at":       datetime.utcnow().isoformat(),
        }
        if len(cache) > 10:
            del cache[next(iter(cache))]
        meta["_compression_cache"] = cache
        self.metadata = meta
        self.commit()

        tokens_after = self.lollmsClient.count_tokens(
            self.export("lollms_text", self.active_branch_id, 999999)
        )
        _cb(callback,
            json.dumps({"type": "summary_generated", "messages_pruned": len(to_prune),
                        "tokens_before": tokens_before, "tokens_after": tokens_after,
                        "budget": budget, "cache_key": cache_key,
                        "artefact_pressure": artefact_pressure}),
            MSG_TYPE.MSG_TYPE_CONTEXT_COMPRESSION,
            {"tokens_before": tokens_before, "tokens_after": tokens_after,
             "budget": budget, "cache_hit": False, "summary_generated": True,
             "messages_pruned": len(to_prune), "artefact_pressure": artefact_pressure})
        _info(callback,
              f"  Compressed {len(to_prune)} messages: "
              f"{tokens_before:,} -> {tokens_after:,} tokens")

        return {"needed": True, "artefact_pressure": artefact_pressure,
                "tokens_before": tokens_before, "tokens_after": tokens_after,
                "budget": budget, "cache_hit": False, "summary_generated": True}

    # ------------------------------------------------------------------ simplified_chat

    def simplified_chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        mcps=None,
        rag_data_stores=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        use_rlm: bool = False,
        decision_temperature: float = 0.2,
        final_answer_temperature: float = 0.7,
        rag_top_k: int = 5,
        rag_min_similarity_percent: float = 0.5,
        enable_image_generation: bool = False,
        enable_image_editing:    bool = False,
        auto_activate_artefacts: bool = True,
        enable_inline_widgets:   bool = True,
        enable_notes:            bool = True,
        enable_skills:           bool = False,
        enable_forms:            bool = True,
        enable_silent_artefact_explanation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        self.scratchpad = ""
        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        object.__setattr__(self, '_active_callback', callback)

        def is_fast(msg):
            m = msg.lower().strip()
            if len(m) < 20 and any(x in m for x in ["bonjour", "salut", "hello", "hi", "hey"]):
                return True
            return m in ["ok", "merci", "thanks", "cool", "yes", "no", "oui", "non"]

        extra_instructions = self._build_artefact_instructions()
        if enable_image_generation or enable_image_editing:
            extra_instructions += self._build_image_generation_instructions()
        if enable_inline_widgets:
            extra_instructions += self._build_inline_widget_instructions()
        if enable_notes:
            extra_instructions += self._build_note_instructions()
        if enable_skills:
            extra_instructions += self._build_skill_instructions()
        if enable_forms:
            extra_instructions += self._build_form_instructions()

        branch_msgs = self.get_branch(branch_tip_id or self.active_branch_id)
        handle_instructions = _build_handle_instructions(branch_msgs)
        if handle_instructions:
            extra_instructions += handle_instructions

        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        user_msg = None
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=user_message,
                images=images,
                **kwargs,
            )

        def _finish(text):
            ai = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content=text,
                parent_id=user_msg.id if user_msg else None,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
            )
            branch_msgs_updated = self.get_branch(ai.id)
            text_after_handles, handle_artefacts = _apply_handles(
                text, branch_msgs_updated, self.artefacts
            )
            if text_after_handles != text:
                ai.content = text_after_handles

            cleaned, affected_pp = self._post_process_llm_response(
                text_after_handles, ai,
                enable_image_generation, enable_image_editing,
                auto_activate_artefacts,
                enable_inline_widgets=enable_inline_widgets,
                enable_notes=enable_notes,
                enable_skills=enable_skills,
                enable_forms=enable_forms,
                enable_silent_artefact_explanation=enable_silent_artefact_explanation,
            )
            affected = handle_artefacts + ss.affected_artefacts + affected_pp
            if cleaned != text_after_handles:
                ai.content = cleaned
            if affected and callback:
                _cb(callback, json.dumps([a.get("title") for a in affected]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
            object.__setattr__(self, '_active_callback', None)
            return {"user_message": user_msg, "ai_message": ai,
                    "sources": [], "artefacts": affected}

        if is_fast(user_message):
            _info(callback, "Simple response path")
            return _finish(self._stream_final_answer(
                callback, images, branch_tip_id, 0.1, **kwargs))

        if self.memory and user_message.lower() in self.memory.lower():
            _info(callback, "Answering from memory")
            return _finish(self._stream_final_answer(
                callback, images, branch_tip_id, final_answer_temperature, **kwargs))

        intent_id = _step_start(callback, "Analyzing intent...")
        intent = self.lollmsClient.generate_structured_content(
            prompt=user_message,
            schema={
                "needs_internal_knowledge": "boolean",
                "needs_full_documents":     "boolean",
                "needs_external_search":    "boolean",
                "reasoning":                "string",
            },
            temperature=decision_temperature,
        )
        _step_end(callback, "Intent analysis complete", intent_id, {"intent": intent})

        scratchpad = ""
        sources: List[str] = []

        if intent and intent.get("needs_full_documents"):
            docs_id = _step_start(callback, "Loading context documents...")
            for zone_name, zone_content in [
                ("user_data",        self.user_data_zone),
                ("discussion_data",  self.discussion_data_zone),
                ("personality_data", self.personality_data_zone),
            ]:
                if zone_content:
                    scratchpad += f"\n--- {zone_name} ---\n{zone_content}\n"
                    sources.append(zone_name)
            _step_end(callback, f"Loaded {len(sources)} zone(s)", docs_id)

        if intent and intent.get("needs_external_search") and rag_data_stores:
            rag_id = _step_start(callback, "Searching external knowledge...")
            for name, fn in rag_data_stores.items():
                if callable(fn):
                    try:
                        res = fn(user_message)
                        if res:
                            scratchpad += f"\n--- {name} ---\n{str(res)}\n"
                            sources.append(name)
                            _info(callback, f"  Retrieved results from `{name}`")
                    except Exception as e:
                        _warning(callback, f"  `{name}` search error: {e}")
            if sources:
                _cb(callback, sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            _step_end(callback, "External search complete", rag_id, {"sources": sources})

        if scratchpad:
            self.scratchpad = scratchpad.strip()

        answer_id  = _step_start(callback, "Generating answer...")
        final_text = self._stream_final_answer(
            callback, images, branch_tip_id, final_answer_temperature, **kwargs)
        _step_end(callback, "Answer generation complete", answer_id)

        if remove_thinking_blocks:
            final_text = self.lollmsClient.remove_thinking_blocks(final_text)

        ai = self.add_message(
            sender=personality.name,
            sender_type="assistant",
            content=final_text,
            parent_id=user_msg.id if user_msg else None,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"sources": sources} if sources else {},
        )

        branch_msgs_updated = self.get_branch(ai.id)
        final_text_after_handles, handle_artefacts = _apply_handles(
            final_text, branch_msgs_updated, self.artefacts
        )
        if final_text_after_handles != final_text:
            ai.content = final_text_after_handles

        cleaned, affected = self._post_process_llm_response(
            final_text_after_handles, ai,
            enable_image_generation, enable_image_editing,
            auto_activate_artefacts,
            enable_inline_widgets=enable_inline_widgets,
            enable_notes=enable_notes,
            enable_skills=enable_skills,
            enable_forms=enable_forms,
            enable_silent_artefact_explanation=enable_silent_artefact_explanation,
        )
        affected = handle_artefacts + affected
        if cleaned != final_text_after_handles:
            ai.content = cleaned
        if affected and callback:
            _cb(callback, json.dumps([a.get("title") for a in affected]),
                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
        self.scratchpad = ""
        object.__setattr__(self, '_active_callback', None)

        return {"user_message": user_msg, "ai_message": ai,
                "sources": sources, "artefacts": affected}

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        user_message: str,
        personality=None,
        branch_tip_id=None,
        tools=None,
        swarm=None,
        swarm_config=None,
        add_user_message: bool = True,
        max_reasoning_steps: int = 20,
        images=None,
        debug: bool = False,
        remove_thinking_blocks: bool = True,
        enable_image_generation: bool = True,
        enable_image_editing:    bool = True,
        auto_activate_artefacts: bool = True,
        enable_show_tools:            bool = True,
        enable_extract_artefact:      bool = True,
        enable_final_answer:          bool = True,
        enable_request_clarification: bool = True,
        enable_repl_tools:            bool = True,
        enable_inline_widgets:        bool = True,
        enable_notes:                 bool = True,
        enable_skills:                bool = False,
        enable_forms:                 bool = True,
        enable_silent_artefact_explanation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        self.scratchpad = ""

        personality = personality or NullPersonality()
        callback    = kwargs.get("streaming_callback")

        if "temperature" in kwargs:
            final_answer_temperature = kwargs.pop("temperature")
        else:
            final_answer_temperature = 0.7

        object.__setattr__(self, '_active_callback', callback)

        # ====================================================================
        #  SWARM DISPATCH
        # ====================================================================
        if swarm:
            from lollms_client.lollms_swarm import SwarmOrchestrator, SwarmConfig as _SC

            _swarm_config = swarm_config if swarm_config is not None else _SC()

            if add_user_message:
                user_msg = self.add_message(
                    sender=kwargs.get("user_name", "user"),
                    sender_type="user",
                    content=user_message,
                    images=images,
                    **kwargs,
                )
            else:
                user_msg = None

            orchestrator = SwarmOrchestrator(
                discussion  = self,
                agents      = swarm,
                config      = _swarm_config,
                callback    = callback,
                user_msg_id = user_msg.id if user_msg else self.active_branch_id,
            )
            result = orchestrator.run(user_message, **kwargs)
            result["user_message"] = user_msg
            object.__setattr__(self, '_active_callback', None)
            self.scratchpad = ""
            if self._is_db_backed and self.autosave:
                self.commit()
            return result

        # ── Effective image flags ────────────────────────────────────────────
        _tti_available = getattr(self.lollmsClient, 'tti', None) is not None
        _eff_img_gen   = enable_image_generation and _tti_available
        _eff_img_edit  = enable_image_editing     and _tti_available

        # ── System-prompt instructions ───────────────────────────────────────
        extra_instructions = self._build_artefact_instructions()
        if _eff_img_gen or _eff_img_edit:
            extra_instructions += self._build_image_generation_instructions()
        if enable_inline_widgets:
            extra_instructions += self._build_inline_widget_instructions()
        if enable_notes:
            extra_instructions += self._build_note_instructions()
        if enable_skills:
            extra_instructions += self._build_skill_instructions()
        if enable_forms:
            extra_instructions += self._build_form_instructions()

        branch_msgs_now = self.get_branch(branch_tip_id or self.active_branch_id)
        handle_instructions = _build_handle_instructions(branch_msgs_now)
        if handle_instructions:
            extra_instructions += handle_instructions

        if extra_instructions.strip():
            original_sp = self._system_prompt or ""
            if extra_instructions not in original_sp:
                object.__setattr__(self, "_system_prompt", original_sp + extra_instructions)

        # ── Generation parameters ────────────────────────────────────────────
        kwargs.pop("temperature", None)
        decision_temperature       = kwargs.get("decision_temperature",       0.3)
        final_answer_temperature   = kwargs.get("final_answer_temperature",   final_answer_temperature)
        rag_top_k                  = kwargs.get("rag_top_k",                  5)
        rag_min_similarity_percent = kwargs.get("rag_min_similarity_percent", 0.5)
        preflight_rag_enabled      = kwargs.get("preflight_rag",              True)

        # ── RLM detection ────────────────────────────────────────────────────
        rlm_enabled          = False
        rlm_context_var_name = "USER_INPUT_CONTEXT"
        actual_user_content  = user_message

        if tools:
            rlm_enabled = (
                any(t.get("name") == "python_exec" for t in tools.values() if isinstance(t, dict))
                and
                any(t.get("name") == "llm_query"   for t in tools.values() if isinstance(t, dict))
            )

        if rlm_enabled and len(user_message) > 10000:
            actual_user_content = "\n".join([
                "<RLM_STUB>",
                f"Large input ({len(user_message):,} chars) stored in `{rlm_context_var_name}`.",
                "PREVIEW:", user_message[:500], "...",
                "Use python_exec() to access the full content.",
                "</RLM_STUB>",
            ])

        # ── Add user message ─────────────────────────────────────────────────
        if add_user_message:
            user_msg = self.add_message(
                sender=kwargs.get("user_name", "user"),
                sender_type="user",
                content=actual_user_content,
                images=images,
                **kwargs,
            )
            if rlm_enabled and len(user_message) > 10000:
                user_msg.metadata["rlm_full_content"] = user_message
                user_msg.metadata["rlm_var_name"]     = rlm_context_var_name
        else:
            if self.active_branch_id not in self._message_index:
                raise ValueError("Regeneration failed: active branch tip not found.")
            user_msg = LollmsMessage(self, self._message_index[self.active_branch_id])
            images   = user_msg.get_active_images()

        # ── Source inference helper ──────────────────────────────────────────
        def _infer_sources_from_json(data: Any, tool_name: str) -> List[Dict]:
            found_sources = []

            def looks_like_source(d: dict) -> bool:
                title_keys = {'title', 'name', 'label', 'header', 'id', 'filename'}
                return any(k in d for k in title_keys)

            def scan(obj: Any):
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict) and looks_like_source(item):
                            found_sources.append(item)
                        else:
                            scan(item)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in ['results', 'sources', 'data', 'items', 'content'] \
                                and isinstance(v, list):
                            scan(v)
                        else:
                            scan(v)

            scan(data)
            if not found_sources and isinstance(data, dict) and looks_like_source(data):
                found_sources.append(data)

            normalized = []
            for item in found_sources:
                title = (item.get('title') or item.get('name') or item.get('label') or
                         item.get('id') or item.get('filename') or f"{tool_name} Result")
                content = (item.get('content') or item.get('summary') or item.get('snippet') or
                           item.get('text') or item.get('description') or "")
                link = (item.get('url') or item.get('link') or item.get('href') or
                        item.get('source') or item.get('pdf_url') or "")
                raw_score = item.get('score') or item.get('relevance', 100 if content else 0)
                try:
                    score = float(raw_score) * 100 if 0 < float(raw_score) <= 1 \
                        else float(raw_score)
                except (TypeError, ValueError):
                    score = 0.0
                metadata = {k: v for k, v in item.items()
                            if k not in ['title', 'content', 'url', 'link', 'score']}
                normalized.append({
                    "title": str(title), "content": str(content), "source": str(link),
                    "relevance_score": score, "metadata": metadata, "tool": tool_name
                })
            return normalized

        # ── Document zone extraction helper ──────────────────────────────────
        def _extract_docs(zone_content):
            if not zone_content:
                return []
            return [
                {
                    "name":        m[0].strip(),
                    "content":     m[1].strip(),
                    "size":        len(m[1].strip()),
                    "token_count": self.lollmsClient.count_tokens(m[1].strip()),
                }
                for m in re.findall(
                    r"--- Document: (.+?) ---\n(.*?)\n--- End Document: \1 ---",
                    zone_content, re.DOTALL,
                )
            ]

        all_documents: List[Dict] = []
        for zone_content, zone_label in [
            (self.discussion_data_zone, "discussion"),
            (self.user_data_zone,       "user"),
            (None if personality.has_data else self.personality_data_zone, "personality"),
        ]:
            if zone_content:
                for d in _extract_docs(zone_content):
                    d["zone"] = zone_label
                    all_documents.append(d)

        # ====================================================================
        #  Tool registry
        # ====================================================================
        tool_registry:     Dict[str, Any] = {}
        tool_descriptions: List[str]      = []
        rag_registry:      Dict[str, Any] = {}
        rag_tool_specs:    Dict[str, Any] = {}

        composable_answer  = {"sections": [], "complete": False, "last_updated": None}
        scratchpad_state   = {"notes": {}, "history": [], "assumptions": {}, "corrections": []}
        collected_sources: List[Dict] = []
        queries_performed: List[Dict] = []
        self_corrections:  List[Dict] = []

        def get_current_answer():
            active    = [s for s in composable_answer["sections"] if s.get("status") == "active"]
            full_text = "\n\n".join(s["content"] for s in active)
            return {"success": True, "full_text": full_text, "sections": active,
                    "total_sections": len(active), "total_length": len(full_text),
                    "last_updated": composable_answer.get("last_updated")}

        def _make_wrapper(fn: Any, params_spec: List[Dict]) -> Any:
            def _wrapped(**kw):
                try:
                    call_args: Dict[str, Any] = {}
                    for p in params_spec:
                        pn = p.get("name")
                        if pn in kw:
                            call_args[pn] = kw[pn]
                        elif not p.get("optional", False):
                            return {"error": f"Missing required parameter: {pn}",
                                    "success": False}
                        elif "default" in p:
                            call_args[pn] = p["default"]
                    result = fn(**call_args)
                    if not isinstance(result, dict):
                        result = {"output": result}
                    if "success" not in result:
                        result["success"] = "error" not in result or not result.get("error")
                    return result
                except Exception as exc:
                    return {"error": str(exc), "success": False}
            return _wrapped

        def _sig(params: List[Dict]) -> str:
            parts = []
            for p in params:
                pn, pt = p.get("name", "arg"), p.get("type", "any")
                opt, dv = p.get("optional", False), p.get("default")
                if opt and dv is not None:
                    parts.append(f"{pn}: {pt} = {dv}")
                elif opt:
                    parts.append(f"{pn}: {pt} (optional)")
                else:
                    parts.append(f"{pn}: {pt}")
            return ", ".join(parts)

        def _register(name, fn, params, description, output=None):
            tool_registry[name] = _make_wrapper(fn, params)
            tool_descriptions.append(f"- {name}({_sig(params)}): {description}")
            if any(o.get("name") == "sources" for o in (output or [])):
                rag_registry[name]   = tool_registry[name]
                rag_tool_specs[name] = {
                    "default_top_k":   rag_top_k,
                    "default_min_sim": rag_min_similarity_percent,
                }

        # ── Layer 1: caller-supplied tools ───────────────────────────────────
        for tool_name, tool_spec in (tools or {}).items():
            if not isinstance(tool_spec, dict):
                continue
            fn = tool_spec.get("callable")
            if not callable(fn):
                continue
            _register(
                name        = tool_spec.get("name", tool_name),
                fn          = fn,
                params      = tool_spec.get("parameters", []),
                description = tool_spec.get("description", f"Execute {tool_name}"),
                output      = tool_spec.get("output", []),
            )

        # ── Layer 2: personality.tool_specs() ────────────────────────────────
        _pt_specs = {}
        try:
            _pt_specs = personality.tool_specs(
                client_binding=getattr(self.lollmsClient, "tools", None)
            )
        except Exception as _pte:
            _warning(callback, f"  Personality tool discovery failed: {_pte}")
            trace_exception(_pte)

        if _pt_specs:
            _pt_step_id = _step_start(callback,
                                      f"Loading {len(_pt_specs)} personality tool(s)...")
            for pt_name, pt_spec in _pt_specs.items():
                fn = pt_spec.get("callable")
                if not callable(fn):
                    continue
                _register(
                    name        = pt_name,
                    fn          = fn,
                    params      = pt_spec.get("parameters", []),
                    description = pt_spec.get("description", f"Execute {pt_name}"),
                    output      = pt_spec.get("output", []),
                )
            _step_end(callback, f"{len(_pt_specs)} personality tool(s) ready", _pt_step_id,
                      {"tool_count": len(_pt_specs)})

        # ── Layer 3: personality RAG tool ────────────────────────────────────
        if personality.has_data:
            def _personality_rag(query: str) -> Dict[str, Any]:
                result = personality.query_data(query)
                sources_filtered = sorted(
                    (s for s in result.get("sources", [])),
                    key=lambda x: x.get("score", 0), reverse=True
                )
                for s in sources_filtered:
                    score = float(s.get("score", 1.0))
                    if score > 1.0:
                        score = 1.0 if score > 100.0 else score / 100.0
                    s["score"] = score
                sources_filtered = [
                    s for s in sources_filtered[:rag_top_k]
                    if s.get("score", 0) >= rag_min_similarity_percent
                ]
                result["sources"] = sources_filtered
                result["count"]   = len(sources_filtered)
                return result

            _register(
                name        = "search_personality_knowledge",
                fn          = _personality_rag,
                params      = [{"name": "query", "type": "str",
                                "description": "Search query"}],
                description = "Search the personality's knowledge base",
                output      = [{"name": "sources", "type": "list"}],
            )

        # ── Personality system prompt ─────────────────────────────────────────
        if personality.system_prompt:
            veracity = (
                "\n=== VERACITY & ATTRIBUTION REQUIREMENTS ===\n"
                "Cite retrieved sources as [1],[2]... "
                "Use 'From my understanding...' for general knowledge.\n"
                "Never fabricate facts. Say 'I don't know' when uncertain.\n"
                "=== END ===\n"
            )
            object.__setattr__(
                self, "_system_prompt",
                personality.system_prompt + veracity + extra_instructions,
            )

        # ── Pre-flight RAG ───────────────────────────────────────────────────
        if preflight_rag_enabled and personality.has_data:
            preflight_id = _step_start(callback, "Pre-flight knowledge retrieval...")
            ctx = self.export("markdown", suppress_system_prompt=True)
            try:
                query_json = self.lollmsClient.generate_structured_content(
                    prompt=ctx[-2000:] + "\nGenerate a concise search query (JSON).",
                    schema={"query": "Your concise search query string"},
                    system_prompt="Output only JSON.",
                    temperature=0.1,
                )
                if query_json and "query" in query_json:
                    rag_result = personality.query_data(query_json["query"])
                    if rag_result.get("success"):
                        fmt = ""
                        for idx, chunk in enumerate(rag_result.get("sources", [])):
                            src  = chunk.get("source", "")
                            meta = chunk.get("metadata", {})
                            title = (
                                chunk.get("title")
                                or meta.get("title")
                                or meta.get("filename")
                                or meta.get("name")
                                or (src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if src else "")
                                or f"Source {idx + 1}"
                            )
                            fmt += (
                                f"[Source {idx+1}] ({src}, "
                                f"{chunk.get('score', 0):.2f})\n"
                                f"{chunk.get('content', '')}\n\n"
                            )
                            collected_sources.append({
                                "title":           title,
                                "content":         chunk.get("content", ""),
                                "source":          src,
                                "query":           query_json["query"],
                                "relevance_score": chunk.get("score", 0),
                                "index":           idx + 1,
                                "phase":           "preflight",
                                "metadata":        meta,
                            })
                        if fmt:
                            self.scratchpad = (
                                fmt.strip() + "\n\nIMPORTANT: Cite sources as [1],[2],..."
                            )
                        if collected_sources:
                            _cb(callback, collected_sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
            except Exception as e:
                trace_exception(e)
            _step_end(callback, "Pre-flight retrieval complete", preflight_id,
                      {"source_count": len(collected_sources)})

        # ====================================================================
        #  FAST PATH — no external tools registered
        # ====================================================================
        _has_external_tools = bool(tool_registry)

        if not _has_external_tools:
            ss = _StreamState(
                discussion            = self,
                callback              = callback,
                ai_message            = None,
                enable_notes          = enable_notes,
                enable_skills         = enable_skills,
                enable_inline_widgets = enable_inline_widgets,
                enable_forms          = enable_forms,
                auto_activate_artefacts = auto_activate_artefacts,
            )

            ai_message = self.add_message(
                sender=personality.name,
                sender_type="assistant",
                content="",
                parent_id=user_msg.id,
                model_name=self.lollmsClient.llm.model_name,
                binding_name=self.lollmsClient.llm.binding_name,
                metadata={"mode": "direct"},
            )
            ss.ai_message = ai_message

            def _fast_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    return ss.feed(chunk)
                return True

            # _stream_final_answer now handles artefact image merging internally
            raw_text = self._stream_final_answer(
                _fast_relay, images,
                branch_tip_id or self.active_branch_id,
                final_answer_temperature, **kwargs,
            )

            ss.flush_remaining_buffer()

            if raw_text and not ai_message.content:
                for ch in raw_text:
                    ss.feed(ch)
                ss.flush_remaining_buffer()

            raw_text = ai_message.content

            if remove_thinking_blocks:
                raw_text = self.lollmsClient.remove_thinking_blocks(raw_text)

            branch_for_handles = self.get_branch(ai_message.id)
            raw_after_handles, handle_arts = _apply_handles(
                raw_text, branch_for_handles, self.artefacts
            )

            ai_message.content = raw_after_handles

            cleaned, affected_pp = self._post_process_llm_response(
                raw_after_handles, ai_message, _eff_img_gen, _eff_img_edit,
                auto_activate_artefacts,
                enable_inline_widgets=enable_inline_widgets,
                enable_notes=enable_notes,
                enable_skills=enable_skills,
                enable_forms=enable_forms,
                enable_silent_artefact_explanation=enable_silent_artefact_explanation,
            )
            affected = handle_arts + ss.affected_artefacts + affected_pp
            if cleaned != raw_after_handles:
                ai_message.content = cleaned
            if affected and callback:
                _cb(callback, json.dumps([a.get("title") for a in affected]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED, {"artefacts": affected})
            if self._is_db_backed and self.autosave:
                self.commit()
            self.scratchpad = ""
            object.__setattr__(self, '_active_callback', None)
            return {
                "user_message":     user_msg,
                "ai_message":       ai_message,
                "sources":          collected_sources,
                "scratchpad":       None,
                "self_corrections": None,
                "artefacts":        affected,
            }

        # ====================================================================
        #  Built-in tools (only when external tools exist)
        # ====================================================================

        if enable_show_tools:
            def _show_tools_impl():
                catalogue: List[Dict[str, Any]] = []
                for _tname, _tspec in (tools or {}).items():
                    if not isinstance(_tspec, dict):
                        continue
                    catalogue.append({
                        "name":        _tspec.get("name", _tname),
                        "description": _tspec.get("description", ""),
                        "parameters":  _tspec.get("parameters", []),
                        "output":      _tspec.get("output", []),
                        "source":      "user",
                    })
                for pt_name, pt_spec in _pt_specs.items():
                    catalogue.append({
                        "name":        pt_name,
                        "description": pt_spec.get("description", ""),
                        "parameters":  pt_spec.get("parameters", []),
                        "output":      pt_spec.get("output", []),
                        "source":      "personality",
                        "binding":     pt_spec.get("_binding", ""),
                    })
                _cb(callback, json.dumps(catalogue, indent=2),
                    MSG_TYPE.MSG_TYPE_TOOLS_LIST, {"tools": catalogue})
                return {"success": True, "tool_count": len(catalogue), "tools": catalogue}

            tool_registry["show_tools"] = _show_tools_impl
            tool_descriptions.append("- show_tools(): Display the full list of available tools")

        if enable_extract_artefact:
            def _extract_artefact_text_impl(
                source_title: str, new_title: str,
                start_line_hint: str, end_line_hint: str,
                occurrence: int = 1,
                artefact_type: str = "document",
                language: str = "",
            ) -> Dict[str, Any]:
                source = self.artefacts.get(source_title)
                if source is None:
                    return {"success": False,
                            "error": f"Artifact '{source_title}' not found."}
                all_lines = source.get("content", "").splitlines()
                total = len(all_lines)
                if total == 0:
                    return {"success": False,
                            "error": f"Artifact '{source_title}' is empty."}
                sh = start_line_hint.strip().lower()
                eh = end_line_hint.strip().lower()
                if not sh:
                    return {"success": False, "error": "start_line_hint must not be empty."}
                if not eh:
                    return {"success": False, "error": "end_line_hint must not be empty."}
                start_idx, hit = None, 0
                for i, ln in enumerate(all_lines):
                    if ln.strip().lower().startswith(sh):
                        hit += 1
                        if hit == occurrence:
                            start_idx = i
                            break
                if start_idx is None:
                    return {"success": False,
                            "error": (f"start_line_hint {start_line_hint!r} not found "
                                      f"(occurrence {occurrence} of {hit} found)."),
                            "total_lines": total}
                end_idx = None
                for i in range(start_idx, total):
                    if all_lines[i].strip().lower().startswith(eh):
                        end_idx = i
                        break
                if end_idx is None:
                    return {"success": False,
                            "error": (f"end_line_hint {end_line_hint!r} not found "
                                      f"after line {start_idx + 1}."),
                            "start_line_no": start_idx + 1, "total_lines": total}
                from ._artefacts import ArtefactType as _AT
                resolved_type = (artefact_type if artefact_type in _AT.ALL
                                 else source.get("type", _AT.DOCUMENT))
                extracted = "\n".join(all_lines[start_idx:end_idx + 1])
                new_art   = self.artefacts.add(
                    title=new_title, artefact_type=resolved_type,
                    content=extracted,
                    language=language or source.get("language") or None,
                    active=True,
                )
                return {
                    "success": True, "source_title": source_title, "new_title": new_title,
                    "start_line_no": start_idx + 1, "end_line_no": end_idx + 1,
                    "total_lines": total, "lines_extracted": end_idx - start_idx + 1,
                    "artefact_id": new_art.get("id"),
                }

            tool_registry["extract_artifact_text"] = _extract_artefact_text_impl
            tool_descriptions.append(
                "- extract_artifact_text(source_title: str, new_title: str, "
                "start_line_hint: str, end_line_hint: str, occurrence: int = 1, "
                "artefact_type: str = 'document', language: str = ''): "
                "Extract a range from an artifact by line-prefix anchors"
            )

        if enable_repl_tools:
            try:
                from ._repl_tools import TextBuffer, register_repl_tools as _reg_repl
                _reg_repl(tool_registry, tool_descriptions, TextBuffer(), self.artefacts)
            except ImportError as _e:
                _warning(callback, f"REPL text tools unavailable: {_e}")

        if enable_final_answer:
            tool_registry["final_answer"] = lambda: {
                "status":  "final",
                "answer":  get_current_answer()["full_text"]
                           if composable_answer["sections"] else None,
                "success": True,
            }
            tool_descriptions.append("- final_answer(): Signal that the answer is ready")

        if enable_request_clarification:
            tool_registry["request_clarification"] = lambda question: {
                "status": "clarification", "question": question, "success": True,
            }
            tool_descriptions.append(
                "- request_clarification(question: str): Ask user for clarification"
            )

        object.__setattr__(
            self, "_system_prompt",
            _build_tool_system_prompt(
                self._system_prompt or "",
                tool_descriptions,
            )
        )

        # ── Context compression ──────────────────────────────────────────────
        if self.max_context_size is not None:
            _cr = self._compress_context(callback, self.max_context_size)
            if _cr["needed"] and _cr["artefact_pressure"]:
                def _deactivate_artefacts_impl(titles: List[str]) -> Dict[str, Any]:
                    deactivated, not_found = [], []
                    for t in titles:
                        if self.artefacts.get(t) is None:
                            not_found.append(t)
                        else:
                            self.artefacts.deactivate(t)
                            deactivated.append(t)
                    if deactivated:
                        if callback:
                            _cb(callback, json.dumps(deactivated),
                                MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                                {"artefacts": deactivated,
                                 "action": "deactivated_for_compression"})
                        tokens_after = self._compress_context(
                            callback, self.max_context_size
                        ).get("tokens_after", _cr["tokens_before"])
                    else:
                        tokens_after = _cr["tokens_before"]
                    return {
                        "success": bool(deactivated), "deactivated": deactivated,
                        "not_found": not_found,
                        "tokens_freed_estimate": _cr["tokens_before"] - tokens_after,
                    }

                tool_registry["deactivate_artefacts"] = _deactivate_artefacts_impl
                tool_descriptions.insert(0,
                    "- deactivate_artefacts(titles: list[str]): "
                    "CONTEXT PRESSURE -- deactivate unneeded artifacts first")
                object.__setattr__(
                    self, "_system_prompt",
                    _build_tool_system_prompt(
                        (self._system_prompt or ""),
                        tool_descriptions,
                    )
                )

        # ====================================================================
        #  Agentic loop
        # ====================================================================
        start_time            = datetime.now()
        is_agentic_turn       = False
        tool_calls_this_turn: List[Dict] = []
        all_events:           List[Dict] = []
        _accumulated_full     = ""
        _clean_text_so_far    = ""
        _round                = 0
        _temp_msg_ids:        List[str] = []
        _current_branch_tip   = branch_tip_id or self.active_branch_id

        _completed_tool_calls:    List[str] = []
        _created_artefact_titles: List[str] = []

        _MAX_IDENTICAL_REPEATS              = 2
        _identical_call_counts: Dict[str, int] = {}

        _round1_no_tool_call = False

        ai_message = self.add_message(
            sender=personality.name,
            sender_type="assistant",
            content="",
            parent_id=user_msg.id,
            model_name=self.lollmsClient.llm.model_name,
            binding_name=self.lollmsClient.llm.binding_name,
            metadata={"mode": "agentic", "events": []}
        )
        if self._is_db_backed:
            self.commit()
        _cb(callback, ai_message.id, MSG_TYPE.MSG_TYPE_NEW_MESSAGE,
            {"message_id": ai_message.id})

        while _round < max_reasoning_steps:
            _round += 1

            _saved_scratchpad = self.scratchpad
            state_lines = []

            if _completed_tool_calls or _created_artefact_titles:
                state_lines.append(
                    "=== AGENT STATE (already completed this turn — DO NOT repeat) ==="
                )
                if _completed_tool_calls:
                    state_lines.append("Tool calls already made:")
                    state_lines.extend(f"  ✓ {c}" for c in _completed_tool_calls)
                if _created_artefact_titles:
                    state_lines.append("Artifacts / notes already created:")
                    state_lines.extend(f"  ✓ {t}" for t in _created_artefact_titles)
                state_lines.append("=== END AGENT STATE ===")

            state_lines.append(_TOOL_CALL_REMINDER)

            if _round == 1 and not _completed_tool_calls:
                state_lines.append(
                    "AVAILABLE TOOLS (quick list): "
                    + ", ".join(tool_registry.keys())
                )

            self.scratchpad = "\n".join(state_lines) + (
                "\n\n" + (self.scratchpad or "") if self.scratchpad else ""
            )

            round_content_start = len(ai_message.content)

            ss = _StreamState(
                discussion            = self,
                callback              = callback,
                ai_message            = ai_message,
                enable_notes          = enable_notes,
                enable_skills         = enable_skills,
                enable_inline_widgets = enable_inline_widgets,
                enable_forms          = enable_forms,
                auto_activate_artefacts = auto_activate_artefacts,
            )

            def _inline_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    result = ss.feed(chunk)
                    if ss.tool_trigger:
                        return False
                    return result
                return True

            # Artefact image merging is handled inside _stream_final_answer
            self._stream_final_answer(
                _inline_relay, images, _current_branch_tip,
                final_answer_temperature, **kwargs,
            )

            ss.flush_remaining_buffer()
            self.scratchpad = _saved_scratchpad

            _so_far        = "".join(ss.stream_buf)
            _accumulated_full += _so_far

            _round_clean   = ai_message.content[round_content_start:]
            _clean_text_so_far += _round_clean

            _tool_trigger  = ss.tool_trigger
            _tool_json_str = ss.get_tool_call_json()

            if _round == 1 and not _tool_trigger and _has_external_tools:
                _output_clean = _round_clean.strip()
                _needs_tool   = bool(
                    len(_output_clean) < 50
                    or re.search(
                        r"i (cannot|can't|don't|am unable|don't have access|"
                        r"have no access|cannot access)",
                        _output_clean, re.IGNORECASE,
                    )
                )
                if _needs_tool:
                    _round1_no_tool_call = True
                    ASCIIColors.yellow(
                        "[chat] Round 1 produced no tool call — injecting correction")
                    _warning(callback,
                             "No tool call detected; reminding the model to use tools.")
                    _corr_call = self.add_message(
                        sender=personality.name, sender_type="assistant",
                        content=_so_far, parent_id=_current_branch_tip
                    )
                    _corr_res = self.add_message(
                        sender="system", sender_type="user",
                        content=_TOOL_CALL_CORRECTION,
                        parent_id=_corr_call.id
                    )
                    _temp_msg_ids.extend([_corr_call.id, _corr_res.id])
                    _current_branch_tip = _corr_res.id
                    _clean_text_so_far = ""
                    _accumulated_full  = ""
                    ai_message.content = ai_message.content[:round_content_start]
                    continue

            if not _tool_trigger:
                break

            is_agentic_turn = True
            try:
                _call_data   = json.loads(_tool_json_str or "{}")
                _tool_name   = _call_data.get("name", "")
                _tool_params = _call_data.get("parameters", {})
            except Exception as e:
                trace_exception(e)
                _warning(callback, f"Failed to parse tool call: {e}")
                break

            _params_summary = ", ".join(
                f"{k}={str(v)[:40]}" for k, v in _tool_params.items()
            )
            _call_signature = f"{_tool_name}({_params_summary})"
            _call_tag       = f"round {_round}: {_call_signature}"

            _identical_call_counts[_call_signature] = \
                _identical_call_counts.get(_call_signature, 0) + 1
            _sig_count = _identical_call_counts[_call_signature]

            if _sig_count > _MAX_IDENTICAL_REPEATS:
                _warning(callback,
                         f"[RUNAWAY] Identical call '{_call_signature}' seen "
                         f"{_sig_count} times — breaking loop.")
                break

            if _sig_count > 1:
                _warning(callback,
                         f"[DEDUP] Blocking duplicate tool call: {_call_signature}")
                _dup_result_str = (
                    f"DUPLICATE CALL BLOCKED: '{_call_signature}' was already executed "
                    f"this turn. Check the AGENT STATE in the scratchpad and either "
                    f"proceed to the next step or write your final answer."
                )
                _temp_call = self.add_message(
                    sender=personality.name, sender_type="assistant",
                    content=_so_far, parent_id=_current_branch_tip
                )
                _temp_res = self.add_message(
                    sender="system", sender_type="user",
                    content=(f"<tool_result name=\"{_tool_name}\">"
                             f"{_dup_result_str}</tool_result>"),
                    parent_id=_temp_call.id
                )
                _temp_msg_ids.extend([_temp_call.id, _temp_res.id])
                _current_branch_tip = _temp_res.id
                continue

            _current_offset = len(_clean_text_so_far)
            _call_id        = str(uuid.uuid4())
            _call_evt = {
                "type": "tool_call", "content": f"Calling {_tool_name}",
                "id": _call_id, "tool": _tool_name,
                "params": _tool_params, "offset": _current_offset,
            }
            _cb(callback, _call_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_CALL, _call_evt)
            all_events.append(_call_evt)

            _marker = f"\n<lollms_event id=\"{_call_id}\" />\n"
            _clean_text_so_far    += _marker
            ai_message.content     = _clean_text_so_far
            ai_message.metadata["events"] = list(all_events)

            _step_lbl = f"Running: {_tool_name.replace('_', ' ').title()}"
            _step_id  = _step_start(callback, _step_lbl,
                                    {"tool": _tool_name, "offset": _current_offset})
            all_events.append({"type": "step_start", "content": _step_lbl,
                               "id": _step_id, "offset": _current_offset})

            if _tool_name not in tool_registry:
                _warning(callback, f"Unknown tool: {_tool_name}")
                _result_str = f"Error: tool '{_tool_name}' not found"
                _result     = {"error": _result_str}
                _err_evt = {
                    "type": "tool_output", "content": _result_str,
                    "id": str(uuid.uuid4()), "tool": _tool_name,
                    "result": _result, "offset": _current_offset,
                }
                _cb(callback, _err_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, _err_evt)
                _step_end(callback, f"Unknown tool '{_tool_name}'",
                          _step_id, {"status": "failed"})
                all_events.extend([_err_evt,
                    {"type": "step_end",
                     "content": f"Unknown tool '{_tool_name}'",
                     "id": _step_id, "offset": _current_offset, "status": "failed"}])
            else:
                try:
                    _result = tool_registry[_tool_name](**_tool_params)

                    inferred_srcs = _infer_sources_from_json(_result, _tool_name)
                    res_label     = _tool_name.replace('_', ' ').title()
                    llm_block     = f"### [Source List: {res_label}]\n"
                    for s in inferred_srcs:
                        s["index"] = len(collected_sources) + 1
                        collected_sources.append(s)
                        llm_block += f"[[{s['index']}]] {s['title']}\n"
                        if s['content']:
                            llm_block += f"Content: {s['content']}\n"
                        if s['metadata']:
                            llm_block += f"Metadata: {json.dumps(s['metadata'])}\n"
                        llm_block += "---\n"
                    if not inferred_srcs:
                        llm_block += json.dumps(_result, indent=2)

                    self.scratchpad = (self.scratchpad or "") + (
                        f"\n--- Tool: {res_label} (round {_round}) ---\n"
                        f"{llm_block}\n"
                        f"--- End {res_label} ---\n"
                    )

                    _completed_tool_calls.append(_call_tag)
                    _created_title = (
                        _result.get("title") or _result.get("name") or
                        _result.get("note_title") or _result.get("artefact_title") or
                        _tool_params.get("title") or _tool_params.get("name") or
                        _tool_params.get("note_title")
                    )
                    if _created_title and str(_created_title) not in _created_artefact_titles:
                        _created_artefact_titles.append(str(_created_title))

                    _result_str = json.dumps(_result, indent=2)[:2000]
                    tool_calls_this_turn.append({
                        "name": _tool_name, "params": _tool_params, "result": _result,
                    })

                    _out_evt = {
                        "type": "tool_output", "content": _result_str,
                        "id": str(uuid.uuid4()), "tool": _tool_name,
                        "result": _result, "offset": _current_offset,
                    }
                    _cb(callback, _out_evt["content"], MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, _out_evt)
                    _step_end(callback, f"Done: {_tool_name}",
                              _step_id, {"status": "success"})
                    all_events.extend([
                        _out_evt,
                        {"type": "step_end", "content": f"Done: {_tool_name}",
                         "id": _step_id, "offset": _current_offset, "status": "success"},
                    ])

                    ai_message.metadata["events"]  = list(all_events)
                    ai_message.metadata["sources"] = list(collected_sources)
                    if self._is_db_backed:
                        self.commit()

                    _q       = _tool_params.get("query", _tool_params.get("prompt", ""))
                    raw_srcs = _result.get("sources", [])
                    if not raw_srcs:
                        cand = _result.get(
                            "results", _result.get(
                                "content",
                                _result if isinstance(_result, list) else []
                            )
                        )
                        if isinstance(cand, list):
                            raw_srcs = cand
                    if raw_srcs and isinstance(raw_srcs, list):
                        queries_performed.append({
                            "step": _round, "tool": _tool_name, "query": _q,
                            "result_count": len(raw_srcs),
                        })
                        for _doc in raw_srcs:
                            if not isinstance(_doc, dict):
                                continue
                            _doc_title   = (_doc.get("title") or _doc.get("name") or
                                            _doc.get("source") or f"{_tool_name} Result")
                            _doc_content = (_doc.get("content") or _doc.get("summary") or
                                            _doc.get("snippet") or "")
                            _doc_link    = (_doc.get("link") or _doc.get("url") or
                                            _doc.get("source") or "")
                            collected_sources.append({
                                "title":           _doc_title,
                                "content":         _doc_content,
                                "source":          _doc_link,
                                "query":           _q,
                                "relevance_score": _doc.get("score",
                                    _doc.get("relevance", 100 if _doc_content else 0)),
                                "index":           len(collected_sources) + 1,
                                "tool":            _tool_name,
                            })

                except Exception as e:
                    trace_exception(e)
                    _warning(callback, f"Tool error ({_tool_name}): {e}")
                    _step_end(callback, f"Error: {e}", _step_id, {"status": "error"})
                    all_events.append({
                        "type": "step_end", "content": f"Error: {e}",
                        "id": _step_id, "offset": _current_offset, "status": "error"
                    })
                    _result_str = f"Error: {e}"
                    _result     = {"error": _result_str}

            _temp_call = self.add_message(
                sender=personality.name, sender_type="assistant",
                content=_so_far, parent_id=_current_branch_tip
            )
            _temp_res = self.add_message(
                sender="system", sender_type="user",
                content=f"<tool_result name=\"{_tool_name}\">{_result_str}</tool_result>",
                parent_id=_temp_call.id
            )
            _temp_msg_ids.extend([_temp_call.id, _temp_res.id])
            _current_branch_tip = _temp_res.id

            ai_message.content = _clean_text_so_far

        # ====================================================================
        #  Forced final-answer pass
        # ====================================================================
        if is_agentic_turn and not _clean_text_so_far.strip():
            _final_id  = _step_start(callback, "Generating final answer...")

            ss_final = _StreamState(
                callback              = callback,
                ai_message            = ai_message,
                enable_notes          = enable_notes,
                enable_skills         = enable_skills,
                enable_inline_widgets = enable_inline_widgets,
                enable_forms          = enable_forms,
                discussion            = self,
            )
            ai_message.content = ""

            def _final_relay(chunk, msg_type=None, meta=None):
                if msg_type is not None and msg_type != MSG_TYPE.MSG_TYPE_CHUNK:
                    return ss_final.passthrough(chunk, msg_type, meta)
                if isinstance(chunk, str):
                    return ss_final.feed(chunk)
                return True

            _scratch_before_final = self.scratchpad
            self.scratchpad = (
                (_scratch_before_final or "")
                + "\n\n[SYSTEM INSTRUCTION] All tool calls are complete. "
                "Write your final answer to the user in plain text. "
                "Do NOT emit any more <tool_call> tags."
            )
            self._stream_final_answer(
                _final_relay, images, _current_branch_tip, final_answer_temperature, **kwargs
            )
            ss_final.flush_remaining_buffer()
            self.scratchpad = _scratch_before_final

            _accumulated_full += "".join(ss_final.stream_buf)
            _clean_text_so_far = ai_message.content
            _step_end(callback, "Final answer generated", _final_id)

        # ── Clean up temporary tool-history messages ─────────────────────────
        for mid in reversed(_temp_msg_ids):
            if hasattr(self, 'remove_message'):
                self.remove_message(mid)
            elif hasattr(self, 'delete_message'):
                self.delete_message(mid)
            else:
                self.db_manager.delete_message(mid)

        import re as _re
        _clean = _re.sub(
            r"<tool_call>.*?(?:</tool_call>|$)", "", _clean_text_so_far, flags=_re.DOTALL
        ).strip()
        if remove_thinking_blocks:
            _clean = self.lollmsClient.remove_thinking_blocks(_clean)

        end_time    = datetime.now()
        duration    = (end_time - start_time).total_seconds()
        token_count = self.lollmsClient.count_tokens(_clean)
        tok_per_sec = (token_count / duration) if duration > 0 else 0

        message_meta: Dict[str, Any] = {
            "mode": ("rlm_agentic" if rlm_enabled
                     else ("agentic" if is_agentic_turn else "direct")),
            "duration_seconds":  duration,
            "token_count":       token_count,
            "tokens_per_second": tok_per_sec,
        }
        if tool_calls_this_turn:
            message_meta["tool_calls"]           = tool_calls_this_turn
        if all_events:
            message_meta["events"]               = all_events
        if collected_sources:
            message_meta["sources"]              = collected_sources
        if queries_performed:
            message_meta["query_history"]        = queries_performed
        if is_agentic_turn:
            message_meta["scratchpad"]           = scratchpad_state
        if self_corrections:
            message_meta["self_corrections"]     = self_corrections
        if _pt_specs:
            message_meta["personality_tools_used"] = [
                n for n in _pt_specs
                if any(tc["name"] == n for tc in tool_calls_this_turn)
            ]
        if _round1_no_tool_call:
            message_meta["round1_correction_applied"] = True

        self.scratchpad = ""

        ai_message.content          = _clean
        ai_message.raw_content      = _accumulated_full
        ai_message.tokens           = token_count
        ai_message.generation_speed = tok_per_sec
        ai_message.metadata         = message_meta

        branch_for_final_handles = self.get_branch(ai_message.id)
        _clean_after_handles, handle_arts = _apply_handles(
            _clean, branch_for_final_handles, self.artefacts
        )
        if _clean_after_handles != _clean:
            ai_message.content = _clean_after_handles

        cleaned_content, affected_pp = self._post_process_llm_response(
            _clean_after_handles, ai_message, _eff_img_gen, _eff_img_edit,
            auto_activate_artefacts,
            enable_inline_widgets=enable_inline_widgets,
            enable_notes=enable_notes,
            enable_skills=enable_skills,
            enable_forms=enable_forms,
            enable_silent_artefact_explanation=enable_silent_artefact_explanation,
        )
        affected_artefacts = handle_arts + affected_pp
        if cleaned_content != _clean_after_handles:
            ai_message.content = cleaned_content

        if affected_artefacts:
            message_meta["artefacts_modified"] = [a.get("title") for a in affected_artefacts]
            ai_message.metadata = message_meta
            if callback:
                _cb(callback, json.dumps(message_meta["artefacts_modified"]),
                    MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED,
                    {"artefacts": affected_artefacts})

        if self._is_db_backed and self.autosave:
            self.commit()

        object.__setattr__(self, '_active_callback', None)

        return {
            "user_message":     user_msg,
            "ai_message":       ai_message,
            "sources":          collected_sources,
            "scratchpad":       scratchpad_state if is_agentic_turn else None,
            "self_corrections": self_corrections or None,
            "artefacts":        affected_artefacts,
        }


# ---------------------------------------------------------------------------
# Handle resolution — called after generation completes
# ---------------------------------------------------------------------------

def _apply_handles(
    text: str,
    branch_messages: List,
    artefacts_manager: Any,
) -> tuple:
    from ._artefacts import ArtefactType

    handle_pattern = re.compile(
        r'<use_handle\s+([^/]*)/>', re.DOTALL | re.IGNORECASE
    )
    affected: List[Dict] = []
    cleaned  = text

    def _parse_attrs_local(attr_str: str) -> Dict[str, str]:
        return {m.group(1): m.group(2)
                for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str)}

    def _handle_match(match: re.Match) -> str:
        attrs = _parse_attrs_local(match.group(1))

        ref      = attrs.get("ref", "").strip()
        name     = attrs.get("name", "").strip()
        atype    = attrs.get("type",     ArtefactType.CODE)
        language = attrs.get("language", "").strip()

        if not ref or not name:
            ASCIIColors.warning(
                f"<use_handle> missing ref or name attribute: {match.group(0)}")
            return match.group(0)

        block = _resolve_handle(ref, branch_messages)
        if block is None:
            ASCIIColors.warning(
                f"<use_handle ref='{ref}'> — handle not found in branch; "
                "check msg_idx and block_idx.")
            return f"[handle {ref} not found]"

        content  = block["content"]
        eff_lang = language or block.get("language") or ""
        eff_type = atype if atype in ArtefactType.ALL else ArtefactType.CODE

        existing = artefacts_manager.get(name)
        if existing is None:
            art = artefacts_manager.add(
                title=name, artefact_type=eff_type,
                content=content, language=eff_lang or None, active=True,
            )
            ASCIIColors.success(
                f"[use_handle] Created artefact '{name}' from handle {ref}")
        else:
            art = artefacts_manager.update(
                title=name, new_content=content,
                language=eff_lang or None, bump_version=True, active=True,
            )
            ASCIIColors.success(
                f"[use_handle] Updated artefact '{name}' from handle {ref} "
                f"→ v{art.get('version','?')}")

        affected.append(art)
        return ""

    cleaned = handle_pattern.sub(_handle_match, cleaned)
    return cleaned.strip(), affected