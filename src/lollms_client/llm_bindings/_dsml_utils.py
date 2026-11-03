"""
Shared DSML (DeepSeek Modeling Language) tool-call parsing utilities.

DeepSeek models (and some vLLM/SGLang/llama.cpp deployments) leak DSML markup
into the text stream instead of using the structured `tool_calls` field. These
helpers detect those blocks and translate them to Lollms `<tool>` XML so the
standard agentic loop can execute them.

Two wrapper shapes are recognised:

  1. Batched wrapper (documented DeepSeek form):
        <｜DSML｜ calls>
            <｜DSML｜ invoke name="foo">
                <｜DSML｜ parameter name="x">1</｜DSML｜ parameter>
            </｜DSML｜ invoke>
            <｜DSML｜ invoke name="bar">
                <｜DSML｜ parameter name="y">"z"</｜DSML｜ parameter>
            </｜DSML｜ invoke>
        </｜DSML｜ calls>

  2. Bare invoke (some deployments omit the wrapper):
        <｜DSML｜ invoke name="foo">
            <｜DSML｜ parameter name="x">1</｜DSML｜ parameter>
        </｜DSML｜ invoke>

Both map to one `<tool>{"name":...,"parameters":{...}}</tool>` per invoke.

Also provides `_format_tool_calls_to_lollms_xml` for translating native
OpenAI `tool_calls` objects into the same Lollms `<tool>` XML format.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


# ── DSML token tolerance ──────────────────────────────────────────────────
# DeepSeek's real special token uses the fullwidth pipe ｜ (U+FF5C). Streaming
# consumers, tokenizers or chat templates may normalize it to ASCII `|`, drop
# it entirely, or inject an underscore between the pipe and the tag name
# (`<｜DSML｜` vs `<|DSML|_calls>`). Accept every observed permutation:
#
#   <｜DSML｜ calls>       <|DSML|_calls>       <｜DSML｜_calls>
#   <｜DSML｜calls>    <｜DSML|_calls>        <｜DSML｜ invoke>
#   <｜DSML｜ invoke>   <｜DSML｜_invoke>
#
# The delimiter class below matches either pipe once (or not at all), and the
# `_?` after it tolerates the underscore-prefixed variant.
_DSML_OPEN_HEAD = r"<[｜\|]?DSML[｜\|]?_?"
_DSML_CLOSE_HEAD = r"</[｜\|]?DSML[｜\|]?_?"

# Wrapper: <...calls> ... </...calls>
_DSML_CALLS_OPEN_RE = re.compile(_DSML_OPEN_HEAD + r"\s*calls\s*>", re.IGNORECASE)
_DSML_CALLS_CLOSE_RE = re.compile(_DSML_CLOSE_HEAD + r"\s*calls\s*>", re.IGNORECASE)
_DSML_CALLS_BLOCK_RE = re.compile(
    _DSML_OPEN_HEAD + r"\s*calls\s*>.*?" + _DSML_CLOSE_HEAD + r"\s*calls\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Individual invoke: <...invoke name="x"> ... </...invoke>
_DSML_INVOKE_RE = re.compile(
    _DSML_OPEN_HEAD + r"\s*invoke\s+name\s*=\s*[\"'](.*?)[\"']\s*>(.*?)"
    + _DSML_CLOSE_HEAD + r"\s*invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Parameter: <...parameter name="x">value</...parameter>
_DSML_PARAM_RE = re.compile(
    _DSML_OPEN_HEAD + r"\s*parameter\s+name\s*=\s*[\"'](.*?)[\"']\s*>(.*?)"
    + _DSML_CLOSE_HEAD + r"\s*parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Bare-invoke closer — used by the stream filter to know when a single invoke
# (without a wrapping <...calls> block) is complete.
_DSML_INVOKE_CLOSE_RE = re.compile(_DSML_CLOSE_HEAD + r"\s*invoke\s*>", re.IGNORECASE)
_DSML_CLOSE_HEAD_PATTERN = _DSML_INVOKE_CLOSE_RE

# Stray-tag sweep: any leftover invoke/parameter/calls tags that escape the
# wrapper (e.g. when a stream cuts a batch across two recoveries, or the model
# double-emits). These never carry useful content on their own.
_DSML_STRAY_TAG_RE = re.compile(
    r"<[｜\|]?DSML[｜\|]?_?\s*(?:calls|invoke|parameter)\b[^>]*>"
    r"|</[｜\|]?DSML[｜\|]?_?\s*(?:calls|invoke|parameter)\b[^>]*>",
    re.IGNORECASE,
)


def _coerce_dsml_param_value(raw: str) -> Any:
    """Best-effort type coercion for a DSML parameter body."""
    val = raw.strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null":
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


def _parse_dsml_block(block: str) -> List[Dict[str, Any]]:
    """Parse DSML `<...invoke>` children (from a wrapper or bare) into tool dicts."""
    tools: List[Dict[str, Any]] = []
    for func_name, inner in _DSML_INVOKE_RE.findall(block):
        params: Dict[str, Any] = {}
        for p_name, p_val in _DSML_PARAM_RE.findall(inner):
            params[p_name.strip()] = _coerce_dsml_param_value(p_val)
        tools.append({"name": func_name.strip(), "parameters": params})
    return tools


def _dsml_tools_to_lollms_xml(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    return "\n".join(f"<tool>{json.dumps(t)}</tool>" for t in tools)


def _extract_dsml_from_text(text: str) -> Tuple[str, str]:
    """
    Scan `text` for DSML blocks, return (cleaned_text, lollms_tool_xml).

    Handles BOTH the batched `<...calls>...</...calls>` wrapper AND bare
    `<...invoke ...>...</...invoke>` emissions. Any residual DSML markup that
    survives the block extraction (e.g. a stray opener, closer or child tag the
    model double-emitted) is stripped via a final catch-all sweep.

    Non-DSML text is preserved verbatim.
    """
    if not text:
        return text, ""

    has_wrapper = bool(_DSML_CALLS_OPEN_RE.search(text))
    has_bare_invoke = bool(_DSML_INVOKE_RE.search(text))
    if not has_wrapper and not has_bare_invoke:
        return text, ""

    all_tools: List[Dict[str, Any]] = []

    if has_wrapper:
        for block in _DSML_CALLS_BLOCK_RE.findall(text):
            all_tools.extend(_parse_dsml_block(block))
        cleaned = _DSML_CALLS_BLOCK_RE.sub("", text)
    else:
        # Bare invokes: parse each, then strip them all.
        for func_name, inner in _DSML_INVOKE_RE.findall(text):
            params: Dict[str, Any] = {}
            for p_name, p_val in _DSML_PARAM_RE.findall(inner):
                params[p_name.strip()] = _coerce_dsml_param_value(p_val)
            all_tools.append({"name": func_name.strip(), "parameters": params})
        cleaned = _DSML_INVOKE_RE.sub("", text)

    # Catch-all: strip any leftover DSML markup (stray wrappers, orphan closers,
    # orphaned parameter tags that survived because the invoke closed early).
    cleaned = _DSML_STRAY_TAG_RE.sub("", cleaned).strip()

    return cleaned, _dsml_tools_to_lollms_xml(all_tools)


def _format_tool_calls_to_lollms_xml(tool_calls: List[Any]) -> str:
    """
    Converts OpenAI-style tool_calls list to Lollms <tool> XML format.

    Each entry is expected to expose `.function.name` and `.function.arguments`
    (a JSON string). Malformed entries are skipped silently.
    """
    if not tool_calls:
        return ""

    xml_parts: List[str] = []
    for tc in tool_calls:
        try:
            name = tc.function.name
            args_str = tc.function.arguments
            try:
                params = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                params = {"raw": args_str}

            tool_obj = {"name": name, "parameters": params}
            xml_parts.append(f"<tool>{json.dumps(tool_obj)}</tool>")
        except Exception:
            continue

    return "\n".join(xml_parts)


class _DSMLStreamFilter:
    """
    Streaming-safe DSML interceptor.

    Holds back text when a potential DSML opener is being formed, captures the
    full DSML block (wrapper OR bare invoke), and emits the translated Lollms
    `<tool>` XML to the downstream handler as MSG_TYPE_CHUNK. Non-DSML text is
    forwarded immediately.

    The `handler` must expose:
      - `.process_content(text) -> bool`  (returns False to abort)
      - `.callback` (may be None)
      - `.output` (str, mutated in place)
    """

    # Literal prefixes for opener AND those that could be the start of an
    # opener variant (both pipe styles, with/without trailing underscore).
    _OPENER_PREFIXES = (
        "<｜DSML｜",
        "<|DSML|",
        "<|DSML|_",
        "<｜DSML｜",
        "<｜DSML|",
        "<|DSML｜",
        "<｜DSML｜_",
        "<｜DSML|_",
        "<|DSML｜_",
        "<｜DSML｜",
    )

    # Literal prefixes of a closing tag we must hold to avoid a leaked
    # `</｜DSML｜` fragment when the `>` arrives in a later chunk. Note that this
    # is NOT covered by the block-capture machinery (which only engages when
    # an OPENER is found), so we must guard it explicitly here.
    _CLOSER_PREFIXES = (
        "</｜DSML｜",
        "</|DSML|",
        "</|DSML|_",
        "</｜DSML｜",
        "</｜DSML|",
        "</|DSML｜",
        "</｜DSML｜_",
        "</｜DSML|_",
        "</|DSML｜_",
        "</｜DSML｜",
    )

    def __init__(self, handler: Any):
        self.handler = handler
        self.buffer = ""
        self.capturing = False
        self.capture_buf = ""

    def _flush_plain(self, text: str) -> bool:
        if not text:
            return True
        return self.handler.process_content(text)

    def _emit_tool_xml(self, tool_xml: str) -> bool:
        if not tool_xml:
            return True
        # Import lazily to avoid a hard dependency on lollms_types at module load.
        from lollms_client.lollms_types import MSG_TYPE
        if self.handler.callback:
            if self.handler.callback(tool_xml, MSG_TYPE.MSG_TYPE_CHUNK) is False:
                return False
        self.handler.output += tool_xml
        return True

    def _finalize_capture(self) -> bool:
        cleaned, tool_xml = _extract_dsml_from_text(self.capture_buf)
        self.capturing = False
        self.capture_buf = ""
        if cleaned and not self._flush_plain(cleaned):
            return False
        return self._emit_tool_xml(tool_xml)

    def _is_partial_prefix(self, tail: str, prefixes: Tuple[str, ...]) -> bool:
        return any(p.startswith(tail) for p in prefixes)

    def feed(self, chunk: str) -> bool:
        if not chunk:
            return True

        if self.capturing:
            self.capture_buf += chunk
            if _DSML_CALLS_CLOSE_RE.search(self.capture_buf):
                return self._finalize_capture()
            return True

        # Not capturing: check for opener in buffer+chunk
        self.buffer += chunk
        m = _DSML_CALLS_OPEN_RE.search(self.buffer)
        if not m:
            # Also check for bare-invoke opener (no wrapper).
            m = _DSML_INVOKE_RE.search(self.buffer)
        if m:
            pre = self.buffer[: m.start()]
            if pre and not self._flush_plain(pre):
                return False
            self.capturing = True
            self.capture_buf = self.buffer[m.start():]
            self.buffer = ""
            # If the complete block already landed in this chunk, finalize now.
            if _DSML_CALLS_CLOSE_RE.search(self.capture_buf):
                return self._finalize_capture()
            # Bare-invoke path: an invoke is complete when its own closer appears.
            if _DSML_INVOKE_CLOSE_RE.search(self.capture_buf):
                return self._finalize_capture()
            return True

        # No opener yet: hold any tail that could be a partial OPENER or CLOSER.
        max_opener = max(len(p) for p in self._OPENER_PREFIXES)
        max_closer = max(len(p) for p in self._CLOSER_PREFIXES)
        max_prefix = max(max_opener, max_closer)

        hold = 0
        for i in range(min(len(self.buffer), max_prefix), 0, -1):
            tail = self.buffer[-i:]
            if self._is_partial_prefix(tail, self._OPENER_PREFIXES) or self._is_partial_prefix(tail, self._CLOSER_PREFIXES):
                hold = i
                break

        if hold:
            emit = self.buffer[:-hold]
            self.buffer = self.buffer[-hold:]
        else:
            emit = self.buffer
            self.buffer = ""

        if emit:
            return self._flush_plain(emit)
        return True

    def flush(self) -> bool:
        if self.capturing:
            # Incomplete DSML block at end of stream: attempt best-effort
            # extraction. If it can't be parsed, do NOT leak the raw markup —
            # strip any residual DSML tags from the buffer before flushing.
            cleaned, tool_xml = _extract_dsml_from_text(self.capture_buf)
            if not cleaned:
                # Total strip (was probably a stray leak) — skip raw forward.
                pass
            else:
                # Text wrapped around the unclosed block survives — but the
                # block itself is guaranteed stripped by _extract.
                if cleaned and not self._flush_plain(cleaned):
                    return False
            if not self._emit_tool_xml(tool_xml):
                return False
            self.capturing = False
            self.capture_buf = ""

        if self.buffer:
            # Residual buffer is either a partial opener or partial closer that
            # never completed. Strip any DSML remnant; forward the rest.
            scrubbed = _DSML_STRAY_TAG_RE.sub("", self.buffer)
            self.buffer = ""
            if scrubbed and not self._flush_plain(scrubbed):
                return False
        return True