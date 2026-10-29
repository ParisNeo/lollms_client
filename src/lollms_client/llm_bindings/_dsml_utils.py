"""
Shared DSML (DeepSeek Modeling Language) tool-call parsing utilities.

DeepSeek models (and some vLLM/SGLang/llama.cpp deployments) leak DSML markup
into the text stream instead of using the structured `tool_calls` field. These
helpers detect those blocks and translate them to Lollms `<tool>` XML so the
standard agentic loop can execute them.

Also provides `_format_tool_calls_to_lollms_xml` for translating native OpenAI
`tool_calls` objects into the same Lollms `<tool>` XML format.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


# ── DSML regexes ──────────────────────────────────────────────────────────
# DeepSeek uses the fullwidth pipe ｜ (U+FF5C) in its special tokens, but
# tokenization / chat templates sometimes strip it to ASCII |. Handle both.
_DSML_PIPE = r"[｜\|]"
_DSML_CALLS_OPEN_RE = re.compile(rf"<{_DSML_PIPE}DSML{_DSML_PIPE}\s*calls>", re.IGNORECASE)
_DSML_CALLS_CLOSE_RE = re.compile(rf"</{_DSML_PIPE}DSML{_DSML_PIPE}\s*calls>", re.IGNORECASE)
_DSML_CALLS_BLOCK_RE = re.compile(
    rf"<{_DSML_PIPE}DSML{_DSML_PIPE}\s*calls>.*?</{_DSML_PIPE}DSML{_DSML_PIPE}\s*calls>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE_RE = re.compile(
    rf"<{_DSML_PIPE}DSML{_DSML_PIPE}\s*invoke\s+name=[\"'](.*?)[\"']>(.*?)</{_DSML_PIPE}DSML{_DSML_PIPE}\s*invoke>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM_RE = re.compile(
    rf"<{_DSML_PIPE}DSML{_DSML_PIPE}\s*parameter\s+name=[\"'](.*?)[\"']>(.*?)</{_DSML_PIPE}DSML{_DSML_PIPE}\s*parameter>",
    re.DOTALL | re.IGNORECASE,
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
    """Parse a single <｜DSML｜ calls>...</｜DSML｜ calls> block into Lollms tool dicts."""
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
    Non-DSML text is preserved verbatim.
    """
    if not text:
        return text, ""
    if not _DSML_CALLS_OPEN_RE.search(text):
        return text, ""

    all_tools: List[Dict[str, Any]] = []
    for block in _DSML_CALLS_BLOCK_RE.findall(text):
        all_tools.extend(_parse_dsml_block(block))

    cleaned = _DSML_CALLS_BLOCK_RE.sub("", text).strip()
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
    full <｜DSML｜ calls>...</｜DSML｜ calls> block, and emits the translated Lollms
    <tool> XML to the downstream handler as MSG_TYPE_CHUNK. Non-DSML text is
    forwarded immediately.

    The `handler` must expose:
      - `.process_content(text) -> bool`  (returns False to abort)
      - `.callback` (may be None)
      - `.output` (str, mutated in place)
    """

    _OPENER_PREFIXES = (
        "<｜DSML｜",
        "<|DSML|",
        "<｜DSML｜",
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
        if m:
            pre = self.buffer[: m.start()]
            if pre and not self._flush_plain(pre):
                return False
            self.capturing = True
            self.capture_buf = self.buffer[m.start():]
            self.buffer = ""
            if _DSML_CALLS_CLOSE_RE.search(self.capture_buf):
                return self._finalize_capture()
            return True

        # No opener yet: check if the tail could be a partial opener
        max_prefix = max(len(p) for p in self._OPENER_PREFIXES)
        hold = 0
        for i in range(min(len(self.buffer), max_prefix), 0, -1):
            tail = self.buffer[-i:]
            if any(p.startswith(tail) for p in self._OPENER_PREFIXES):
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
            # Incomplete DSML block: emit as-is (best effort)
            if self.capture_buf:
                if not self._flush_plain(self.capture_buf):
                    return False
            self.capturing = False
            self.capture_buf = ""
        if self.buffer:
            if not self._flush_plain(self.buffer):
                return False
            self.buffer = ""
        return True