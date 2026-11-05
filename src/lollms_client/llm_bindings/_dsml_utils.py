"""
Shared OpenAI-native tool-call helpers.

The bindings forward the OpenAI Chat Completions ``tools`` array verbatim to
the server. The server (OpenAI, vLLM with ``--tool-call-parser``, llama.cpp,
etc.) is the sole authority on tool-call parsing: it emits structured
``tool_calls`` in the wire format defined by the OpenAI spec. The bindings
accumulate streaming deltas, translate them to Lollms ``<tool>`` XML for the
callback, and preserve the wire state (``tool_calls`` / ``tool_call_id``) so
that the multi-round managed loop can proceed.

There is intentionally **no** client-side DSML / Markdown / Jinja tolerance
here: if a server leaks tool-call markup into ``content`` instead of emitting
``tool_calls``, that is a server configuration problem. Configure the server
with the proper ``--tool-call-parser`` (vLLM) or the correct chat template
(llama.cpp / Ollama) so it emits native ``tool_calls``. This module will NOT
paper over such leaks.

Only the native-OpenAI-shaped helpers live here:

    _format_tool_calls_to_lollms_xml            → tool_calls  → Lollms XML
    _format_accumulated_tool_calls_to_lollms_xml → deltas     → Lollms XML
    _collect_streamed_tool_deltas               → accumulate streamed deltas
    _extract_native_tool_calls                  → read tool_calls from message
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _format_tool_calls_to_lollms_xml(tool_calls: List[Any]) -> str:
    """Convert an OpenAI-native ``tool_calls`` list to Lollms ``<tool>`` XML.

    Each entry is expected to expose ``.function.name`` and
    ``.function.arguments`` (a JSON string), which is the shape produced by
    both the OpenAI SDK and any OpenAI-compatible server. Dict-shaped entries
    are also accepted. Malformed entries are skipped silently so a single bad
    call cannot poison the whole batch.
    """
    if not tool_calls:
        return ""

    xml_parts: List[str] = []
    for tc in tool_calls:
        try:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or ""
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") or ""
                raw_args = getattr(fn, "arguments", "") or ""
            if not name:
                continue
            try:
                params = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                params = {"raw": raw_args}
            xml_parts.append(
                f"<tool>{json.dumps({'name': name, 'parameters': params})}</tool>"
            )
        except Exception:
            continue

    return "\n".join(xml_parts)


def _collect_streamed_tool_deltas(accumulated: Dict[int, Dict], delta_tool_calls: Any) -> None:
    """Accumulate OpenAI-compatible streamed ``tool_call`` deltas by index.

    Handles both pydantic objects and plain dicts. Idempotent: safe to call
    with ``None``. The resulting ``accumulated`` mapping conforms to the
    OpenAI ``tool_calls`` wire shape (``id`` / ``name`` / ``arguments``) with
    argument fragments concatenated in arrival order.
    """
    if not delta_tool_calls:
        return
    for tc_delta in delta_tool_calls:
        if isinstance(tc_delta, dict):
            idx = tc_delta.get("index")
            tc_id = tc_delta.get("id")
            fn = tc_delta.get("function")
        else:
            idx = getattr(tc_delta, "index", None)
            tc_id = getattr(tc_delta, "id", None)
            fn = getattr(tc_delta, "function", None)

        if idx is None:
            idx = 0
        entry = accumulated.setdefault(idx, {"id": None, "name": "", "arguments": ""})

        if tc_id:
            entry["id"] = tc_id

        if fn is not None:
            if isinstance(fn, dict):
                fn_name = fn.get("name")
                fn_args = fn.get("arguments")
            else:
                fn_name = getattr(fn, "name", None)
                fn_args = getattr(fn, "arguments", None)
            if fn_name:
                entry["name"] = fn_name
            if fn_args:
                entry["arguments"] += fn_args


_NIM_FUNCTION_NAME_PLACEHOLDER = "23d4f03a-b8a6-4adb-a183-7daa083a09cc"


def _extract_native_tool_calls(message_obj: Any) -> List[Any]:
    """Read the native ``tool_calls`` list from a message object or dict."""
    if message_obj is None:
        return []
    if isinstance(message_obj, dict):
        tcs = message_obj.get("tool_calls")
    else:
        tcs = getattr(message_obj, "tool_calls", None)
    return list(tcs) if tcs else []