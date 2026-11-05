"""
Shared tool-calling loop utilities for LLM bindings.

These helpers implement the OpenAI Chat Completions tool-calling lifecycle so
that individual provider bindings do not need to reimplement the DSML/native
tool-call wiring, callback-result normalization, and iteration-bounded
re-prompting that every OpenAI-compatible endpoint requires.

Design contract
---------------
A streaming callback may return:

    True / None                → continue streaming (backward compat)
    False                      → abort the stream immediately
    {"tool_results": [         → binding appends the results to the message
        {"tool_call_id": "…",      history and re-invokes the model for the
         "content": "…"},          next round (bounded by MAX_TOOL_ROUNDS).
        …                      ]}

Each ``tool_results`` entry MUST carry a ``tool_call_id`` matching one of the
``tool_calls`` returned by the previous assistant turn, and a ``content``
string (the serialized tool output — usually ``json.dumps(result)``). Entries
without a ``tool_call_id`` are dropped because the wire protocol cannot
correlate them. A structured dict without ``tool_results`` (or with an empty
list) is treated as ``True`` so downstream parsing is unaffected.

The utilities are intentionally pure functions over plain Python types so
they can be exercised without a live model.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Default maximum number of consecutive tool-call rounds a single generate_*
# call may spawn. Prevents runaway models from looping on tool executions
# indefinitely. This is only the DEFAULT — callers can raise it per-call via
# ``tool_rounds_budget(n)`` (ambient, flows through the opaque binding stack)
# or per-invocation via an explicit ``max_tool_rounds`` kwarg. The chat mixin
# uses the ambient contextvar so its configured budget (e.g. 50) is honored by
# every binding without touching a single provider file.
MAX_TOOL_ROUNDS = 8

# Hard ceiling to prevent a misconfigured caller from exhausting memory or
# triggering a runaway loop. Well above any realistic agentic budget.
_MAX_TOOL_ROUNDS_CEILING = 1000

# Ambient per-call override. ``None`` means "use MAX_TOOL_ROUNDS". Set via
# ``tool_rounds_budget()`` by the caller (chat mixin) so the budget flows
# through the binding's opaque call stack without every provider binding
# needing to forward it explicitly.
_MAX_TOOL_ROUNDS_OVERRIDE: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "lollms_max_tool_rounds_override", default=None
)


@contextlib.contextmanager
def tool_rounds_budget(rounds: Optional[int]) -> Iterator[None]:
    """Ambiently set the max tool-call rounds for the duration of the block.

    The chat mixin wraps the binding's ``generate_from_messages`` call in this
    context manager so the binding's internal native-tool-calling loop honors
    the chat-level budget (e.g. 50) instead of the module default (8). The
    previous value is restored on exit, so nesting is safe.

    Passing ``None`` or a non-positive value clears the override (falls back
    to ``MAX_TOOL_ROUNDS``). Values above ``_MAX_TOOL_ROUNDS_CEILING`` are
    clamped to the ceiling.
    """
    previous = _MAX_TOOL_ROUNDS_OVERRIDE.get()
    if rounds is None:
        _MAX_TOOL_ROUNDS_OVERRIDE.set(None)
    else:
        try:
            normalized = int(rounds)
        except (TypeError, ValueError):
            normalized = None
        if normalized is None or normalized <= 0:
            _MAX_TOOL_ROUNDS_OVERRIDE.set(None)
        else:
            _MAX_TOOL_ROUNDS_OVERRIDE.set(min(normalized, _MAX_TOOL_ROUNDS_CEILING))
    try:
        yield
    finally:
        _MAX_TOOL_ROUNDS_OVERRIDE.set(previous)


def resolve_max_tool_rounds(explicit: Optional[int] = None) -> int:
    """Resolve the effective max tool-call rounds for the current call.

    Precedence (highest first):
      1. ``explicit`` argument (a binding that reads ``max_tool_rounds`` from
         ``**kwargs`` and forwards it here).
      2. Ambient ``tool_rounds_budget()`` contextvar (set by the chat mixin).
      3. Module default ``MAX_TOOL_ROUNDS`` (8).

    Always returns a positive int, clamped to ``_MAX_TOOL_ROUNDS_CEILING``.
    """
    if explicit is not None:
        try:
            normalized = int(explicit)
        except (TypeError, ValueError):
            normalized = None
        if normalized is not None and normalized > 0:
            return min(normalized, _MAX_TOOL_ROUNDS_CEILING)

    ambient = _MAX_TOOL_ROUNDS_OVERRIDE.get()
    if ambient is not None and ambient > 0:
        return ambient

    return MAX_TOOL_ROUNDS


def normalize_callback_result(result: Any) -> Tuple[bool, Optional[List[Dict[str, str]]]]:
    """
    Normalize a callback's return value into ``(continue_stream, tool_results)``.

    - ``None`` / ``True``  → ``(True,  None)``
    - ``False``            → ``(False, None)``
    - ``{"tool_results": [...]}`` → ``(True,  [normalized entries])``

    Normalized entries always carry a non-empty string ``tool_call_id`` and a
    string ``content``. Malformed entries are discarded silently. Non-dict
    results, dicts without ``tool_results``, and dicts whose ``tool_results``
    list normalizes to empty all degrade to ``(True, None)``.
    """
    if result is False:
        return False, None
    if not isinstance(result, dict):
        return True, None

    raw = result.get("tool_results")
    if not isinstance(raw, list) or not raw:
        return True, None

    normalized: List[Dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        tool_call_id = entry.get("tool_call_id")
        content = entry.get("content")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            continue
        if content is None:
            content = ""
        elif not isinstance(content, str):
            try:
                content = json.dumps(content)
            except (TypeError, ValueError):
                content = str(content)
        normalized.append({"tool_call_id": tool_call_id, "content": content})

    return True, normalized or None


def extract_native_tool_calls(message_obj: Any) -> List[Any]:
    """
    Return the native ``tool_calls`` list from a non-streaming message object
    or dict. Returns an empty list if absent or falsy.
    """
    if message_obj is None:
        return []
    if isinstance(message_obj, dict):
        tcs = message_obj.get("tool_calls")
    else:
        tcs = getattr(message_obj, "tool_calls", None)
    return list(tcs) if tcs else []


def accumulate_streamed_tool_deltas(accumulated: Dict[int, Dict[str, Any]], delta_tool_calls: Any) -> None:
    """
    Accumulate OpenAI-compatible streamed ``tool_call`` deltas by index.

    Handles both pydantic objects and plain dicts. Idempotent: safe to call
    with ``None``. The resulting ``accumulated`` mapping conforms to the
    OpenAI ``tool_calls`` wire shape (id/name/arguments) with fragments
    concatenated in arrival order.
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


def build_assistant_tool_calls_message(
    content: str,
    accumulated: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build the OpenAI assistant message that carries the accumulated tool calls.

    Returns ``None`` if no complete tool call (with a name) was accumulated.
    Missing per-call ids are synthesized as ``call_{index}`` so the following
    ``tool`` messages can correlate by ``tool_call_id`` even when the provider
    omitted ids on streaming deltas.
    """
    if not accumulated:
        return None

    wire_calls: List[Dict[str, Any]] = []
    for idx in sorted(accumulated.keys()):
        tc = accumulated[idx]
        name = (tc.get("name") or "").strip()
        if not name:
            continue
        tc_id = tc.get("id") or f"call_{idx}"
        raw_args = tc.get("arguments") or ""
        try:
            args_str = json.dumps(json.loads(raw_args)) if raw_args else "{}"
        except (json.JSONDecodeError, TypeError):
            args_str = json.dumps({"raw": raw_args})
        wire_calls.append(
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )

    if not wire_calls:
        return None

    return {
        "role": "assistant",
        "content": content if content else None,
        "tool_calls": wire_calls,
    }


def build_tool_result_messages(tool_results: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert normalized callback ``tool_results`` into OpenAI ``tool`` messages
    ready to append to the conversation.
    """
    return [
        {"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]}
        for r in tool_results
    ]


def build_tool_call_callback_payloads(
    accumulated: Dict[int, Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """
    Return one ``(tool_call_id, tool_name, tool_xml)`` tuple per accumulated
    streaming tool call, in index order.

    Used by the bindings to invoke ``streaming_callback`` **once per tool call**
    with ``meta={"tool_call_id": ..., "tool_name": ...}`` so a consumer can
    correlate the returned ``tool_results`` entry to the correct call — which
    is what the OpenAI Chat Completions protocol requires. Entries missing a
    ``name`` are skipped. Missing ids are synthesized as ``call_{index}`` to
    match :func:`build_assistant_tool_calls_message`.
    """
    if not accumulated:
        return []

    payloads: List[Tuple[str, str, str]] = []
    for idx in sorted(accumulated.keys()):
        tc = accumulated[idx]
        name = (tc.get("name") or "").strip()
        if not name:
            continue
        tc_id = tc.get("id") or f"call_{idx}"
        raw_args = tc.get("arguments") or ""
        try:
            params = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            params = {"raw": raw_args}
        xml = f"<tool>{json.dumps({'name': name, 'parameters': params})}</tool>"
        payloads.append((tc_id, name, xml))
    return payloads


def _callback_accepts_meta(callback: Callable) -> bool:
    """Return True iff *callback* can be invoked with ``(text, msg_type, meta)``.

    Detected once per callback object via ``inspect.signature`` and cached on
    the callable itself. Handles ``*args`` (accepts meta), positional-only,
    and keyword-only parameters correctly. Falls back to True (3-arg call) if
    the signature cannot be inspected (rare C callables).
    """
    cached = getattr(callback, "_lollms_accepts_meta", None)
    if cached is not None:
        return cached

    try:
        import inspect
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())
    except (ValueError, TypeError):
        # Cannot introspect: assume the modern 3-arg contract.
        try:
            setattr(callback, "_lollms_accepts_meta", True)
        except Exception:
            pass
        return True

    # A VAR_POSITIONAL (*args) accepts anything.
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        try:
            setattr(callback, "_lollms_accepts_meta", True)
        except Exception:
            pass
        return True

    # Count parameters that can receive a positional slot (pos only, pos-or-kw).
    positional_capable = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts = len(positional_capable) >= 3

    # If fewer than 3 positional slots exist, check whether a 'meta' or
    # 'metadata' keyword-only parameter exists (some callbacks declare
    # ``def cb(text, msg_type, *, meta=None)``).
    if not accepts:
        for p in params:
            if p.kind == inspect.Parameter.KEYWORD_ONLY and p.name in ("meta", "metadata"):
                accepts = True
                break

    try:
        setattr(callback, "_lollms_accepts_meta", accepts)
    except Exception:
        pass
    return accepts


def emit_callback_safe(
    callback: Optional[Callable],
    text: str,
    msg_type: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Invoke *callback* with ``(text, msg_type, meta)``, degrading to
    ``(text, msg_type)`` when the callback only accepts two positional args.

    The arity is detected ONCE via :func:`_callback_accepts_meta` and cached
    on the callback object, so a genuine ``TypeError`` raised INSIDE a
    well-formed 3-arg callback propagates verbatim to the caller (nothing is
    masked by retrying with fewer arguments). This is the compatibility shim
    for the callback contract upgrade: pre-existing user callbacks declared
    as ``def cb(text, msg_type)`` keep working untouched.
    """
    if callback is None:
        return True
    if _callback_accepts_meta(callback):
        return callback(text, msg_type, meta or {})
    return callback(text, msg_type)