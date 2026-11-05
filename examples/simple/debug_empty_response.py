"""
Diagnostic script: isolate why a lollms-binding generation returns an empty answer.

Runs three probes against the ACTIVE model profile (as configured by
``get_client_from_env()``):

    1. ``generate_from_messages`` with NO tools, NO streaming.
    2. ``generate_from_messages`` with NO tools, WITH streaming.
    3. ``generate_text`` with NO tools, NO streaming.

For each probe it monkey-patches ``binding.client.chat.completions.create``
to print the EXACT payload being sent to the server, so we can see whether
sampling fields (``reasoning_effort``, ``chat_template_kwargs``, ``tools``,
``max_tokens`` vs ``max_completion_tokens``, ``temperature`` …) that strict
OpenAI-compatible endpoints reject are slipping through.

Then it prints the returned value verbatim, including an explicit type tag
so a subtle ``None`` / ``""`` / error-dict distinction is impossible to miss.

Usage
-----
    python examples/simple/debug_empty_response.py
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

from ascii_colors import ASCIIColors, trace_exception

from lollms_client import LollmsClient, MSG_TYPE
from lollms_client.lollms_config_cli_env import get_client_from_env


def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a short, readable summary of an OpenAI chat.completions payload."""
    summary: Dict[str, Any] = {
        "model": payload.get("model"),
        "top_level_keys": sorted(k for k in payload.keys() if k not in ("messages", "extra_body")),
    }
    if "messages" in payload:
        summary["messages"] = [
            {
                "role": m.get("role"),
                "content_repr": repr(m.get("content"))[:120],
                "has_tool_calls": bool(m.get("tool_calls")),
            }
            for m in payload["messages"]
        ]
    if "extra_body" in payload:
        summary["extra_body"] = payload["extra_body"]
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty",
                "max_tokens", "max_completion_tokens", "reasoning_effort",
                "tools", "tool_choice", "stream"):
        if key in payload:
            val = payload[key]
            summary[key] = val if key != "tools" else f"<{len(val)} tools>"
    return summary


def _describe_result(label: str, value: Any) -> None:
    ASCIIColors.green(f"\n[{label}] Result type: {type(value).__name__}")
    if isinstance(value, dict):
        ASCIIColors.green(f"[{label}] Dict value: {value}")
    elif isinstance(value, str):
        ASCIIColors.green(f"[{label}] String length: {len(value)}")
        if value.strip():
            ASCIIColors.green(f"[{label}] Repr: {value!r}")
        else:
            ASCIIColors.error(f"[{label}] String is EMPTY (repr={value!r})")
    else:
        ASCIIColors.error(f"[{label}] Unexpected type: {value!r}")


def _install_payload_sniffer(binding: Any, label: str) -> None:
    """
    Monkey-patch the binding's openai client to log every ``create()`` call.

    IMPORTANT: installs are IDEMPOTENT. If a sniffer was already installed by a
    previous probe, we first unwrap it back to the original ``create`` and then
    install a fresh sniffer. Without this guard, each probe would chain onto
    the previous probe's sniffer, so by Probe 4 a single ``create()`` call
    would be logged 4× (once per still-installed layer) — producing a payload
    dump that looks like a runaway retry loop but is really 1 request.
    """
    # Restore the original create first (robust against repeated installs).
    original_create = getattr(
        binding.client.chat.completions,
        "_original_create_for_sniffer",
        binding.client.chat.completions.create,
    )

    def sniffer(*args, **kwargs):
        try:
            ASCIIColors.cyan(f"\n[{label}] → Payload sent to server:")
            ASCIIColors.cyan(json.dumps(_summarize_payload(kwargs), indent=2, default=str))
        except Exception:
            ASCIIColors.warning(f"[{label}] Could not summarize payload: {kwargs.keys()}")

        return original_create(*args, **kwargs)

    # Remember the original so a subsequent probe can unwrap cleanly.
    sniffer._original_create_for_sniffer = original_create  # type: ignore[attr-defined]
    binding.client.chat.completions.create = sniffer
    binding.client.chat.completions._original_create_for_sniffer = original_create  # type: ignore[attr-defined]


def main() -> int:
    client: LollmsClient = get_client_from_env()

    binding = client.llm
    if binding is None:
        ASCIIColors.error("No active LLM binding. Configure a model profile first.")
        return 1

    ASCIIColors.info(f"Binding name: {getattr(binding, 'binding_name', type(binding).__name__)}")
    ASCIIColors.info(f"Model name:   {getattr(binding, 'model_name', '<unknown>')}")
    ASCIIColors.info(f"Host/addr:    {getattr(binding, 'base_address', getattr(binding, 'host_address', '<unknown>'))}")

    simple_messages = [
        {"role": "user", "content": "Reply with exactly one word: hello"},
    ]

    # ── Probe 1: generate_from_messages (no tools, no stream) ──────────
    ASCIIColors.yellow("\n=== Probe 1: generate_from_messages (no tools, no stream) ===")
    _install_payload_sniffer(binding, "Probe1")
    try:
        result1 = binding.generate_from_messages(
            messages=simple_messages,
            n_predict=64,
            stream=False,
            temperature=0.2,
        )
    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"[Probe1] Exception: {e}")
        return 1
    _describe_result("Probe1", result1)

    # ── Probe 2: generate_from_messages (no tools, streaming) ──────────
    ASCIIColors.yellow("\n=== Probe 2: generate_from_messages (no tools, streaming) ===")

    def cb(text: str, msg_type: MSG_TYPE) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(text, end="", flush=True)
        return True

    _install_payload_sniffer(binding, "Probe2")
    try:
        result2 = binding.generate_from_messages(
            messages=simple_messages,
            n_predict=64,
            stream=True,
            streaming_callback=cb,
            temperature=0.2,
        )
    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"[Probe2] Exception: {e}")
        return 1
    print()
    _describe_result("Probe2", result2)

    # ── Probe 3: generate_text (no tools, no stream) ───────────────────
    ASCIIColors.yellow("\n=== Probe 3: generate_text (no tools, no stream) ===")
    _install_payload_sniffer(binding, "Probe3")
    try:
        result3 = binding.generate_text(
            prompt="Reply with exactly one word: hello",
            n_predict=64,
            stream=False,
            temperature=0.2,
        )
    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"[Probe3] Exception: {e}")
        return 1
    _describe_result("Probe3", result3)

    # ── Probe 4: generate_from_messages WITH tools, streaming ──────────
    ASCIIColors.yellow("\n=== Probe 4: generate_from_messages (tools, streaming) ===")
    demo_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Return the current time.",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "strict": False,
            },
        }
    ]

    def tool_cb(text: str, msg_type: MSG_TYPE, meta: dict) -> Any:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(text, end="", flush=True)
        return True
    # ── NOTE ─────────────────────────────────────────────────────────────
    # The 3-argument callback signature (``text``, ``msg_type``, ``meta``) is
    # supported by the binding's ``emit_callback_safe`` shim inside
    # ``_StreamThinkingHandler.process_content`` / ``process_reasoning``,
    # so 2-arg and 3-arg callbacks both work transparently.

    _install_payload_sniffer(binding, "Probe4")
    try:
        result4 = binding.generate_from_messages(
            messages=[
                {"role": "user", "content": "What time is it? Use the get_time tool."}
            ],
            tools=demo_tools,
            tool_choice="auto",
            n_predict=256,
            stream=True,
            streaming_callback=tool_cb,
            temperature=0.2,
        )
    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"[Probe4] Exception: {e}")
        result4 = None
    print()
    _describe_result("Probe4", result4)

    # ── Verdict ────────────────────────────────────────────────────────
    # A tool-loop diagnostic dict (``{"status": "error", "message": ...}``) is
    # the CORRECT contract when the server has no `--tool-call-parser` — the
    # binding surfaces the actionable root cause instead of silently
    # returning "". Treat that dict as "structurally-OK, diagnostically
    # informative" so the verdict does not flag a working guard as a failure.
    results = {"Probe1": result1, "Probe2": result2, "Probe3": result3, "Probe4": result4}

    def _extract_server_error_text(v: Any) -> str:
        """Best-effort flatten of an error dict/exception into a searchable string."""
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            parts = [str(v.get("message", ""))]
            body = v.get("server_body")
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    parts.append(str(err.get("message", "")))
                parts.append(str(body.get("message", "")))
            return " ".join(p for p in parts if p)
        return str(v)

    def _probe_ok(v: Any) -> bool:
        # A diagnostic dict is a CORRECT binding response only for Probe 4
        # (tool round) — for the non-tool probes (1, 2, 3) it means the same
        # failure the diagnostic describes. We report it as "diagnostic ok"
        # (not a hard failure) but the verdict text distinguishes the cases.
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, dict) and v.get("status") == "error":
            return "diagnostic"
        return False

    status_map = {k: _probe_ok(v) for k, v in results.items()}
    ok = [k for k, s in status_map.items() if s is True]
    diagnostic = [k for k, s in status_map.items() if s == "diagnostic"]
    bad = [k for k, s in status_map.items() if s is False]

    ASCIIColors.yellow("\n=== Verdict ===")

    # ── Specific detection of the vLLM tool-calling misconfiguration ────
    # The vLLM server 400-rejects tool_choice="auto" when it was launched
    # without --enable-auto-tool-choice / --tool-call-parser. Detect that
    # exact signature and print the remediation command instead of a
    # generic "tools don't work" warning.
    combined_error_text = " ".join(
        _extract_server_error_text(v) for v in results.values()
    ).lower()
    vllm_tool_misconfig = (
        "requires --enable-auto-tool-choice" in combined_error_text
        or "--tool-call-parser" in combined_error_text
        or "empty response with tools enabled" in combined_error_text
    )

    if vllm_tool_misconfig:
        ASCIIColors.error(
            "\n❌ SERVER-SIDE MISCONFIGURATION DETECTED (vLLM):\n"
            "   The vLLM backend was launched WITHOUT tool-calling support.\n"
            "   Every tool request is rejected with:\n"
            "       \"auto\" tool choice requires --enable-auto-tool-choice\n"
            "       and --tool-call-parser to be set\n\n"
            "   Fix: relaunch vLLM with the tool-calling flags, e.g.:\n"
            "       python -m vllm.entrypoints.openai.api_server \\\n"
            "           --model <model-path> \\\n"
            "           --enable-auto-tool-choice \\\n"
            "           --tool-call-parser deepseek_v3\n"
            "   (use `hermes` for Qwen/Hermes, `mistral` for Mistral,\n"
            "    `llama3_json` for Llama-3, `internlm` for InternLM).\n\n"
            "   This is NOT a client bug — the client payload is valid."
        )
        return 4

    if not bad and not diagnostic:
        ASCIIColors.green(
            "✅ All probes returned non-empty text AND the tool round "
            "produced a real tool call or answer. The endpoint is fully "
            "tool-capable."
        )
        return 0
    if diagnostic and not bad and set(diagnostic) <= {"Probe4"}:
        # Probes 1/2/3 are good; Probe 4 returned the diagnostic dict (which
        # is the binding's honest report of "server didn't emit tool_calls").
        # This means the endpoint is tool-INCAPABLE, not that the binding is
        # broken. Exit code 3 (warning) still distinguishes this from success.
        ASCIIColors.warning(
            "⚠️  Text generation works (Probes 1/2/3 green) but tool calling "
            "does not. The binding correctly reports the failure via a "
            "diagnostic dict. Actionable fix: configure the server with a "
            "tool-call parser (vLLM: --enable-auto-tool-choice "
            "--tool-call-parser <family>; llama.cpp/Ollama: a tool-capable "
            "chat template)."
        )
        return 3
    if not ok and not diagnostic:
        ASCIIColors.error(
            "❌ Every probe returned empty/None/error. The bug is UPSTREAM "
            "(request payload / endpoint), not tool-loop-specific. Look at "
            "the payload dumps above for offending fields and check the "
            "server response body."
        )
        return 2
    ASCIIColors.warning(
        f"⚠️  Mixed results. OK: {ok}. Diagnostic: {diagnostic}. Bad: {bad}. "
        f"This usually means the streaming vs non-streaming paths behave "
        f"differently, OR the generate_text vs generate_from_messages paths "
        f"use a different payload shape."
    )
    return 3


if __name__ == "__main__":
    sys.exit(main())