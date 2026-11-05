"""
End-to-end test for the binding-level tool-calling lifecycle.

Demonstrates:
    • OpenAI Chat Completions tool schema definition.
    • Native binding-managed loop via ``tools=`` parameter.
    • Callback upgrade: return ``{"tool_results": [...]}`` to hand execution
      results back to the binding so it can re-invoke the model.
    • Backward compatibility: when the callback returns ``True`` the binding
      streams the ``<tool>`` XML and stops (legacy mode).

The bindings forward ``tools`` to the server verbatim and rely on the
server to emit native ``tool_calls`` (vLLM requires ``--tool-call-parser``,
llama.cpp requires the correct chat template, etc.). There is no client-side
DSML tolerance: if the server leaks tool-call markup into ``content``, fix
the server configuration. This example validates that behaviour end-to-end.

Run against any OpenAI-compatible endpoint (OpenAI, LoLLMS, vLLM, llama.cpp,
Groq, Ollama, …) by exporting the appropriate environment variables.

Usage
-----
    python examples/simple/simple_tool_calling_agent.py

Environment variables
---------------------
    LOLLMS_BINDING_NAME   e.g. "openai", "lollms", "vllm", "ollama"
    LOLLMS_HOST_ADDRESS   e.g. "http://localhost:9624" (neutral for OpenAI)
    LOLLMS_MODEL_NAME     e.g. "gpt-4o-mini" or a local GGUF name
    OPENAI_API_KEY        when the binding is "openai"
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from ascii_colors import ASCIIColors

# Imports are done lazily so the file remains importable even when the
# environment does not have a local LoLLMS checkout wired up.
from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE

from lollms_client.lollms_config_cli_env import get_client_from_env

# ═══════════════════════════════════════════════════════════════════════════
# 1. Define the tool schema (OpenAI Chat Completions format)
# ═══════════════════════════════════════════════════════════════════════════
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris, France",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit (default: celsius)",
                    },
                },
                "required": ["location"],
            },
            "strict": False,  # Broadest compatibility; set True for OpenAI SO
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL",
                    },
                },
                "required": ["ticker"],
            },
            "strict": False,
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Implement the local tools
# ═══════════════════════════════════════════════════════════════════════════
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Mock data source. Replace with a real weather API for production use."""
    # A real implementation would call an external service and return a JSON
    # string. Keeping the payload compact protects the context window.
    return json.dumps(
        {"location": location, "temperature": 22, "unit": unit, "condition": "clear"}
    )


def get_stock_price(ticker: str) -> str:
    """Mock data source. Replace with a real market data API for production."""
    # Simple deterministic mock so the example is reproducible.
    price_table = {"AAPL": 189.42, "MSFT": 412.10, "GOOGL": 152.87}
    price = price_table.get(ticker.upper(), 0.0)
    return json.dumps({"ticker": ticker.upper(), "price": price, "currency": "USD"})


# ═══════════════════════════════════════════════════════════════════════════
# 3. Build the tool registry (name → callable)
# ═══════════════════════════════════════════════════════════════════════════
TOOL_REGISTRY = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
}


def execute_tool_call(tool_name: str, parameters: Dict[str, Any]) -> str:
    """Execute a tool call safely, returning a JSON string or an error payload."""
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return TOOL_REGISTRY[tool_name](**parameters)
    except Exception as e:
        # Errors must NOT crash the loop — surface them to the model so it can
        # decide whether to retry, ask the user, or give up gracefully.
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ═══════════════════════════════════════════════════════════════════════════
# 4. Streaming callback — executes every requested tool and returns results
# ═══════════════════════════════════════════════════════════════════════════
def tool_executing_callback(text: str, msg_type: MSG_TYPE, meta: dict) -> Any:
    """
    Handles all events emitted by the binding during a tool-calling round.

    Contract:
      - MSG_TYPE_CHUNK / MSG_TYPE_THOUGHT_CHUNK: normal text/thought chunks.
      - When the server emits a native ``tool_call``, the binding translates
        it to ``<tool>{"name": ..., "parameters": ...}</tool>`` XML and
        dispatches it to this callback with
        ``meta={"tool_call_id": ..., "tool_name": ...}``. We parse it, execute
        locally, and return
        ``{"tool_results": [{"tool_call_id": ..., "content": ...}, ...]}``.

    The OpenAI Chat Completions protocol correlates tool results to tool
    calls via ``tool_call_id``. The binding always provides that id in
    ``meta`` when the tool call originated from a native ``tool_calls``
    entry, which is the only path — there is no client-side DSML fallback.
    """
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        # Detect a tool call block.
        stripped = text.strip()
        if stripped.startswith("<tool>") and stripped.endswith("</tool>"):
            try:
                payload = json.loads(stripped[len("<tool>"):-len("</tool>")])
            except json.JSONDecodeError:
                ASCIIColors.warning(f"[example] Malformed tool payload: {stripped!r}")
                return True

            tool_name = payload.get("name", "")
            parameters = payload.get("parameters") or {}
            tool_call_id = meta.get("tool_call_id")  # present for native calls

            ASCIIColors.magenta(
                f"\n[example] Model requested '{tool_name}' "
                f"(tool_call_id={tool_call_id}) with {parameters}"
            )

            result = execute_tool_call(tool_name, parameters)
            ASCIIColors.magenta(f"[example] Tool returned: {result}")

            if tool_call_id:
                # Structured contract: the binding will append the results
                # to the conversation and re-invoke the model.
                return {"tool_results": [{"tool_call_id": tool_call_id, "content": result}]}

            # Native tool_calls always carry an id from the server. If it is
            # missing, this indicates a misbehaving server; return True and
            # let the surrounding run fail visibly rather than silently
            # mangling the wire state.
            ASCIIColors.warning(
                "[example] Server emitted a tool call without an id; "
                "the binding cannot correlate tool_results. Check your "
                "server's --tool-call-parser configuration."
            )
            print(f"[tool output] {result}")
            return True

        # Normal content chunk — stream to stdout.
        print(text, end="", flush=True)
        return True

    if msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
        # Silently drop thinking chunks in this example; a UI could show them.
        return True

    if msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"[example] Exception from binding: {text}")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# 5. Wiring + smoke test
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    # Make sure you configure an openai binding ,then an open ai model profile and set it as a default before calling this 
    
    client: LollmsClient = get_client_from_env()

    # ── Test 1: Managed tool loop (multi-round) ─────────────────────────────
    ASCIIColors.yellow("\n=== Test 1: Managed tool loop (multi-round) ===")
    messages = [
        {"role": "system", "content": "You are a helpful weather and finance assistant."},
        {
            "role": "user",
            "content": (
                "What is the current weather in Paris (celsius) and the AAPL "
                "stock price? Then summarize both answers in one sentence."
            ),
        },
    ]

    try:
        answer = client.llm.generate_from_messages(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            streaming_callback=tool_executing_callback,
            n_predict=512,
            temperature=0.2,
        )
    except Exception as e:
        ASCIIColors.error(f"[example] Managed tool loop failed: {e}")
        return 1

    if isinstance(answer, dict) and answer.get("status") == "error":
        ASCIIColors.error(f"[example] Binding error: {answer.get('message')}")
        return 1

    print()  # newline after the streamed content
    ASCIIColors.green("\nFinal answer:")
    print(answer)

    if answer is None:
        ASCIIColors.error(
            "\n[example] Binding returned None (method not implemented).\n"
            "The active model profile's binding does not override "
            "generate_from_messages(). Use a binding that supports tool "
            "calling: openai, lollms, vllm, ollama, or llama_cpp_server."
        )
        return 1

    if not (isinstance(answer, str) and answer.strip()):
        ASCIIColors.error(
            "\n[example] Binding returned an EMPTY string. Likely causes:\n"
            "  • Reasoning-only model (e.g. deepseek-r1 via Ollama, "
            "deepseek-reasoner via OpenAI) that never emits visible content "
            "when think=False AND the reasoning rescue did not fire.\n"
            "  • The server rejected the request (tools unsupported, or "
            "sampling params incompatible) and the retry path also failed.\n"
            "  • vLLM requires --tool-call-parser; llama.cpp / Ollama "
            "require a tool-capable chat template.\n"
            "Enable think=True on the binding or check the binding logs for "
            "the underlying 4xx/5xx payload."
        )
        return 1

    # ── Test 2: Backward compatibility — callback returns True ──────────────
    # The binding should stream the tool XML to stdout and stop without a
    # second round. This validates the legacy boolean contract still works.
    ASCIIColors.yellow("\n=== Test 2: Legacy callback contract (returns True) ===")

    def legacy_callback(text: str, msg_type: MSG_TYPE, meta: dict) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(text, end="", flush=True)
        return True

    try:
        legacy_answer = client.llm.generate_from_messages(
            messages=[
                {"role": "user", "content": "What is the weather in Tokyo right now?"},
            ],
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            streaming_callback=legacy_callback,
            n_predict=256,
            temperature=0.2,
        )
    except Exception as e:
        ASCIIColors.error(f"[example] Legacy contract test failed: {e}")
        return 1

    if isinstance(legacy_answer, dict) and legacy_answer.get("status") == "error":
        ASCIIColors.error(f"[example] Binding error: {legacy_answer.get('message')}")
        return 1

    print()
    ASCIIColors.green("\nRaw output (including <tool> XML):")
    print(legacy_answer)

    if legacy_answer is None:
        ASCIIColors.error(
            "\n[example] Legacy mode returned None — the binding does not "
            "override generate_from_messages(). Switch the model profile to "
            "an OpenAI-compatible binding."
        )
        return 1

    if not (isinstance(legacy_answer, str) and legacy_answer.strip()):
        ASCIIColors.error(
            "\n[example] Legacy mode also returned an EMPTY answer. The "
            "reasoning rescue did not fire (no reasoning captured) and no "
            "content was generated. Check the binding's server logs — the "
            "server likely rejected the request or emitted no visible output."
        )
        return 1
    ASCIIColors.info(
        "\nNote: in legacy mode the consumer MUST parse the <tool> XML and "
        "drive the second round itself."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())