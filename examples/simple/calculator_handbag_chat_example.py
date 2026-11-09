"""
Interactive calculator handbag chat — end-to-end conversation with tool calling.

Compared to ``simple_handbag_tools_example.py`` (which is a non-interactive
pipeline smoke-test), THIS example exercises the tools *in a live conversation*:

  1. Builds a handbag folder on disk (SOUL.md + tools/calculator_tools/TOOLS.py).
  2. Loads it via ``LollmsPersonality.from_handbag(...)``.
  3. Drives a real conversation:
       * Mode A (default): ``LollmsDiscussion`` — DB-backed multi-turn chat
         that owns the tool-calling loop, streaming, and persistence.
       * Mode B: the personality's own ``chat()`` agentic loop (in-memory).
     Flip ``USE_DISCUSSION`` below to compare.
  4. The LLM decides when to call ``tool_calc`` or ``tool_greet`` — the tools
     are real, and their return values appear in the assistant's answer.

Requirements
------------
This example needs a LIVE ``lollms_client`` (LLM backend must be reachable).
It is acquired via ``lollms_client.lollms_config_cli_env.get_client_from_env()``,
the same helper used by ``simple_image_generation_agent.py`` and
``simple_chat_app.py``. Configure your backend first with:

    python -m lollms_client.lollms_config_cli_env --configure

or by setting the usual environment variables (see ``examples/.env.example``).

Run
---
    python examples/simple/calculator_handbag_chat_example.py

Type a math question ("what is (3+5)*7?") or a greeting ("hi, I'm Alice").
Type ``exit`` or ``quit`` to leave.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from ascii_colors import ASCIIColors

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

#: Set to False to use the personality's own chat() loop instead of a
#: DB-backed LollmsDiscussion. Both exercise the same tool pipeline; the
#: discussion variant additionally persists turns to disk.
USE_DISCUSSION: bool = True

#: Optional: cap how many rounds the LLM may use per user prompt.
MAX_ROUNDS: int = 12


# ────────────────────────────────────────────────────────────────────────────
# 1. Handbag contents written to disk (same calculator toolset as the
#    non-interactive example, so both can be diffed side-by-side).
# ────────────────────────────────────────────────────────────────────────────

_TOOLSET_SOURCE = '''\
"""
Calculator toolset — interactive example.

Conventions:
  * Canonical entry point: ``TOOLS.py`` inside the toolset folder.
  * Public functions named ``tool_<something>`` are auto-registered.
  * The docstring is the tool description shown to the LLM.
  * Type-annotated parameters become the tool schema automatically.
  * Return ``{"success": True, "output": ...}`` on success, or
    ``{"success": False, "error": ...}`` on failure.
"""

from __future__ import annotations

import ast
import operator


# Whitelisted operators — the tool must NEVER eval() untrusted input, so we
# rebuild the arithmetic tree from a strict whitelist via AST.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def tool_calc(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Args:
        expression (str): Arithmetic expression using + - * / // % ** and parentheses.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
        return {"success": True, "output": f"{expression} = {result}"}
    except Exception as exc:
        return {"success": False, "error": f"Invalid expression: {exc}"}


def tool_greet(name: str) -> dict:
    """Return a friendly greeting for the given name.

    Args:
        name (str): The name of the person to greet.
    """
    clean = (name or "").strip() or "stranger"
    return {"success": True, "output": f"Hello, {clean}! 👋"}
'''

_SOUL_MD = """\
---
name: Calculator Chat Agent
author: lollms_client
version: 1.0
category: tools_demo
description: Interactive agent that leverages a calculator and a greeting tool.
---

You are a concise, friendly calculator agent.
Whenever the user asks for an arithmetic computation, you MUST call the
`tool_calc` tool rather than guessing the answer. Always pass the raw
expression string (e.g. ``(3 + 5) * 7``) to the tool — do not try to simplify
it yourself.

When the user introduces themselves or asks for a greeting, call
`tool_greet` with their name.

After receiving the tool result, reply to the user with the final answer.
Do not mention "tool_calc" or "tool_greet" by name in your reply to the user.
"""


def build_handbag(root: Path) -> Path:
    """Creates a complete, valid handbag folder and returns its path."""
    handbag_dir = root / "calculator_chat_handbag"
    handbag_dir.mkdir(parents=True)

    (handbag_dir / "SOUL.md").write_text(_SOUL_MD, encoding="utf-8")

    toolset_dir = handbag_dir / "tools" / "calculator_tools"
    toolset_dir.mkdir(parents=True)
    (toolset_dir / "TOOLS.py").write_text(_TOOLSET_SOURCE, encoding="utf-8")

    # Give the conversation a persistent workspace (so LollmsDiscussion can
    # keep its SQLite DB there in Mode A).
    (handbag_dir / "workspace").mkdir(exist_ok=True)

    return handbag_dir


# ────────────────────────────────────────────────────────────────────────────
# 2. Streaming callback — renders chunks to the console
# ────────────────────────────────────────────────────────────────────────────

def _make_stream_cb():
    from lollms_client import MSG_TYPE

    def cb(chunk, msg_type=None, meta=None) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            return True
        # Surface tool start/end events so the user can SEE the calculator
        # being invoked — the whole point of this example.
        if msg_type == MSG_TYPE.MSG_TYPE_TOOL_START:
            tool = (meta or {}).get("tool_name", "?")
            params = (meta or {}).get("parameters", {})
            ASCIIColors.cyan(f"\n   🔧 Calling tool: {tool}({params})\n")
            return True
        if msg_type == MSG_TYPE.MSG_TYPE_TOOL_END:
            tool = (meta or {}).get("tool_name", "?")
            ok = (meta or {}).get("success", True)
            status = "✅" if ok else "❌"
            ASCIIColors.cyan(f"   {status} {tool} returned.\n")
            return True
        return True

    return cb


# ────────────────────────────────────────────────────────────────────────────
# 3. Mode A — LollmsDiscussion-backed conversation
# ────────────────────────────────────────────────────────────────────────────

def run_discussion_mode(personality, workspace_path: Path) -> None:
    """Interactive REPL backed by a real LollmsDiscussion."""
    from lollms_client.lollms_discussion import LollmsDiscussion

    ASCIIColors.info("\nUsing LollmsDiscussion (persisted, DB-backed).")
    discussion = LollmsDiscussion(personality.lollms_client, workspace_path=workspace_path)

    # Seed the discussion with the personality's system prompt.
    try:
        discussion._system_prompt = personality.system_prompt
    except Exception:
        pass

    # Tool specs: LollmsDiscussion consumes the flat active_tools dict format.
    tool_specs = personality.tool_specs()
    if not tool_specs:
        ASCIIColors.error(
            "The personality discovered NO tools. Cannot run the discussion "
            "example meaningfully. Check that the handbag built correctly."
        )
        return
    ASCIIColors.success(f"Loaded {len(tool_specs)} tool(s): {sorted(tool_specs.keys())}")

    cb = _make_stream_cb()

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", ":q"):
            break

        print("\nai> ", end="", flush=True)
        try:
            result = discussion.chat(
                user_input,
                tools=tool_specs,
                streaming_callback=cb,
                enforce_end_tag=True,
            )
        except Exception as exc:
            ASCIIColors.error(f"\n[chat error] {exc}")
            traceback.print_exc()
            continue

        print()  # newline after the streamed answer

        _report_tools(result)


# ────────────────────────────────────────────────────────────────────────────
# 4. Mode B — Personality's own agentic loop
# ────────────────────────────────────────────────────────────────────────────

def run_personality_mode(personality) -> None:
    """Interactive REPL backed by the personality's internal chat() loop."""
    ASCIIColors.info("\nUsing LollmsPersonality.chat() (in-memory history).")

    # Quick sanity — the personality should already have tools attached.
    specs = personality.tool_specs()
    if not specs:
        ASCIIColors.error(
            "The personality discovered NO tools. Cannot run the personality "
            "example meaningfully. Check the handbag build."
        )
        return
    ASCIIColors.success(f"Loaded {len(specs)} tool(s): {sorted(specs.keys())}")

    cb = _make_stream_cb()

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", ":q"):
            break

        print("\nai> ", end="", flush=True)
        try:
            result = personality.chat(
                user_input,
                streaming_callback=cb,
                max_nb_rounds=MAX_ROUNDS,
                enforce_end_tag=True,
            )
        except Exception as exc:
            ASCIIColors.error(f"\n[chat error] {exc}")
            traceback.print_exc()
            continue

        print()
        _report_tools(result)


# ────────────────────────────────────────────────────────────────────────────
# 5. Helpers
# ────────────────────────────────────────────────────────────────────────────

def _report_tools(result) -> None:
    """Summarises the tools invoked during the last turn (if any)."""
    if not isinstance(result, dict):
        return
    tool_calls = result.get("tool_calls") or []
    if not tool_calls:
        return
    parts = []
    for call in tool_calls:
        name = call.get("name", "?")
        params = call.get("parameters", {})
        parts.append(f"{name}({params})")
    ASCIIColors.magenta(f"   [tools used this turn: {'; '.join(parts)}]")


def _acquire_client():
    """Returns a live LollmsClient, or None (with a clear error message)."""
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        # create_tti/create_tts/etc. left off — we only need the LLM binding
        # for the calculator example. Tools come from the handbag.
        return get_client_from_env()
    except Exception as exc:
        ASCIIColors.error(
            "Failed to create a LollmsClient from the environment.\n"
            f"Details: {exc}\n\n"
            "This example needs a live LLM backend. Configure one with:\n"
            "    python -m lollms_client.lollms_config_cli_env --configure\n"
            "or copy examples/.env.example → .env and fill it in."
        )
        traceback.print_exc()
        return None


# ────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ASCIIColors.cyan("=" * 70)
    ASCIIColors.cyan("Calculator Handbag Chat — interactive tool-calling example")
    ASCIIColors.cyan("=" * 70)

    client = _acquire_client()
    if client is None:
        return 1

    tmp_root = Path(tempfile.mkdtemp(prefix="calc_handbag_chat_"))
    handbag_dir = None
    try:
        handbag_dir = build_handbag(tmp_root)
        ASCIIColors.info(f"\nHandbag built at: {handbag_dir}")
        ASCIIColors.info("  ├── SOUL.md")
        ASCIIColors.info("  └── tools/calculator_tools/TOOLS.py")

        # Load the personality from the handbag, wired to the live client.
        from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

        # Give the personality the handbag's workspace folder so any
        # artefact/scratchpad state stays contained in the temp dir.
        workspace = handbag_dir / "workspace"
        personality = LollmsPersonality.from_handbag(
            handbag_dir,
            lollms_client=client,
            workspace_path=workspace,
        )

        ASCIIColors.success(f"\nPersonality ready: '{personality.name}'")
        ASCIIColors.info("Type a math question or a greeting. 'exit' / 'quit' to stop.")
        ASCIIColors.info("Example prompts:")
        ASCIIColors.info("  • What is (12 + 8) * 3?")
        ASCIIColors.info("  • Hi, I'm Alice — nice to meet you!")
        ASCIIColors.info("  • Compute 2 ** 10 / 4")
        print()

        if USE_DISCUSSION:
            run_discussion_mode(personality, workspace)
        else:
            run_personality_mode(personality)

    except Exception as exc:
        ASCIIColors.error(f"\nUnexpected failure: {exc}")
        traceback.print_exc()
        return 1
    finally:
        # Clean up the temp handbag. Change this to `pass` if you want to
        # inspect the generated folder after the session ends.
        if tmp_root.exists():
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
                ASCIIColors.info(f"\nCleaned up temp handbag: {tmp_root}")
            except Exception:
                pass

    ASCIIColors.success("\nSession ended. Bye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())