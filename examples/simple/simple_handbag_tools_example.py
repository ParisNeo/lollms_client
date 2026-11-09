"""
Simple handbag tools example — end-to-end, zero external dependencies.

This example demonstrates the FULL documented handbag lifecycle in one file:

    1. Create a handbag folder on disk.
    2. Write a SOUL.md (personality metadata + system prompt).
    3. Add a toolset using the canonical layout: tools/<name>/TOOLS.py
    4. Load the handbag via LollmsPersonality.from_handbag() — default config.
    5. Verify the tool is actually DISCOVERED and actually EXECUTES.

Why this exists
---------------
The single most common user report against handbags is "the agent doesn't see
my tools." That symptom has many possible upstream causes (folder layout,
TOOL_LIBRARY_LOAD_FLAG gating, AST-vs-regex parse failure, gate resolution,
active_tools merge drop, ...). This script walks every stage in order and
prints a ✓/✗ per stage, so a failure tells you EXACTLY which layer broke —
instead of dumping an opaque traceback.

No LLM. No network. No API key. No persistency outside the temp dir.

Run
---
    python examples/simple/simple_handbag_tools_example.py

The temp handbag path is printed at the start so you can inspect it on disk
while the script runs, and it is removed at the end (unless it fails, in which
case it is preserved for debugging).
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

from ascii_colors import ASCIIColors


# ────────────────────────────────────────────────────────────────────────────
# 1. Handbag contents written to disk
# ────────────────────────────────────────────────────────────────────────────

# A real toolset (not a stub): users get a working calculator immediately and
# can copy this file as a template for their own toolsets.
_TOOLSET_SOURCE = '''\
"""
Calculator toolset — a minimal but WORKING example.

Conventions demonstrated here (all optional):
  * Canonical entry point is TOOLS.py inside the toolset folder.
  * Each public function must be named ``tool_<something>``.
  * The function's docstring is the tool's description shown to the LLM.
  * Type-annotated parameters become the tool schema automatically.
  * Return a dict with ``success``, ``output`` (and optionally ``error``).
"""

from __future__ import annotations

import ast
import operator


# Whitelisted binary operators. The tool must never eval() untrusted input,
# so we rebuild the arithmetic tree from a strict whitelist via AST.
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
name: Calculator Handbag
author: lollms_client
version: 1.0
category: tools_demo
description: A handbag demonstrating the canonical toolset layout.
---

You are a focused agent whose only job is to help with math and greetings.
Use the available tools whenever the user asks for an arithmetic computation
or a greeting. Always prefer a tool call over guessing an answer.
"""


# ────────────────────────────────────────────────────────────────────────────
# 2. Handbag construction on disk
# ────────────────────────────────────────────────────────────────────────────

def build_handbag(root: Path) -> Path:
    """Creates a complete, valid handbag folder and returns its path."""
    handbag_dir = root / "calculator_handbag"
    handbag_dir.mkdir(parents=True)

    (handbag_dir / "SOUL.md").write_text(_SOUL_MD, encoding="utf-8")

    toolset_dir = handbag_dir / "tools" / "calculator_tools"
    toolset_dir.mkdir(parents=True)
    (toolset_dir / "TOOLS.py").write_text(_TOOLSET_SOURCE, encoding="utf-8")

    return handbag_dir


# ────────────────────────────────────────────────────────────────────────────
# 3. Stage-by-stage verification
# ────────────────────────────────────────────────────────────────────────────

_EXPECTED_TOOLS = {"tool_calc", "tool_greet"}


def _ok(stage: str, detail: str = "") -> None:
    ASCIIColors.success(f"  ✓ {stage}{(': ' + detail) if detail else ''}")


def _fail(stage: str, detail: str) -> None:
    ASCIIColors.error(f"  ✗ {stage}: {detail}")


def verify_pipeline(handbag_dir: Path) -> bool:
    """Runs every stage and returns True only if ALL stages pass."""
    from lollms_client.lollms_personality.handbag import Handbag
    from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

    all_ok = True

    # ── Stage 1 — Handbag scans the toolset folder ──
    ASCIIColors.info("\nStage 1 — Handbag._load_tools() finds the toolset")
    hb = Handbag(handbag_dir)
    if not hb.tool_files:
        _fail("Handbag scanning", f"tool_files is empty (expected 1 folder) in {handbag_dir / 'tools'}")
        return False
    if len(hb.tool_files) != 1 or hb.tool_files[0].name != "calculator_tools":
        _fail("Handbag scanning", f"unexpected tool_files: {hb.tool_files}")
        return False
    _ok("Handbag yields folder", str(hb.tool_files[0].name))

    # ── Stage 2 — from_handbag attaches a working tool binding ──
    ASCIIColors.info("\nStage 2 — LollmsPersonality.from_handbag() builds the personality")
    personality = LollmsPersonality.from_handbag(handbag_dir)
    binding = personality._tool_binding
    if binding is None:
        _fail("from_handbag", "personality._tool_binding is None")
        return False
    discovered = {t.get("name") for t in getattr(binding, "discovered_tools", [])}
    missing = _EXPECTED_TOOLS - discovered
    if missing:
        _fail("LCPBinding discovery", f"missing {sorted(missing)}; got {sorted(discovered)}")
        all_ok = False
    else:
        _ok("LCPBinding discovered tools", ", ".join(sorted(discovered)))

    # ── Stage 3 — The personality's active_tools contains the handbag tools ──
    ASCIIColors.info("\nStage 3 — _discover_tools() surfaces handbag tools in active_tools")
    active_tools = personality._discover_tools()
    missing_active = _EXPECTED_TOOLS - set(active_tools.keys())
    if missing_active:
        _fail(
            "active_tools merge",
            f"missing {sorted(missing_active)}; active tool names: {sorted(active_tools.keys())}",
        )
        all_ok = False
    else:
        _ok("active_tools contains handbag tools", ", ".join(sorted(_EXPECTED_TOOLS)))

    # ── Stage 4 — The system prompt actually lists the tools ──
    ASCIIColors.info("\nStage 4 — _build_system_prompt() renders the tools for the LLM")
    if missing_active:
        _fail("system prompt", "skipped (active_tools incomplete)")
        return False
    prompt = personality._build_system_prompt(active_tools)
    prompt_missing = [t for t in _EXPECTED_TOOLS if t not in prompt]
    if prompt_missing:
        _fail("system prompt", f"tool names missing from prompt: {prompt_missing}")
        all_ok = False
    else:
        _ok("system prompt contains all handbag tool names")

    # ── Stage 5 — The tool's callable actually executes ──
    ASCIIColors.info("\nStage 5 — Handbag tools are EXECUTABLE (not just discoverable)")
    try:
        calc_spec = active_tools["tool_calc"]
        calc_result = calc_spec["callable"](expression="(2 + 3) * 4")
        # Callable may return {'output':..., 'success':...} (LCP normalized)
        inner = calc_result.get("output", calc_result) if isinstance(calc_result, dict) else calc_result
        if isinstance(inner, dict):
            rendered = inner.get("output") or inner.get("error") or ""
        else:
            rendered = str(inner)
        if "20" not in rendered:
            _fail("tool_calc execution", f"expected '20' in response; got: {rendered!r}")
            all_ok = False
        else:
            _ok("tool_calc executes", "(2 + 3) * 4 = 20")

        greet_spec = active_tools["tool_greet"]
        greet_result = greet_spec["callable"](name="World")
        inner_g = greet_result.get("output", greet_result) if isinstance(greet_result, dict) else greet_result
        rendered_g = inner_g.get("output", "") if isinstance(inner_g, dict) else str(inner_g)
        if "World" not in rendered_g:
            _fail("tool_greet execution", f"expected 'World' in response; got: {rendered_g!r}")
            all_ok = False
        else:
            _ok("tool_greet executes", "Hello, World! 👋")
    except Exception as exc:
        _fail("tool execution", str(exc))
        traceback.print_exc()
        all_ok = False

    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# 4. Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ASCIIColors.cyan("=" * 70)
    ASCIIColors.cyan("Handbag Tools — end-to-end example")
    ASCIIColors.cyan("=" * 70)

    tmp_root = Path(tempfile.mkdtemp(prefix="handbag_example_"))
    ASCIIColors.info(f"\nTemp handbag root: {tmp_root}")

    ok = False
    try:
        handbag_dir = build_handbag(tmp_root)
        ASCIIColors.info(f"Built handbag at: {handbag_dir}")
        ASCIIColors.info("  ├── SOUL.md")
        ASCIIColors.info("  └── tools/calculator_tools/TOOLS.py")

        ok = verify_pipeline(handbag_dir)
    except Exception as exc:
        ASCIIColors.error(f"\nUnexpected failure: {exc}")
        traceback.print_exc()
    finally:
        if ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            ASCIIColors.info(f"\nCleaned up temp handbag: {tmp_root}")
        else:
            ASCIIColors.warning(
                f"\n⚠️  Preserving temp handbag for inspection: {tmp_root}"
            )

    print()
    if ok:
        ASCIIColors.success("=" * 70)
        ASCIIColors.success("✅ ALL STAGES PASSED — handbag tools are fully visible to the LLM")
        ASCIIColors.success("=" * 70)
        return 0
    else:
        ASCIIColors.error("=" * 70)
        ASCIIColors.error("❌ PIPELINE BROKEN — see the ✗ lines above for the failing stage")
        ASCIIColors.error("=" * 70)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())