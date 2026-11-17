"""tool_editor_agent.py — a conversational agent that edits its own tool.

Same ergonomics as ``simple_image_generation_agent.py``: build the persona,
then talk to it in a REPL. The agent can BOTH call the target toolset AND
edit its source. The two intents are declared on separate parameters:

    * the target   (``unit_converter``)  — a native handbag tool,
    * ``toolset_editor``                 — the editing surface, passed via
      ``extra_tool_staging_paths`` (NOT ``extra_tools``).

The three wiring invariants (from ``docs/TOOLSET_STAGING_AND_SANDBOX.md``):

1. ``extra_tool_staging_paths`` carries the editor toolset; the target is a
   native handbag tool discovered from ``handbag/tools/``.
2. ``workspace_path`` must be an ANCESTOR of the target folder — the editor
   enumerates candidates via ``workspace.rglob("TOOLS.py")``.
3. The editing runs on the agent that OWNS the binding (never a sub-agent).

The agentic loop is delegated to ``LollmsPersonality.chat()`` — the library's
canonical entry point. It owns native tool-call dispatch (``env`` injection,
session-flag semantics, artefact sync, context compression, checkpointing and
the artefact/scratchpad stream state machine). The callback passed here only
RENDERS already-dispatched events — it never executes tools.

    python examples/simple/tool_editor_agent.py
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Any

from ascii_colors import ASCIIColors

# Make the repo's ``src`` importable when run from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lollms_client import MSG_TYPE
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

# ── Layout ────────────────────────────────────────────────────────────────
DEMO_ROOT = Path(__file__).resolve().parent / "_tool_editor_demo"
HANDBAG_DIR = DEMO_ROOT / "editor_handbag"
TARGET_TOOLSET_DIR = HANDBAG_DIR / "tools" / "unit_converter"
TARGET_ENTRY = TARGET_TOOLSET_DIR / "TOOLS.py"
WORKSPACE_DIR = HANDBAG_DIR  # must be an ANCESTOR of TARGET_TOOLSET_DIR


def _editor_toolset_dir() -> Path:
    """Resolve the shipped ``toolset_editor`` folder from the installed package.

    The folder has no ``__init__.py``, so importing it yields a namespace
    package whose ``__file__`` is ``None`` — never read ``__file__`` here.
    Resolution: ``find_spec(...).submodule_search_locations`` first, then a
    filesystem fallback rooted at the imported ``lollms_client`` package.
    """
    package_name = "lollms_client.tools_bindings.lcp.default_tools.toolset_editor"
    spec = importlib.util.find_spec(package_name)
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = Path(location).resolve()
            if (candidate / "TOOLS.py").is_file():
                return candidate

    import lollms_client

    package_root = Path(lollms_client.__file__).resolve().parent
    fallback = (
        package_root
        / "tools_bindings" / "lcp" / "default_tools" / "toolset_editor"
    ).resolve()
    if (fallback / "TOOLS.py").is_file():
        return fallback

    raise RuntimeError(
        f"Could not locate the shipped 'toolset_editor' toolset folder "
        f"(searched through spec={spec!r} and fallback={fallback})."
    )


STARTER_TOOL_SOURCE = textwrap.dedent('''\
    """unit_converter — starter toolset for the tool-editor demo."""

    from typing import Any, Dict

    TOOL_LIBRARY_NAME = "Unit Converter"
    TOOL_LIBRARY_DESC = "Convert values between units (demo target for editing)."
    TOOL_LIBRARY_ICON = "🔄"


    def init_tools_library(config: dict = None) -> None:
        return None


    def tool_celsius_to_fahrenheit(celsius: float) -> Dict[str, Any]:
        """Convert a temperature from Celsius to Fahrenheit.

        Args:
            celsius: Temperature in degrees Celsius.

        Returns:
            dict: {"success": True, "fahrenheit": <float>}
        """
        fahrenheit = (float(celsius) * 9.0 / 5.0) + 32.0
        return {"success": True, "fahrenheit": fahrenheit}


    def tool_km_to_miles(kilometers: float) -> Dict[str, Any]:
        """Convert a distance from kilometers to miles.

        Args:
            kilometers: Distance in kilometers.

        Returns:
            dict: {"success": True, "miles": <float>}
        """
        miles = float(kilometers) * 0.621371
        return {"success": True, "miles": miles}
''')


SOUL_MD = textwrap.dedent("""\
    ---
    name: ToolSmith
    author: lollms-client
    version: '1.0'
    category: engineering
    description: A focused agent that inspects, edits, and validates a single LCP toolset.
    skills_mode: loadable
    ---

    You are ToolSmith, a dedicated LCP toolset editor.

    Your ONLY job is to inspect, edit, and validate the target toolset
    ``__TARGET_TOOLSET__``. You must NOT perform unrelated tasks.

    THE TARGET IS EXACTLY ONE FILE, ON DISK:
        __TARGET_ENTRY_ABS__

    ## The mandatory edit loop

    1. READ the source with ``tool_edit_toolset_read``. It returns the full
       ``TOOLS.py`` source directly. Do NOT use a file tool, a sub-agent, or an
       ``<artifact>`` to read it.
    2. OBSERVE current behavior with ``tool_edit_toolset_test`` (call one
       ``tool_*`` with a representative input) BEFORE editing.
    3. REASON about the defect: state the exact failing behavior in one sentence.
    4. EDIT via ``tool_edit_toolset_patch`` (Aider SEARCH/REPLACE blocks) or
       ``tool_edit_toolset_rewrite`` (full file). Keep every other ``tool_*``
       unchanged.
    5. VALIDATE with ``tool_edit_toolset_test`` using the SAME inputs. Compare
       to the pre-edit output. State PASS or FAIL explicitly.
    6. Iterate 4–5 until PASS. Then summarise the final diff and stop.

    ## Hard rules

    - Edit ONLY through ``tool_edit_toolset_patch`` / ``tool_edit_toolset_rewrite``.
      NEVER emit an ``<artifact>`` to write ``TOOLS.py``: that bypasses the AST
      validation + atomic patch + hot-reload this toolset provides, and it
      writes to the wrong (CWD-relative) path.
    - If a ``tool_edit_toolset_*`` call reports "not found in module" or any
      dispatch error, do NOT invent an alternative editor and do NOT fall back
      to artifacts. Stop and report the exact error text to the user — it is a
      wiring/config issue for the operator, not a task to improvise around.
    - Never edit files outside the attached toolset folder.
    - Never write a broken ``TOOLS.py``; the toolset rejects invalid Python.
    - Reload happens automatically after patch/rewrite. If a tool reports
      ``reloaded: false``, call ``tool_edit_toolset_reload`` once.
    - The edited ``TOOLS.py`` on disk IS the deliverable — do not copy it.
""")


def write_starter_toolset() -> None:
    """Materialize the starter tool (idempotent — preserves existing edits)."""
    TARGET_TOOLSET_DIR.mkdir(parents=True, exist_ok=True)
    if not TARGET_ENTRY.is_file():
        TARGET_ENTRY.write_text(STARTER_TOOL_SOURCE, encoding="utf-8")


# Hard, runtime-appended contract. It is appended AFTER the templated SOUL so
# it survives any edit to the base body. It exists because a live run showed
# the model (a) inventing a `sub` argument `tool_edit_toolset_read` does not
# accept, then (b) printing a fabricated "PASS" validation table without ever
# calling the tools. Both are failures the base persona failed to prevent.
_TOOL_CONTRACT = """

## TOOL CONTRACTS (exact — do not invent arguments)

- `tool_edit_toolset_read()` takes NO arguments. Call it as
  `tool_edit_toolset_read()` with an empty parameter object `{}`. Never pass
  `sub`, `path`, `file`, or any other key.
- `tool_edit_toolset_test(tool_name=..., arguments_json=...)` — `tool_name`
  is one of the `tool_*` names returned by `tool_edit_toolset_read`; the
  second argument is a JSON STRING of the tool's arguments.
- `tool_edit_toolset_patch(patches=...)` — `patches` is an Aider
  SEARCH/REPLACE string; every block must apply or nothing is written.
- `tool_edit_toolset_rewrite(new_source=...)` — full file replacement.

## FORBIDDEN BEHAVIOURS (a live incident)

- NEVER print a validation table, "PASS/FAIL", or a diff summary unless you
  actually received a result dict from `tool_edit_toolset_test` /
  `tool_edit_toolset_patch` in THIS turn. A fabricated PASS is a hard
  failure — it is worse than admitting you could not edit.
- NEVER repeat a tool call with identical arguments after it already
  returned. If a call's result is unclear, read it again; do not re-issue it.
- If a tool result is an error, quote the error text verbatim and stop to
  reason about it. Do not narrate progress you did not observe.

## The only valid sequence

1. `tool_edit_toolset_read()`   → returns `{"source": ...}`
2. `tool_edit_toolset_test(tool_name=..., arguments_json="...")`  (before)
3. `tool_edit_toolset_patch(...)` OR `tool_edit_toolset_rewrite(...)`
4. `tool_edit_toolset_test(tool_name=..., arguments_json="...")`  (after)
5. Report PASS/FAIL using ONLY the actual outputs from steps 2 and 4.
"""


# Hard, runtime-appended contract. Appended AFTER the templated SOUL so it
# survives any edit to the base body. It exists because a live run showed the
# model (a) inventing a `sub` argument `tool_edit_toolset_read` does not
# accept, then (b) printing a fabricated "PASS" table without ever calling the
# tools. Both are failures the base persona failed to prevent.
_TOOL_CONTRACT = """

## TOOL CONTRACTS (exact — do not invent arguments)

- `tool_edit_toolset_read()` takes NO arguments. Call it with an empty
  parameter object `{}`. Never pass `sub`, `path`, `file`, or any other key.
- `tool_edit_toolset_test(tool_name=..., arguments_json=...)` — `tool_name`
  is one of the `tool_*` names returned by `tool_edit_toolset_read`; the
  second argument is a JSON STRING of that tool's arguments.
- `tool_edit_toolset_patch(patches=...)` — `patches` is an Aider
  SEARCH/REPLACE string; every block must apply or nothing is written.
- `tool_edit_toolset_rewrite(new_source=...)` — full file replacement.

## FORBIDDEN BEHAVIOURS (learned from a live incident)

- NEVER print a validation table, "PASS/FAIL", or a diff summary unless you
  actually received a result dict from `tool_edit_toolset_test` /
  `tool_edit_toolset_patch` in THIS turn. A fabricated PASS is a hard failure.
- NEVER re-issue a tool call with identical arguments after it already
  returned. If a result is unclear, read it again; do not resend it.
- If a tool result is an error, quote the error text verbatim and stop to
  reason about it. Do not narrate progress you did not observe.

## The only valid sequence

1. `tool_edit_toolset_read()`               -> returns {"source": ...}
2. `tool_edit_toolset_test(...)`            (before)
3. `tool_edit_toolset_patch(...)` OR `tool_edit_toolset_rewrite(...)`
4. `tool_edit_toolset_test(...)`            (after)
5. Report PASS/FAIL using ONLY the real outputs from steps 2 and 4.
"""


def _render_soul() -> str:
    """Render SOUL.md with the ABSOLUTE target path baked in, plus the
    runtime tool contract.

    Two prior live failures are closed here:
      * The model invented ``tools/unit_converter/TOOLS.py`` (CWD-relative)
        because the persona never named the real file -> bake the absolute
        path in.
      * The model guessed non-existent tool arguments and fabricated a PASS
        table -> append ``_TOOL_CONTRACT`` and the exact activation call.
    """
    rendered = SOUL_MD.replace(
        "__TARGET_ENTRY_ABS__", str(TARGET_ENTRY.resolve())
    ).replace(
        "__TARGET_TOOLSET__", TARGET_TOOLSET_DIR.name
    )
    activation = (
        "\n## If activation is ever required\n\n"
        "Prefer `tool_edit_toolset_read()` first — it auto-activates the sole "
        "target. Only if read() returns an activation error, call this exact "
        "form (unambiguous, always resolves):\n\n"
        f'  tool_edit_toolset_activate(toolset_name="{TARGET_TOOLSET_DIR.name}", '
        f'toolset_folder="{TARGET_TOOLSET_DIR.resolve()}")\n'
    )
    return rendered.rstrip() + "\n" + _TOOL_CONTRACT + activation
    activation = (
        "\n## If activation is ever required\n\n"
        "Call this exact form (it is unambiguous and always resolves):\n\n"
        f'  tool_edit_toolset_activate(toolset_name="{TARGET_TOOLSET_DIR.name}", '
        f'toolset_folder="{TARGET_TOOLSET_DIR.resolve()}")\n\n'
        "Prefer `tool_edit_toolset_read()` first — it auto-activates the sole "
        "target. Only call activate explicitly if read() returned an "
        "activation error.\n"
    )
    return rendered.rstrip() + "\n" + _TOOL_CONTRACT + activation


def _soul_md() -> str:
    """Render SOUL.md with the ABSOLUTE target path baked in.

    The model cannot guess the target folder from relative hints — a prior run
    invented ``tools/unit_converter/TOOLS.py`` (CWD-relative) and thrashed.
    Baking the real path into the persona removes that failure mode, and the
    "no artifact fallback" clause stops it from hand-writing the file when the
    editor surface answers slowly.
    """
    return SOUL_MD.replace("__TARGET_TOOLSET__", TARGET_TOOLSET_DIR.name).replace(
        "__TARGET_ENTRY_ABS__", str(TARGET_ENTRY.resolve())
    )


def write_handbag() -> None:
    HANDBAG_DIR.mkdir(parents=True, exist_ok=True)
    (HANDBAG_DIR / "SOUL.md").write_text(_render_soul(), encoding="utf-8")
    (HANDBAG_DIR / "handbag.yaml").write_text(
        textwrap.dedent("""\
            name: Tool Editor Handbag
            version: '1.0'
            description: Handbag for a single-toolset editor persona.
            skills_mode: loadable
            """),
        encoding="utf-8",
    )


def cb(chunk: str, msg_type: int, metadata: dict) -> bool:
    """Stream callback passed to ``chat(streaming_callback=...)``.

    ``chat()`` emits already-dispatched tool events; this callback only
    renders. Tool execution, ``env`` injection, and model re-invocation all
    belong to the discussion/personality layer — do NOT execute tools here.
    A rendering bug must never kill the REPL, so every branch is guarded.
    """
    try:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if chunk:
                print(chunk, end="", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
            ASCIIColors.error(f"\n[error] {chunk}")
            return False
        # Thinking/other chunks are dropped in this example; a UI could show them.
    except Exception as exc:  # noqa: BLE001 — never let rendering kill the REPL
        ASCIIColors.warning(f"[callback] {type(exc).__name__}: {exc}")
    return True


def build_editor(client: Any) -> LollmsPersonality:
    """Wire the editor persona: target is a native handbag tool, editor is mounted.

    ``extra_tools`` carries ONLY the editor toolset; the target is discovered
    from ``handbag/tools/unit_converter``. ``workspace_path`` is an ancestor of
    the target so the editor's ``rglob("TOOLS.py")`` scan finds it.
    """
    return LollmsPersonality.from_handbag(
        handbag_path=HANDBAG_DIR,
        lollms_client=client,
        # ``extra_tools`` is ONLY the set of toolsets the agent CALLS. The
        # editing surface is declared separately via
        # ``extra_tool_staging_paths`` so we do not overload one parameter
        # with two distinct intents. The target stays a native handbag tool
        # discovered from ``handbag/tools/``.
        # ``extra_tools`` is ONLY the toolsets the agent CALLS. The editing
        # surface is declared separately via ``extra_tool_staging_paths`` so
        # one parameter never carries two intents.
        # ``extra_tools`` is ONLY the toolsets the agent CALLS. The editing
        # surface is declared separately via ``extra_tool_staging_paths`` so
        # one parameter never carries two intents. The target stays a native
        # handbag tool under ``tools/``.
        extra_tool_staging_paths=[_editor_toolset_dir()],
        workspace_path=WORKSPACE_DIR,       # ANCESTOR of TARGET_TOOLSET_DIR
        max_tokens_per_turn=8192,
    )


def main() -> None:
    write_starter_toolset()
    write_handbag()

    # Initialize client from environment/config (same as the image agent).
    lc = get_client_from_env()
    editor = build_editor(lc)

    mounted = sorted(
        t["name"] for t in editor.list_tools_structured()
        if t["name"].startswith(("tool_convert", "tool_celsius", "tool_km", "tool_edit_toolset"))
    )
    ASCIIColors.info(f"ToolSmith ready. Editing: {TARGET_ENTRY}")
    ASCIIColors.info(f"Mounted tools: {', '.join(mounted)}")
    ASCIIColors.info(
        "Ask the agent to improve the converter, e.g. "
        "'add miles->km and feet->m, then prove it works'."
    )

    print("\n--- Chat Session Started ---")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue

        try:
            editor.chat(
                prompt=user_input,
                streaming_callback=cb,
                enable_workspace_tools=False,   # editor tools are the only edit surface
                enable_python_exec=False,       # no raw-Python fallback into the sandbox
            )
            print()
        except Exception as exc:  # noqa: BLE001
            ASCIIColors.error(f"Error during chat: {exc}")


if __name__ == "__main__":
    main()