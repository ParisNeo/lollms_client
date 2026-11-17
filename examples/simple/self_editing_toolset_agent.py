"""Self-editing toolset agent — a handbag-driven persona that edits its OWN tools.

This example demonstrates the complete loop, both for a TOOLSET and for a
SKILL the agent itself carries:

    1. Build a handbag whose ``tools/`` folder contains a small toolset,
       and a ``skills/`` folder holding an editable skill plus a
       read-only skill that refuses agent edits.
    2. Construct the personality with ``extra_tools`` (so the toolset is
       callable), ``extra_tool_staging_paths`` (so it is editable),
       ``extra_skills_dirs`` (so its skills flow into ``skills_manager``).
    3. Call ``discussion.chat(
           enable_toolset_editing=True,
           enable_skill_editing=True,
       )`` — ChatMixin mounts ``toolset_staging`` and ``skill_editor`` and
       injects the corresponding editing protocols.
    4. PHASE 1 (toolset): the agent lists candidates, stages its own target
       toolset, reads the source through ``tool_stage_read``, patches it,
       validates with ``tool_stage_test``, and reports. It then attempts to
       stage the read-only ``locked_units`` toolset and reports the refusal.
    5. PHASE 2 (skill): the agent lists skills (auto-activating on the sole
       editable candidate ``temperature_helper``), reads it via
       ``tool_edit_skill_read``, patches it with Aider blocks, validates with
       ``tool_edit_skill_test``, and reports. It then attempts to activate
       the ``locked_helper`` skill and reports the refusal.
    6. The caller inspects the on-disk ``TOOLS.py`` and ``SKILL.md`` — they
       are the deliverables. The two ``locked_*`` files remain untouched.

Run:
    python examples/simple/self_editing_toolset_agent.py

Environment:
    Configure an LLM binding (e.g. via the standard LLM_* env vars) or pass
    a pre-built ``LollmsClient`` into ``build_agent()``. The script sets up
    the workspace on disk BEFORE constructing the personality so the
    ``extra_tools`` path exists at construction time (from_handbag silently
    drops non-existent paths).
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client import LollmsClient
from lollms_client.lollms_personality.lollms_personality import LollmsPersonality


# ── Layout ────────────────────────────────────────────────────────────────────
ROOT          = Path("./_self_editing_demo").resolve()
HANDBAG_DIR   = ROOT / "editor_handbag"
WORKSPACE_DIR = ROOT / "workspace"
TARGET_DIR    = WORKSPACE_DIR / "units"          # <-- the toolset the agent OWNS
TARGET_ENTRY  = TARGET_DIR / "TOOLS.py"

# Skill editing targets — one editable, one protected, mirroring the tool cases.
SKILLS_DIR         = WORKSPACE_DIR / "skills"
SKILL_TARGET_DIR   = SKILLS_DIR / "temperature_helper"     # editable skill
SKILL_TARGET_ENTRY = SKILL_TARGET_DIR / "SKILL.md"
SKILL_LOCKED_DIR   = SKILLS_DIR / "locked_helper"          # SKILL_LIBRARY_READONLY
SKILL_LOCKED_ENTRY = SKILL_LOCKED_DIR / "SKILL.md"


# ── Starter toolset source (idempotent — preserved if already edited) ─────────
STARTER_TOOLSET = textwrap.dedent('''\
    """units — a small measure-conversion toolset.

    The example agent will add a `tool_convert_temperature` function and
    validate it after patching. Because the file is the deliverable, later
    runs against the same workspace keep the edits.
    """
    from __future__ import annotations
    from typing import Any, Dict

    TOOL_LIBRARY_NAME = "Units"
    TOOL_LIBRARY_DESC = "Simple unit conversions."


    def init_tools_library(config: Optional[dict] = None) -> None:
        return None


    def tool_kilometers_to_miles(kilometers: float) -> Dict[str, Any]:
        """Convert kilometers to miles.

        Args:
            kilometers: Distance in kilometers.

        Returns:
            {"success": True, "output": "<miles>", "miles": <float>}
        """
        miles = float(kilometers) * 0.621371
        return {"success": True, "output": f"{miles:.4f}", "miles": miles}
    ''')


STARTER_READONLY_TOOLSET = textwrap.dedent('''\
    """locked_units — a READ-ONLY toolset demonstrating the self-protection flag.

    Because it declares `TOOL_LIBRARY_READONLY = True`, tool_stage will refuse
    to stage it and tool_stage_patch will refuse to write to it — the agent
    cannot edit it even if it asks.
    """
    from __future__ import annotations
    from typing import Any, Dict

    TOOL_LIBRARY_NAME = "Locked Units"
    TOOL_LIBRARY_DESC = "Read-only sample toolset guarded by TOOL_LIBRARY_READONLY."
    TOOL_LIBRARY_READONLY = True


    def init_tools_library(config: Optional[dict] = None) -> None:
        return None


    def tool_inch_to_cm(inches: float) -> Dict[str, Any]:
        """Convert inches to centimeters.

        Args:
            inches: Distance in inches.

        Returns:
            {"success": True, "output": "<cm>", "cm": <float>}
        """
        cm = float(inches) * 2.54
        return {"success": True, "output": f"{cm:.4f}", "cm": cm}
    ''')


# ── Starter skill sources (skills the agent OWNS; demonstrated as editable) ───
STARTER_SKILL = textwrap.dedent("""\
    ---
    name: temperature_helper
    description: How to answer temperature conversion questions using the units toolset.
    ---

    # Temperature Helper

    When the user asks to convert a temperature:

    1. Call `tool_convert_temperature(value, from_unit, to_unit)` from the
       `units` toolset.
    2. Report the result with the ORIGINAL and CONVERTED units spelled out
       (e.g. `25 °C = 77 °F`).
    3. Never round beyond one decimal place unless the user asks.
    """)

# A read-only skill demonstrating BOTH accepted protection flags. The
# canonical one is `modifiable: false` — that is what SkillsManager's own
# write paths honour. `SKILL_LIBRARY_READONLY: true` is the explicit
# toolset-style marker (see the comment in the body).
PROTECTED_SKILL = textwrap.dedent("""\
    ---
    name: locked_helper
    description: A read-only skill that refuses agent-driven edits.
    modifiable: false
    SKILL_LIBRARY_READONLY: true
    ---

    # Locked Helper

    This skill is read-only. Its frontmatter declares BOTH accepted flags:

    - `modifiable: false` — the canonical SkillsManager gate (honoured by
      update_skill / append_to_skill / remove_skill / set_skill_visibility;
      also the default when the key is absent).
    - `SKILL_LIBRARY_READONLY: true` — the explicit toolset-style marker,
      mirroring `TOOL_LIBRARY_READONLY` in toolset_staging.

    `tool_edit_skill_activate` refuses to activate it and
    `tool_edit_skill_patch` / `tool_edit_skill_rewrite` re-check the flag at
    write time. The agent must report the refusal, never bypass it.
    """)


# ── Handbag SOUL ──────────────────────────────────────────────────────────────
SOUL_MD = textwrap.dedent("""\
    ---
    name: UnitSmith
    author: lollms-client
    version: '1.0'
    category: engineering
    description: A focused tool-author agent that edits its own `units` toolset.
    skills_mode: loadable
    ---

    You are UnitSmith, the author of the `units` toolset AND of the
    `temperature_helper` skill.

    Your ONLY job is to extend and maintain the toolset attached to you
    (folder name: `units`) and the editable skill (`temperature_helper`).
    You do NOT perform unrelated tasks.

    ## The mandatory TOOLSET edit loop

    1. tool_stage_list_stageable         — confirm `units` is present
    2. tool_stage(name="units")          — take the single focus slot
    3. tool_stage_read                   — see the current source
    4. tool_stage_test(tool_name="tool_kilometers_to_miles",
                       arguments_json='{"kilometers": 10}')
                                         — observe current behavior
    5. tool_stage_patch (Aider blocks)   — add the new tool
    6. tool_stage_test same tool new args — validate
    7. tool_stage_unstage                — release the focus slot
    8. Summarize the change and stop.

    ## The mandatory SKILL edit loop (when skill editing is enabled)

    1. tool_edit_skill_list              — enumerate skills; note `protected`
    2. tool_edit_skill_read              — auto-activates on `temperature_helper`
    3. tool_edit_skill_patch (Aider blocks) — extend the skill body
    4. tool_edit_skill_test              — validate the change
    5. tool_edit_skill_close             — release the session
    6. Summarize and stop.

    ## Hard rules

    - Read source with tool_stage_read / tool_edit_skill_read. Do NOT call
      execute_python or a file tool to inspect the sources — read them through
      the dedicated editing tools.
    - Do NOT touch any toolset other than `units`, and only the skill
      `temperature_helper`.
    - Do NOT remove existing tool_* functions or skill sections unless asked.
    - If tool_stage or tool_edit_skill_activate refuses a target as
      `protected`, do not attempt to bypass it — report the exact refusal.
    """)


# ── Filesystem setup ──────────────────────────────────────────────────────────
def _materialize_layout() -> None:
    """Create the workspace, handbag, and starter toolsets BEFORE from_handbag.

    ``from_handbag(extra_tools=[...])`` silently drops paths that do not exist
    at construction time, so the target toolset must be on disk first.
    """
    HANDBAG_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    (HANDBAG_DIR / "SOUL.md").write_text(SOUL_MD, encoding="utf-8")
    (HANDBAG_DIR / "handbag.yaml").write_text(
        "name: Self-Editing Toolset Demo\n"
        "version: '1.0'\n"
        "description: Handbag for a persona that edits its own toolset.\n"
        "skills_mode: loadable\n",
        encoding="utf-8",
    )

    if not TARGET_ENTRY.is_file():
        TARGET_ENTRY.write_text(STARTER_TOOLSET, encoding="utf-8")

    locked_dir = WORKSPACE_DIR / "locked_units"
    locked_dir.mkdir(parents=True, exist_ok=True)
    locked_entry = locked_dir / "TOOLS.py"
    if not locked_entry.is_file():
        locked_entry.write_text(STARTER_READONLY_TOOLSET, encoding="utf-8")

    # Skill layout inside the workspace so `extra_skills_dirs=` sees them.
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    if not SKILL_TARGET_ENTRY.is_file():
        SKILL_TARGET_ENTRY.write_text(STARTER_SKILL, encoding="utf-8")
    if not SKILL_LOCKED_ENTRY.is_file():
        SKILL_LOCKED_ENTRY.write_text(PROTECTED_SKILL, encoding="utf-8")


def _resolve_toolset_staging_dir() -> Path:
    """Locate the shipped `toolset_staging` folder from the installed package."""
    package_name = "lollms_client.tools_bindings.lcp.default_tools.toolset_staging"
    spec = importlib.util.find_spec(package_name)
    if spec is not None and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            candidate = Path(loc).resolve()
            if (candidate / "TOOLS.py").is_file():
                return candidate
    import lollms_client

    fallback = (
        Path(lollms_client.__file__).resolve().parent
        / "tools_bindings" / "lcp" / "default_tools" / "toolset_staging"
    )
    if (fallback / "TOOLS.py").is_file():
        return fallback
    raise RuntimeError(
        "Could not locate the shipped `toolset_staging` toolset; "
        "ensure lollms_client is installed with its default_tools folder."
    )


# ── Agent construction ────────────────────────────────────────────────────────
def build_agent(client: Optional[LollmsClient] = None) -> LollmsPersonality:
    """Build a persona whose OWN toolset is both callable and editable."""
    _materialize_layout()
    staging_dir = _resolve_toolset_staging_dir()

    if client is None:
        client = LollmsClient(
            llm_binding_name=os.environ.get("LOLLMS_LLM_BINDING", "ollama"),
            llm_binding_config={
                "model_name": os.environ.get("LOLLMS_MODEL", "qwen2.5-coder:7b"),
            },
        )

    agent = LollmsPersonality.from_handbag(
        handbag_path=HANDBAG_DIR,
        lollms_client=client,
        # Toolset the agent MAY CALL (its own tools).
        extra_tools=[TARGET_DIR],
        # Editing surface: the staging toolset itself, plus the workspace
        # (which contains the target). Declaring the staging toolset here is
        # what makes `tool_stage_*` a callable tool from chat().
        extra_tool_staging_paths=[staging_dir, WORKSPACE_DIR],
        # Skills the agent HOLDS: mounted on the personality's skills_manager.
        # When chat() is later called with enable_skill_editing=True, these
        # become auto-activation candidates — the agent can update any of
        # them in-place (subject to SKILL_LIBRARY_READONLY).
        extra_skills_dirs=[SKILLS_DIR],
        # `workspace_path` MUST be an ancestor of the target for the rglob
        # discovery to find it — keep it equal to WORKSPACE_DIR.
        workspace_path=WORKSPACE_DIR,
        max_tokens_per_turn=8192,
    )
    return agent


# ── Streaming callback ────────────────────────────────────────────────────────
def cb(chunk: str, msg_type: Any, meta: Dict[str, Any]) -> bool:
    """Print streamed text and show tool-call boundaries in the console."""
    from lollms_client.lollms_types import MSG_TYPE  # local import: constants only

    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and isinstance(chunk, str):
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_START:
        name = (meta or {}).get("tool_name", "?")
        ASCIIColors.magenta(f"\n[TOOL] {name}({meta.get('parameters') or {}})")
    elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_END:
        name = (meta or {}).get("tool_name", "?")
        ok = "OK" if (meta or {}).get("success") else "FAIL"
        ASCIIColors.magenta(f"[TOOL] {name} → {ok}")
    return True


# ── The self-edit task ────────────────────────────────────────────────────────
EDIT_PROMPT = (
    "TWO PHASES, both in one turn.\n\n"
    "PHASE 1 — TOOLSET EDIT:\n"
    "Extend the `units` toolset with a `tool_convert_temperature(value: float, "
    "from_unit: str = 'C', to_unit: str = 'F')` function. Support 'C', 'F', "
    "and 'K'. Validate it with a few conversions. Then attempt "
    "`tool_stage(name=\"locked_units\")` and report the exact refusal — do NOT "
    "try to bypass it.\n\n"
    "PHASE 2 — SKILL EDIT:\n"
    "Update the `temperature_helper` skill so it references the new "
    "`tool_convert_temperature` (instead of the old tool_kilometers_to_miles "
    "example). Keep the frontmatter intact. Validate with tool_edit_skill_test. "
    "Then attempt `tool_edit_skill_activate(skill_name=\"locked_helper\")` and "
    "report the exact refusal message — do NOT try to bypass it.\n\n"
    "Summarize both phases and stop."
)


def self_edit(agent: LollmsPersonality) -> Dict[str, Any]:
    """Drive ONE turn of the self-editing loop via ChatMixin.chat()."""
    discussion = agent.discussion
    return discussion.chat(
        user_message=EDIT_PROMPT,
        personality=agent,
        streaming_callback=cb,
        # The two CHAT PARAMETERS the feature adds: they mount
        # `toolset_staging` and `skill_editor` respectively and admit their
        # `tool_stage_*` / `tool_edit_skill_*` tools into this turn.
        enable_toolset_editing=True,
        enable_skill_editing=True,
        max_nb_rounds=30,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ASCIIColors.cyan("Building self-editing agent...")
    agent = build_agent()
    ASCIIColors.cyan(
        f"Agent has {len(agent.list_tools_structured())} callable tool(s)."
    )

    ASCIIColors.cyan("\nDriving the self-edit turn:\n")
    result = self_edit(agent)

    ASCIIColors.cyan("\nTurn finished. Final on-disk TOOLS.py:\n")
    print(TARGET_ENTRY.read_text(encoding="utf-8"))

    ASCIIColors.cyan("\nProtected `locked_units` representation in candidates:")
    # Show that locked_units is discoverable but refused.
    draft = agent.discussion.chat(
        user_message="Call tool_stage_list_stageable and print the raw JSON.",
        personality=agent,
        streaming_callback=cb,
        enable_toolset_editing=True,
        max_nb_rounds=4,
    )
    _ = draft  # output already streamed

    ASCIIColors.cyan("\nFinal on-disk SKILL.md (temperature_helper):\n")
    print(SKILL_TARGET_ENTRY.read_text(encoding="utf-8"))

    ASCIIColors.cyan("\nProtected `locked_helper` skill remains untouched.\n")
    print(SKILL_LOCKED_ENTRY.read_text(encoding="utf-8"))

    ASCIIColors.cyan(
        f"\nDeliverables:\n"
        f"  toolset : {TARGET_ENTRY}\n"
        f"  skill   : {SKILL_TARGET_ENTRY}\n"
        f"  turn_status={(result.get('ai_message') or {}).metadata.get('turn_status', '?')}"
    )


if __name__ == "__main__":
    main()