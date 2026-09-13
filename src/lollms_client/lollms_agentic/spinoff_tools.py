# lollms_agentic/spinoff_tools.py
# Spinoff sub-agent tool factories: specialized in-process LLM calls exposed
# to the WORKER persona as callable tools.
#
# DOCTRINE: Spinoffs are worker-tier execution capabilities. The Orchestrator
# persona never receives them in its own registry — it spawns Workers, and the
# Workers inherit the spinoff tools through the tools registry. This preserves
# the structural mimicry immunity of the Orchestrator tier (zero tool syntax
# in its context).
#
# These tools spawn focused, low-temperature LLM generations that inherit the
# discussion's active artifact zone without polluting the primary conversation
# context or breaking KV-cache alignment.

from typing import Any, Callable, Dict

from ascii_colors import ASCIIColors


def build_spinoff_agent_tools(
    discussion,
    images: list,
    orchestrator_mode: bool = False,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically registers specialized sub-agents as executable in-process tools.
    Enables the LLM to delegate heavy cognitive, formatting, or parsing tasks on-demand
    without breaking the main stream or bloating the primary conversation context.

    Args:
        discussion: The active LollmsDiscussion instance.
        images: Active image payloads forwarded to spinoff generations.
        orchestrator_mode: When True (Orchestrator persona), returns an EMPTY
            dict — the Orchestrator must never be handed executable tool syntax.
            Workers spawned by the Orchestrator receive the spinoffs through the
            tools registry instead.
    """
    spinoffs: Dict[str, Dict[str, Any]] = {}

    if orchestrator_mode:
        ASCIIColors.info(
            "[SpinoffTools] Orchestrator persona active — spinoff sub-agent "
            "tools withheld from its registry (Workers inherit them)."
        )
        return spinoffs

    def tool_spinoff_code_specialist(task_instructions: str) -> dict:
        """
        Spawns a specialized Surgical Code Specialist in a focused, low-temperature sandbox.
        Ideal for generating complete Python scripts, performing exact aider patches, or refactoring logic.

        Args:
            task_instructions (str): The specific coding or refactoring instructions for the specialist.
        """
        custom_system = (
            "You are an expert Surgical Code Specialist.\n"
            "You operate in a hyper-focused sandbox isolated from the main conversation's noise.\n"
            "Your sole task is to implement the requested code modifications perfectly.\n"
            "You see ONLY the artifacts listed in the context zone below. Work strictly within them.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY a valid <artifact> block containing your code or SEARCH/REPLACE patch.\n"
            "2. Do NOT use markdown fences or write introductory/concluding prose outside the tags.\n"
            "3. Ensure character-for-character accuracy in aider SEARCH/REPLACE blocks."
        )
        art_zone = discussion.artefacts.build_artefacts_context_zone()
        payload = f"=== CONTEXT ARTIFACTS ===\n{art_zone}\n\n=== SPECIALIST TASK ===\n{task_instructions}"
        try:
            res = discussion.lollmsClient.generate_text(
                prompt=payload,
                system_prompt=custom_system,
                images=images,
                temperature=0.1,
                **{k: v for k, v in kwargs.items() if k not in ("temperature", "streaming_callback")}
            )
            return {"success": True, "output": res.strip()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    spinoffs["tool_spinoff_code_specialist"] = {
        "name": "tool_spinoff_code_specialist",
        "description": "Spawns a specialized Surgical Code specialist in a focused, low-temperature sandbox to write, patch, or refactor code artifacts.",
        "parameters": [{"name": "task_instructions", "type": "str", "description": "Specific code or patch instructions."}],
        "callable": tool_spinoff_code_specialist,
    }

    def tool_spinoff_presentation_designer(style: str, slide_count: int, structure_hints: str) -> dict:
        """
        Spawns a specialized HTML Slide Presentation Designer in a focused sandbox.
        Converts active artifacts into a styled, structured multi-slide HTML5 presentation deck.

        Args:
            style (str): The design theme (e.g. 'dark', 'light', 'creative').
            slide_count (int): Expected number of slides.
            structure_hints (str): Specific topics or structural outlines to focus on.
        """
        custom_system = (
            "You are an expert HTML Slide Presentation Designer.\n"
            "You design beautiful, modern 16:9 slideshows using semantic HTML5 and CSS.\n"
            "You see ONLY the artifacts listed in the context zone below. Work strictly within them.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY a single <artifact> tag containing your complete, valid HTML document.\n"
            "2. Do NOT write conversational prose or use markdown code blocks outside the tags."
        )
        art_zone = discussion.artefacts.build_artefacts_context_zone()
        payload = (
            f"=== CONTEXT ARTIFACTS ===\n{art_zone}\n\n"
            f"=== DESIGN REQUIREMENTS ===\n"
            f"• Style Theme: {style}\n"
            f"• Slides Count: {slide_count}\n"
            f"• Structure Outlines: {structure_hints}"
        )
        try:
            res = discussion.lollmsClient.generate_text(
                prompt=payload,
                system_prompt=custom_system,
                temperature=0.3,
                **{k: v for k, v in kwargs.items() if k not in ("temperature", "streaming_callback")}
            )
            return {"success": True, "output": res.strip()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    spinoffs["tool_spinoff_presentation_designer"] = {
        "name": "tool_spinoff_presentation_designer",
        "description": "Spawns a specialized HTML Slide Presentation Designer in a focused sandbox to synthesize active datasets/artifacts into a highly styled multi-slide HTML5 presentation deck.",
        "parameters": [
            {"name": "style", "type": "str", "description": "Design theme (dark, light, creative, minimal)."},
            {"name": "slide_count", "type": "int", "description": "Expected number of slides."},
            {"name": "structure_hints", "type": "str", "description": "Outlines and structural hints."},
        ],
        "callable": tool_spinoff_presentation_designer,
    }

    ASCIIColors.info(
        f"[SpinoffTools] Registered {len(spinoffs)} spinoff sub-agent tool(s) for the "
        f"worker persona: {sorted(spinoffs.keys())}"
    )

    return spinoffs