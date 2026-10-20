"""
pptx_builder.py
LCP toolset for building, editing, and reading PowerPoint (.pptx)
presentations via sandboxed python-pptx "recipe" code.

The LLM writes pure-Python recipe code with NO import statements. The tool
embeds that code into a function body, pre-populates the namespace with the
full python-pptx design system, execs it, saves the deck to the workspace,
and verifies the result by re-reading the saved file.

Workflow:
  1. tool_pptx_build   -> create a fresh 16:9 deck from a recipe.
  2. tool_pptx_edit    -> open an existing deck and mutate it with a recipe.
  3. pptx_read         -> inspect an existing deck (slides, texts, tables).
"""

import re
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "PPTX Builder"
TOOL_LIBRARY_DESC = (
    "Create, edit and read PowerPoint (.pptx) presentations via sandboxed "
    "python-pptx recipe code (no imports allowed in recipes). The recipe is "
    "plain Python operating on the Presentation object with a pre-imported "
    "design system: Presentation, Inches, Pt, Emu, RGBColor, MSO_SHAPE, "
    "MSO_CONNECTOR, MSOAN_ANCHOR, qn."
)
TOOL_LIBRARY_ICON = "📊"

_MAX_RECIPE_CHARS = 60_000
_DEFAULT_FILE_NAME = "presentation.pptx"

_FORBIDDEN_TOKENS = (
    "import ",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "os.",
    "sys.",
    "subprocess",
    "shutil",
    "pathlib",
    "__",
    "globals(",
    "locals(",
    "vars(",
    "getattr(",
    "setattr(",
    "delattr(",
)


def init_tools_library(config: Optional[dict] = None) -> None:
    """Auto-installs python-pptx via pipmaster before first tool execution."""
    try:
        import pipmaster as pm
        pm.ensure_packages("python-pptx")
    except Exception as e:
        ASCIIColors.warning(f"[pptx_builder] Could not auto-install python-pptx via pipmaster: {e}")


# ─────────────────────── Design system namespace ──────────────────────────

def _build_recipe_namespace() -> Dict[str, Any]:
    """
    Builds the controlled namespace in which recipe code executes.
    Only python-pptx symbols are exposed; no imports are possible.
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn

    return {
        "Presentation": Presentation,
        "Inches": Inches,
        "Pt": Pt,
        "Emu": Emu,
        "RGBColor": RGBColor,
        "MSO_SHAPE": MSO_SHAPE,
        "MSO_CONNECTOR": MSO_CONNECTOR,
        "MSO_ANCHOR": MSO_ANCHOR,
        "PP_ALIGN": PP_ALIGN,
        "qn": qn,
    }