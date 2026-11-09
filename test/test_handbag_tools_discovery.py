"""
End-to-end discovery test for handbag toolsets.

Verifies the entire pipeline from ``Handbag._load_tools()`` down to the
tool name appearing in the LLM-visible system prompt. Each stage is
asserted independently so that a regression in one layer reports
precisely which stage broke.

Covers the exact symptom class that causes "my handbag tools are invisible":

    tools/<toolset>/TOOLS.py  →  LCPBinding.discovered_tools
                              →  to_chat_tool_specs()
                              →  LollmsPersonality._discover_tools()
                              →  _build_system_prompt()

Uses a real temporary handbag directory on disk (no mocking) so the
canonical ``TOOLS.py`` layout is exercised through the same code path
production uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src package is importable when the test is run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOOLSET_PY = '''\
"""ArXiv toolset — single tool for test coverage."""


def tool_search_arxiv(query: str, max_results: int = 5) -> dict:
    """Search the arXiv repository for papers matching the query.

    Args:
        query (str): Natural language search query.
        max_results (int): Maximum number of papers to return.
    """
    return {
        "success": True,
        "output": f"[stub] {max_results} ArXiv result(s) for '{query}'.",
    }
'''


@pytest.fixture
def handbag_dir(tmp_path: Path) -> Path:
    """Builds a minimal, production-shaped handbag on disk.

    Layout::

        <handbag>/
            SOUL.md
            tools/
                arxiv_tools/
                    TOOLS.py         (defines tool_search_arxiv)
    """
    hb = tmp_path / "test_handbag"
    hb.mkdir()
    (hb / "SOUL.md").write_text(
        "---\n"
        "name: Test Handbag Agent\n"
        "author: pytest\n"
        "category: research\n"
        "description: Discovery test.\n"
        "---\n\n"
        "You are a test agent.\n",
        encoding="utf-8",
    )
    toolset_dir = hb / "tools" / "arxiv_tools"
    toolset_dir.mkdir(parents=True)
    (toolset_dir / "TOOLS.py").write_text(_TOOLSET_PY, encoding="utf-8")
    return hb


# ---------------------------------------------------------------------------
# Stage 1 — Handbag scans and yields the toolset FOLDER
# ---------------------------------------------------------------------------

def test_handbag_yields_toolset_folder(handbag_dir: Path) -> None:
    """Handbag._load_tools() must yield the toolset FOLDER (not the .py file),
    so that the LCP resolver can find TOOLS.py inside it."""
    from lollms_client.lollms_personality.handbag import Handbag

    hb = Handbag(handbag_dir)
    assert hb.tool_files, (
        f"Handbag found zero tools in '{handbag_dir}/tools'. "
        "Expected one entry per toolset folder."
    )
    assert len(hb.tool_files) == 1, (
        f"Expected exactly 1 toolset, got {len(hb.tool_files)}: {hb.tool_files}"
    )
    entry = hb.tool_files[0]
    assert entry.name == "arxiv_tools", (
        f"Expected folder 'arxiv_tools', got '{entry.name}'. "
        "Handbag._load_tools must return the folder, not the file."
    )
    assert entry.is_dir(), (
        f"Handbag._load_tools returned a FILE ('{entry}') instead of the "
        "toolset FOLDER. The LCP resolver expects folder paths."
    )
    assert (entry / "TOOLS.py").is_file(), (
        f"Expected TOOLS.py inside '{entry}'."
    )


# ---------------------------------------------------------------------------
# Stage 2 — LCPBinding discovers the tool from the folder
# ---------------------------------------------------------------------------

def test_lcp_binding_discovers_tool_from_folder(handbag_dir: Path) -> None:
    """LCPBinding(tool_files=[<folder>]) must populate ``discovered_tools``."""
    from lollms_client.lollms_personality.handbag import Handbag
    from lollms_client.tools_bindings.lcp import LCPBinding

    hb = Handbag(handbag_dir)
    binding = LCPBinding(tool_files=[str(f) for f in hb.tool_files])

    tool_names = {t.get("name") for t in binding.discovered_tools}
    assert "tool_search_arxiv" in tool_names, (
        f"LCPBinding discovered {sorted(tool_names)} but not 'tool_search_arxiv'. "
        f"Binding.tool_files={[str(p) for p in binding.tool_files]}. "
        "Check _parse_tool_via_ast and _is_library_admitted/_is_library_permitted "
        "for '{entry}'.".format(entry=hb.tool_files[0])
    )


# ---------------------------------------------------------------------------
# Stage 3 — to_chat_tool_specs() converts discovered tool → LLM spec
# ---------------------------------------------------------------------------

def test_lcp_binding_to_chat_tool_specs(handbag_dir: Path) -> None:
    """``to_chat_tool_specs()`` must return a full spec including ``callable``."""
    from lollms_client.lollms_personality.handbag import Handbag
    from lollms_client.tools_bindings.lcp import LCPBinding

    hb = Handbag(handbag_dir)
    binding = LCPBinding(tool_files=[str(f) for f in hb.tool_files])
    specs = binding.to_chat_tool_specs()

    assert "tool_search_arxiv" in specs, (
        f"to_chat_tool_specs() returned {sorted(specs.keys())} — missing tool. "
        f"Raw discovered_tools={[t.get('name') for t in binding.discovered_tools]}"
    )
    spec = specs["tool_search_arxiv"]
    assert callable(spec.get("callable")), (
        "Spec is missing a 'callable' field — the agentic loop cannot invoke "
        "the tool without it."
    )
    assert spec.get("description"), "Spec description is empty."
    params = {p["name"] for p in spec.get("parameters", [])}
    assert {"query", "max_results"}.issubset(params), (
        f"Expected parameters 'query' and 'max_results' in spec; got {params}."
    )


# ---------------------------------------------------------------------------
# Stage 4 — LollmsPersonality discovers the tool in active_tools
# ---------------------------------------------------------------------------

def test_personality_discover_tools_includes_handbag_tool(handbag_dir: Path) -> None:
    """The full ``from_handbag`` → ``_discover_tools`` pipeline must return
    the handbag tool in the active tool registry."""
    from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

    personality = LollmsPersonality.from_handbag(handbag_dir)

    # Sanity: binding attached?
    assert personality._tool_binding is not None, (
        "from_handbag did not attach a tool binding. "
        "Check Handbag._load_tools returned non-empty and LCPBinding was created."
    )
    discovered = getattr(personality._tool_binding, "discovered_tools", None) or []
    discovered_names = {t.get("name") for t in discovered}
    assert "tool_search_arxiv" in discovered_names, (
        f"Personality's tool binding discovered {sorted(discovered_names)} "
        f"but not 'tool_search_arxiv'. The binding exists but failed to load "
        f"the toolset. tool_files={[str(p) for p in getattr(personality._tool_binding, 'tool_files', [])]}"
    )

    active_tools = personality._discover_tools()
    assert "tool_search_arxiv" in active_tools, (
        f"_discover_tools() returned {sorted(active_tools.keys())} — handbag "
        "tool is missing from the active registry. Check the merge block at the "
        "bottom of _discover_tools for silent drops."
    )


# ---------------------------------------------------------------------------
# Stage 5 — tool name appears in the LLM-visible system prompt
# ---------------------------------------------------------------------------

def test_personality_build_system_prompt_renders_handbag_tool(handbag_dir: Path) -> None:
    """The rendered system prompt must list the handbag tool so the LLM
    actually knows it exists."""
    from lollms_client.lollms_personality.lollms_personality import LollmsPersonality

    personality = LollmsPersonality.from_handbag(handbag_dir)
    active_tools = personality._discover_tools()
    prompt = personality._build_system_prompt(active_tools)

    assert "=== ACTIVE TOOLS" in prompt, (
        "System prompt does not contain an '=== ACTIVE TOOLS ===' section — "
        "active_tools is empty before prompt build."
    )
    assert "tool_search_arxiv" in prompt, (
        "Tool name 'tool_search_arxiv' is absent from the rendered system prompt. "
        "The agentic loop will never call it."
    )
    # Spot-check that parameters made it into the prompt (lean schema).
    assert "query" in prompt, "Parameter 'query' missing from rendered prompt."


# ---------------------------------------------------------------------------
# Diagnostic — legacy <folder>.py layout remains supported
# ---------------------------------------------------------------------------

def test_legacy_folder_py_layout_still_works(tmp_path: Path) -> None:
    """A toolset using the legacy ``<folder>/<folder>.py`` layout must still
    be discovered — the ``TOOLS.py`` change must not break old handbags."""
    from lollms_client.lollms_personality.handbag import Handbag
    from lollms_client.tools_bindings.lcp import LCPBinding

    hb = tmp_path / "legacy_handbag"
    hb.mkdir()
    (hb / "SOUL.md").write_text("---\nname: Legacy\n---\n\nLegacy agent.\n", encoding="utf-8")
    toolset_dir = hb / "tools" / "websearch"
    toolset_dir.mkdir(parents=True)
    (toolset_dir / "websearch.py").write_text(
        'def tool_web_search(query: str) -> dict:\n'
        '    """Search the web."""\n'
        '    return {"success": True, "output": f"stub: {query}"}\n',
        encoding="utf-8",
    )

    handbag = Handbag(hb)
    assert handbag.tool_files, "Legacy toolset not picked up by Handbag."
    binding = LCPBinding(tool_files=[str(f) for f in handbag.tool_files])
    names = {t.get("name") for t in binding.discovered_tools}
    assert "tool_web_search" in names, (
        f"Legacy layout discovery failed. Discovered: {sorted(names)}. "
        "resolve_library_entry fallback should pick <folder>/<folder>.py."
    )


# ---------------------------------------------------------------------------
# Diagnostic — non-toolset folders must not pollute the registry
# ---------------------------------------------------------------------------

def test_assets_folder_is_not_treated_as_toolset(tmp_path: Path) -> None:
    """``assets/`` and ``references/`` folders inside ``tools/`` must be
    skipped (they hold binary blobs, not tool libraries)."""
    from lollms_client.lollms_personality.handbag import Handbag

    hb = tmp_path / "handbag_with_assets"
    hb.mkdir()
    (hb / "SOUL.md").write_text("---\nname: Assets\n---\n\nAgent.\n", encoding="utf-8")
    tools_dir = hb / "tools"
    tools_dir.mkdir()
    (tools_dir / "assets").mkdir()
    (tools_dir / "assets" / "logo.py").write_text("x = 1\n", encoding="utf-8")
    (tools_dir / "references").mkdir()
    (tools_dir / "references" / "note.py").write_text("y = 2\n", encoding="utf-8")
    real_toolset = tools_dir / "real_tools"
    real_toolset.mkdir()
    (real_toolset / "TOOLS.py").write_text(
        'def tool_real() -> dict:\n    """Real tool."""\n    return {"success": True}\n',
        encoding="utf-8",
    )

    handbag = Handbag(hb)
    names = {p.name for p in handbag.tool_files}
    assert "assets" not in names, "'assets' folder must be excluded from toolsets."
    assert "references" not in names, "'references' folder must be excluded from toolsets."
    assert "real_tools" in names, "Real toolset was skipped by the assets filter."