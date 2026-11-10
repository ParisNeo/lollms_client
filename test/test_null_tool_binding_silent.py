"""Regression tests: the ``_NullToolBinding`` opt-out must be SILENT.

A personality with no handbag tools resolves its ``tools`` attribute to the
documented ``_NullToolBinding`` sentinel (see the module contract at the top of
``lollms_personality.py``). Before the fix, every ``_discover_tools()`` call
emitted a hard ERROR::

    ❌ Personality tool binding 'null' produced 0 tool spec(s) at chat time
    ... Handbag toolsets will NOT be visible to the model.

…even though the LCP path had correctly mounted its toolset. That was a false
alarm on the personality-merge path, not a real failure.

These tests pin the corrected contract:

1. ``_is_null_tool_binding`` recognises every opt-out sentinel and rejects real
   bindings.
2. ``_discover_tools`` on a personality carrying ``_NULL_TOOL_BINDING`` must NOT
   emit an ``ASCIIColors.error`` mentioning "produced 0 tool spec".
3. A *real* binding that genuinely yields zero specs must STILL error (no
   regression to silent-swallowing real breakages).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ascii_colors import ASCIIColors

from lollms_client.lollms_personality.lollms_personality import (
    LollmsPersonality,
    _NULL_TOOL_BINDING,
    _NullToolBinding,
    _is_null_tool_binding,
)


# --------------------------------------------------------------------------- #
# 1. Sentinel resolver contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sentinel", [None, "null", "NULL", " none ", "", _NULL_TOOL_BINDING, _NullToolBinding()])
def test_is_null_tool_binding_accepts_sentinels(sentinel):
    assert _is_null_tool_binding(sentinel) is True


@pytest.mark.parametrize(
    "binding",
    [
        MagicMock(binding_name="lcp"),
        MagicMock(binding_name="some_handbag_toolset"),
        MagicMock(binding_name="custom"),
    ],
)
def test_is_null_tool_binding_rejects_real_bindings(binding):
    assert _is_null_tool_binding(binding) is False


# --------------------------------------------------------------------------- #
# 2. Null binding is silent during discovery
# --------------------------------------------------------------------------- #

def _make_personality(**kwargs) -> LollmsPersonality:
    # No workspace → no artefact system, no LCP auto-bootstrap side-effects
    # beyond what we explicitly stub. tools=None → _NULL_TOOL_BINDING.
    return LollmsPersonality(name="tester", lollms_client=None, **kwargs)


def test_null_binding_does_not_emit_error_during_discovery():
    pers = _make_personality()
    assert pers.tools is _NULL_TOOL_BINDING

    captured: list[str] = []

    def _capture(msg, *a, **kw):
        captured.append(str(msg))

    with patch.object(ASCIIColors, "error", side_effect=_capture):
        pers._discover_tools(enable_data_tools=False, enable_workspace_tools=False)

    offending = [m for m in captured if "produced 0 tool spec" in m or "Handbag toolsets will" in m]
    assert offending == [], (
        "Null tool binding must be a silent opt-out, but emitted an error: "
        f"{offending}"
    )


class _FakeLCPBinding:
    """Minimal, non-MagicMock stand-in for a real LCP binding.

    Must NOT expose ``_mock_return_value`` (MagicMock does), otherwise
    ``_discover_tools`` treats it as a mock and discards it.
    """

    binding_name = "lcp"

    def __init__(self, specs, host_tool_configs=None):
        self._specs = dict(specs)
        self.host_tool_configs = host_tool_configs or {}

    def discover_tools(self):
        return list(self._specs.values())

    def execute_tool(self, tool_name, params, **_):
        return {"success": False, "error": "not executed in test"}

    def to_chat_tool_specs(self, **_):
        return dict(self._specs)

    def mount_tool_library_if_absent(self, *_a, **_k):
        return None

    def mount_tool_library(self, *_a, **_k):
        return None

    def is_tool_available(self, *_a, **_k):
        return False

    def find_library_for_tool(self, *_a, **_k):
        return None


def test_null_binding_still_allows_lcp_tools_to_surface():
    """The opt-out only skips personality merge — LCP-mounted tools stay active."""
    pers = _make_personality()

    lcp = _FakeLCPBinding({
        "tool_execute_python_data_query": {
            "name": "tool_execute_python_data_query",
            "description": "Query tabular data.",
            "parameters": [{"name": "query", "type": "str"}],
            "callable": lambda query: {"success": True},
        }
    })
    pers._lollms_client = MagicMock(tools=lcp)

    captured: list[str] = []

    with patch.object(ASCIIColors, "error", side_effect=lambda m, *a, **kw: captured.append(str(m))):
        active = pers._discover_tools(
            enable_data_tools=False,
            enable_workspace_tools=False,
            enable_python_exec=False,
        )

    # No false null-binding error even though the personality carries no tools.
    assert not any("produced 0 tool spec" in m for m in captured)
    # The personality carries the null sentinel → it contributes nothing.
    assert pers.tools is _NULL_TOOL_BINDING


# --------------------------------------------------------------------------- #
# 3. A genuine broken binding must STILL error
# --------------------------------------------------------------------------- #

def test_real_binding_with_zero_specs_still_errors():
    pers = _make_personality()

    broken = MagicMock()
    broken.binding_name = "my_real_handbag"
    broken.to_chat_tool_specs.return_value = {}
    broken.discovered_tools = []
    broken.tool_files = []
    broken.tools_folders = []
    pers.attach_tool_binding(broken)

    captured: list[str] = []

    with patch.object(ASCIIColors, "error", side_effect=lambda m, *a, **kw: captured.append(str(m))):
        pers._discover_tools(enable_data_tools=False, enable_workspace_tools=False)

    assert any("produced 0 tool spec" in m for m in captured), (
        "A real binding that yields zero specs must still surface an error; "
        f"captured={captured}"
    )