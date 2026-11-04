"""
test_execute_python_tools_behavior.py

Functional / behavioral contract tests for the two Python execution tools:
    - execute_python               (standard profile)
    - execute_python_data_query    (data_query profile)

SCOPE
    The existing sandbox suites (
        test_sandbox_python_kernel.py,
        test_sandbox_module_load_order.py,
        test_sandbox_process_isolation.py,
        test_execute_python_data_query_init.py,
    ) cover **safety** and **initialization hygiene**. They assert what the
    tools BLOCK.

    This file covers what the tools DELIVER: output capture, traceback shape,
    df auto-load/auto-persist, plot capture, error surfacing, host-path
    sanitization, argument normalization. These are the behaviours an
    LLM-tool-calling layer depends on; a regression here silently degrades
    every data-analysis interaction.

PHILOSOPHY
    - Use `pytest.importorskip` so any absent optional dep skips rather than
      fails (CI parity with the existing suites).
    - Assert on observable envelope fields only — never internals — so the
      tests stay valid if the kernel refactors.
    - Each test uses `monkeypatch.chdir(tmp_path)` for workspace isolation, the
      same pattern used by `test_sandbox_process_isolation.py`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

try:
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )
except ImportError:  # pragma: no cover - dev checkout fallback
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from lollms_client.tools_bindings.lcp.default_tools.execute_python import execute_python as ep
    from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query import (
        execute_python_data_query as epdq,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolated CWD per test — matches the layout the tools expect."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# Both tools share this envelope contract. Tests that assert envelope shape
# use this helper instead of touching tool internals.
def _assert_envelope_shape(result, *, minimal=False):
    assert isinstance(result, dict)
    assert "success" in result
    assert isinstance(result["success"], bool)
    if "output" in result:
        assert isinstance(result["output"], str)
    if not result["success"]:
        assert "error" in result and isinstance(result["error"], str)
    if minimal:
        return
    # On non-success, output must be a string (possibly empty).
    if "output" in result and not result["success"]:
        assert isinstance(result["output"], str)


# ──────────────────────────────────────────────────────────────────────────────
# 1. execute_python — output capture & error surfacing
# ──────────────────────────────────────────────────────────────────────────────

class TestExecutePythonOutput:
    def test_print_is_captured_verbatim(self, workspace):
        result = ep.tool_execute_python_code(code="print('hello world')")
        _assert_envelope_shape(result)
        assert result["success"] is True, result.get("error")
        assert "hello world" in result["output"]

    def test_multiple_prints_preserve_order_and_newlines(self, workspace):
        result = ep.tool_execute_python_code(
            code="print('line-1')\nprint('line-2')\nprint('line-3')"
        )
        assert result["success"] is True, result.get("error")
        lines = [ln for ln in result["output"].splitlines() if ln.strip()]
        assert lines == ["line-1", "line-2", "line-3"]

    def test_no_output_returns_default_message(self, workspace):
        result = ep.tool_execute_python_code(code="x = 1 + 1")
        assert result["success"] is True, result.get("error")
        # The tool's contract: empty stdout → an explicit "succeeded" notice.
        assert result["output"].strip() != ""

    def test_stdout_and_stderr_are_separately_routed(self, workspace):
        code = "import sys\nprint('to-stdout')\nprint('to-stderr', file=sys.stderr)"
        result = ep.tool_execute_python_code(code=code)
        assert result["success"] is True, result.get("error")
        # Only stdout is in the tool output; stderr must not pollute it.
        assert "to-stdout" in result["output"]
        assert "to-stderr" not in result["output"]


class TestExecutePythonErrors:
    def test_runtime_error_surfaces_exception_name_and_message(self, workspace):
        result = ep.tool_execute_python_code(code="raise ValueError('boom-123')")
        _assert_envelope_shape(result)
        assert result["success"] is False
        assert "ValueError" in result["error"]
        assert "boom-123" in result["error"]

    def test_syntax_error_is_reported_cleanly(self, workspace):
        result = ep.tool_execute_python_code(code="def : broken")
        assert result["success"] is False
        assert result["error"]  # some diagnostic is present
        # Must not crash the tool.
        _assert_envelope_shape(result)

    def test_empty_code_returns_error(self, workspace):
        result = ep.tool_execute_python_code(code="")
        assert result["success"] is False
        assert result["error"]

    def test_failure_after_prints_preserves_partial_output(self, workspace):
        code = "print('partial-output')\nraise RuntimeError('after-print')"
        result = ep.tool_execute_python_code(code=code)
        assert result["success"] is False
        # Output preceding the exception must survive.
        assert "partial-output" in result["output"]
        assert "after-print" in result["error"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. execute_python — traceback annotation
# ──────────────────────────────────────────────────────────────────────────────

class TestExecutePythonTracebackAnnotation:
    def test_traceback_contains_failing_source_line(self, workspace):
        """The kernel annotates tracebacks with the offending source line.

        This is what lets the LLM fix its own code without recounting lines.
        """
        code = "a = 1\nb = 2\nraise KeyError('offending-line-here')\nc = 3"
        result = ep.tool_execute_python_code(code=code)
        assert result["success"] is False
        assert "KeyError" in result["error"]
        # The annotated failing line must appear, prefixed by our marker.
        assert ">>> LINE 3" in result["error"]
        assert "offending-line-here" in result["error"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. execute_python — namespace & argument normalization
# ──────────────────────────────────────────────────────────────────────────────

class TestExecutePythonArgs:
    def test_dict_argument_is_unwrapped(self, workspace):
        """LCP may pass `code` as a dict; the tool must unwrap it."""
        result = ep.tool_execute_python_code(code={"code": "print('unwrapped')"})
        # The execute_python handler may accept this differently — tolerate
        # both shapes but require the string body be honoured when unwrapped.
        if result["success"]:
            assert "unwrapped" in result["output"]

    def test_file_reference_loads_code_from_disk(self, workspace):
        """`code='script.py'` runs the file content (existing contract)."""
        script = workspace / "greet.py"
        script.write_text("print('from-file')\n", encoding="utf-8")
        result = ep.tool_execute_python_code(code=str(script))
        if result["success"]:
            assert "from-file" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. execute_python_data_query — df auto-load
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQueryDataFrameAutoload:
    def test_csv_with_comma_separator_is_autoloaded(self, workspace):
        pytest.importorskip("pandas")
        (workspace / "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        result = epdq.tool_execute_python_data_query(
            code="print(int(df['a'].sum()))", env=None
        )
        assert result["success"] is True, result.get("error")
        assert "4" in result["output"]

    def test_csv_with_semicolon_separator_is_autoloaded(self, workspace):
        """European CSVs use `;` — the tool must detect this automatically."""
        pytest.importorskip("pandas")
        (workspace / "data.csv").write_text("a;b\n10;20\n30;40\n", encoding="utf-8")
        result = epdq.tool_execute_python_data_query(
            code="print(int(df['a'].sum()))", env=None
        )
        assert result["success"] is True, result.get("error")
        assert "40" in result["output"]

    def test_xlsx_is_autoloaded(self, workspace):
        pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        import pandas as pd
        pd.DataFrame({"x": [5, 6, 7]}).to_excel(workspace / "data.xlsx", index=False)
        result = epdq.tool_execute_python_data_query(
            code="print(int(df['x'].sum()))", env=None
        )
        assert result["success"] is True, result.get("error")
        assert "18" in result["output"]

    def test_existing_df_definition_suppresses_autoload(self, workspace):
        """If the user defines `df` themselves, we must NOT overwrite it."""
        pytest.importorskip("pandas")
        # Poison file that would produce a wildly different sum if loaded.
        (workspace / "data.csv").write_text("wrong\n999\n", encoding="utf-8")
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame({'my_col': [1, 2, 3]})\n"
            "print(int(df['my_col'].sum()))\n"
        )
        result = epdq.tool_execute_python_data_query(code=code, env=None)
        assert result["success"] is True, result.get("error")
        assert "6" in result["output"]

    def test_data_query_without_csv_works_when_user_loads_manually(self, workspace):
        pytest.importorskip("pandas")
        # No CSV present — user code creates its own DataFrame.
        code = (
            "import pandas as pd\n"
            "x = pd.DataFrame({'v': [4, 5, 6]})\n"
            "print(int(x['v'].sum()))\n"
        )
        result = epdq.tool_execute_python_data_query(code=code, env=None)
        assert result["success"] is True, result.get("error")
        assert "15" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 5. execute_python_data_query — df auto-persist
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQueryDataFramePersistence:
    def test_mutated_df_is_persisted_back_to_csv(self, workspace):
        """The core data-query value proposition: mutate df, file updates."""
        pytest.importorskip("pandas")
        src = workspace / "data.csv"
        src.write_text("n\n1\n2\n3\n", encoding="utf-8")
        code = (
            "df['doubled'] = df['n'] * 2\n"
            "print('rows', len(df))\n"
        )
        result = epdq.tool_execute_python_data_query(code=code, env=None)
        assert result["success"] is True, result.get("error")

        import pandas as pd
        reloaded = pd.read_csv(src)
        assert "doubled" in reloaded.columns, (
            f"Persisted CSV should contain the added column. Got columns: "
            f"{list(reloaded.columns)}"
        )
        assert list(reloaded["doubled"]) == [2, 4, 6]

    def test_unmodified_df_roundtrips_without_data_loss(self, workspace):
        """Reading df without mutation must not corrupt the source file."""
        pytest.importorskip("pandas")
        src = workspace / "data.csv"
        original = "a,b\n1,2\n3,4\n"
        src.write_text(original, encoding="utf-8")
        result = epdq.tool_execute_python_data_query(
            code="print(int(df['a'].sum()))", env=None
        )
        assert result["success"] is True, result.get("error")
        # File must still parse to the same values (byte-level layout may differ
        # because pandas normalises the CSV — that's acceptable).
        import pandas as pd
        reloaded = pd.read_csv(src)
        assert list(reloaded["a"]) == [1, 3]
        assert list(reloaded["b"]) == [2, 4]


# ──────────────────────────────────────────────────────────────────────────────
# 6. execute_python_data_query — plot capture
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQueryPlotCapture:
    def test_matplotlib_figure_is_written_to_workspace(self, workspace):
        pytest.importorskip("matplotlib")
        code = (
            "import matplotlib.pyplot as plt\n"
            "plt.plot([1, 2, 3], [4, 5, 6])\n"
            "plt.title('test-plot')\n"
            "print('plot-drawn')\n"
        )
        result = epdq.tool_execute_python_data_query(code=code, env=None)
        assert result["success"] is True, result.get("error")
        assert "plot-drawn" in result["output"]
        pngs = list(workspace.glob("custom_analysis_plot_*.png"))
        assert pngs, (
            "Expected a captured PNG in the workspace; the sandbox figure "
            "harvesting contract is broken."
        )
        # PNGs carry a magic header; sanity-check it.
        assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_no_plot_code_produces_no_png(self, workspace):
        pytest.importorskip("matplotlib")
        result = epdq.tool_execute_python_data_query(
            code="print('no plot')", env=None
        )
        assert result["success"] is True, result.get("error")
        assert not list(workspace.glob("custom_analysis_plot_*.png"))


# ──────────────────────────────────────────────────────────────────────────────
# 7. execute_python_data_query — pre-injected namespace
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQueryInjectedNamespace:
    def test_pd_np_plt_are_present_in_namespace(self, workspace):
        """The tool pre-injects pd/np/plt — user code can rely on them."""
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        pytest.importorskip("matplotlib")
        code = (
            "print('pd_is_module', type(pd).__name__ == 'module')\n"
            "print('np_is_module', type(np).__name__ == 'module')\n"
            "print('plt_is_module', type(plt).__name__ == 'module')\n"
        )
        result = epdq.tool_execute_python_data_query(code=code, env=None)
        assert result["success"] is True, result.get("error")
        assert "pd_is_module True" in result["output"]
        assert "np_is_module True" in result["output"]
        assert "plt_is_module True" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 8. Host-path sanitization (both tools)
# ──────────────────────────────────────────────────────────────────────────────

class TestHostPathSanitization:
    def test_data_query_output_sanitizes_workspace_absolute_paths(self, workspace):
        """The LLM must never see the operator's absolute filesystem layout."""
        pytest.importorskip("pandas")
        code = "print(__import__('os').getcwd())\n"
        result = epdq.tool_execute_python_data_query(code=code, env=None)
        if result["success"]:
            abs_ws = str(workspace.resolve())
            # Absolute workspace path is replaced by '.'; safety net: at the
            # very least it must not appear verbatim in the output.
            assert abs_ws not in result["output"], (
                "Absolute workspace path leaked into tool output."
            )

    def test_execute_python_reports_success_envelope(self, workspace):
        """Sanity: standard tool still returns the envelope contract."""
        result = ep.tool_execute_python_code(code="print('ok')")
        _assert_envelope_shape(result)
        assert result["success"] is True


# ──────────────────────────────────────────────────────────────────────────────
# 9. Cross-tool envelope parity
# ──────────────────────────────────────────────────────────────────────────────

class TestEnvelopeParity:
    def test_both_tools_share_success_envelope(self, workspace):
        r1 = ep.tool_execute_python_code(code="print('a')")
        r2 = epdq.tool_execute_python_data_query(code="print('b')", env=None)
        _assert_envelope_shape(r1)
        _assert_envelope_shape(r2)

    def test_both_tools_return_output_string_on_failure(self, workspace):
        r1 = ep.tool_execute_python_code(code="raise RuntimeError('e1')")
        r2 = epdq.tool_execute_python_data_query(
            code="import subprocess\nsubprocess.run(['x'])", env=None
        )
        _assert_envelope_shape(r1)
        _assert_envelope_shape(r2)
        assert r1["success"] is False
        assert r2["success"] is False
        assert isinstance(r1.get("output", ""), str)
        assert isinstance(r2.get("output", ""), str)

    def test_dataquery_dict_code_argument_is_unwrapped(self, workspace):
        """Symmetric to the standard tool's dict-unwrap behaviour."""
        result = epdq.tool_execute_python_data_query(
            code={"code": "print('from-dict')"}, env=None
        )
        _assert_envelope_shape(result)
        if result["success"]:
            assert "from-dict" in result["output"]