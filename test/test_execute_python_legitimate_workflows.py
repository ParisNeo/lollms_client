"""
test_execute_python_legitimate_workflows.py

CERTIFICATION: legitimate work can actually be done with these tools.

This suite answers a question no other test file answers:
    "Can an agent use execute_python / execute_python_data_query to perform
     real analytical tasks end-to-end — the way a user would ask — without
     the sandbox falsely rejecting legitimate library calls?"

Coverage axes:
    - safety (what the sandbox BLOCKS):        test_sandbox_python_kernel.py
    - init hygiene (no pipmaster at init):     test_execute_python_data_query_init.py
    - envelope shape & output capture:         test_execute_python_tools_behavior.py
    - LEGITIMATE END-TO-END WORKFLOWS:         this file

If any test in this file fails, the sandbox has over-blocked a legitimate
operation and the tools are silently less useful than advertised. That is a
correctness bug in the sandbox, not a test error.

PHILOSOPHY
    - Each test simulates a plausible user request ("analyze this CSV",
      "join these two files", "produce a chart", "save an xlsx report").
    - Assertions are on OBSERVABLE ARTEFACTS (files on disk with expected
      content) and DELIVERED NUMBERS (in `output`), never on internals.
    - `pytest.importorskip` for optional deps: a minimal env still passes.
    - `monkeypatch.chdir(tmp_path)` per test: workspace isolation.
"""
from __future__ import annotations

import csv
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
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_data_query(workspace, code, **kw):
    """Run data-query code, print diagnostics on failure for fast triage."""
    result = epdq.tool_execute_python_data_query(code=code, env=None, **kw)
    if not result.get("success"):
        raise AssertionError(
            f"data_query failed:\n"
            f"  code=\n{code}\n"
            f"  error=\n{result.get('error')}\n"
            f"  output=\n{result.get('output')}"
        )
    return result


def _run_execute_python(workspace, code, **kw):
    result = ep.tool_execute_python_code(code=code, **kw)
    if not result.get("success"):
        raise AssertionError(
            f"execute_python failed:\n"
            f"  code=\n{code}\n"
            f"  error=\n{result.get('error')}\n"
            f"  output=\n{result.get('output')}"
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 1. ETL — the classic "analyze this sales CSV" request
# ──────────────────────────────────────────────────────────────────────────────

class TestETLWorkflow:
    def test_filter_groupby_aggregate_write_report(self, workspace):
        """Plausible agent task: load sales, keep Q1, group by region, write report.

        Exercises: read_csv (auto-load), boolean filter, groupby().sum(),
        to_csv persistence, summary printed.
        """
        pytest.importorskip("pandas")
        (workspace / "sales.csv").write_text(
            "region,quarter,revenue\n"
            "EU,Q1,100\nEU,Q2,200\n"
            "NA,Q1,300\nNA,Q2,150\n"
            "APAC,Q1,50\nAPAC,Q2,75\n",
            encoding="utf-8",
        )
        code = (
            "q1 = df[df['quarter'] == 'Q1']\n"
            "summary = q1.groupby('region')['revenue'].sum().reset_index()\n"
            "summary.to_csv('q1_by_region.csv', index=False)\n"
            "print('total_q1', int(q1['revenue'].sum()))\n"
            "print('rows_summary', len(summary))\n"
        )
        result = _run_data_query(workspace, code)
        assert "total_q1 450" in result["output"]
        assert "rows_summary 3" in result["output"]

        out_path = workspace / "q1_by_region.csv"
        assert out_path.is_file(), "ETL did not produce q1_by_region.csv"
        rows = list(csv.DictReader(out_path.open(encoding="utf-8")))
        by_region = {r["region"]: int(r["revenue"]) for r in rows}
        assert by_region == {"EU": 100, "NA": 300, "APAC": 50}

    def test_chained_cleaning_pipeline(self, workspace):
        """Tasks agents perform: fill NaN, cast, sort, then persist."""
        pytest.importorskip("pandas")
        (workspace / "events.csv").write_text(
            "user,score\nann,10\nbob,\nbob,20\ncy,15\n", encoding="utf-8"
        )
        code = (
            "df['score'] = df['score'].fillna(0).astype(int)\n"
            "df = df.sort_values('score', ascending=False).reset_index(drop=True)\n"
            "df.to_csv('cleaned.csv', index=False)\n"
            "print('top_user', df.iloc[0]['user'])\n"
            "print('top_score', int(df.iloc[0]['score']))\n"
            "print('total', int(df['score'].sum()))\n"
        )
        result = _run_data_query(workspace, code)
        assert "top_user bob" in result["output"]
        assert "top_score 20" in result["output"]
        assert "total 45" in result["output"]
        assert (workspace / "cleaned.csv").is_file()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-file join — merging datasets (a very common task)
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiFileJoin:
    def test_join_two_csv_files_and_aggregate(self, workspace):
        """Load two files with pd.read_csv, merge on a key, sum a measure."""
        pytest.importorskip("pandas")
        (workspace / "orders.csv").write_text(
            "order_id,customer_id,amount\n"
            "1,10,50\n2,11,75\n3,10,25\n4,12,100\n",
            encoding="utf-8",
        )
        (workspace / "customers.csv").write_text(
            "customer_id,name\n"
            "10,Alice\n11,Bob\n12,Carl\n",
            encoding="utf-8",
        )
        code = (
            "import pandas as pd\n"
            "orders = pd.read_csv('orders.csv')\n"
            "customers = pd.read_csv('customers.csv')\n"
            "merged = orders.merge(customers, on='customer_id')\n"
            "per_customer = merged.groupby('name')['amount'].sum().to_dict()\n"
            "print('alice', int(per_customer.get('Alice', 0)))\n"
            "print('carl', int(per_customer.get('Carl', 0)))\n"
        )
        result = _run_data_query(workspace, code)
        assert "alice 75" in result["output"]
        assert "carl 100" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Excel round-trip — write / read / validate
# ──────────────────────────────────────────────────────────────────────────────

class TestExcelRoundTrip:
    def test_write_and_read_xlsx_through_office_guard(self, workspace):
        """Write an xlsx, re-open it in the same call, verify contents.

        Exercises the office guard's happy path: a valid xlsx must NOT be
        rolled back. This is the critical false-positive check.
        """
        pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame({'product': ['A', 'B', 'C'], 'qty': [3, 5, 7]})\n"
            "df.to_excel('inventory.xlsx', index=False)\n"
            "back = pd.read_excel('inventory.xlsx')\n"
            "print('rows', len(back))\n"
            "print('total', int(back['qty'].sum()))\n"
        )
        result = _run_data_query(workspace, code)
        assert "rows 3" in result["output"]
        assert "total 15" in result["output"]

        xlsx_path = workspace / "inventory.xlsx"
        assert xlsx_path.is_file(), "xlsx was not persisted"
        # Office guard must leave the file intact (it's a valid workbook).
        assert xlsx_path.stat().st_size > 0

    def test_xlsx_report_with_formula_works(self, workspace):
        """Agents often write formula cells (e.g. '=A1*B1'). Must succeed."""
        pytest.importorskip("openpyxl")
        code = (
            "from openpyxl import Workbook\n"
            "wb = Workbook()\n"
            "ws = wb.active\n"
            "ws['A1'] = 4\n"
            "ws['B1'] = 5\n"
            "ws['C1'] = '=A1*B1'\n"
            "wb.save('formula.xlsx')\n"
            "print('saved')\n"
        )
        result = _run_data_query(workspace, code)
        assert "saved" in result["output"]
        assert (workspace / "formula.xlsx").is_file()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Numeric / scientific compute
# ──────────────────────────────────────────────────────────────────────────────

class TestNumericCompute:
    def test_numpy_statistics_on_real_data(self, workspace):
        """Baseline numeric pipeline: mean / std / correlation via numpy."""
        pytest.importorskip("numpy")
        code = (
            "import numpy as np\n"
            "x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])\n"
            "print('mean', round(float(np.mean(x)), 2))\n"
            "print('std', round(float(np.std(x)), 4))\n"
            "print('corr', round(float(np.corrcoef(x, y)[0, 1]), 4))\n"
        )
        result = _run_data_query(workspace, code)
        assert "mean 3.0" in result["output"]
        assert "std 1.4142" in result["output"]
        assert "corr 1.0" in result["output"]

    def test_linear_regression_slope(self, workspace):
        """A common agent task: fit a line, report slope and intercept."""
        pytest.importorskip("numpy")
        code = (
            "import numpy as np\n"
            "x = np.array([0, 1, 2, 3, 4], dtype=float)\n"
            "y = np.array([1, 3, 5, 7, 9], dtype=float)  # y = 2x + 1\n"
            "slope, intercept = np.polyfit(x, y, 1)\n"
            "print('slope', round(float(slope), 4))\n"
            "print('intercept', round(float(intercept), 4))\n"
        )
        result = _run_data_query(workspace, code)
        assert "slope 2.0" in result["output"]
        assert "intercept 1.0" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Plot generation from real data
# ──────────────────────────────────────────────────────────────────────────────

class TestPlotFromRealData:
    def test_plot_renders_and_png_is_plausible(self, workspace):
        """An agent asked to "plot the trend" must get a real PNG."""
        pytest.importorskip("pandas")
        pytest.importorskip("matplotlib")
        (workspace / "trend.csv").write_text(
            "month,sales\nJan,100\nFeb,150\nMar,230\nApr,210\n",
            encoding="utf-8",
        )
        code = (
            "plt.figure(figsize=(6, 4))\n"
            "plt.bar(df['month'], df['sales'])\n"
            "plt.title('Monthly Sales')\n"
            "plt.xlabel('Month')\n"
            "plt.ylabel('Sales')\n"
            "print('chart_ready')\n"
        )
        result = _run_data_query(workspace, code)
        assert "chart_ready" in result["output"]
        pngs = list(workspace.glob("custom_analysis_plot_*.png"))
        assert len(pngs) >= 1, "matplotlib capture failed to produce a PNG"
        # A 6x4 chart is typically 5–30 KB; assert it's a real image (>1 KB).
        assert pngs[0].stat().st_size > 1024, (
            f"PNG is suspiciously small ({pngs[0].stat().st_size} bytes) — "
            "matplotlib may have rendered a blank figure."
        )
        assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Filesystem operations INSIDE the workspace (legitimate, must succeed)
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkspaceFilesystemOperations:
    def test_path_write_text_and_mkdir_work(self, workspace):
        """Path.write_text / Path.mkdir inside the workspace MUST succeed.

        This is the pivotal false-positive test: a sandbox that over-blocks
        workspace-scoped file writes makes half the tool's useful work
        impossible. Inside = allowed, always.
        """
        code = (
            "from pathlib import Path\n"
            "report_dir = Path('reports')\n"
            "report_dir.mkdir(exist_ok=True)\n"
            "(report_dir / 'summary.txt').write_text('analysis complete')\n"
            "print('written', (report_dir / 'summary.txt').read_text())\n"
        )
        result = _run_data_query(workspace, code)
        assert "written analysis complete" in result["output"]
        assert (workspace / "reports" / "summary.txt").read_text() == "analysis complete"

    def test_builtin_open_write_then_read_inside_workspace(self, workspace):
        """Plain `open(...,'w')` inside workspace must work."""
        code = (
            "with open('log.txt', 'w', encoding='utf-8') as f:\n"
            "    f.write('line1\\nline2\\n')\n"
            "with open('log.txt') as f:\n"
            "    content = f.read()\n"
            "print('lines', len(content.strip().splitlines()))\n"
        )
        result = _run_data_query(workspace, code)
        assert "lines 2" in result["output"]
        assert (workspace / "log.txt").read_text(encoding="utf-8").strip() == "line1\nline2"

    def test_os_listdir_of_workspace(self, workspace):
        """Enumerating the workspace must be allowed."""
        (workspace / "a.csv").write_text("x\n1\n", encoding="utf-8")
        (workspace / "b.csv").write_text("y\n2\n", encoding="utf-8")
        code = (
            "import os\n"
            "csvs = sorted(f for f in os.listdir('.') if f.endswith('.csv'))\n"
            "print('found', csvs)\n"
        )
        result = _run_data_query(workspace, code)
        assert "['a.csv', 'b.csv']" in result["output"] or "'a.csv', 'b.csv'" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 7. SQLite — a legitimate local database workflow
# ──────────────────────────────────────────────────────────────────────────────

class TestSqliteWorkflow:
    def test_sqlite_create_table_insert_query(self, workspace):
        """A standard agent task: persist results into a local sqlite db."""
        code = (
            "import sqlite3\n"
            "conn = sqlite3.connect('results.db')\n"
            "conn.execute('CREATE TABLE m (k TEXT, v INT)')\n"
            "for k, v in [('x', 1), ('y', 2), ('z', 3)]:\n"
            "    conn.execute('INSERT INTO m VALUES (?, ?)', (k, v))\n"
            "conn.commit()\n"
            "total = conn.execute('SELECT SUM(v) FROM m').fetchone()[0]\n"
            "print('total', total)\n"
            "conn.close()\n"
        )
        result = _run_data_query(workspace, code)
        assert "total 6" in result["output"]
        assert (workspace / "results.db").is_file()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Script-from-disk workflow
# ──────────────────────────────────────────────────────────────────────────────

class TestScriptFromDisk:
    def test_data_query_runs_a_py_file_from_workspace(self, workspace):
        """An agent may save a script and re-run it. `code='script.py'` must work."""
        pytest.importorskip("pandas")
        script = workspace / "analyze.py"
        script.write_text(
            "import pandas as pd\n"
            "df = pd.DataFrame({'v': [10, 20, 30]})\n"
            "print('sum', int(df['v'].sum()))\n",
            encoding="utf-8",
        )
        result = _run_data_query(workspace, str(script))
        assert "sum 60" in result["output"]

    def test_execute_python_runs_a_py_file_from_workspace(self, workspace):
        script = workspace / "hello.py"
        script.write_text("print('hello-from-script')\n", encoding="utf-8")
        result = _run_execute_python(workspace, str(script))
        assert "hello-from-script" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 9. execute_python (standard profile) does real work too
# ──────────────────────────────────────────────────────────────────────────────

class TestExecutePythonLegitimateWork:
    def test_standard_tool_runs_full_analysis_script(self, workspace):
        """The standard profile must ALSO support end-to-end analysis."""
        pytest.importorskip("pandas")
        (workspace / "scores.csv").write_text(
            "name,score\nann,88\nbob,72\ncy,95\n", encoding="utf-8"
        )
        code = (
            "import pandas as pd\n"
            "df = pd.read_csv('scores.csv')\n"
            "df['grade'] = df['score'].apply(lambda s: 'A' if s >= 90 else 'B')\n"
            "df.to_csv('graded.csv', index=False)\n"
            "print('top', df.sort_values('score', ascending=False).iloc[0]['name'])\n"
            "print('a_count', int((df['grade'] == 'A').sum()))\n"
        )
        result = _run_execute_python(workspace, code)
        assert "top cy" in result["output"]
        assert "a_count 1" in result["output"]
        assert (workspace / "graded.csv").is_file()

    def test_standard_tool_writes_files_inside_workspace(self, workspace):
        code = (
            "from pathlib import Path\n"
            "Path('notes').mkdir(exist_ok=True)\n"
            "Path('notes/todo.txt').write_text('finish report')\n"
            "print('done')\n"
        )
        result = _run_execute_python(workspace, code)
        assert "done" in result["output"]
        assert (workspace / "notes" / "todo.txt").read_text() == "finish report"


# ──────────────────────────────────────────────────────────────────────────────
# 10. Cross-tool consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossToolConsistency:
    def test_same_computation_both_tools_agree(self, workspace):
        """Running the same numerical task on both tools must yield the same answer."""
        pytest.importorskip("numpy")
        code = (
            "import numpy as np\n"
            "vals = np.array([3.0, 5.0, 7.0, 9.0])\n"
            "print('mean', float(np.mean(vals)))\n"
        )
        r_dq = _run_data_query(workspace, code)
        r_ep = _run_execute_python(workspace, code)
        assert "mean 6.0" in r_dq["output"]
        assert "mean 6.0" in r_ep["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 11. Idempotence: repeated calls see prior persistent state
# ──────────────────────────────────────────────────────────────────────────────

class TestIdempotenceAndContinuity:
    def test_second_call_sees_first_call_persisted_file(self, workspace):
        """Real agents iterate: read → write → re-read in a second call."""
        pytest.importorskip("pandas")
        (workspace / "base.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        _run_data_query(
            workspace,
            "df['s'] = df['a'] + df['b']\ndf.to_csv('step1.csv', index=False)",
        )
        assert (workspace / "step1.csv").is_file()
        result = _run_data_query(
            workspace,
            "import pandas as pd\n"
            "d = pd.read_csv('step1.csv')\n"
            "print('col_s_sum', int(d['s'].sum()))\n",
        )
        assert "col_s_sum 10" in result["output"]

    def test_repeated_identical_call_is_stable(self, workspace):
        """Same benign call twice must produce the same output and succeed."""
        code = "print('stable-result', sum([1, 2, 3]))"
        r1 = _run_data_query(workspace, code)
        r2 = _run_data_query(workspace, code)
        assert "stable-result 6" in r1["output"]
        assert "stable-result 6" in r2["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 12. Unicode / UTF-8 — non-English work must succeed
# ──────────────────────────────────────────────────────────────────────────────

class TestUnicodeWorkflows:
    def test_utf8_data_roundtrips_through_csv(self, workspace):
        """Agents must support accented and non-Latin content."""
        pytest.importorskip("pandas")
        (workspace / "users.csv").write_text(
            "name,city\nÉmile,Montréal\n李雷,北京\n",
            encoding="utf-8",
        )
        code = (
            "names = list(df['name'])\n"
            "print('names', names)\n"
            "df['greeting'] = ['hi-' + n for n in names]\n"
            "df.to_csv('greeted.csv', index=False, encoding='utf-8-sig')\n"
        )
        result = _run_data_query(workspace, code)
        assert "Émile" in result["output"]
        assert "李雷" in result["output"]
        assert (workspace / "greeted.csv").is_file()

    def test_nonascii_print_survives(self, workspace):
        result = _run_data_query(workspace, "print('café ☕ ✔')")
        assert "café" in result["output"]
        assert "☕" in result["output"]


# ──────────────────────────────────────────────────────────────────────────────
# 13. Performance sanity — real work must finish in reasonable time
# ──────────────────────────────────────────────────────────────────────────────

class TestPerformanceSanity:
    def test_moderate_dataframe_operation_completes_quickly(self, workspace):
        """A realistic 10k-row pipeline must finish well under the hard timeout.

        This guards against sandbox guard overhead making legitimate work
        unusable. 10k rows × a handful of column ops should be <10s on any
        modern machine; we assert <30s to leave CI headroom.
        """
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        import time
        code = (
            "import numpy as np\n"
            "rng = np.random.default_rng(0)\n"
            "df = pd.DataFrame({\n"
            "    'k': rng.integers(0, 5, 10_000),\n"
            "    'v': rng.random(10_000),\n"
            "})\n"
            "df['bucket'] = (df['v'] * 10).astype(int)\n"
            "summary = df.groupby(['k', 'bucket'])['v'].agg(['sum', 'count']).reset_index()\n"
            "print('summary_rows', len(summary))\n"
        )
        t0 = time.perf_counter()
        result = _run_data_query(workspace, code)
        elapsed = time.perf_counter() - t0
        assert "summary_rows" in result["output"]
        assert elapsed < 30.0, (
            f"10k-row pipeline took {elapsed:.1f}s — sandbox guard overhead "
            "is degrading legitimate work."
        )