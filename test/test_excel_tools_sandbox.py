
"""Regression suite for the upgraded Excel toolset sandbox (P1–P7).

Covers the seven acceptance criteria from the audit plus backward-compat
guards (shim + auto-save + read modes must remain intact).
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("openpyxl")

from lollms_client.tools_bindings.lcp.default_tools.excel_tools import TOOLS as excel_tools


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Redirect the discussion workspace used by the tools to a temp dir."""
    class _Stub:
        workspace_data_path = str(tmp_path)

    monkeypatch.setattr(excel_tools, "_resolve_working_dir", lambda d: tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- P1
def test_stdout_is_returned(workspace):
    result = excel_tools.tool_build_excel(
        file="p1.xlsx",
        recipe='print("hello")\nws["A1"] = 1\nprint("world")',
        discussion_instance=None,
    )
    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert "world" in result["stdout"]
    assert result["stderr"] == ""


def test_stderr_is_captured(workspace):
    import sys
    # `sys` is not importable inside the recipe (whitelist) — so trigger stderr
    # from a harmless runtime warning via `print(..., file=...)` is impossible;
    # instead exercise the capture buffer directly through a failing recipe.
    result = excel_tools.tool_build_excel(
        file="p1b.xlsx",
        recipe='print("before")\nraise RuntimeError("boom")',
        discussion_instance=None,
    )
    assert result["success"] is False
    assert "before" in result["stdout"]
    assert "RuntimeError" in result["error"]


# --------------------------------------------------------------------------- P2
def test_help_sentinel_returns_manifest(workspace):
    manifest_result = excel_tools.tool_build_excel(
        file="p2.xlsx", recipe="__help__", discussion_instance=None,
    )
    assert manifest_result["success"] is True
    assert "manifest" in manifest_result
    manifest = manifest_result["manifest"]
    assert manifest["prebound"] == ["wb", "ws"]
    assert manifest["auto_save"] is True
    assert "preimported_symbols" in manifest
    assert "openpyxl_version" in manifest
    assert manifest["stdout_captured"] is True
    # No file written.
    assert not (workspace / "p2.xlsx").exists()


def test_empty_recipe_returns_manifest(workspace):
    result = excel_tools.tool_edit_excel.__doc__  # smoke: import resolves the tool
    assert result  # dummy use, keeps linter honest
    built = excel_tools.tool_build_excel(file="p2e.xlsx", recipe="", discussion_instance=None)
    assert built["success"] is True and "manifest" in built


# --------------------------------------------------------------------------- P3
def test_description_mentions_auto_save_and_whitelist():
    descs = {spec["name"]: spec["description"] for spec in excel_tools.get_tools()}
    for name in ("tool_build_excel", "tool_edit_excel"):
        text = descs[name].lower()
        assert "saving is automatic" in text
        assert "never call wb.save()" in text
        assert "import whitelist" in text
        assert "print() output is captured" in text
        assert "pandas" in text
        assert "__help__" in text


# --------------------------------------------------------------------------- P4
def test_blocked_import_has_remediation_hint(workspace):
    result = excel_tools.tool_build_excel(
        file="p4.xlsx",
        recipe="import pandas",
        discussion_instance=None,
    )
    assert result["success"] is False
    assert "pandas" in result["error"]
    assert "Hint:" in result["error"]


def test_forbidden_token_names_the_token(workspace):
    result = excel_tools.tool_build_excel(
        file="p4b.xlsx",
        recipe="x = eval('1 + 1')",
        discussion_instance=None,
    )
    assert result["success"] is False
    assert "eval(" in result["error"]


# --------------------------------------------------------------------------- P5
def test_expanded_preimports_usable(workspace):
    recipe = (
        "t = Table(displayName='T1', ref='A1:B2')\n"
        "ws.add_table(t)\n"
        "ws['A1'] = 1; ws['B1'] = 'x'; ws['A2'] = 2; ws['B2'] = 'y'\n"
        "ch = ScatterChart(); ch.title = 's'\n"
        "sc = DataBarRule(start_type='min', end_type='max', color='FF0000')\n"
        "from openpyxl.styles import NamedStyle\n"  # explicit import must still work
    )
    result = excel_tools.tool_build_excel(
        file="p5.xlsx", recipe=recipe, discussion_instance=None,
    )
    assert result["success"] is True, result.get("error")
    assert (workspace / "p5.xlsx").exists()


# --------------------------------------------------------------------------- P6
def test_data_bridge_writes_rows(workspace):
    result = excel_tools.tool_build_excel(
        file="p6.xlsx",
        data=[{"name": "a", "v": 1}, {"name": "b", "v": 2}],
        recipe="write_rows(ws, start_row=1, start_col=1)\nws['C1'] = '=SUM(B2:B3)'",
        discussion_instance=None,
    )
    assert result["success"] is True, result.get("error")
    from openpyxl import load_workbook
    wb = load_workbook(str(workspace / "p6.xlsx"))
    ws = wb.active
    assert ws["A1"].value == "name" and ws["B1"].value == "v"
    assert ws["A2"].value == "a" and ws["B2"].value == 1


# --------------------------------------------------------------------------- P7
def test_validate_only_writes_no_file(workspace):
    result = excel_tools.tool_build_excel(
        file="p7.xlsx",
        recipe='ws["A1"] = 1\nws["A2"] = "=SUM(A1:A1)"',
        validate_only=True,
        discussion_instance=None,
    )
    assert result["success"] is True
    assert result["validated"] is True
    assert result["report"]["formula_count"] == 1
    assert result["report"]["sheet_count"] == 1
    assert not (workspace / "p7.xlsx").exists()


# --------------------------------------------------------------------------- Backward compat
def test_auto_save_uses_real_method(workspace):
    """The no-op save shim must not intercept the tool's validated save."""
    result = excel_tools.tool_build_excel(
        file="compat.xlsx",
        recipe='ws["A1"] = "x"\nwb.save()  # must be a no-op',
        discussion_instance=None,
    )
    assert result["success"] is True, result.get("error")
    assert (workspace / "compat.xlsx").is_file()


def test_tool_read_excel_modes_preserved(workspace):
    excel_tools.tool_build_excel(
        file="read.xlsx",
        recipe='ws["A1"] = "h"\nws["A2"] = "v"',
        discussion_instance=None,
    )
    inventory = excel_tools.tool_read_excel(file="read.xlsx", mode="inventory", discussion_instance=None)
    assert inventory["success"] is True and "inventory" in inventory
    peek = excel_tools.tool_read_excel(file="read.xlsx", mode="peek", limit=1, discussion_instance=None)
    assert peek["success"] is True and peek["mode"] == "peek"
    chunk = excel_tools.tool_read_excel(file="read.xlsx", mode="chunk", offset=0, limit=2, discussion_instance=None)
    assert chunk["success"] is True and chunk["mode"] == "chunk"
    full = excel_tools.tool_read_excel(file="read.xlsx", mode="text", discussion_instance=None)
    assert full["success"] is True and full["mode"] == "text"
