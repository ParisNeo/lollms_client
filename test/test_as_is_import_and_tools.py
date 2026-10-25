"""
Regression gate for the Excel toolset (excel_tools.py) and the AS-IS
import-and-tools pipeline.

Locks in the fixes for the "broken Excel file" defect class:
- The save shim installed on the root object must never shadow the REAL
  openpyxl save performed by _save_and_verify.
- A recipe that rebinds `wb = Workbook()` must have its REBOUND workbook
  saved (root-rebind regression).
- Validation is STRUCTURAL ONLY: formula cells (no cached values) must pass.
"""

import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

openpyxl = pytest.importorskip("openpyxl")


from lollms_client.tools_bindings.lcp.default_tools.excel_tools.excel_tools import (
    tool_build_excel,
    tool_read_excel,
)


class _StubDiscussion:
    """Minimal discussion stand-in exposing the workspace contract."""

    def __init__(self, workspace_data_path: str):
        self.workspace_data_path = workspace_data_path


class TestExcelToolBuild(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_excel_tool_test_")
        self.discussion = _StubDiscussion(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_excel_valid_xlsx(self):
        recipe = (
            'ws["A1"] = "Name"\n'
            'ws["B1"] = 10\n'
            'ws["C1"] = "=SUM(B1:B1)"\n'
        )
        result = tool_build_excel(
            file="report.xlsx",
            recipe=recipe,
            discussion_instance=self.discussion,
        )

        self.assertTrue(result.get("success"), f"build failed: {result.get('error')}")
        saved_path = Path(result["workspace_file"])
        self.assertTrue(saved_path.is_file(), "workspace_file does not exist on disk")

        # Container integrity: must be a valid ZIP, never BadZipFile.
        with zipfile.ZipFile(saved_path) as zf:
            self.assertIsNone(zf.testzip())
            self.assertIn("[Content_Types].xml", zf.namelist())

        # OOXML integrity + cell round-trip.
        wb = openpyxl.load_workbook(saved_path)
        ws = wb.active
        self.assertEqual(ws["A1"].value, "Name")
        self.assertEqual(ws["B1"].value, 10)
        self.assertEqual(ws["C1"].value, "=SUM(B1:B1)")

    def test_rebound_wb_is_saved(self):
        recipe = (
            "wb = Workbook()\n"
            "ws = wb.active\n"
            'ws["A1"] = "Rebound"\n'
            'ws["B1"] = 7\n'
            'ws["C1"] = "=SUM(B1:B1)"\n'
            "wb.save()\n"
        )
        result = tool_build_excel(
            file="rebound.xlsx",
            recipe=recipe,
            discussion_instance=self.discussion,
        )

        self.assertTrue(result.get("success"), f"build failed: {result.get('error')}")
        saved_path = Path(result["workspace_file"])
        self.assertTrue(saved_path.is_file())

        wb = openpyxl.load_workbook(saved_path)
        ws = wb.active
        self.assertEqual(ws["A1"].value, "Rebound", "rebound workbook content was lost")
        self.assertEqual(ws["C1"].value, "=SUM(B1:B1)")

    def test_read_excel_inventory(self):
        build_result = tool_build_excel(
            file="inventory_source.xlsx",
            recipe='ws["A1"] = "Name"\nws["B1"] = 10\nws["C1"] = "=SUM(B1:B1)"\n',
            discussion_instance=self.discussion,
        )
        self.assertTrue(build_result.get("success"))

        read_result = tool_read_excel(
            file="inventory_source.xlsx",
            discussion_instance=self.discussion,
        )
        self.assertTrue(read_result.get("success"), f"read failed: {read_result.get('error')}")
        inventory = read_result.get("inventory", "")
        self.assertTrue(inventory, "inventory is empty")
        self.assertIn("Sheet", inventory)


if __name__ == "__main__":
    unittest.main()