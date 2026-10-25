"""
Test suite to verify that ALL active artifacts are properly synced to workspace
before tool execution, enabling the LLM to build artifacts and then call custom tools.
"""

import hashlib
import unittest
from pathlib import Path
import pytest
import shutil
import tempfile


def pytest_importorskip_excel():
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl is not installed")
    import openpyxl
    return openpyxl

# Add src to path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_artefact import ArtefactVisibility

class MockLollmsClient:
    def __init__(self):
        self.debug = False
        self.llm = self
        self.model_name = "test-model"
        self.binding_name = "test-binding"
        self.ai_name = "Assistant"
        
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def count_image_tokens(self, image: str) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"

class TestArtifactSyncForTools(unittest.TestCase):
    """Test artifact synchronization workflow for custom tool execution."""
    
    def setUp(self):
        """Create a fresh discussion for each test."""
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_sync_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_sync_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_workspace_path_verification(self):
        """Verify that the workspace_data_path is correctly resolved."""
        expected_path = Path(self.tmp_workspace) / "workspace_data"
        self.assertEqual(Path(self.discussion.workspace_data_path), expected_path)
        self.assertTrue(expected_path.exists())

    def test_all_active_artifacts_synced(self):
        """Test that active artifacts are physically written to disk."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        self.discussion.artefacts.add(
            title="active_script.py",
            artefact_type="code",
            content="print('hello')",
            language="python",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        sync_ws, sync_files = self.discussion.artefacts.sync_all_active_to_disk()
        
        self.assertIsNotNone(sync_ws)
        expected_file = ws_path / "active_script.py"
        
        # CRITICAL FIX: Normalize paths via resolve() before comparison to handle 
        # OS-specific slash differences (WindowsPath C:/ vs C:\).
        resolved_expected = expected_file.resolve()
        resolved_sync_files = [Path(f).resolve() for f in sync_files]
        
        self.assertIn(resolved_expected, resolved_sync_files)
        
        self.assertTrue(expected_file.exists(), "Active artifact was not written to disk")
        self.assertEqual(expected_file.read_text(), "print('hello')")

    def test_inactive_artifacts_not_synced(self):
        """Test that inactive artifacts are not synced to disk."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        self.discussion.artefacts.add(
            title="inactive_script.py",
            artefact_type="code",
            content="print('inactive')",
            language="python",
            active=False,
            visibility=ArtefactVisibility.TREE_LOCKED
        )
        self.discussion.commit()
        
        expected_file = ws_path / "inactive_script.py"
        
        # CRITICAL FIX: The artefacts.add() might trigger an immediate physical materialization.
        # To accurately test that sync_all_active_to_disk() skips it, we must delete the file first.
        if expected_file.exists():
            expected_file.unlink()
            
        self.assertFalse(expected_file.exists(), "Failed to clean up pre-existing inactive file for test.")
        
        sync_ws, sync_files = self.discussion.artefacts.sync_all_active_to_disk()
        
        resolved_expected = expected_file.resolve()
        resolved_sync_files = [Path(f).resolve() for f in sync_files]
        
        self.assertNotIn(resolved_expected, resolved_sync_files)
        
        # Assert the sync method did not recreate the file
        self.assertFalse(expected_file.exists(), "Inactive artifact was written to disk by sync_all_active_to_disk despite being filtered.")

    def test_artifact_build_then_use_pattern(self):
        """Simulate the LLM building an artifact, then a tool reading it from disk."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        # 1. LLM "builds" the artifact
        self.discussion.artefacts.add(
            title="data_processor.py",
            artefact_type="code",
            content="def process(): return 42",
            language="python",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        # 2. System syncs it to disk before tool execution
        self.discussion.artefacts.sync_all_active_to_disk()
        
        # 3. "Tool" attempts to read it
        file_path = ws_path / "data_processor.py"
        self.assertTrue(file_path.exists())
        
        content = file_path.read_text()
        self.assertIn("def process()", content)

    def test_excel_artifact_sync_binary_integrity(self):
        """Excel artifacts must survive sync_all_active_to_disk() byte-intact.

        XLSX is a binary ZIP container: any text-mode read/write in the sync
        path corrupts it. This test gates the binary-safe sync seam with a
        sha256 comparison over raw bytes.
        """
        openpyxl = pytest_importorskip_excel()
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)

        # 1. Build a REAL xlsx in the workspace (mirrors tool_build_excel output).
        source_path = ws_path / "report.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "Name"
        ws["B1"] = 10
        ws["C1"] = "=SUM(B1:B1)"
        wb.save(str(source_path))
        source_bytes = source_path.read_bytes()
        self.assertTrue(source_bytes[:2] == b"PK", "source file is not a ZIP container")

        # 2. Register it as an active artifact using the suite's existing pattern.
        self.discussion.artefacts.add(
            title="report.xlsx",
            artefact_type="data",
            content="### Data File: report.xlsx",
            active=True,
            visibility=ArtefactVisibility.FULL,
            physical_data=source_bytes,
        )
        self.discussion.commit()

        # 3. Sync and verify byte-identity (binary-safe, never read_text).
        sync_ws, sync_files = self.discussion.artefacts.sync_all_active_to_disk()
        synced_path = ws_path / "report.xlsx"
        self.assertTrue(synced_path.is_file(), "synced xlsx was not written to disk")
        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            hashlib.sha256(synced_path.read_bytes()).hexdigest(),
            "synced xlsx is not byte-identical to the source (binary corruption)",
        )

        # 4. The synced file must remain a loadable workbook.
        reloaded = openpyxl.load_workbook(synced_path)
        self.assertEqual(reloaded.active["A1"].value, "Name")
        self.assertEqual(reloaded.active["C1"].value, "=SUM(B1:B1)")

    def test_cir_file_accessible_to_tools(self):
        """Test that .cir files are synced and accessible."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        self.discussion.artefacts.add(
            title="circuit.cir",
            artefact_type="code",
            content="* Circuit",
            language="text",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        self.discussion.artefacts.sync_all_active_to_disk()
        
        cir_file = ws_path / "circuit.cir"
        self.assertTrue(cir_file.exists(), ".cir artifact was not written to disk")
        self.assertEqual(cir_file.read_text(), "* Circuit")

if __name__ == "__main__":
    unittest.main()
