import unittest
import sys
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_artefact import ArtefactType


class MockClient:
    """Mock client for testing artifact operations without an LLM."""
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock"
        self.binding_name = "mock"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 0

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "ok"

    def reset_cancel(self):
        pass


class TestImportConflictResolution(unittest.TestCase):
    """Validates the on_conflict strategies for file import."""

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_conflict_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="conflict_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )
        
        self.file_a = Path(self.tmp_dir) / "README.md"
        self.file_a.write_text("# Original Content\n\nThis is the first file.")
        
        self.file_b = Path(self.tmp_dir) / "README_from_elsewhere.md"
        self.file_b.write_text("# Overwritten Content\n\nThis is the second file.")
        # We will manually pass title="README.md" for file_b to trigger the conflict

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_suffix_strategy(self):
        """
        Scenario: Import file A, then import file B with the same title using on_conflict='suffix'.
        Expected: File B is saved as 'README_1.md'. Both physical files exist.
        """
        self.discussion.import_file(self.file_a, mode="text", title="README.md", on_conflict="suffix")
        
        res_b = self.discussion.import_file(self.file_b, mode="text", title="README.md", on_conflict="suffix")
        art_b = res_b["text_artefact"]
        
        self.assertEqual(art_b["title"], "README_1.md", "Second file should be renamed with _1 suffix")
        
        ws_data_dir = Path(self.discussion.workspace_data_path)
        self.assertTrue((ws_data_dir / "README.md").exists(), "Original file should still exist")
        self.assertTrue((ws_data_dir / "README_1.md").exists(), "Suffixed file should exist")
        
        # Ensure content is distinct
        self.assertIn("Original Content", self.discussion.artefacts.get("README.md")["content"])
        self.assertIn("Overwritten Content", self.discussion.artefacts.get("README_1.md")["content"])

    def test_version_strategy(self):
        """
        Scenario: Import file A, then import file B with the same title using on_conflict='version'.
        Expected: File B overwrites the physical 'README.md', and the DB version is bumped to 2.
        """
        self.discussion.import_file(self.file_a, mode="text", title="README.md", on_conflict="version")
        
        res_b = self.discussion.import_file(self.file_b, mode="text", title="README.md", on_conflict="version")
        art_b = res_b["text_artefact"]
        
        self.assertEqual(art_b["version"], 2, "Version should be bumped to 2")
        self.assertEqual(art_b["title"], "README.md", "Title should remain the same")
        
        ws_data_dir = Path(self.discussion.workspace_data_path)
        self.assertTrue((ws_data_dir / "README.md").exists(), "Physical file should exist")
        
        # The physical file now contains the new content
        content = (ws_data_dir / "README.md").read_text()
        self.assertIn("Overwritten Content", content, "Physical file should contain the new content")
        
        # The DB should reflect the new content
        self.assertIn("Overwritten Content", self.discussion.artefacts.get("README.md")["content"])

    def test_overwrite_strategy(self):
        """
        Scenario: Import file A, then import file B with the same title using on_conflict='overwrite'.
        Expected: File B overwrites the physical 'README.md', and the DB version remains 1.
        """
        self.discussion.import_file(self.file_a, mode="text", title="README.md", on_conflict="overwrite")
        
        res_b = self.discussion.import_file(self.file_b, mode="text", title="README.md", on_conflict="overwrite")
        art_b = res_b["text_artefact"]
        
        self.assertEqual(art_b["version"], 1, "Version should NOT be incremented on overwrite")
        self.assertEqual(art_b["title"], "README.md", "Title should remain the same")
        
        ws_data_dir = Path(self.discussion.workspace_data_path)
        self.assertTrue((ws_data_dir / "README.md").exists(), "Physical file should exist")
        
        # The physical file now contains the new content
        content = (ws_data_dir / "README.md").read_text()
        self.assertIn("Overwritten Content", content, "Physical file should contain the new content")

    def test_replace_strategy(self):
        """
        Scenario: Import file A (creates v1), update it (creates v2), then import file B with on_conflict='replace'.
        Expected: All previous versions are purged. File B becomes the new v1. History is reset.
        """
        # 1. Initial import
        self.discussion.import_file(self.file_a, mode="text", title="README.md", on_conflict="replace")
        art_a = self.discussion.artefacts.get("README.md")
        self.assertEqual(art_a["version"], 1)
        
        # 2. Simulate an update creating v2
        self.discussion.artefacts.update("README.md", new_content="# Updated\n\nv2 content", bump_version=True)
        art_v2 = self.discussion.artefacts.get("README.md")
        self.assertEqual(art_v2["version"], 2, "Precondition: version should be 2 before replace")
        
        # Verify history has 2 entries
        history = self.discussion.artefacts.get_version_history("README.md")
        self.assertEqual(len(history), 2, "Precondition: should have 2 versions in history")
        
        # 3. Execute replace strategy
        res_b = self.discussion.import_file(self.file_b, mode="text", title="README.md", on_conflict="replace")
        art_b = res_b["text_artefact"]
        
        # 4. Assertions
        self.assertEqual(art_b["version"], 1, "Version should be reset to 1 after replace")
        self.assertEqual(art_b["title"], "README.md", "Title should remain the same")
        self.assertIn("Overwritten Content", art_b["content"], "Content should be from file B")
        
        # History should be wiped clean (only 1 version now)
        history_after = self.discussion.artefacts.get_version_history("README.md")
        self.assertEqual(len(history_after), 1, "History should be purged, leaving only 1 version")
        
        # Physical file should contain the new content
        ws_data_dir = Path(self.discussion.workspace_data_path)
        content = (ws_data_dir / "README.md").read_text()
        self.assertIn("Overwritten Content", content, "Physical file should contain the new content")


if __name__ == "__main__":
    unittest.main()
