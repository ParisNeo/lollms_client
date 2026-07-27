import unittest
import tempfile
import shutil
from pathlib import Path

# Add src to path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_artefact import ArtefactType, ArtefactVisibility


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


class TestArtefactMdDuplication(unittest.TestCase):
    """Test to ensure markdown artifacts are not written twice (with and without .md)."""
    
    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_md_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_md_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_no_duplicate_md_file(self):
        """Verifies that creating a document artifact named 'README' produces only 'README.md'."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Simulate creating a stale, extensionless file first (mimicking old behavior or external sync)
        stale_file = ws_path / "README"
        stale_file.write_text("stale content")
        self.assertTrue(stale_file.exists(), "Failed to create stale file for test setup.")
        
        # 2. Add the artifact through the manager
        self.discussion.artefacts.add(
            title="README",
            artefact_type=ArtefactType.DOCUMENT,
            content="# Hello World",
            language="markdown",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        # 3. Sync to disk
        self.discussion.artefacts.sync_all_active_to_disk()
        
        # 4. Assertions
        expected_file = ws_path / "README.md"
        unexpected_file = ws_path / "README"
        
        self.assertTrue(expected_file.exists(), "README.md was not written to disk.")
        self.assertFalse(unexpected_file.exists(), "Stale README file was not cleaned up.")
        self.assertEqual(expected_file.read_text(), "# Hello World")


if __name__ == "__main__":
    unittest.main()
