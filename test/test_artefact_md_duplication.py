import unittest
from pathlib import Path
import tempfile
import shutil
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


class TestArtefactMdDuplication(unittest.TestCase):
    """Test suite to verify that importing skills/documents without extensions does not create duplicate files."""
    
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

    def test_skill_import_no_duplicate(self):
        """Test that importing a skill titled 'SKILL' creates only 'SKILL.md' on disk."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        self.discussion.artefacts.add(
            title="SKILL",
            artefact_type="skill",
            content="Skill content",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        self.discussion.artefacts.sync_all_active_to_disk()
        
        expected_file = ws_path / "SKILL.md"
        stale_file = ws_path / "SKILL"
        
        self.assertTrue(expected_file.exists(), "Active artifact was not written to disk with .md extension")
        self.assertFalse(stale_file.exists(), "Stale, extensionless duplicate file was created!")
        self.assertEqual(expected_file.read_text(), "Skill content")

    def test_document_import_no_duplicate(self):
        """Test that importing a document titled 'README' creates only 'README.md' on disk."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        self.discussion.artefacts.add(
            title="README",
            artefact_type="document",
            content="This is a readme.",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        self.discussion.artefacts.sync_all_active_to_disk()
        
        expected_file = ws_path / "README.md"
        stale_file = ws_path / "README"
        
        self.assertTrue(expected_file.exists(), "Active artifact was not written to disk with .md extension")
        self.assertFalse(stale_file.exists(), "Stale, extensionless duplicate file was created!")
        self.assertEqual(expected_file.read_text(), "This is a readme.")

    def test_update_skill_no_duplicate(self):
        """Test that updating a skill does not create an extensionless stale file."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        self.discussion.artefacts.add(
            title="SKILL",
            artefact_type="skill",
            content="Skill content v1",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        self.discussion.artefacts.update(
            title="SKILL",
            new_content="Skill content v2",
            bump_version=True
        )
        self.discussion.commit()
        
        self.discussion.artefacts.sync_all_active_to_disk()
        
        expected_file = ws_path / "SKILL.md"
        stale_file = ws_path / "SKILL"
        
        self.assertTrue(expected_file.exists(), "Updated artifact was not written to disk with .md extension")
        self.assertFalse(stale_file.exists(), "Stale, extensionless duplicate file was created during update!")
        self.assertEqual(expected_file.read_text(), "Skill content v2")

    def test_simulated_stale_file_cleanup(self):
        """Test that the cleanup logic actively removes an existing stale file during sync."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        # Simulate a pre-existing stale file (e.g., from an older version of the library)
        stale_file = ws_path / "README"
        stale_file.write_text("stale content")
        self.assertTrue(stale_file.exists())
        
        self.discussion.artefacts.add(
            title="README",
            artefact_type="document",
            content="Fresh content",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        self.discussion.artefacts.sync_all_active_to_disk()
        
        expected_file = ws_path / "README.md"
        
        self.assertTrue(expected_file.exists(), "Active artifact was not written to disk with .md extension")
        self.assertFalse(stale_file.exists(), "Stale, extensionless duplicate file was NOT cleaned up during sync!")
        self.assertEqual(expected_file.read_text(), "Fresh content")

if __name__ == "__main__":
    unittest.main()
