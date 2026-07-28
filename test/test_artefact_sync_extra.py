import unittest
import tempfile
import shutil
from pathlib import Path

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


class TestArtefactSyncExtra(unittest.TestCase):
    """Tests for verifying no on-disk duplication for artifacts without explicit file extensions."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_artefact_sync_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_sync_extra_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )
        self.ws_path = Path(self.discussion.workspace_data_path)
        self.ws_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def _count_files_in_workspace(self) -> int:
        """Counts the number of files directly in the workspace_data directory."""
        return len([f for f in self.ws_path.iterdir() if f.is_file()])

    def test_scratchpad_creation_no_duplication(self):
        """Create a scratchpad (no extension) and ensure it doesn't duplicate on disk."""
        self.discussion.artefacts.add(
            title="my_scratchpad",
            artefact_type=ArtefactType.SCRATCHPAD,
            content="Initial thoughts",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        # Check that exactly one file exists
        self.assertEqual(self._count_files_in_workspace(), 1, "Scratchpad created duplicate files or no file.")

        # Check it has the exact name and no extension was appended
        expected_file = self.ws_path / "my_scratchpad"
        self.assertTrue(expected_file.exists(), "Scratchpad file with exact name not found.")
        self.assertEqual(expected_file.read_text(encoding="utf-8"), "Initial thoughts")

    def test_skill_import_no_duplication(self):
        """Create a skill (no extension) and ensure it doesn't duplicate on disk."""
        skill_content = "This is a skill definition."
        self.discussion.artefacts.add(
            title="my_custom_skill",
            artefact_type=ArtefactType.SKILL,
            content=skill_content,
            active=True,
            visibility=ArtefactVisibility.FULL,
            description="A test skill",
            category="test"
        )
        self.discussion.commit()

        # Check that exactly one file exists
        self.assertEqual(self._count_files_in_workspace(), 1, "Skill created duplicate files or no file.")

        # Architectural standard: Skills must be saved as Markdown files.
        expected_file = self.ws_path / "my_custom_skill.md"
        self.assertTrue(expected_file.exists(), "Skill file with .md extension not found.")
        self.assertEqual(expected_file.read_text(encoding="utf-8"), skill_content)

    def test_code_artifact_with_extension_no_duplication(self):
        """Create a code artifact (with extension) and ensure it doesn't duplicate."""
        self.discussion.artefacts.add(
            title="main.py",
            artefact_type=ArtefactType.CODE,
            content="print('hello')",
            language="python",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        self.assertEqual(self._count_files_in_workspace(), 1, "Code artifact created duplicate files or no file.")
        expected_file = self.ws_path / "main.py"
        self.assertTrue(expected_file.exists(), "Code file with exact name not found.")

    def test_document_artifact_without_extension_no_duplication(self):
        """Create a document artifact (no extension, defaults to .md) and ensure no stale extensionless file remains."""
        self.discussion.artefacts.add(
            title="README",
            artefact_type=ArtefactType.DOCUMENT,
            content="# Hello World",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        self.assertEqual(self._count_files_in_workspace(), 1, "Document artifact created duplicate files or no file.")
        
        # It should be saved as README.md
        expected_file = self.ws_path / "README.md"
        self.assertTrue(expected_file.exists(), "Document file with .md extension not found.")
        
        # And the extensionless version should NOT exist
        stale_file = self.ws_path / "README"
        self.assertFalse(stale_file.exists(), "Stale extensionless file was created/kept.")

    def test_update_scratchpad_no_duplication(self):
        """Update a scratchpad and ensure it still doesn't duplicate on disk."""
        self.discussion.artefacts.add(
            title="update_scratch_test",
            artefact_type=ArtefactType.SCRATCHPAD,
            content="v1 content",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        self.discussion.artefacts.update(
            title="update_scratch_test",
            new_content="v2 content",
            bump_version=True,
            active=True
        )
        self.discussion.commit()

        self.assertEqual(self._count_files_in_workspace(), 1, "Scratchpad update created duplicate files.")
        expected_file = self.ws_path / "update_scratch_test"
        self.assertTrue(expected_file.exists(), "Updated scratchpad file not found.")
        self.assertEqual(expected_file.read_text(encoding="utf-8"), "v2 content")


if __name__ == "__main__":
    unittest.main()
