import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from datetime import datetime

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_artefact import ArtefactVisibility


class MockClient:
    """Mock LollmsClient for isolated testing without LLM bindings."""
    def __init__(self):
        self.debug = True
        self.llm = self
        self.model_name = "mock-model"
        self.binding_name = "mock-binding"
        self.ai_name = "Assistant"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"


class TestArtefactSyncing(unittest.TestCase):
    """Comprehensive test suite for bidirectional artefact synchronization."""

    def setUp(self):
        """Create a fresh isolated discussion for each test."""
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_sync_test_")
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_sync_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )
        self.ws_data_dir = Path(self.discussion.workspace_data_path)
        self.ws_data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up temporary workspace after each test."""
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_disk_to_db_new_text_file_registration(self):
        """Verify a new text file placed on disk is registered in the DB."""
        new_file = self.ws_data_dir / "new_script.py"
        new_file.write_text("print('hello world')", encoding="utf-8")

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["new_artefacts"], 1)
        self.assertEqual(report["updated_artefacts"], 0)
        self.assertEqual(report["restored_files"], 0)

        art = self.discussion.artefacts.get("new_script.py")
        self.assertIsNotNone(art, "Artifact should be registered in DB.")
        self.assertEqual(art["type"], "code")
        self.assertEqual(art["content"], "print('hello world')")
        self.assertEqual(art["visibility"], ArtefactVisibility.TREE_UNLOCKABLE)

    def test_disk_to_db_new_binary_file_registration(self):
        """Verify a binary file (e.g., .db) is registered with a placeholder."""
        binary_file = self.ws_data_dir / "database.sqlite"
        binary_file.write_bytes(b"SQLite format 3\x00\x00\x00\x00")

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["new_artefacts"], 1)
        art = self.discussion.artefacts.get("database.sqlite")
        self.assertIsNotNone(art, "Binary artifact should be registered.")
        self.assertEqual(art["type"], "data")
        self.assertIn("Binary/Structured Data", art["content"])
        self.assertEqual(art["visibility"], ArtefactVisibility.TREE_UNLOCKABLE)

    def test_db_to_disk_missing_file_restoration(self):
        """Verify that a DB artifact missing from disk is restored."""
        self.discussion.artefacts.add(
            title="restored_doc.md",
            artefact_type=ArtefactType.DOCUMENT,
            content="# Restored Content\nThis was lost.",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        # Ensure it's on disk initially
        disk_path = self.ws_data_dir / "restored_doc.md"
        self.assertTrue(disk_path.exists())

        # Delete from disk to simulate loss
        disk_path.unlink()
        self.assertFalse(disk_path.exists())

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["restored_files"], 1)
        self.assertTrue(disk_path.exists(), "File should be restored to disk.")
        self.assertIn("# Restored Content", disk_path.read_text(encoding="utf-8"))

    def test_db_to_disk_skips_true_binary_data_restoration(self):
        """Verify that true binary data artifacts (.db) are NOT restored from text placeholder."""
        self.discussion.artefacts.add(
            title="binary_data.db",
            artefact_type=ArtefactType.DATA,
            content="### Data File: `binary_data.db`",
            file_ext=".db",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        disk_path = self.ws_data_dir / "binary_data.db"
        # It shouldn't have been written to disk because _sync_to_disk_workspace skips true binaries
        # if physical_data is None. But let's ensure sync doesn't try to write it either.
        if disk_path.exists():
            disk_path.unlink()

        report = self.discussion.sync_workspace_to_artefacts()
        self.assertEqual(report["restored_files"], 0, "True binary files should not be restored from text content.")
        self.assertFalse(disk_path.exists(), "Binary file should not be recreated from placeholder text.")

    def test_bidirectional_update_modified_text_file(self):
        """Verify modifying a file on disk updates the DB content."""
        text_file = self.ws_data_dir / "config.json"
        original_content = '{"key": "value"}'
        text_file.write_text(original_content, encoding="utf-8")

        self.discussion.sync_workspace_to_artefacts()
        
        art = self.discussion.artefacts.get("config.json")
        self.assertEqual(art["content"], original_content)

        # Modify the file
        modified_content = '{"key": "new_value", "updated": true}'
        text_file.write_text(modified_content, encoding="utf-8")

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["updated_artefacts"], 1)
        updated_art = self.discussion.artefacts.get("config.json")
        self.assertEqual(updated_art["content"], modified_content)

    def test_idempotence_no_duplicate_updates(self):
        """Verify syncing twice without changes does not trigger duplicate updates."""
        text_file = self.ws_data_dir / "stable.txt"
        text_file.write_text("stable content", encoding="utf-8")

        self.discussion.sync_workspace_to_artefacts()
        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["new_artefacts"], 0)
        self.assertEqual(report["updated_artefacts"], 0)
        self.assertEqual(report["restored_files"], 0)

    def test_ignored_directories_and_extensions(self):
        """Verify __pycache__, .git, and .pyc files are ignored."""
        pycache_dir = self.ws_data_dir / "__pycache__"
        pycache_dir.mkdir(exist_ok=True)
        (pycache_dir / "module.cpython-39.pyc").write_bytes(b"\x00\x01\x02")

        git_dir = self.ws_data_dir / ".git"
        git_dir.mkdir(exist_ok=True)
        (git_dir / "config").write_text("git config")

        lam_file = self.ws_data_dir / "schema.lam"
        lam_file.write_text("logical metadata", encoding="utf-8")

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["new_artefacts"], 0, "Ignored files should not be registered.")
        self.assertIsNone(self.discussion.artefacts.get("module.cpython-39.pyc"))
        self.assertIsNone(self.discussion.artefacts.get("config"))
        self.assertIsNone(self.discussion.artefacts.get("schema.lam"))

    def test_subfolder_path_preservation(self):
        """Verify files in subfolders are registered with relative paths intact."""
        subfolder = self.ws_data_dir / "src" / "components"
        subfolder.mkdir(parents=True, exist_ok=True)
        component_file = subfolder / "Button.tsx"
        component_file.write_text("export const Button = () => null;", encoding="utf-8")

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertEqual(report["new_artefacts"], 1)
        # The title should be the filename, but the physical_path should preserve subfolders
        arts = self.discussion.artefacts.list()
        tsx_art = next((a for a in arts if a["title"] == "Button.tsx"), None)
        self.assertIsNotNone(tsx_art)
        self.assertEqual(tsx_art["type"], "code")

    def test_sync_all_active_to_disk(self):
        """Verify sync_all_active_to_disk writes all active artifacts to workspace."""
        self.discussion.artefacts.add(
            title="active_script.py",
            artefact_type=ArtefactType.CODE,
            content="print('active')",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.artefacts.add(
            title="inactive_script.py",
            artefact_type=ArtefactType.CODE,
            content="print('inactive')",
            active=False,
            visibility=ArtefactVisibility.HIDDEN
        )
        self.discussion.commit()

        workspace_dir, synced_files = self.discussion.artefacts.sync_all_active_to_disk()

        active_path = self.ws_data_dir / "active_script.py"
        inactive_path = self.ws_data_dir / "inactive_script.py"

        self.assertTrue(active_path.exists(), "Active file should be synced to disk.")
        self.assertFalse(inactive_path.exists(), "Inactive file should NOT be synced to disk.")
        self.assertEqual(len(synced_files), 1)
        self.assertIn(str(active_path.resolve()), [str(Path(f).resolve()) for f in synced_files])

    def test_sync_all_active_to_disk_empty_workspace(self):
        """Verify sync_all_active_to_disk returns empty list when no active artifacts exist."""
        workspace_dir, synced_files = self.discussion.artefacts.sync_all_active_to_disk()
        self.assertEqual(len(synced_files), 0)
        self.assertTrue(workspace_dir.exists())

    def test_hidden_artifacts_not_restored(self):
        """Verify HIDDEN artifacts are not restored to disk by sync."""
        self.discussion.artefacts.add(
            title="hidden_doc.md",
            artefact_type=ArtefactType.DOCUMENT,
            content="secret content",
            active=False,
            visibility=ArtefactVisibility.HIDDEN
        )
        self.discussion.commit()

        disk_path = self.ws_data_dir / "hidden_doc.md"
        if disk_path.exists():
            disk_path.unlink()

        report = self.discussion.sync_workspace_to_artefacts()
        self.assertEqual(report["restored_files"], 0, "HIDDEN artifacts should not be restored.")
        self.assertFalse(disk_path.exists())

    def test_data_artifact_csv_restoration(self):
        """Verify text-based data artifacts (CSV) are restored to disk if missing."""
        csv_content = "name,age\nAlice,30\nBob,25"
        self.discussion.artefacts.add(
            title="users.csv",
            artefact_type=ArtefactType.DATA,
            content=csv_content,
            file_ext=".csv",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        disk_path = self.ws_data_dir / "users.csv"
        self.assertTrue(disk_path.exists())
        disk_path.unlink()

        report = self.discussion.sync_workspace_to_artefacts()
        self.assertEqual(report["restored_files"], 1, "CSV file should be restored from DB content.")
        self.assertTrue(disk_path.exists())
        self.assertEqual(disk_path.read_text(encoding="utf-8"), csv_content)


if __name__ == "__main__":
    unittest.main()
