import unittest
import tempfile
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager


class MockClient:
    """Mock LollmsClient for isolated testing without LLM bindings."""
    def __init__(self):
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


class TestWorkspacePathPersistence(unittest.TestCase):
    """
    Validates the workspace path persistence lifecycle.
    
    The physical file path is the absolute truth, and binding a specific 
    folder to a discussion must survive application restarts (closing and 
    re-opening the discussion object) without the host application needing 
    to re-inject the path.
    """

    def setUp(self):
        self.client = MockClient()
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_ws_persist_")
        # Use a file-backed SQLite DB so we can simulate a full restart cleanly
        self.db_file = Path(tempfile.mkdtemp(prefix="lollms_ws_db_")) / "discussion.db"
        self.db_manager = LollmsDataManager(f"sqlite:///{self.db_file}")

    def tearDown(self):
        if self.db_manager:
            try:
                # Close session if open
                if self.db_manager._session and self.db_manager._session.is_active:
                    self.db_manager._session.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)
        shutil.rmtree(self.db_file.parent, ignore_errors=True)

    def test_workspace_path_persists_across_restarts(self):
        """
        Scenario: User sets a custom workspace folder for a discussion.
        Action: The application closes. User reopens the discussion.
        Expected: The discussion automatically points to the custom folder.
        """
        custom_folder = Path(self.tmp_workspace) / "my_custom_folder"
        custom_folder.mkdir(parents=True)

        # 1. Initial creation & explicit path binding
        discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="persistent_ws_test",
            workspace_path=str(custom_folder),
            autosave=True
        )

        # Set the workspace path (simulating user changing it via UI or API)
        new_folder = Path(self.tmp_workspace) / "switched_folder"
        new_folder.mkdir(parents=True)
        discussion.set_workspace_path(str(new_folder))

        # Force a commit to ensure the metadata is flushed to the DB
        discussion.commit()
        
        # Verify the path is currently correct in memory
        self.assertEqual(discussion.workspace_path, str(new_folder.resolve()))
        
        # Close the discussion (simulating app exit)
        discussion.close()

        # 2. Simulate Restart: Create a NEW discussion instance from the DB
        # NOTE: We deliberately DO NOT pass workspace_path here. 
        # The host app only provides the discussion_id.
        reloaded_discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="persistent_ws_test"
        )
        
        # 3. CRITICAL ASSERTION: The workspace path must have been restored from metadata
        expected_resolved = str(new_folder.resolve())
        actual_path = reloaded_discussion.workspace_path
        
        self.assertIsNotNone(actual_path, "Workspace path was None after restart!")
        self.assertEqual(
            actual_path, 
            expected_resolved, 
            f"Workspace path was lost after restart. Expected: {expected_resolved}, Got: {actual_path}"
        )
        
        # 4. Verify the workspace_data_path is correctly derived from the restored path
        expected_data_path = str((new_folder / "workspace_data").resolve())
        self.assertEqual(
            reloaded_discussion.workspace_data_path,
            expected_data_path,
            "workspace_data_path was not correctly derived from the restored workspace_path."
        )

        reloaded_discussion.close()

    def test_default_workspace_path_if_no_metadata(self):
        """
        Regression test: If a discussion never had a custom workspace path set,
        it should fall back to the default generated path upon restart.
        """
        # 1. Create discussion without explicit workspace_path
        discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="default_ws_test",
            autosave=True
        )
        
        initial_path = discussion.workspace_path
        self.assertIsNotNone(initial_path)
        
        # Add a message to ensure touch/commit happens naturally
        discussion.add_message(sender="user", sender_type="user", content="hello")
        discussion.commit()
        discussion.close()

        # 2. Restart
        reloaded_discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="default_ws_test"
        )
        
        # 3. Assertion: The path should match the default structure 
        #    (data_workspace/discussions/{id}) OR the persisted default
        #    Because the default path is technically set in __init__ before the first commit,
        #    it WILL be persisted in the metadata by touch().
        self.assertEqual(
            reloaded_discussion.workspace_path,
            initial_path,
            "Default workspace path should also persist if it was explicitly set during __init__."
        )
        
        reloaded_discussion.close()


if __name__ == "__main__":
    unittest.main()