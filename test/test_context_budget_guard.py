import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion._mixin_chat import _StreamState
from lollms_client.lollms_artefact import ArtefactType, ArtefactVisibility


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


class TestContextBudgetGuard(unittest.TestCase):
    """Validates the Context Budget Guard and Tool-Generated File Visibility Doctrine."""

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_budget_guard_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="budget_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_large_tool_generated_file_visibility(self):
        """
        Scenario: A tool generates a large file (>100KB).
        Expected: The file is registered with visibility=TREE_UNLOCKABLE and active=False.
        """
        large_content = "A" * 250_000  # 250KB > 100KB threshold, ~62,500 tokens
        
        # Simulate the post-execution artifact sync logic
        file_size_kb = len(large_content.encode('utf-8')) / 1024
        is_large_file = file_size_kb > 100
        
        # Apply the Visibility Doctrine
        visibility = ArtefactVisibility.TREE_UNLOCKABLE if is_large_file else ArtefactVisibility.FULL
        active = not is_large_file
        
        art = self.discussion.artefacts.add(
            title="large_output.json",
            artefact_type=ArtefactType.DOCUMENT,
            content=large_content,
            active=active,
            visibility=visibility
        )
        self.discussion.commit()
        
        retrieved = self.discussion.artefacts.get("large_output.json")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get("visibility"), ArtefactVisibility.TREE_UNLOCKABLE)
        self.assertFalse(retrieved.get("active"))
        # Mock client calculates tokens as len(text) // 4
        expected_tokens = len(large_content) // 4
        self.assertEqual(retrieved.get("token_count", 0), expected_tokens)
        self.assertGreater(expected_tokens, 50000)

    def test_unlock_file_blocked_for_large_file(self):
        """
        Scenario: The LLM attempts to <unlock_file> a large file (>50k tokens).
        Expected: The unlock is blocked. The file remains TREE_UNLOCKABLE.
                  The UI message contains the blocked warning.
        """
        # Create a large artifact that simulates a tool-generated file
        large_content = "B" * 250_000  # ~62,500 tokens
        self.discussion.artefacts.add(
            title="huge_data.json",
            artefact_type=ArtefactType.DOCUMENT,
            content=large_content,
            active=False,
            visibility=ArtefactVisibility.TREE_UNLOCKABLE
        )
        self.discussion.commit()
        
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_1")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )
        
        # Feed the unlock_file tag in streaming chunks to simulate real LLM output
        ss.feed("<unlock_file>")
        ss.feed("huge_data.json")
        ss.feed("</unlock_file>")
        ss.flush_remaining_buffer()
        
        # Assert the file was NOT unlocked
        retrieved = self.discussion.artefacts.get("huge_data.json")
        self.assertEqual(retrieved.get("visibility"), ArtefactVisibility.TREE_UNLOCKABLE)
        self.assertFalse(retrieved.get("active"))
        
        # Assert the UI message contains the blocked warning
        self.assertIn("BLOCKED", ai_msg.content)
        self.assertIn("too large for context", ai_msg.content)
        self.assertIn("huge_data.json", ai_msg.content)
        
        # Assert that the continuation flag was set so the LLM receives guidance
        self.assertTrue(ss.context_unlock_requested)
        self.assertIn("huge_data.json", ss.context_unlocked_files)

    def test_unlock_file_allowed_for_small_file(self):
        """
        Scenario: The LLM attempts to <unlock_file> a small file (<50k tokens).
        Expected: The unlock succeeds. The file becomes FULL visibility.
        """
        small_content = "C" * 1000  # ~250 tokens
        self.discussion.artefacts.add(
            title="small_config.txt",
            artefact_type=ArtefactType.DOCUMENT,
            content=small_content,
            active=False,
            visibility=ArtefactVisibility.TREE_UNLOCKABLE
        )
        self.discussion.commit()
        
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_2")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )
        
        # Feed the unlock_file tag in streaming chunks to simulate real LLM output
        # CRITICAL: The tag must start on a new line to pass the Start-Of-Line check in the parser
        ss.feed("\n<unlock_file>\n")
        ss.feed("small_config.txt\n")
        ss.feed("</unlock_file>")
        ss.flush_remaining_buffer()
        
        retrieved = self.discussion.artefacts.get("small_config.txt")
        self.assertEqual(retrieved.get("visibility"), ArtefactVisibility.FULL)
        self.assertTrue(retrieved.get("active"))
        self.assertNotIn("BLOCKED", ai_msg.content)
        self.assertIn("✅ Unlocking", ai_msg.content)


if __name__ == "__main__":
    unittest.main()
