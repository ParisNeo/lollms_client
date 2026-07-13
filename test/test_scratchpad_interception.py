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


class TestScratchpadInterception(unittest.TestCase):
    """Validates that <scratchpad> tags are intercepted and persisted correctly."""

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_scratchpad_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="scratchpad_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_scratchpad_streaming_interception(self):
        """
        Scenario: The LLM emits a <scratchpad> tag with intermediate hypotheses.
        Expected: _StreamState intercepts the tag, buffers the body, and dispatches it
                  when the closing tag arrives. The artifact is saved as SCRATCHPAD type.
        """
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_1")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )

        chunks = [
            "I will now write my intermediate thoughts.\n",
            "<scratchpad title=\"analysis_notes\">\n",
            "Hypothesis: The user wants to test the scratchpad.\n",
            "Next step: Verify the artifact is saved.\n",
            "</scratchpad>\n"
        ]

        for chunk in chunks:
            ss.feed(chunk)

        ss.flush_remaining_buffer()

        self.assertFalse(ss._is_accumulating_secondary,
                         "Secondary accumulation should be False after closing tag.")
        self.assertTrue(ss._action_dispatched,
                        "Action should be marked as dispatched after scratchpad closure.")
        
        retrieved = self.discussion.artefacts.get("analysis_notes")
        self.assertIsNotNone(retrieved, "Scratchpad artifact should exist in the database.")
        self.assertEqual(retrieved.get("type"), ArtefactType.SCRATCHPAD,
                         "Artifact type must be SCRATCHPAD.")
        self.assertIn("Hypothesis: The user wants to test the scratchpad.", retrieved.get("content", ""),
                       "Scratchpad content must be preserved.")

        self.assertNotIn("<scratchpad", ai_msg.content,
                         "Raw <scratchpad> XML must not leak into ai_msg.content.")
        self.assertIn("<processing type=\"scratchpad\"", ai_msg.content,
                       "A processing block for the scratchpad should be present in the UI content.")

    def test_scratchpad_truncated_recovery(self):
        """
        Scenario: The LLM emits a <scratchpad> tag but stops before the closing tag.
        Expected: flush_remaining_buffer() synthesizes the closing tag and dispatches
                  the scratchpad artifact as a best-effort save.
        """
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_2")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )

        chunks = [
            "<scratchpad title=\"truncated_notes\">\n",
            "Partial thought that never finished..."
        ]

        for chunk in chunks:
            ss.feed(chunk)

        self.assertTrue(ss._is_accumulating_secondary,
                        "Should be accumulating secondary tag before flush.")

        ss.flush_remaining_buffer()

        self.assertFalse(ss._is_accumulating_secondary,
                         "Secondary accumulation should be False after flush recovery.")
        self.assertTrue(ss._action_dispatched,
                        "Action should be marked as dispatched after recovery flush.")
        
        retrieved = self.discussion.artefacts.get("truncated_notes")
        self.assertIsNotNone(retrieved, "Truncated scratchpad artifact should still be saved.")
        self.assertEqual(retrieved.get("type"), ArtefactType.SCRATCHPAD)
        self.assertIn("Partial thought that never finished...", retrieved.get("content", ""))


if __name__ == "__main__":
    unittest.main()
