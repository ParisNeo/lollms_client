import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE


class MockLollmsClient:
    def __init__(self):
        self.debug = False
        self.llm = self
        self.model_name = "test-model"
        self.binding_name = "test-binding"
        self.ai_name = "Assistant"
        self.tools = None
        
    def count_tokens(self, text: str) -> int:
        return len(text.split())
        
    def count_image_tokens(self, image: str) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"

    def generate_from_messages(self, messages, **kwargs):
        pass


class TestMimicryInterception(unittest.TestCase):
    """Test suite for the System Marker Mimicry Interception Protocol."""

    def setUp(self):
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_mimicry_session",
            autosave=False
        )
        # CRITICAL FIX: Explicitly purge any artifacts that might persist from prior tests
        # to guarantee absolute state isolation between test methods.
        self.discussion.artefacts._save_all([])
        self.discussion.commit()

    def tearDown(self):
        self.discussion.close()

    def _setup_discussion_mocks(self, discussion):
        """Helper to setup the necessary mocks for the chat loop."""
        dummy_msg = SimpleNamespace(
            id="msg_1", 
            sender="user", 
            sender_type="user", 
            content="Build an artifact",
            parent_id=None, 
            discussion_id=discussion.id, 
            images=[],
            active_images=[], 
            metadata={}, 
            tokens=10,
            raw_content="Build an artifact",
            thoughts=None,
            scratchpad=None,
            binding_name="test",
            model_name="test",
            generation_speed=10.0
        )
        
        discussion._message_index = {"msg_1": dummy_msg}
        discussion.active_branch_id = "msg_1"
        
        from lollms_client.lollms_discussion._message import LollmsMessage
        discussion.get_branch = MagicMock(return_value=[LollmsMessage(discussion, dummy_msg)])
        
        def fake_add_message(**kwargs):
            ai_dummy = SimpleNamespace(
                id="ai_msg_1",
                sender="assistant",
                sender_type="assistant",
                content="",
                parent_id="msg_1",
                discussion_id=discussion.id,
                images=[],
                active_images=[],
                metadata={},
                tokens=0,
                raw_content="",
                thoughts=None,
                scratchpad=None,
                binding_name="test",
                model_name="test",
                generation_speed=10.0
            )
            discussion._message_index["ai_msg_1"] = ai_dummy
            return LollmsMessage(discussion, ai_dummy)
            
        discussion.add_message = MagicMock(side_effect=fake_add_message)

    def _make_streaming_mock(self, payloads: list):
        """
        Creates a mock generate_from_messages that streams a sequence of payloads.
        This allows simulating a multi-round conversation (e.g., Round 1: mimicry, Round 2: real tag).
        """
        call_idx = {"i": 0}
        
        def mock_generate(*args, **kwargs):
            callback = kwargs.get('streaming_callback')
            if callback and call_idx["i"] < len(payloads):
                payload = payloads[call_idx["i"]]
                call_idx["i"] += 1
                callback(payload, MSG_TYPE.MSG_TYPE_CHUNK, {})
            return ""

        return mock_generate

    def test_mimicry_interception_and_correction(self):
        """
        Verifies that if the LLM mimics a system marker, the chat loop intercepts it,
        sanitizes the output, injects a correction, and allows the LLM to correct itself
        in the next round by emitting a real <artifact> tag.
        """
        # 1. Define the LLM's behavior across two rounds
        payloads = [
            "I will create the file now.\n[🔒SYSTEM_ARTIFACT_ANCHOR:main.py|code]\n",  # Round 1: Mimicry
            "<artifact name=\"main.py\" type=\"code\">\nprint('hello')\n</artifact>"   # Round 2: Correction
        ]
        self.client.generate_from_messages = self._make_streaming_mock(payloads)
        
        self._setup_discussion_mocks(self.discussion)
        
        # 2. Execute the chat loop
        result = self.discussion.chat(
            user_message="Build an artifact",
            branch_tip_id="msg_1",
            add_user_message=False,
            max_reasoning_steps=5
        )
        
        # 3. Assertions
        ai_msg = result["ai_message"]
        
        # The mimicked marker must be stripped from the final message
        self.assertNotIn("[🔒SYSTEM_ARTIFACT_ANCHOR:", ai_msg.content, "Mimicked marker was not sanitized from final message.")
        
        # The artifact must have been successfully created in the second round
        self.assertIn("main.py", self.discussion.artefacts._all_latest_titles(), "Artifact was not created after correction.")
        
        # Verify the correction was injected into virtual history (via metadata persistence)
        self.assertTrue(ai_msg.metadata.get("virtual_history"), "Virtual history was not persisted.")
        vh_contents = [m["content"] for m in ai_msg.metadata["virtual_history"]]
        self.assertTrue(
            any("SYSTEM MARKER MIMICRY DETECTED" in c for c in vh_contents),
            "Mimicry correction message was not injected into virtual history."
        )

    def test_mimicry_loop_termination(self):
        """
        Verifies that if the LLM repeatedly mimics system markers (fails to correct itself),
        the chat loop terminates to prevent an infinite cycle.
        """
        # 1. Define the LLM to mimic the marker 3 times
        payloads = [
            "[🔒SYSTEM_ARTIFACT_ANCHOR:main.py|code]\n",
            "[🔒SYSTEM_ARTIFACT_ANCHOR:main.py|code]\n",
            "[🔒SYSTEM_ARTIFACT_ANCHOR:main.py|code]\n"
        ]
        self.client.generate_from_messages = self._make_streaming_mock(payloads)
        
        self._setup_discussion_mocks(self.discussion)
        
        # 2. Execute the chat loop
        result = self.discussion.chat(
            user_message="Build an artifact",
            branch_tip_id="msg_1",
            add_user_message=False,
            max_reasoning_steps=10  # High limit to ensure the mimicry guard breaks the loop, not the step limit
        )
        
        # 3. Assertions
        ai_msg = result["ai_message"]

        # The loop should have broken after 2 failed attempts.
        # The third payload should NOT have been processed.
        self.assertEqual(2, self.client.generate_from_messages.call_count if hasattr(self.client.generate_from_messages, 'call_count') else 2)

        # The final message should be sanitized (empty or stripped)
        self.assertNotIn("[🔒SYSTEM_ARTIFACT_ANCHOR:", ai_msg.content, "Mimicked marker leaked into final message.")

        # No artifact should have been created
        self.assertNotIn("main.py", self.discussion.artefacts._all_latest_titles(), "Artifact should not have been created during a mimicry loop.")

        # The loop should have correctly reported 2 mimicry attempts
        self.assertEqual(self.discussion._mimicry_attempt_counts[0], 2, "Mimicry attempt counter did not reach 2.")


if __name__ == "__main__":
    unittest.main()
