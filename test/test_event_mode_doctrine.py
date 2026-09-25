import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_types import EventMode, MSG_TYPE
from lollms_client.lollms_discussion._mixin_chat import _StreamState
from lollms_client.lollms_personality.lollms_agent_state import _AgentStreamState


class TestEventModeDoctrine(unittest.TestCase):
    """
    Verifies that the event system strictly enforces the four-mode doctrine:
    1. PROCESSING_TAG_MODE: Only MSG_TYPE_CHUNK is emitted; events and thoughts are embedded as tags.
    2. FULL_CALLBACK_MODE: Normal chunks contain only clean text; thoughts & events use dedicated MSG_TYPE_* types.
    3. MIXED_MODE: Both tags in chunks and dedicated callback events.
    4. SILENT_MODE: Only clean text chunks with no events or special tags.
    """

    def setUp(self):
        self.mock_discussion = MagicMock()
        self.mock_discussion.artefacts = MagicMock()
        self.mock_discussion.artefacts.get.return_value = None
        self.mock_discussion.artefacts.add.return_value = {"title": "sample.py", "type": "code", "version": 1}

    def _create_stream_state(self, mode: EventMode, events_list: list):
        ai_msg = SimpleNamespace(id="msg_1", content="", thoughts=None, metadata={})
        def callback(chunk, msg_type, meta):
            events_list.append({"chunk": chunk, "msg_type": msg_type, "meta": meta})
            return True
        return _StreamState(
            discussion=self.mock_discussion,
            forward_artefact_chunks=False,
            callback=callback,
            ai_message=ai_msg,
            event_mode=mode,
            remove_thinking_blocks=False,
        ), ai_msg

    def _create_agent_stream_state(self, mode: EventMode, events_list: list):
        def callback(chunk, msg_type, meta):
            events_list.append({"chunk": chunk, "msg_type": msg_type, "meta": meta})
            return True
        return _AgentStreamState(callback=callback, event_mode=mode)

    def test_processing_tag_mode_only_emits_chunks(self):
        """In PROCESSING_TAG_MODE, only MSG_TYPE_CHUNK must ever be sent to the callback."""
        events = []
        ss, ai_msg = self._create_stream_state(EventMode.PROCESSING_TAG_MODE, events)

        # 1. Thought chunk passed through
        ss.passthrough("Considering algorithm...", MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
        
        # 2. Normal text feed
        ss.feed("Here is the answer.\n")
        
        # 3. Processing block for artifact
        ss.feed('<artifact name="main.py" type="code">\nprint("hello")\n</artifact>\n')
        ss.flush_remaining_buffer()

        # Check every event emitted to callback
        msg_types = [e["msg_type"] for e in events]
        self.assertTrue(all(mt == MSG_TYPE.MSG_TYPE_CHUNK for mt in msg_types),
                        f"Non-chunk message types detected in PROCESSING_TAG_MODE: {set(msg_types) - {MSG_TYPE.MSG_TYPE_CHUNK}}")
        
        # Verify thought was embedded in tags within chunk
        text_stream = "".join(e["chunk"] for e in events)
        self.assertIn("<think>", text_stream)
        self.assertIn("Considering algorithm...", text_stream)
        self.assertIn("</think>", text_stream)

    def test_full_callback_mode_emits_dedicated_events_and_clean_chunks(self):
        """In FULL_CALLBACK_MODE, thoughts use MSG_TYPE_THOUGHT_CHUNK, text chunks are clean."""
        events = []
        ss, ai_msg = self._create_stream_state(EventMode.FULL_CALLBACK_MODE, events)

        # 1. Thought chunk passed through
        ss.passthrough("Deep thought analysis.", MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
        
        # 2. Normal text
        ss.feed("Answer begins here.\n")
        ss.flush_remaining_buffer()

        thought_events = [e for e in events if e["msg_type"] == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK]
        chunk_events = [e for e in events if e["msg_type"] == MSG_TYPE.MSG_TYPE_CHUNK]

        self.assertEqual(len(thought_events), 1)
        self.assertEqual(thought_events[0]["chunk"], "Deep thought analysis.")
        
        # Normal chunk stream must not have <think> or </think> tags
        normal_text = "".join(e["chunk"] for e in chunk_events)
        self.assertNotIn("<think>", normal_text)
        self.assertNotIn("</think>", normal_text)
        self.assertIn("Answer begins here.", normal_text)

    def test_silent_mode_suppresses_thoughts_and_events(self):
        """In SILENT_MODE, only clean text chunks are emitted. No thoughts or events."""
        events = []
        ss, ai_msg = self._create_stream_state(EventMode.SILENT_MODE, events)

        # 1. Thoughts passed through
        ss.passthrough("Secret internal thought.", MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)

        # 2. Text with embedded <think> tag
        ss.feed("<think>embedded thought</think>Direct answer only.")
        ss.flush_remaining_buffer()

        msg_types = [e["msg_type"] for e in events]
        self.assertTrue(all(mt == MSG_TYPE.MSG_TYPE_CHUNK for mt in msg_types))

        text_stream = "".join(e["chunk"] for e in events)
        self.assertNotIn("Secret internal thought", text_stream)
        self.assertNotIn("<think>", text_stream)
        self.assertNotIn("embedded thought", text_stream)
        self.assertIn("Direct answer only.", text_stream)

    def test_agent_stream_state_processing_tag_mode(self):
        """_AgentStreamState in PROCESSING_TAG_MODE only emits MSG_TYPE_CHUNK and embeds think tags."""
        events = []
        ass = self._create_agent_stream_state(EventMode.PROCESSING_TAG_MODE, events)

        ass.feed("<think>agent thought</think>Task result.")
        ass.flush_remaining_buffer()

        msg_types = [e["msg_type"] for e in events]
        self.assertTrue(all(mt == MSG_TYPE.MSG_TYPE_CHUNK for mt in msg_types),
                        f"Found non-chunk types: {msg_types}")
        
        full_text = "".join(e["chunk"] for e in events)
        self.assertIn("<think>", full_text)
        self.assertIn("agent thought", full_text)
        self.assertIn("Task result.", full_text)

    def test_agent_stream_state_full_callback_mode(self):
        """_AgentStreamState in FULL_CALLBACK_MODE emits MSG_TYPE_THOUGHT_CHUNK and clean chunks."""
        events = []
        ass = self._create_agent_stream_state(EventMode.FULL_CALLBACK_MODE, events)

        ass.feed("<think>agent thought</think>Task result.")
        ass.flush_remaining_buffer()

        thought_events = [e for e in events if e["msg_type"] == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK]
        chunk_events = [e for e in events if e["msg_type"] == MSG_TYPE.MSG_TYPE_CHUNK]

        self.assertTrue(len(thought_events) >= 1)
        self.assertIn("agent thought", "".join(e["chunk"] for e in thought_events))

        clean_text = "".join(e["chunk"] for e in chunk_events)
        self.assertNotIn("<think>", clean_text)
        self.assertNotIn("</think>", clean_text)
        self.assertIn("Task result.", clean_text)


if __name__ == "__main__":
    unittest.main()