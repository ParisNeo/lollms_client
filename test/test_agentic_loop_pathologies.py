import unittest
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from types import SimpleNamespace
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion._mixin_chat import _StreamState
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_artefact import ArtefactType


class PathologicalMockClient:
    """
    Mock client that streams specific pathological payloads to test loop guards.
    """
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock-pathological"
        self.binding_name = "mock"
        self.payload_type = "empty"
        self._payloads = {
            "empty": [],
            "truncated_artifact": [
                'I will build it.\n',
                '<artifact name="test.py" type="code" language="python">\n',
                '<<<<<<< SEARCH\n',
                'def old():\n',
                '    pass\n',
                '=======\n',
                'def new():\n',
                '    return True\n'
            ],
            "processing_mimicry": [
                'Sure. Here is the result.\n',
                '<processing type="tool" title="Fake Execution">\n',
                '* Running...\n',
                '</processing>\n'
            ]
        }

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 0

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "ok"

    def generate_structured_content(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"title": "Mock"}

    def reset_cancel(self):
        pass

    def generate_from_messages(self, messages: list, **kwargs) -> str:
        callback = kwargs.get("streaming_callback")
        chunks = self._payloads.get(self.payload_type, [])

        if self.payload_type == "empty":
            if callback:
                callback("", MSG_TYPE.MSG_TYPE_CHUNK, {})
            return ""

        for chunk in chunks:
            if callback:
                keep_going = callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK, {})
                if keep_going is False:
                    break
            else:
                pass
        return ""


class TestAgenticLoopPathologies(unittest.TestCase):

    def setUp(self):
        self.client = PathologicalMockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        import tempfile
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_pathology_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="pathology_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )
        self.discussion.artefacts.add(
            title="test.py",
            artefact_type=ArtefactType.CODE,
            content="def old():\n    pass\n",
            language="python",
            version=1
        )
        self.discussion.commit()

    def tearDown(self):
        self.discussion.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_empty_response_guard(self):
        """
        Scenario: LLM generates 0 tokens.
        Expected: Loop breaks immediately, virtual_history gets [No output generated].
        """
        self.client.payload_type = "empty"

        round_count_holder = {"count": 0}
        original_generate = self.client.generate_from_messages

        def counting_generate(*args, **kwargs):
            round_count_holder["count"] += 1
            return original_generate(*args, **kwargs)

        self.client.generate_from_messages = counting_generate

        res = self.discussion.chat(
            user_message="Run test",
            max_reasoning_steps=5,
            enable_artefacts=True
        )

        self.assertEqual(round_count_holder["count"], 1,
                         "Loop should break after 1 round on empty response, not iterate up to max_reasoning_steps.")

        ai_msg = res.get("ai_message")
        self.assertIsNotNone(ai_msg)
        self.assertEqual(ai_msg.content.strip(), "",
                         "ai_msg.content should be empty because no real content was generated.")

    def test_processing_block_mimicry(self):
        """
        Scenario: LLM attempts to generate a <processing> block.
        Expected: _StreamState.feed() returns False, halting generation.
                  The <processing> tag is stripped from the buffer.
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
            "Sure. Here is the result.\n",
            "<processing type=\"tool\" title=\"Fake Execution\">\n",
            "* Running...\n"
        ]

        results = []
        for chunk in chunks:
            keep_going = ss.feed(chunk)
            results.append(keep_going)

        self.assertTrue(results[0], "Normal text should return True.")
        self.assertFalse(results[1], "Generation should halt (return False) when <processing> is detected.")

        self.assertNotIn("<processing", ss._pending_buffer.lower(),
                         "The <processing> tag should be stripped from the pending buffer.")
        self.assertNotIn("<processing", ai_msg.content.lower(),
                         "The <processing> tag should not leak into ai_msg.content.")

    def test_truncated_artifact_recovery(self):
        """
        Scenario: LLM emits a SEARCH/REPLACE block but stops before </artifact>.
        Expected: flush_remaining_buffer() synthesizes the closing tag and dispatches.
        """
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_1")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )

        chunks = self.client._payloads["truncated_artifact"]
        for chunk in chunks:
            ss.feed(chunk)

        self.assertTrue(ss.artefact_tracker.is_inside_artefact,
                        "Tracker should be inside artefact before flush.")

        # CRITICAL FIX: Explicitly call flush_remaining_buffer() to simulate
        # the end of the LLM generation stream. This triggers the truncated
        # artifact recovery logic in _StreamState.
        ss.flush_remaining_buffer()

        self.assertFalse(ss.artefact_tracker.is_inside_artefact,
                         "Tracker should be closed after flush.")
        self.assertTrue(ss._action_dispatched,
                        "Action should be marked as dispatched after recovery.")
        self.assertIn("status:finished", ai_msg.content,
                      "Processing block should be closed with status:finished.")

        updated_art = self.discussion.artefacts.get("test.py")
        self.assertIsNotNone(updated_art, "Artifact should exist after best-effort dispatch.")
        self.assertGreater(updated_art.get("version", 1), 1,
                           "Artifact version should be bumped if patch was applied.")
        self.assertIn("def new():", updated_art.get("content", ""),
                      "Truncated patch content should have been applied.")


if __name__ == "__main__":
    unittest.main()
