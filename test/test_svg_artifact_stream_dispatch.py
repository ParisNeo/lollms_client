import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_discussion._mixin_chat import _StreamState
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
        return len(text) // 4

    def count_image_tokens(self, image: str) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"

    def generate_from_messages(self, messages, **kwargs):
        pass


class TestSVGArtifactStreamDispatch(unittest.TestCase):
    """Test that SVG artifacts emitted with type='image' are written to disk via StreamState dispatch."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_svg_dispatch_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_svg_dispatch_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )
        self.ws_path = Path(self.discussion.workspace_data_path)
        self.ws_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def _setup_stream_state(self) -> _StreamState:
        """Initializes a _StreamState instance with a mocked AI message."""
        ai_dummy = SimpleNamespace(
            id="ai_msg_1",
            sender="assistant",
            sender_type="assistant",
            content="",
            parent_id="msg_1",
            discussion_id=self.discussion.id,
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
        
        return _StreamState(
            discussion=self.discussion,
            forward_artefact_chunks=False,
            callback=None,
            ai_message=ai_dummy,
            enable_artefacts=True
        )

    def test_svg_dispatch_writes_to_disk(self):
        """
        Simulates the LLM emitting <artifact name="circuit.svg" type="image">...</artifact>
        and verifies the _dispatch_closed_tag method correctly writes it to disk as text.
        """
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>'
        
        # Simulate the exact arguments _StreamState.feed() would pass to _dispatch_closed_tag
        tag_name = "artifact"
        attrs_str = '<artifact name="circuit.svg" type="image" language="svg">'
        body = svg_content
        full_match_text = f'{attrs_str}{body}</artifact>'
        
        ss = self._setup_stream_state()
        
        # Directly invoke the dispatch logic
        ss._dispatch_closed_tag(tag_name, attrs_str, body, full_match_text)
        
        # 1. Verify it exists in DB
        db_art = self.discussion.artefacts.get("circuit.svg")
        self.assertIsNotNone(db_art, "Artifact was not saved to database.")
        self.assertEqual(db_art["content"], svg_content)
        self.assertEqual(db_art["type"], "image")
        
        # 2. Verify it exists on disk
        svg_file = self.ws_path / "circuit.svg"
        self.assertTrue(svg_file.exists(), "SVG artifact was NOT written to disk by dispatch!")
        
        # 3. Verify content integrity
        disk_content = svg_file.read_text(encoding="utf-8")
        self.assertEqual(disk_content, svg_content, "Disk content does not match SVG payload.")


if __name__ == "__main__":
    unittest.main()
