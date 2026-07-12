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


class TestSearchReplaceDoubleVersion(unittest.TestCase):
    """
    Regression test: Ensures a SEARCH/REPLACE patch on an existing artifact
    creates exactly ONE new version with the patched content, and does NOT
    fall through to create a second version containing the raw patch block.
    """

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_sr_double_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="sr_double_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )
        self.discussion.artefacts.add(
            title="my_skill.md",
            artefact_type=ArtefactType.SKILL,
            content="# My Skill\n\nThis is the original content.\nIt has a bug here.\n",
            language="markdown",
            version=1
        )
        self.discussion.commit()

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_patch_creates_single_version(self):
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_1")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )

        patch_body = """<<<<<<< SEARCH
It has a bug here.
=======
The bug is now fixed.
>>>>>>> REPLACE"""

        full_tag = f'<artifact name="my_skill.md" type="skill" language="markdown">\n{patch_body}\n</artifact>'

        ss.feed(full_tag)
        ss.flush_remaining_buffer()

        history = self.discussion.artefacts.get_version_history("my_skill.md")

        self.assertEqual(len(history), 2,
                         f"Expected exactly 2 versions (original + 1 patch), but got {len(history)}.")

        v2 = self.discussion.artefacts.get("my_skill.md", version=2)
        self.assertIsNotNone(v2, "Version 2 should exist.")
        self.assertIn("The bug is now fixed.", v2["content"],
                      "Version 2 should contain the patched content.")
        self.assertNotIn("<<<<<<< SEARCH", v2["content"],
                         "Version 2 must NOT contain the raw SEARCH/REPLACE block text.")

        v3 = self.discussion.artefacts.get("my_skill.md", version=3)
        self.assertIsNone(v3, "Version 3 should NOT exist (double-version bug).")


if __name__ == "__main__":
    unittest.main()
