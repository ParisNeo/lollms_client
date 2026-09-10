import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE


class MockClient:
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock"
        self.binding_name = "mock"
        self.tools = None

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


class TestInlineCodeFenceLeak(unittest.TestCase):
    """
    Regression test suite for StreamState code-fence / inline-code lockout bugs.

    A stray backtick or an unclosed ``` fence must never permanently trap the
    parser: functional <artifact> tags emitted afterwards at the start of a
    line MUST still be intercepted and dispatched.
    """

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_fence_leak_")
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="fence_leak_test",
            workspace_path=self.tmp_workspace,
            autosave=True
        )
        self.chunks = []
        self.callback = lambda chunk, msg_type, meta=None: self.chunks.append((chunk, msg_type)) or True
        self.ai_message = SimpleNamespace(content="", thoughts=None, id="msg_fence")

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def _make_stream_state(self, **kwargs):
        from lollms_client.lollms_discussion._mixin_chat import _StreamState
        return _StreamState(
            discussion=self.discussion,
            callback=self.callback,
            forward_artefact_chunks=False,
            ai_message=self.ai_message,
            processed_tags=set(),
            **kwargs
        )

    def test_single_backtick_followed_by_newline_does_not_lockout_artifact(self):
        """
        Reproduces the original bug:
        LLM emits stray backtick + newline, then a functional `<artifact>` tag.
        The parser must dispatch the artifact, not leak it to the UI.
        """
        ss = self._make_stream_state(enable_artefacts=True)

        ss.feed("`")
        ss.feed("\n")
        artifact_chunk = '<artifact name="test.html" type="html"><div>content</div></artifact>'
        ss.feed(artifact_chunk)
        ss.flush_remaining_buffer()

        self.assertTrue(ss._action_dispatched, "Action was not dispatched! Artifact leaked.")
        self.assertTrue(len(ss.affected_artefacts) > 0, "affected_artefacts is empty.")

        created = self.discussion.artefacts.get("test.html")
        self.assertIsNotNone(created, "Artifact was not created in the discussion.")
        self.assertEqual(created.get("content", ""), "<div>content</div>")

        self.assertNotIn("<artifact", self.ai_message.content, "Raw <artifact> tag leaked to UI!")
        self.assertNotIn("</artifact>", self.ai_message.content, "Raw </artifact> tag leaked to UI!")
        self.assertNotIn("<div>content</div>", self.ai_message.content, "Raw artifact body leaked to UI!")

        self.assertIn("`", self.ai_message.content, "Stray backtick was lost!")
        self.assertIn("<processing", self.ai_message.content, "Processing block was not emitted!")

    def test_artifact_inside_closed_code_fence_is_not_intercepted(self):
        """
        Counter-guard: an <artifact> tag inside a PROPERLY CLOSED markdown code
        fence is documentation, not a live functional tag. It must leak verbatim
        to the UI and NOT be dispatched.
        """
        ss = self._make_stream_state(enable_artefacts=True)

        ss.feed("Docs:\n```html\n")
        ss.feed('<artifact name="doc.html" type="html">demo</artifact>\n')
        ss.feed("```\nEnd docs.")
        ss.flush_remaining_buffer()

        self.assertFalse(ss._action_dispatched,
                         "Artifact inside a closed code fence must NOT be dispatched.")
        self.assertEqual(len(ss.affected_artefacts), 0)
        self.assertIsNone(self.discussion.artefacts.get("doc.html"))
        self.assertIn("<artifact", self.ai_message.content,
                      "Documented artifact tag must remain visible as verbatim text.")

    def test_single_backtick_inline_closed_on_same_line(self):
        """
        Ensures legitimate inline code (e.g., `<tool>` inside markdown) is still
        captured correctly and does not trigger a lockout.
        """
        ss = self._make_stream_state()

        ss.feed("Here is code: `")
        ss.feed("<tool>")
        ss.feed("` end")
        ss.flush_remaining_buffer()

        self.assertFalse(ss._in_inline_code, "Parser stuck in inline code mode after close.")
        self.assertEqual(self.ai_message.content, "Here is code: `<tool>` end")

    def test_single_backtick_closed_on_next_chunk(self):
        """
        Ensures that if the closing backtick arrives in a later chunk (but no
        newline has been encountered yet), the inline state is maintained and
        correctly closed.
        """
        ss = self._make_stream_state()

        ss.feed("Run `")
        ss.feed("ls -l")
        ss.feed("` now")
        ss.flush_remaining_buffer()

        self.assertFalse(ss._in_inline_code, "Parser stuck in inline code mode.")
        self.assertEqual(self.ai_message.content, "Run `ls -l` now")

    def test_double_backtick_followed_by_newline_unlocks_artifact(self):
        """
        Guards against stray triple backticks: if the LLM emits an opening
        fence ` ``` ` and never closes it (a common LLM failure), functional
        tags must still be intercepted when they appear at the start of a line.
        """
        ss = self._make_stream_state(enable_artefacts=True)

        ss.feed("Here is some text\n```python")
        ss.feed("\n")
        artifact_chunk = '<artifact name="test.html" type="html"><div>content</div></artifact>'
        ss.feed(artifact_chunk)
        ss.flush_remaining_buffer()

        self.assertTrue(ss._action_dispatched,
                        "Action was not dispatched! Artifact leaked (code fence lockout bug).")
        self.assertTrue(len(ss.affected_artefacts) > 0,
                        "affected_artefacts is empty (code fence lockout bug).")
        created = self.discussion.artefacts.get("test.html")
        self.assertIsNotNone(created, "Artifact was not created (code fence lockout bug).")
        self.assertNotIn("<artifact", self.ai_message.content,
                         "Raw <artifact> tag leaked to UI (code fence lockout)!")


if __name__ == "__main__":
    unittest.main()