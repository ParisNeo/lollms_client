import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from lollms_client.lollms_types import MSG_TYPE

class TestInlineCodeFenceLeak(unittest.TestCase):
    """
    🐛 Regression test for the StreamState single-backtick lockout bug.
    
    The parser must not enter a permanent _in_inline_code state if the LLM 
    emits a single backtick followed by a newline (e.g., raw HTML), as this 
    caused the `<artifact>` tag interceptor to be bypassed, leaking raw XML.
    """
    def setUp(self):
        self.discussion = MagicMock()
        self.discussion.artefacts = MagicMock()
        
        self.ai_message = SimpleNamespace(content="", thoughts=None)
        
        # Minimal callback to capture chunks
        self.chunks = []
        self.callback = lambda chunk, msg_type, meta=None: self.chunks.append((chunk, msg_type))

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

    def _feed_chunks(self, ss, chunks_list):
        for c in chunks_list:
            ss.feed(c)

    def test_single_backtick_followed_by_newline_does_not_lockout_artifact(self):
        """
        Reproduces the original bug:
        LLM emits stray backtick + newline, then a functional `<artifact>` tag.
        The parser must dispatch the artifact, not leak it to the UI.
        """
        ss = self._make_stream_state(enable_artefacts=True)
        
        # Simulate the LLM output:
        # A stray backtick
        ss.feed("`")
        # Newline (this used to trap the parser in _in_inline_code forever)
        ss.feed("\n")
        # A raw artifact tag that MUST be intercepted
        artifact_chunk = '<artifact name="test.html" type="html"><div>content</div></artifact>'
        ss.feed(artifact_chunk)
        ss.flush_remaining_buffer()
        
        # Verify the artifact was dispatched
        self.assertTrue(ss._action_dispatched, "Action was not dispatched! Artifact leaked.")
        self.assertTrue(len(ss.affected_artefacts) > 0, "affected_artefacts is empty.")
        
        # Verify the artifact content is not present in the final UI content
        self.assertNotIn("<artifact", self.ai_message.content, "Raw <artifact> tag leaked to UI!")
        self.assertNotIn("</artifact>", self.ai_message.content, "Raw </artifact> tag leaked to UI!")
        self.assertNotIn("<div>content</div>", self.ai_message.content, "Raw artifact body leaked to UI!")
        
        # The UI should contain the stray backtick and the subsequent processing block
        self.assertIn("`", self.ai_message.content, "Stray backtick was lost!")
        self.assertIn("<processing", self.ai_message.content, "Processing block was not emitted!")

    def test_single_backtick_inline_closed_on_same_line(self):
        """
        Ensures that legitimate inline code (e.g., `<tool>` inside markdown)
        is still captured correctly and does not trigger a lockout.
        """
        ss = self._make_stream_state()
        
        # Simulate LLM: "Here is code: `<tool>` end"
        ss.feed("Here is code: `")
        ss.feed("<tool>")
        ss.feed("` end")
        ss.flush_remaining_buffer()
        
        self.assertFalse(ss._in_inline_code, "Parser stuck in inline code mode after close.")
        self.assertEqual(self.ai_message.content, "Here is code: `<tool>` end")

    def test_single_backtick_closed_on_next_chunk(self):
        """
        Ensures that if the closing backtick arrives in a later chunk
        (but no newline has been encountered yet), the state is still maintained
        and correctly closed.
        """
        ss = self._make_stream_state()
        
        # Simulate LLM output streaming in small fragments
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

        # The LLM opens a code fence but never closes it
        ss.feed("Here is some text\n```python")
        ss.feed("\n")
        # Now the LLM emits a functional <artifact> tag on a new line.
        # The parser must break out of _in_code_fence and intercept it.
        artifact_chunk = '<artifact name="test.html" type="html"><div>content</div></artifact>'
        ss.feed(artifact_chunk)
        ss.flush_remaining_buffer()

        self.assertTrue(ss._action_dispatched, "Action was not dispatched! Artifact leaked (code fence lockout bug).")
        self.assertTrue(len(ss.affected_artefacts) > 0, "affected_artefacts is empty (code fence lockout bug).")
        self.assertNotIn("<artifact", self.ai_message.content, "Raw <artifact> tag leaked to UI (code fence lockout)!")
