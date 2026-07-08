import unittest
import tempfile
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE


class CancelableMockClient:
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock"
        self.binding_name = "mock"
        self._cancel_flag = False
    def count_tokens(self, text): return len(text) // 4
    def count_image_tokens(self, img): return 0
    def remove_thinking_blocks(self, text): return text
    def generate_text(self, prompt, **kwargs): return "ok"
    def reset_cancel(self): self._cancel_flag = False
    def cancel(self): self._cancel_flag = True
    def generate_from_messages(self, messages, **kwargs):
        callback = kwargs.get("streaming_callback")
        if callback:
            for _ in range(5):
                if self._cancel_flag:
                    return ""
                callback("chunk ", MSG_TYPE.MSG_TYPE_CHUNK, {})
                time.sleep(0.1)
        return ""


class TestCognitiveInterruptAndCancel(unittest.TestCase):
    """Tests the thread-safe cancellation protocol."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_cancel_")
        self.client = CancelableMockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_cancel",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_cancel_during_streaming(self):
        """Verify calling cancel_generation() halts the stream and marks the message."""
        def run_chat():
            self.res = self.discussion.chat(user_message="Long generation")

        thread = threading.Thread(target=run_chat)
        thread.start()

        time.sleep(0.2)
        self.discussion.cancel_generation()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive(), "Thread should have terminated.")
        self.assertTrue(self.res["was_cancelled"])
        self.assertIn("[Generation cancelled by user]", self.res["ai_message"].content)

    def test_cancel_resets_state(self):
        """Verify the cancel flag resets after a cancelled turn, allowing subsequent turns."""
        self.discussion.cancel_generation()
        res1 = self.discussion.chat(user_message="Test 1")
        self.assertTrue(res1["was_cancelled"])

        res2 = self.discussion.chat(user_message="Test 2")
        self.assertFalse(res2["was_cancelled"], "Second turn should not be cancelled.")


if __name__ == "__main__":
    unittest.main()
