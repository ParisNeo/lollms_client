import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE


class VisionMockClient:
    """Mock client that captures the images passed to generate_from_messages."""
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock-vision"
        self.binding_name = "mock"
        self.captured_images = None

    def count_tokens(self, text): return len(text) // 4
    def count_image_tokens(self, img): return 0
    def remove_thinking_blocks(self, text): return text
    def generate_text(self, prompt, **kwargs): return "ok"
    def reset_cancel(self): pass

    def generate_from_messages(self, messages, **kwargs):
        self.captured_images = kwargs.get("images")
        callback = kwargs.get("streaming_callback")
        if callback:
            callback("Response text", MSG_TYPE.MSG_TYPE_CHUNK, {})
        return ""


class TestImageSuppression(unittest.TestCase):
    """Tests the suppress_images flag for non-vision LLMs."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_img_suppress_")
        self.client = VisionMockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_img_suppress",
            workspace_path=self.tmp_workspace,
            autosave=True
        )
        # Add a dummy image to the discussion to ensure it exists in context
        self.dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        self.discussion.add_discussion_image(self.dummy_b64)

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_images_passed_by_default(self):
        """Verify images are passed to the LLM when suppress_images is False (default)."""
        self.discussion.chat(
            user_message="Describe this image",
            images=[self.dummy_b64]
        )
        self.assertIsNotNone(self.client.captured_images)
        self.assertIn(self.dummy_b64, self.client.captured_images)

    def test_images_suppressed_when_flagged(self):
        """Verify NO images are passed to the LLM when suppress_images is True."""
        self.discussion.chat(
            user_message="Describe this image",
            images=[self.dummy_b64],
            suppress_images=True
        )
        self.assertIsNone(self.client.captured_images, "Images should not be passed to a non-vision LLM.")


if __name__ == "__main__":
    unittest.main()
