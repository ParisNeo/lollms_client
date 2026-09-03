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


    def test_vlm_bundle_description_substitution_and_caching(self):
        """Verify VLM generates image description, substitutes it into text-only model, and caches it."""
        from lollms_client import LollmsClient, LollmsBindingProfile, LollmsModelProfile

        # Setup mock VLM and Text-Only LLMs
        client = LollmsClient(
            llm_binding_profiles={
                "mock_engine": LollmsBindingProfile(
                    name="mock_engine",
                    binding_name="ollama"
                )
            },
            llm_model_profiles={
                "text_only": LollmsModelProfile(
                    name="text_only",
                    binding_profile_name="mock_engine",
                    model_name="llama3-text",
                    vision_enabled=False,
                    is_default=True
                ),
                "vlm_model": LollmsModelProfile(
                    name="vlm_model",
                    binding_profile_name="mock_engine",
                    model_name="llava-vision",
                    vision_enabled=True,
                    is_default=False
                )
            }
        )

        # Mock the text generation on both models
        mock_text_llm = MagicMock()
        mock_text_llm.vision_enabled = False
        mock_text_llm.generate_text.return_value = "Answer about image"

        mock_vlm = MagicMock()
        mock_vlm.vision_enabled = True
        mock_vlm.generate_text.return_value = "A red sports car on a racetrack"

        client.llms["text_only"] = mock_text_llm
        client.llms["vlm_model"] = mock_vlm
        client.llm = mock_text_llm
        client._active_llm_alias = "text_only"

        test_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        # Call generate_text with an image on the text-only model
        response = client.generate_text(prompt="What is this?", images=[test_img])

        # 1. Check that the VLM was called once to produce the description
        mock_vlm.generate_text.assert_called_once()

        # 2. Check that the text LLM received the prompt with the description substituted and NO images
        called_kwargs = mock_text_llm.generate_text.call_args.kwargs
        self.assertIn("[Image Description: A red sports car on a racetrack]", called_kwargs["prompt"])
        self.assertIsNone(called_kwargs["images"])

        # 3. Call generate_text again with the same image: VLM should NOT be called again (cached)
        mock_vlm.generate_text.reset_mock()
        response2 = client.generate_text(prompt="Second question", images=[test_img])
        mock_vlm.generate_text.assert_not_called()

        called_kwargs2 = mock_text_llm.generate_text.call_args.kwargs
        self.assertIn("[Image Description: A red sports car on a racetrack]", called_kwargs2["prompt"])
        self.assertIsNone(called_kwargs2["images"])


if __name__ == "__main__":
    unittest.main()
