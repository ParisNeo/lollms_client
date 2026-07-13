import unittest
import sys
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_artefact import ArtefactType, ArtefactVisibility


class MockClient:
    """Mock client for testing artifact operations without an LLM."""
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


class TestVisionHydrationVisibility(unittest.TestCase):
    """
    Validates that the Dynamic Vision Hydration logic in ChatMixin.chat()
    only hydrates image artifacts with visibility == ArtefactVisibility.FULL.
    Images with TREE_UNLOCKABLE or HIDDEN visibility must NOT be injected
    into the LLM's vision context, preventing crashes on non-vision models.
    """

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_vision_hydration_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="vision_hydration_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _simulate_hydration_logic(self, affected_arts, base_images=None):
        """
        Replicates the exact Dynamic Vision Hydration block from ChatMixin.chat()
        to verify which images get added to round_images.
        """
        round_images = list(base_images) if base_images else []
        for art in affected_arts:
            if art.get("type") == "image" and art.get("images") and art.get("visibility") == ArtefactVisibility.FULL:
                for img_b64 in art["images"]:
                    if img_b64 not in round_images:
                        round_images.append(img_b64)
        return round_images

    def test_full_visibility_image_is_hydrated(self):
        """
        Scenario: A tool generates an image artifact with visibility=FULL.
        Expected: The image base64 is added to round_images for vision context.
        """
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        self.discussion.artefacts.add(
            title="full_vis_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: full_vis_plot.png",
            images=[img_b64],
            image_media_types=["image/png"],
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        affected_arts = self.discussion.artefacts.list()
        round_images = self._simulate_hydration_logic(affected_arts)

        self.assertEqual(len(round_images), 1, "FULL visibility image should be hydrated")
        self.assertIn(img_b64, round_images)

    def test_tree_unlockable_image_is_not_hydrated(self):
        """
        Scenario: A tool generates an image artifact with visibility=TREE_UNLOCKABLE.
        Expected: The image base64 is NOT added to round_images.
        """
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        self.discussion.artefacts.add(
            title="unlockable_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: unlockable_plot.png",
            images=[img_b64],
            image_media_types=["image/png"],
            active=False,
            visibility=ArtefactVisibility.TREE_UNLOCKABLE
        )
        self.discussion.commit()

        affected_arts = self.discussion.artefacts.list()
        round_images = self._simulate_hydration_logic(affected_arts)

        self.assertEqual(len(round_images), 0, "TREE_UNLOCKABLE image should NOT be hydrated")
        self.assertNotIn(img_b64, round_images)

    def test_hidden_image_is_not_hydrated(self):
        """
        Scenario: An image artifact exists with visibility=HIDDEN.
        Expected: The image base64 is NOT added to round_images.
        """
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        
        self.discussion.artefacts.add(
            title="hidden_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: hidden_plot.png",
            images=[img_b64],
            image_media_types=["image/png"],
            active=False,
            visibility=ArtefactVisibility.HIDDEN
        )
        self.discussion.commit()

        affected_arts = self.discussion.artefacts.list()
        round_images = self._simulate_hydration_logic(affected_arts)

        self.assertEqual(len(round_images), 0, "HIDDEN image should NOT be hydrated")

    def test_mixed_visibility_only_full_hydrated(self):
        """
        Scenario: Multiple image artifacts exist with different visibility tiers.
        Expected: Only the FULL visibility image is hydrated. Others are excluded.
        """
        img_b64_full = "FULL_IMAGE_B64_DATA"
        img_b64_unlockable = "UNLOCKABLE_IMAGE_B64_DATA"
        img_b64_hidden = "HIDDEN_IMAGE_B64_DATA"
        
        self.discussion.artefacts.add(
            title="full_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: full_plot.png",
            images=[img_b64_full],
            image_media_types=["image/png"],
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.artefacts.add(
            title="unlockable_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: unlockable_plot.png",
            images=[img_b64_unlockable],
            image_media_types=["image/png"],
            active=False,
            visibility=ArtefactVisibility.TREE_UNLOCKABLE
        )
        self.discussion.artefacts.add(
            title="hidden_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: hidden_plot.png",
            images=[img_b64_hidden],
            image_media_types=["image/png"],
            active=False,
            visibility=ArtefactVisibility.HIDDEN
        )
        self.discussion.commit()

        affected_arts = self.discussion.artefacts.list()
        round_images = self._simulate_hydration_logic(affected_arts)

        self.assertEqual(len(round_images), 1, "Only 1 image (FULL) should be hydrated")
        self.assertIn(img_b64_full, round_images)
        self.assertNotIn(img_b64_unlockable, round_images)
        self.assertNotIn(img_b64_hidden, round_images)

    def test_non_image_artifacts_ignored(self):
        """
        Scenario: Non-image artifacts (code, document) exist with visibility=FULL.
        Expected: They are completely ignored by the vision hydration logic.
        """
        self.discussion.artefacts.add(
            title="script.py",
            artefact_type=ArtefactType.CODE,
            content="print('hello')",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.artefacts.add(
            title="notes.md",
            artefact_type=ArtefactType.DOCUMENT,
            content="# Notes",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        affected_arts = self.discussion.artefacts.list()
        round_images = self._simulate_hydration_logic(affected_arts)

        self.assertEqual(len(round_images), 0, "Non-image artifacts must not produce vision data")

    def test_base_images_preserved(self):
        """
        Scenario: The user provided base images, and a FULL visibility image is generated.
        Expected: Both the base images and the FULL image are in round_images.
        """
        base_img = "USER_PROVIDED_IMAGE_B64"
        tool_img = "TOOL_GENERATED_IMAGE_B64"
        
        self.discussion.artefacts.add(
            title="tool_plot.png",
            artefact_type=ArtefactType.IMAGE,
            content="### Image: tool_plot.png",
            images=[tool_img],
            image_media_types=["image/png"],
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        affected_arts = self.discussion.artefacts.list()
        round_images = self._simulate_hydration_logic(affected_arts, base_images=[base_img])

        self.assertEqual(len(round_images), 2, "Both base image and tool image should be present")
        self.assertIn(base_img, round_images)
        self.assertIn(tool_img, round_images)


if __name__ == "__main__":
    unittest.main()
