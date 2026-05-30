import unittest
from pathlib import Path
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_core import LollmsClient
from lollms_client.lollms_types import LCPResult

class DummyClient:
    def __init__(self):
        self.debug = True
        self.llm = self
        self.model_name = "unknown"
        self.binding_name = "unknown"
    def count_tokens(self, text):
        return len(text) // 4
    def count_image_tokens(self, img):
        return 256
    def remove_thinking_blocks(self, text):
        return text
    def generate_text(self, prompt, **kwargs):
        return "Simulated response"
    def chat(self, discussion, **kwargs):
        return "Chat response"

class TestLollmsArtifactTools(unittest.TestCase):
    def setUp(self):
        # In-memory db manager for test isolation
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = DummyClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_session",
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()

    def test_lcp_result_serialization(self):
        # Test basic dataclass serialization
        res = LCPResult(
            success=True,
            output="Done",
            prompt_injection="Injected Prompt",
            images=["img1_b64"],
            code_blocks=[{"content": "print('hello')", "language": "python"}],
            paths=["data_workspace/file.csv"],
            sources=[{"title": "Doc", "content": "Text", "source": "http"}]
        )
        data = res.to_dict()
        self.assertTrue(data["success"])
        self.assertEqual(data["output"], "Done")
        self.assertEqual(data["prompt_injection"], "Injected Prompt")
        self.assertEqual(data["images"][0], "img1_b64")
        self.assertEqual(data["code_blocks"][0]["language"], "python")
        self.assertEqual(data["paths"][0], "data_workspace/file.csv")
        self.assertEqual(data["sources"][0]["title"], "Doc")

        # Test from_dict
        restored = LCPResult.from_dict(data)
        self.assertEqual(restored.output, "Done")
        self.assertEqual(restored.sources[0]["title"], "Doc")

    def test_ephemeral_artifact_and_promotion(self):
        # Create an ephemeral artifact
        art = self.discussion.artefacts.add(
            title="temp_config",
            artefact_type=ArtefactType.DOCUMENT,
            content="temp content",
            ephemeral=True
        )
        self.assertTrue(art.get("ephemeral"))
        
        # Verify it is in database but flagged
        db_art = self.discussion.artefacts.get("temp_config")
        self.assertTrue(db_art.get("ephemeral"))

        # In server.py, we exclude ephemeral files from list. Let's verify list filtering logic:
        latest = [db_art]
        filtered = [a for a in latest if not a.get("ephemeral")]
        self.assertEqual(len(filtered), 0) # Hidden from sidebar list!

        # Promote artifact to persistent using promote_artifact
        # First, ensure the tool is registered in the discussion
        active_tools = {}
        # Simulate registration
        def _promote_artifact_impl(title: str):
            target = self.discussion.artefacts.get(title)
            if not target:
                return {"success": False, "error": "Not found"}
            self.discussion.artefacts.update(
                title=title,
                active=True,
                ephemeral=False,
                bump_version=False
            )
            self.discussion.commit()
            return {"success": True}

        # Run promotion
        res = _promote_artifact_impl("temp_config")
        self.assertTrue(res["success"])

        # Verify it is now persistent / visible in list
        promoted_art = self.discussion.artefacts.get("temp_config")
        self.assertFalse(promoted_art.get("ephemeral"))
        
        latest_after = [promoted_art]
        filtered_after = [a for a in latest_after if not a.get("ephemeral")]
        self.assertEqual(len(filtered_after), 1) # Now visible!

    def test_read_only_flag_and_metadata_rendering(self):
        # Create a read-only artifact
        art = self.discussion.artefacts.add(
            title="sales_data",
            artefact_type="data",
            content="data schema description",
            file_ext=".xlsx",
            read_only=True
        )
        self.assertTrue(art.get("read_only"))

        # Verify the context builder omits version string and appends extension for data artifacts
        self.discussion.artefacts.activate("sales_data", version=1)
        self.discussion.commit()
        zone = self.discussion.artefacts.build_artefacts_context_zone()

        # Displays as: sales_data.xlsx instead of sales_data, and omits v1/version
        self.assertIn("sales_data.xlsx", zone)
        self.assertNotIn("v1", zone)

        # Test toggle endpoint logic
        existing = self.discussion.artefacts.get("sales_data")
        new_state = not existing.get("read_only", False)
        self.discussion.artefacts.update(
            title="sales_data",
            read_only=new_state,
            bump_version=False
        )
        self.discussion.commit()
        toggled = self.discussion.artefacts.get("sales_data")
        self.assertFalse(toggled.get("read_only"))

if __name__ == "__main__":
    unittest.main()
