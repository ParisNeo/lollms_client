import unittest
import tempfile
import shutil
from pathlib import Path

# Add src to path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_artefact import ArtefactType, ArtefactVisibility

class MockLollmsClient:
    def __init__(self):
        self.debug = False
        self.llm = self
        self.model_name = "test-model"
        self.binding_name = "test-binding"
        self.ai_name = "Assistant"
        
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def count_image_tokens(self, image: str) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"

class TestSVGArtifactSync(unittest.TestCase):
    """Test that SVG artifacts are correctly written to disk even if classified as IMAGE type."""
    
    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_svg_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_svg_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_svg_artifact_with_image_type_is_written_to_disk(self):
        """
        Simulates the exact bug condition:
        - LLM emits `<artifact name="circuit.svg" type="image">...</artifact>`
        - The system must write the SVG XML content to disk as text, not skip it.
        """
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>'
        
        # Add the artifact exactly as the LLM would (type="image")
        art = self.discussion.artefacts.add(
            title="circuit.svg",
            artefact_type=ArtefactType.IMAGE,  # The bug trigger
            content=svg_content,
            language="svg",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()
        
        # Verify it exists in DB
        db_art = self.discussion.artefacts.get("circuit.svg")
        self.assertIsNotNone(db_art)
        self.assertEqual(db_art["content"], svg_content)
        
        # Verify it exists on disk
        svg_file = ws_path / "circuit.svg"
        self.assertTrue(svg_file.exists(), "SVG artifact was NOT written to disk!")
        
        # Verify the content is correct and not corrupted
        disk_content = svg_file.read_text(encoding="utf-8")
        self.assertEqual(disk_content, svg_content)

if __name__ == "__main__":
    unittest.main()  
