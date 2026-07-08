import unittest
import tempfile
import shutil
from pathlib import Path

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_artefact import ArtefactVisibility


class MockClient:
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock"
        self.binding_name = "mock"
    def count_tokens(self, text): return len(text) // 4
    def count_image_tokens(self, img): return 0
    def remove_thinking_blocks(self, text): return text
    def generate_text(self, prompt, **kwargs): return "ok"


class TestDualStreamDataProtocol(unittest.TestCase):
    """Tests the .lam (Logical) vs physical (Raw) separation for data files."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_dual_stream_")
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_dual_stream",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_csv_generates_lam_and_physical_twin(self):
        """Verify adding a CSV creates both a .lam metadata file and the raw .csv file."""
        csv_content = "name,age\nAlice,30\nBob,25"
        raw_bytes = csv_content.encode('utf-8')
        
        self.discussion.artefacts.add(
            title="users.csv",
            artefact_type=ArtefactType.DATA,
            content="Columns: name (str), age (int)",
            file_ext=".csv",
            physical_data=raw_bytes,
            logical_content="Columns: name (str), age (int)"
        )
        self.discussion.commit()

        ws_data = Path(self.discussion.workspace_data_path)
        meta_dir = Path(self.discussion.artefacts_metadata_path)

        self.assertTrue((ws_data / "users.csv").exists(), "Physical CSV twin must exist.")
        
        lam_files = list(meta_dir.rglob("users.lam"))
        self.assertEqual(len(lam_files), 1, "Logical .lam twin must exist.")
        self.assertIn("Columns: name", lam_files[0].read_text())

    def test_context_zone_injects_lam_not_raw(self):
        """Verify build_artefacts_context_zone injects the .lam content, not the raw bytes."""
        csv_content = "name,age\nAlice,30\nBob,25"
        
        self.discussion.artefacts.add(
            title="employees.csv",
            artefact_type=ArtefactType.DATA,
            content="### Data File: employees.csv\nColumns: name, age",
            file_ext=".csv",
            physical_data=csv_content.encode('utf-8'),
            logical_content="### Data File: employees.csv\nColumns: name, age",
            active=True,
            visibility=ArtefactVisibility.FULL
        )
        self.discussion.commit()

        zone = self.discussion.artefacts.build_artefacts_context_zone()
        self.assertIn("Columns: name, age", zone)
        self.assertNotIn("Alice,30", zone, "Raw CSV rows must not leak into LLM context.")


if __name__ == "__main__":
    unittest.main()
