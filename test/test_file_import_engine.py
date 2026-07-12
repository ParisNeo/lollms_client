import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_artefact import ArtefactVisibility

class DummyClient:
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "dummy"
        self.binding_name = "dummy"
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    def count_image_tokens(self, img) -> int:
        return 0
    def remove_thinking_blocks(self, text: str) -> str:
        return text

class TestFileImportEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = DummyClient()

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_import_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_import_session",
            workspace_path=self.tmp_workspace
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def _create_dummy_file(self, name: str, content: str = "dummy content") -> Path:
        file_path = Path(self.tmp_workspace) / "workspace_data" / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def test_text_mode_csv_import(self):
        """Verify CSV imported in TEXT mode is fully visible and NOT treated as DATA."""
        csv_path = self._create_dummy_file("data.csv", "col1,col2\n1,2\n3,4")

        res = self.discussion.import_file(csv_path, mode="text")
        art = res["text_artefact"]

        self.assertIsNotNone(art)
        self.assertEqual(art["type"], ArtefactType.DOCUMENT)
        self.assertEqual(art["visibility"], ArtefactVisibility.FULL)
        self.assertNotIn("file_ext", art, "file_ext should NOT be injected for text imports.")
        self.assertIn("col1,col2", art["content"], "Full text must be present.")

    def test_text_mode_docx_import(self):
        """Verify DOCX imported in TEXT mode is fully visible and NOT treated as DATA."""
        # Simulate a basic docx file (we don't need real docx parsing for this logic test)
        docx_path = self._create_dummy_file("mock.docx", "PK\x03\x04") 

        # Mock the docx extractor to avoid python-docx parsing errors on dummy data
        with patch("lollms_client.lollms_artefact.file_import._extract_docx_text", return_value="Mocked DOCX Text"):
            res = self.discussion.import_file(docx_path, mode="text")
            art = res["text_artefact"]

        self.assertIsNotNone(art)
        self.assertEqual(art["type"], ArtefactType.DOCUMENT)
        self.assertEqual(art["visibility"], ArtefactVisibility.FULL)
        self.assertNotIn("file_ext", art, "file_ext should NOT be injected for text imports.")

    def test_data_mode_csv_import(self):
        """Verify CSV imported in DATA mode triggers the .lam Dual-Stream system."""
        csv_path = self._create_dummy_file("schema.csv", "id,name\n1,Alice")

        res = self.discussion.import_file(csv_path, mode="data")
        art = res["text_artefact"]

        self.assertIsNotNone(art)
        self.assertEqual(art["type"], ArtefactType.DATA)
        self.assertEqual(art.get("file_ext"), ".csv", "file_ext MUST be injected for data imports.")
        self.assertIsNotNone(art.get("logical_content"), "DATA mode must produce a logical twin (.lam).")
        self.assertIsNotNone(art.get("physical_data"), "DATA mode must retain raw physical bytes.")

    def test_tool_output_default_visibility(self):
        """Verify files created by tool execution default to TREE_UNLOCKABLE [U]."""
        # Simulate a tool creating a new file in the workspace
        tool_output_path = Path(self.discussion.workspace_data_path) / "tool_output.log"
        tool_output_path.parent.mkdir(parents=True, exist_ok=True)
        tool_output_path.write_text("Execution finished successfully.", encoding="utf-8")

        # Run the sync to mimic ChatMixin's post-tool execution scan
        self.discussion.sync_workspace_to_artefacts()

        art = self.discussion.artefacts.get("tool_output.log")
        self.assertIsNotNone(art, "Tool output should be registered as an artifact.")
        self.assertEqual(
            art["visibility"], 
            ArtefactVisibility.TREE_UNLOCKABLE, 
            "Tool outputs MUST default to [U] to prevent context pollution."
        )

    def test_artefact_builder_default_visibility(self):
        """Verify files created by <artifact> tags default to FULL [C]."""
        self.discussion.artefacts.add(
            title="script.py",
            artefact_type=ArtefactType.CODE,
            content="print('hello')",
            active=True # Simulates auto_activate_artefacts=True
        )

        art = self.discussion.artefacts.get("script.py")
        self.assertIsNotNone(art)
        self.assertEqual(
            art["visibility"], 
            ArtefactVisibility.FULL, 
            "Artifact builders MUST default to [C] when auto_activate is True."
        )

    def test_sync_ignores_venv_and_hidden_folders(self):
        """Verify that sync_workspace_to_artefacts ignores venv, .venv, and .folders."""
        ws_dir = Path(self.discussion.workspace_data_path)

        # Create files that should be ignored
        (ws_dir / "venv" / "lib").mkdir(parents=True, exist_ok=True)
        (ws_dir / "venv" / "lib" / "python3.10").write_text("pyc", encoding="utf-8")

        (ws_dir / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (ws_dir / ".venv" / "bin" / "activate").write_text("sh", encoding="utf-8")

        (ws_dir / ".hidden_folder").mkdir(parents=True, exist_ok=True)
        (ws_dir / ".hidden_folder" / "secret.txt").write_text("secret", encoding="utf-8")

        # Create a file that should be indexed
        (ws_dir / "main.py").write_text("print('main')", encoding="utf-8")

        report = self.discussion.sync_workspace_to_artefacts()

        self.assertIsNotNone(self.discussion.artefacts.get("main.py"), "main.py should be indexed.")
        self.assertIsNone(self.discussion.artefacts.get("activate"), "venv/bin/activate should be ignored.")
        self.assertIsNone(self.discussion.artefacts.get("python3.10"), "venv/lib/python3.10 should be ignored.")
        self.assertIsNone(self.discussion.artefacts.get("secret.txt"), ".hidden_folder/secret.txt should be ignored.")

if __name__ == "__main__":
    unittest.main()
