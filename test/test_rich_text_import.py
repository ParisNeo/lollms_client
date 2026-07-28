import unittest
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager


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


class TestRichTextImport(unittest.TestCase):
    """Tests that rich text files (.docx, .pdf, .pptx) are imported as documents with extracted text."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_richtext_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_richtext_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def _create_mock_docx(self, path: Path, text: str):
        """Creates a minimal valid .docx file containing the given text."""
        import zipfile
        import xml.etree.ElementTree as ET

        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        ET.register_namespace("", ns)
        
        body = ET.Element(f"{{{ns}}}body")
        p = ET.SubElement(body, f"{{{ns}}}p")
        r = ET.SubElement(p, f"{{{ns}}}r")
        t = ET.SubElement(r, f"{{{ns}}}t")
        t.text = text
        
        document = ET.Element(f"{{{ns}}}document")
        document.append(body)
        
        xml_str = ET.tostring(document, encoding="utf-8", xml_declaration=True)
        
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
            zf.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
            zf.writestr("word/_rels/document.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
            zf.writestr("word/document.xml", xml_str)

    def test_docx_import_creates_single_artifact_with_text(self):
        """Verifies that importing a .docx file creates a single artifact with extracted text, not a binary metadata placeholder."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        docx_path = ws_path / "lollms_client_doc.docx"
        self._create_mock_docx(docx_path, "This is the real content of the docx file.")
        
        res = self.discussion.import_file(
            path=docx_path,
            mode="text",
            title="lollms_client_doc.docx",
            activate=True
        )
        
        self.assertIsNotNone(res.get("text_artefact"), "Import should return a text artifact.")
        
        art = self.discussion.artefacts.get("lollms_client_doc.docx")
        self.assertIsNotNone(art, "Artifact should exist in the manager.")
        
        self.assertEqual(art["version"], 1, "Should only have one version (v1). No v2 metadata version should be created.")
        
        self.assertNotIn("### Data File:", art["content"], "Content should not be a binary metadata placeholder.")
        self.assertIn("This is the real content of the docx file.", art["content"], "Content should contain the extracted text from the docx.")
        self.assertEqual(art["type"], "document", "Artifact type should be 'document'.")

    def test_py_file_import_remains_asis(self):
        """Verifies that pure text files like .py are imported as-is without conversion."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)
        
        py_path = ws_path / "script.py"
        py_path.write_text("print('hello')", encoding="utf-8")
        
        res = self.discussion.import_file(
            path=py_path,
            mode="text",
            title="script.py",
            activate=True
        )
        
        self.assertIsNotNone(res.get("text_artefact"))
        art = self.discussion.artefacts.get("script.py")
        self.assertIsNotNone(art)
        self.assertEqual(art["version"], 1)
        self.assertEqual(art["content"], "print('hello')")
        self.assertEqual(art["type"], "code")

if __name__ == "__main__":
    unittest.main()
