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


class TestRichTextMdSuffix(unittest.TestCase):
    """Tests that rich text files (.pdf, .docx, .pptx) are imported with a .md suffix to prevent binary data file interception."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_md_suffix_test_")
        self.source_dir = tempfile.mkdtemp(prefix="lollms_md_suffix_src_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_md_suffix_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)
        shutil.rmtree(self.source_dir, ignore_errors=True)

    def _create_mock_pdf(self, path: Path, text: str):
        """Creates a minimal valid .pdf file containing the given text."""
        # A minimal valid PDF structure with text
        pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td ({text}) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000232 00000 n 
0000000301 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
399
%%EOF"""
        path.write_text(pdf_content, encoding="utf-8")

    def test_pdf_import_creates_md_twin_and_preserves_text(self):
        """Verifies that importing a .pdf file creates a .pdf.md twin on disk, preserving extracted text and preventing v2 version bump."""
        ws_path = Path(self.discussion.workspace_data_path)
        ws_path.mkdir(parents=True, exist_ok=True)

        # CRITICAL FIX: Create the source PDF in an isolated source directory, 
        # NOT in the workspace_data directory. This accurately simulates importing 
        # an external file and ensures the workspace remains pristine.
        pdf_path = Path(self.source_dir) / "research_paper.pdf"
        self._create_mock_pdf(pdf_path, "This is the extracted PDF text content.")

        res = self.discussion.import_file(
            path=pdf_path,
            mode="text",
            title="research_paper.pdf",
            activate=True
        )

        self.assertIsNotNone(res.get("text_artefact"), "Import should return a text artifact.")

        art = self.discussion.artefacts.get("research_paper.pdf.md")
        self.assertIsNotNone(art, "Artifact should exist with the .md suffix in the title.")

        self.assertEqual(art["version"], 1, "Should only have one version (v1). No v2 metadata version should be created.")

        self.assertNotIn("### Data File:", art["content"], "Content should not be a binary metadata placeholder.")
        self.assertIn("This is the extracted PDF text content.", art["content"], "Content should contain the extracted text from the pdf.")
        self.assertEqual(art["type"], "document", "Artifact type should be 'document'.")

        expected_md_file = ws_path / "research_paper.pdf.md"
        self.assertTrue(expected_md_file.exists(), "The physical .pdf.md file must exist on disk.")
        self.assertNotIn("### Data File:", expected_md_file.read_text(), "Physical .pdf.md file must contain extracted text, not binary metadata.")

        unexpected_pdf_file = ws_path / "research_paper.pdf"
        self.assertFalse(unexpected_pdf_file.exists(), "The raw binary .pdf file should NOT be copied to the workspace_data directory.")

if __name__ == "__main__":
    unittest.main()
