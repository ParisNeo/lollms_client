import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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


class TestPDFTextModeOCRFix(unittest.TestCase):
    """Tests to ensure that importing a PDF in 'text' mode strictly uses text-layer extraction and never triggers OCR."""

    def setUp(self):
        self.tmp_workspace = Path("./test_workspace_pdf_text")
        self.tmp_workspace.mkdir(parents=True, exist_ok=True)
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_pdf_text_session",
            workspace_path=str(self.tmp_workspace),
            autosave=True
        )
        self.pdf_path = self.tmp_workspace / "test_doc.pdf"
        self.pdf_path.write_bytes(b"%PDF-1.4\n%Test PDF Content")

    def tearDown(self):
        self.discussion.close()
        import shutil
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    @patch('lollms_client.lollms_artefact.file_import._ensure_installed')
    def test_text_mode_uses_fitz_and_avoids_ocr(self, mock_ensure_installed):
        """
        Verify that importing a PDF in mode='text' uses PyMuPDF (fitz) for text extraction
        and does not invoke OCR fallbacks like pdf2image or docling.
        """
        from lollms_client.lollms_artefact import file_import
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is the extracted text layer."
        
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None

        with patch.dict(sys.modules, {'fitz': MagicMock(open=MagicMock(return_value=mock_doc))}):
            result = self.discussion.import_file(
                path=self.pdf_path,
                mode="text",
                activate=True
            )
            
            mock_ensure_installed.assert_any_call("pymupdf", "fitz")
            
            mock_page.get_text.assert_called_once_with("text")
            
            text_art = result.get("text_artefact")
            self.assertIsNotNone(text_art)
            self.assertIn("This is the extracted text layer.", text_art["content"])
            self.assertEqual(result["image_count"], 0)
            self.assertIsNone(result["image_artefact"])

    @patch('lollms_client.lollms_artefact.file_import._ensure_installed')
    def test_text_mode_falls_back_to_pypdf_safely(self, mock_ensure_installed):
        """
        Verify that if fitz fails to import or use, it falls back to pypdf 
        (which is also a strict text extractor, NOT OCR).
        """
        from lollms_client.lollms_artefact import file_import

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text from pypdf layer."
        mock_reader.pages = [mock_page]

        with patch.dict(sys.modules, {
            'fitz': None,
            'pypdf': MagicMock(PdfReader=MagicMock(return_value=mock_reader))
        }):
            result = self.discussion.import_file(
                path=self.pdf_path,
                mode="text",
                activate=True
            )

            mock_ensure_installed.assert_any_call("pymupdf", "fitz")
            mock_ensure_installed.assert_any_call("pypdf")

            mock_page.extract_text.assert_called_once()

            text_art = result.get("text_artefact")
            self.assertIsNotNone(text_art)
            self.assertIn("Text from pypdf layer.", text_art["content"])
            self.assertEqual(result["image_count"], 0)

    def test_text_mode_preserves_raw_text_over_lam_schema(self):
        """
        Verify that importing a file in mode='text' preserves the raw extracted text
        and does NOT trigger the .lam Dual-Stream protocol (which would overwrite the content
        with a binary data schema).
        """
        from lollms_client.lollms_artefact import file_import

        raw_text_content = "This is the raw extracted text that must be preserved."

        with patch.object(file_import, '_extract_pdf_text', return_value=raw_text_content) as mock_extract:
            result = self.discussion.import_file(
                path=self.pdf_path,
                mode="text",
                activate=True
            )

            mock_extract.assert_called_once()

            text_art = result.get("text_artefact")
            self.assertIsNotNone(text_art)

            self.assertEqual(text_art["content"], raw_text_content)
            self.assertNotIn("Data Interface:", text_art["content"])
            self.assertNotIn("Format: SQLite", text_art["content"])


if __name__ == "__main__":
    unittest.main()
