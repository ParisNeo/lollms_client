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


class TestFileImportEngine(unittest.TestCase):
    """Tests YAML frontmatter skill detection and data bundle fusion."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_import_")
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_import",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_yaml_frontmatter_skill_detection(self):
        """Verify .md files with --- name: ... --- are parsed as SKILL type."""
        skill_file = Path(self.tmp_workspace) / "my_skill.md"
        skill_content = """---
name: My Custom Skill
category: testing
description: A test skill
---

# Skill Body
Use this rule when testing.
"""
        skill_file.write_text(skill_content, encoding="utf-8")

        res = self.discussion.import_file(
            path=skill_file,
            mode="text",
            activate=True
        )

        art = res["text_artefact"]
        self.assertEqual(art["type"], ArtefactType.SKILL)
        self.assertEqual(art["title"], "My Custom Skill")
        self.assertIn("Skill Body", art["content"])

    def test_data_bundle_fusion(self):
        """Verify importing a folder of identical CSVs fuses them into one SQLite DB."""
        bundle_dir = Path(self.tmp_workspace) / "data_bundle"
        bundle_dir.mkdir(exist_ok=True)
        
        headers = "id,name,amount\n"
        (bundle_dir / "sales_q1.csv").write_text(headers + "1,Alice,100\n2,Bob,200\n")
        (bundle_dir / "sales_q2.csv").write_text(headers + "3,Charlie,300\n4,Dave,400\n")

        res = self.discussion.import_file(
            path=bundle_dir,
            mode="data_bundle",
            title="consolidated_sales",
            activate=True
        )

        art = res["text_artefact"]
        self.assertEqual(art["type"], ArtefactType.DATA)
        self.assertTrue(art.get("file_ext", "").endswith(".db"))
        
        ws_data = Path(self.discussion.workspace_data_path)
        db_path = ws_data / "consolidated_sales.db"
        self.assertTrue(db_path.exists(), "Consolidated SQLite DB must be created on disk.")
        
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM data_consolidated_sales")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 4, "All 4 rows from both CSVs must be fused into the table.")


if __name__ == "__main__":
    unittest.main()
