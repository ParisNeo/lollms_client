import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_discussion._mixin_chat import _StreamState
from lollms_client.lollms_artefact import ArtefactType


class MockClient:
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


class TestSearchReplaceDoubleVersion(unittest.TestCase):
    """
    Regression test: Ensures a SEARCH/REPLACE patch on an existing artifact
    creates exactly ONE new version with the patched content, and does NOT
    fall through to create a second version containing the raw patch block.
    """

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_sr_double_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="sr_double_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )
        self.discussion.artefacts.add(
            title="my_skill.md",
            artefact_type=ArtefactType.SKILL,
            content="# My Skill\n\nThis is the original content.\nIt has a bug here.\n",
            language="markdown",
            version=1
        )
        self.discussion.commit()

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_patch_creates_single_version(self):
        ai_msg = SimpleNamespace(content="", thoughts="", id="msg_1")
        ss = _StreamState(
            discussion=self.discussion,
            callback=None,
            forward_artefact_chunks=False,
            ai_message=ai_msg,
            enable_artefacts=True
        )

        patch_body = """<<<<<<< SEARCH
It has a bug here.
=======
The bug is now fixed.
>>>>>>> REPLACE"""

        full_tag = f'<artifact name="my_skill.md" type="skill" language="markdown">\n{patch_body}\n</artifact>'

        ss.feed(full_tag)
        ss.flush_remaining_buffer()

        history = self.discussion.artefacts.get_version_history("my_skill.md")

        self.assertEqual(len(history), 2,
                         f"Expected exactly 2 versions (original + 1 patch), but got {len(history)}.")

        v2 = self.discussion.artefacts.get("my_skill.md", version=2)
        self.assertIsNotNone(v2, "Version 2 should exist.")
        self.assertIn("The bug is now fixed.", v2["content"],
                      "Version 2 should contain the patched content.")
        self.assertNotIn("<<<<<<< SEARCH", v2["content"],
                         "Version 2 must NOT contain the raw SEARCH/REPLACE block text.")

        v3 = self.discussion.artefacts.get("my_skill.md", version=3)
        self.assertIsNone(v3, "Version 3 should NOT exist (double-version bug).")


    def test_update_csv_does_not_create_phantom_revert_version(self):
        """
        Scenario: A CSV file is imported as a DATA artifact. The LLM updates the content.
        Expected: The update creates exactly ONE new version (v2) with the new content.
                  It must NOT create an additional version (v3) containing the old content.
        """
        title = "phantom_test.csv"
        csv_bytes = b"id,name\n1,Alice\n2,Bob\n"
        lam_content = "# CSV Schema: phantom_test.csv\nColumns: id (int), name (str)"

        # 1. Add the data artifact (v1)
        self.discussion.artefacts.add(
            title=title,
            artefact_type=ArtefactType.DATA,
            content=lam_content,
            logical_content=lam_content,
            physical_data=csv_bytes,
            file_ext=".csv",
            active=True
        )
        self.discussion.commit()

        # Verify v1
        art_v1 = self.discussion.artefacts.get(title)
        self.assertEqual(art_v1["version"], 1)
        self.assertEqual(art_v1["content"], lam_content)

        # 2. Update the content (should create v2)
        updated_csv_content = "id,name\n1,Alice\n2,Bob\n3,Charlie\n"
        updated_lam = "# CSV Schema: phantom_test.csv\nColumns: id (int), name (str)\nRows: 3"
        
        self.discussion.artefacts.update(
            title=title,
            new_content=updated_csv_content,
            logical_content=updated_lam,
            bump_version=True
        )
        self.discussion.commit()

        # 3. Check version history
        history = self.discussion.artefacts.get_version_history(title)
        self.assertEqual(len(history), 2, "There should be exactly 2 versions (v1 and v2). No phantom v3 should exist.")

        # 4. Verify the latest version has the NEW content
        art_v2 = self.discussion.artefacts.get(title)
        self.assertEqual(art_v2["version"], 2, "The latest version should be 2.")
        self.assertEqual(art_v2["content"], updated_csv_content, "The latest version must contain the updated content.")

        # 5. Verify the physical file on disk has the NEW content
        ws_data_dir = Path(self.discussion.workspace_data_path)
        csv_path = ws_data_dir / title
        self.assertTrue(csv_path.exists())
        disk_content = csv_path.read_text(encoding="utf-8")
        self.assertEqual(disk_content, updated_csv_content, "The physical file on disk must contain the updated content.")

if __name__ == "__main__":
    unittest.main()
