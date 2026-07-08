import unittest
import tempfile
import shutil
import json
from pathlib import Path

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_artefact import ArtefactVisibility


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


class TestArtifactVersioningAndExport(unittest.TestCase):
    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_versioning_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = DummyClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_versioning_session",
            autosave=True,
            workspace_path=self.tmp_workspace
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_version_bumping_and_history(self):
        self.discussion.artefacts.add(
            title="script.py",
            artefact_type=ArtefactType.CODE,
            content="print('v1')",
            language="python",
            version=1,
            active=True
        )
        self.discussion.artefacts.update(
            title="script.py",
            new_content="print('v2')",
            language="python",
            bump_version=True,
            active=True
        )
        self.discussion.artefacts.update(
            title="script.py",
            new_content="print('v3')",
            language="python",
            bump_version=True,
            active=True
        )

        latest = self.discussion.artefacts.get("script.py")
        self.assertEqual(latest["version"], 3)
        self.assertEqual(latest["content"], "print('v3')")

        history = self.discussion.artefacts.get_version_history("script.py")
        self.assertEqual(len(history), 3)
        self.assertTrue(history[2]["is_active"])

    def test_standalone_archive_export_import(self):
        self.discussion.artefacts.add(
            title="archive_test.md",
            artefact_type=ArtefactType.DOCUMENT,
            content="Version 1 content",
            version=1
        )
        self.discussion.artefacts.update(
            title="archive_test.md",
            new_content="Version 2 content",
            bump_version=True
        )

        archive_path = Path(self.tmp_workspace) / "archive_test.laa"
        self.discussion.artefacts.export_artefact_to_archive("archive_test.md", archive_path)
        self.assertTrue(archive_path.exists())

        self.discussion.artefacts.remove("archive_test.md")
        self.assertIsNone(self.discussion.artefacts.get("archive_test.md"))

        imported = self.discussion.artefacts.import_artefact_from_archive(archive_path, activate=True)
        self.assertIsNotNone(imported)
        self.assertEqual(imported["title"], "archive_test.md")
        self.assertEqual(imported["version"], 2)
        self.assertEqual(imported["content"], "Version 2 content")

        history = self.discussion.artefacts.get_version_history("archive_test.md")
        self.assertEqual(len(history), 2)

    def test_multi_version_export_import(self):
        self.discussion.artefacts.add(
            title="multi_ver.py",
            artefact_type=ArtefactType.CODE,
            content="def foo(): pass",
            language="python",
            version=1
        )
        self.discussion.artefacts.update(
            title="multi_ver.py",
            new_content="def bar(): pass",
            language="python",
            bump_version=True
        )

        exported_data = self.discussion.export_artefact("multi_ver.py")
        self.assertEqual(len(exported_data["versions"]), 2)

        self.discussion.artefacts.remove("multi_ver.py")
        self.assertIsNone(self.discussion.artefacts.get("multi_ver.py"))

        imported = self.discussion.import_artefact(exported_data, activate=True)
        self.assertIsNotNone(imported)
        self.assertEqual(imported["version"], 2)
        self.assertEqual(imported["content"], "def bar(): pass")

        history = self.discussion.artefacts.get_version_history("multi_ver.py")
        self.assertEqual(len(history), 2)

    def test_artifact_bundle_export_import(self):
        ws_data = Path(self.discussion.workspace_data_path)
        ws_data.mkdir(parents=True, exist_ok=True)
        
        (ws_data / "main.py").write_text("print('main')")
        sub_dir = ws_data / "utils"
        sub_dir.mkdir(exist_ok=True)
        (sub_dir / "helper.py").write_text("def help(): pass")

        self.discussion.artefacts.add(
            title="main.py",
            artefact_type=ArtefactType.CODE,
            content="print('main')",
            language="python",
            active=True
        )
        self.discussion.artefacts.add(
            title="utils/helper.py",
            artefact_type=ArtefactType.CODE,
            content="def help(): pass",
            language="python",
            active=True
        )
        self.discussion.commit()

        bundle_path = Path(self.tmp_workspace) / "bundle.lab"
        self.discussion.artefacts.export_artefact_bundle(
            paths=[ws_data / "main.py", ws_data / "utils" / "helper.py"],
            output_path=bundle_path
        )
        self.assertTrue(bundle_path.exists())

        self.discussion.artefacts.remove("main.py")
        self.discussion.artefacts.remove("utils/helper.py")
        # remove() automatically cleans up physical files from workspace_data.
        # We only need to clean up the subdirectory if it exists.
        if sub_dir.exists():
            shutil.rmtree(sub_dir, ignore_errors=True)

        imported_arts = self.discussion.artefacts.import_artefact_bundle(bundle_path, activate=True)
        self.assertEqual(len(imported_arts), 2)

        main_art = self.discussion.artefacts.get("main.py")
        helper_art = self.discussion.artefacts.get("utils/helper.py")
        self.assertIsNotNone(main_art)
        self.assertIsNotNone(helper_art)
        self.assertEqual(main_art["content"], "print('main')")
        self.assertEqual(helper_art["content"], "def help(): pass")
        
        self.assertTrue((ws_data / "main.py").exists())
        self.assertTrue((ws_data / "utils" / "helper.py").exists())


if __name__ == "__main__":
    unittest.main()
