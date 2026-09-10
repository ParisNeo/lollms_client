"""
Agentic Mode Doctrine Tests: No .versions/, No .lam (Application-Managed Versioning).

Guards the invariant that LAM twins and version snapshots are DISCUSSION-MODE
ONLY. When `disable_artefact_versioning=True` (agentic mode with potentially
thousands of files), the ArtefactManager must:
  1. Never create the `.versions/` directory.
  2. Never write `.lam` logical twins.
  3. Still write the active file to the workspace root (single source of truth).
  4. Force-disable `include_versions` when exporting `.lab` bundles.

Discussion mode (versioning enabled) must keep the Dual-Stream behavior:
  5. `.versions/` IS created and holds `_vN` snapshots.
"""

import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_artefact import ArtefactVisibility


class MockClient:
    """Mock LollmsClient for isolated testing without LLM bindings."""

    def __init__(self):
        self.llm = self
        self.model_name = "mock-model"
        self.binding_name = "mock-binding"
        self.ai_name = "Assistant"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"


class TestAgenticModeNoVersions(unittest.TestCase):
    """Agentic mode doctrine: the workspace is a plain live tree."""

    def _make_discussion(self, agentic: bool):
        tmp_workspace = tempfile.mkdtemp(prefix="lollms_agentic_test_")
        client = MockClient()
        db_manager = LollmsDataManager("sqlite:///:memory:")
        discussion = LollmsDiscussion.create_new(
            lollms_client=client,
            db_manager=db_manager,
            id="test_agentic_session",
            workspace_path=tmp_workspace,
            autosave=True,
        )
        object.__setattr__(discussion, "disable_artefact_versioning", agentic)
        ws_data_dir = Path(discussion.workspace_data_path)
        ws_data_dir.mkdir(parents=True, exist_ok=True)
        return discussion, ws_data_dir

    def _assert_no_version_artifacts(self, ws_data_dir: Path):
        versions_dir = ws_data_dir / ".versions"
        self.assertFalse(
            versions_dir.exists(),
            "AGENTIC MODE VIOLATION: `.versions/` was created while "
            "`disable_artefact_versioning=True`. Versioning must be owned by the "
            "host application (or Git), never by the ArtefactManager in agentic mode.",
        )
        lam_files = list(ws_data_dir.rglob("*.lam"))
        self.assertFalse(
            lam_files,
            "AGENTIC MODE VIOLATION: `.lam` logical twin(s) were written while "
            "`disable_artefact_versioning=True`.",
        )

    def tearDown(self):
        discussion = getattr(self, "_discussion", None)
        if discussion is not None:
            discussion.close()
        tmp = getattr(self, "_tmp_workspace", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_agentic_add_never_creates_versions_dir(self):
        """add() writes the active file but must NEVER create .versions/ or .lam twins."""
        discussion, ws_data_dir = self._make_discussion(agentic=True)
        self._discussion = discussion
        self._tmp_workspace = str(ws_data_dir.parent)

        discussion.artefacts.add(
            title="src/main.py",
            artefact_type=ArtefactType.CODE,
            content="print('agentic')",
            language="python",
            active=True,
            visibility=ArtefactVisibility.FULL,
        )
        discussion.commit()

        active_file = ws_data_dir / "src" / "main.py"
        self.assertTrue(active_file.exists())
        self.assertEqual(active_file.read_text(encoding="utf-8"), "print('agentic')")
        self._assert_no_version_artifacts(ws_data_dir)

    def test_agentic_update_overwrites_in_place_without_versions(self):
        """update() overwrites the active file in-place; version stays 1; no snapshots."""
        discussion, ws_data_dir = self._make_discussion(agentic=True)
        self._discussion = discussion
        self._tmp_workspace = str(ws_data_dir.parent)

        discussion.artefacts.add(
            title="notes.md",
            artefact_type=ArtefactType.DOCUMENT,
            content="# v1 content",
            active=True,
            visibility=ArtefactVisibility.FULL,
        )
        updated = discussion.artefacts.update(title="notes.md", new_content="# v2 content")
        discussion.commit()

        active_file = ws_data_dir / "notes.md"
        self.assertTrue(active_file.exists())
        self.assertEqual(active_file.read_text(encoding="utf-8"), "# v2 content")
        self.assertEqual(updated.get("version"), 1, "Agentic mode must not bump versions.")
        self._assert_no_version_artifacts(ws_data_dir)

    def test_agentic_sync_all_active_to_disk_skips_version_snapshots(self):
        """The heal pass re-materializes active files WITHOUT touching .versions/."""
        discussion, ws_data_dir = self._make_discussion(agentic=True)
        self._discussion = discussion
        self._tmp_workspace = str(ws_data_dir.parent)

        discussion.artefacts.add(
            title="report.md",
            artefact_type=ArtefactType.DOCUMENT,
            content="# agentic report",
            active=True,
            visibility=ArtefactVisibility.FULL,
        )
        discussion.commit()

        active_file = ws_data_dir / "report.md"
        self.assertTrue(active_file.exists())
        active_file.unlink()

        discussion.artefacts.sync_all_active_to_disk()

        self.assertTrue(
            active_file.exists(),
            "Heal pass must guarantee the active file exists on disk.",
        )
        self._assert_no_version_artifacts(ws_data_dir)

    def test_agentic_bundle_export_forces_include_versions_off(self):
        """export_artefact_bundle(include_versions=True) must be a no-op flag in agentic mode."""
        discussion, ws_data_dir = self._make_discussion(agentic=True)
        self._discussion = discussion
        self._tmp_workspace = str(ws_data_dir.parent)

        discussion.artefacts.add(
            title="bundle_me.py",
            artefact_type=ArtefactType.CODE,
            content="x = 42",
            language="python",
            active=True,
            visibility=ArtefactVisibility.FULL,
        )
        discussion.commit()

        out_path = ws_data_dir.parent / "bundle.lab"
        discussion.artefacts.export_artefact_bundle(
            paths=["bundle_me.py"],
            output_path=out_path,
            include_versions=True,
        )

        self.assertTrue(out_path.exists())
        with zipfile.ZipFile(out_path, "r") as zf:
            names = zf.namelist()

        self.assertIn("bundle_me.py", names)
        self.assertFalse(
            any(n.startswith("_versions/") for n in names),
            "AGENTIC MODE VIOLATION: version snapshots were bundled while "
            "`disable_artefact_versioning=True`. The `include_versions` flag must "
            "be force-disabled in agentic mode.",
        )

    def test_discussion_mode_still_creates_versions(self):
        """Regression guard: discussion mode keeps the Dual-Stream snapshot behavior."""
        discussion, ws_data_dir = self._make_discussion(agentic=False)
        self._discussion = discussion
        self._tmp_workspace = str(ws_data_dir.parent)

        discussion.artefacts.add(
            title="main.py",
            artefact_type=ArtefactType.CODE,
            content="print('discussion mode')",
            language="python",
            active=True,
            visibility=ArtefactVisibility.FULL,
        )
        discussion.commit()

        self.assertTrue((ws_data_dir / "main.py").exists())
        versions_dir = ws_data_dir / ".versions"
        self.assertTrue(
            versions_dir.is_dir(),
            "DISCUSSION MODE REGRESSION: `.versions/` was NOT created while "
            "versioning is enabled. The agentic-mode gate over-scoped and broke "
            "the Dual-Stream protocol.",
        )
        snapshots = [f for f in versions_dir.rglob("main_v1.py") if f.is_file()]
        self.assertTrue(snapshots, "The `_v1` snapshot must exist in discussion mode.")


if __name__ == "__main__":
    unittest.main()