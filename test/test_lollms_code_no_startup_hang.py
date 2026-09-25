import sys
import tempfile
import shutil
import time
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_chat_core import take_workspace_snapshot


class DummyClient:
    def __init__(self):
        self.llm = self
        self.model_name = "test-model"
        self.binding_name = "test"
        self.tools = None

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


class TestLollmsCodeNoStartupHang(unittest.TestCase):
    """
    Guarantees that workspace initialization and personality creation
    complete instantaneously without performing an eager recursive crawl of massive
    folders (.git, venv, node_modules) or importing all workspace files on startup.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_no_hang_ws_")
        self.ws_path = Path(self.tmp_dir)

        # Create simulated massive folders that must never be deeply traversed
        (self.ws_path / ".git" / "objects" / "pack").mkdir(parents=True, exist_ok=True)
        for i in range(100):
            (self.ws_path / ".git" / "objects" / "pack" / f"blob_{i}.pack").write_bytes(b"git_data")

        (self.ws_path / "venv" / "Lib" / "site-packages" / "pkg").mkdir(parents=True, exist_ok=True)
        for i in range(100):
            (self.ws_path / "venv" / "Lib" / "site-packages" / "pkg" / f"mod_{i}.py").write_text("x = 1")

        # Create a few legitimate project files
        (self.ws_path / "main.py").write_text("print('hello world')", encoding="utf-8")
        (self.ws_path / "README.md").write_text("# Project", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_sync_artefact_index_with_disk_is_no_op(self):
        """_sync_artefact_index_with_disk must not eagerly import files into artefacts."""
        client = DummyClient()
        p = LollmsPersonality(
            name="test_agent",
            system_prompt="You are an engineer.",
            workspace_path=self.ws_path,
            lollms_client=client
        )

        start = time.perf_counter()
        p._sync_artefact_index_with_disk()
        elapsed = time.perf_counter() - start

        # Must execute in < 10ms
        self.assertLess(elapsed, 0.1)

        # Artefacts list should be empty initially (on-demand indexing only)
        if hasattr(p, "_artefact_manager") and p._artefact_manager:
            arts = p._artefact_manager.list()
            self.assertEqual(len(arts), 0, "No files should be eagerly imported at startup.")

    def test_personality_initialization_speed_with_big_folders(self):
        """Creating a LollmsPersonality with a large workspace must be sub-second."""
        client = DummyClient()
        start = time.perf_counter()
        p = LollmsPersonality(
            name="fast_coder",
            workspace_path=self.ws_path,
            lollms_client=client
        )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"Initialization took {elapsed:.2f}s, expected < 1s")

    def test_take_workspace_snapshot_ignores_git_and_venv(self):
        """take_workspace_snapshot must ignore .git and venv using os.walk directory pruning."""
        snapshot = take_workspace_snapshot(self.ws_path)
        snapshot_paths = [str(k).replace("\\", "/") for k in snapshot.keys()]

        for path_str in snapshot_paths:
            self.assertFalse(path_str.startswith(".git"), f"Snapshot included ignored folder: {path_str}")
            self.assertFalse(path_str.startswith("venv"), f"Snapshot included ignored folder: {path_str}")

        self.assertIn("main.py", snapshot_paths)
        self.assertIn("README.md", snapshot_paths)

    def test_on_demand_file_unlock_works(self):
        """Files in workspace can still be unlocked on-demand despite eager indexing being disabled."""
        client = DummyClient()
        p = LollmsPersonality(
            name="test_agent",
            workspace_path=self.ws_path,
            lollms_client=client
        )
        res = p.change_file_visibility(["main.py"], "load")
        self.assertIn("✅", res.get("status_str", ""))
        self.assertIn("print('hello world')", res.get("loaded_contents", {}).get("main.py", ""))


if __name__ == "__main__":
    unittest.main()