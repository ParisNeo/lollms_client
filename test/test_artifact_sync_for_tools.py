"""
Test suite to verify that ALL active artifacts are properly synced to workspace
before tool execution, enabling the LLM to build artifacts and then call custom tools.
"""

import unittest
from pathlib import Path
import shutil
from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion


class TestArtifactSyncForTools(unittest.TestCase):
    """Test artifact synchronization workflow for custom tool execution."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.workspace_dir = Path("./data_workspace")
        cls.test_artifacts = [
            {"title": "test_circuit.cir", "type": "document", "content": "* Test SPICE circuit\nV1 1 0 DC 5V\nR1 1 0 1k", "language": None},
            {"title": "analysis_script.py", "type": "code", "content": "print('Hello from analysis script')", "language": "python"},
            {"title": "dataset.csv", "type": "data", "content": "col1,col2,col3\n1,2,3\n4,5,6", "file_ext": ".csv"},
            {"title": "notes.md", "type": "document", "content": "# Test Notes\nThis is a test document", "language": "markdown"},
        ]
        
        # Clean workspace before tests
        if cls.workspace_dir.exists():
            shutil.rmtree(cls.workspace_dir)
        cls.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if cls.workspace_dir.exists():
            shutil.rmtree(cls.workspace_dir)
    
    def setUp(self):
        """Create a fresh discussion for each test."""
        # Mock LollmsClient for testing
        class MockLollmsClient:
            def count_tokens(self, text):
                return len(text.split())
        
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(lollms_client=self.client)
    
    def test_all_active_artifacts_synced(self):
        """Verify that ALL active artifacts (not just data) are synced to workspace."""
        # Add multiple artifact types
        for art_data in self.test_artifacts:
            self.discussion.artefacts.add(
                title=art_data["title"],
                artefact_type=art_data["type"],
                content=art_data["content"],
                language=art_data.get("language"),
                file_ext=art_data.get("file_ext"),
                active=True
            )
        
        # Force sync
        sync_ws, sync_files = self.discussion.artefacts.sync_all_active_to_disk()
        
        # Verify all artifacts are on disk
        for art_data in self.test_artifacts:
            expected_path = self.workspace_dir / art_data["title"]
            self.assertTrue(
                expected_path.exists(),
                f"Artifact '{art_data['title']}' should exist at {expected_path}"
            )
        
        # Verify sync report includes all files
        self.assertEqual(len(sync_files), len(self.test_artifacts))
    
    def test_cir_file_accessible_to_tools(self):
        """Verify .cir files (non-data artifacts) are synced and accessible."""
        cir_content = "* RLC Circuit\nV1 1 0 AC 10V\nR1 1 2 1k\nC1 2 0 1uF\nL1 2 0 10mH"
        
        self.discussion.artefacts.add(
            title="rlc_circuit.cir",
            artefact_type="document",
            content=cir_content,
            active=True
        )
        
        # Sync
        self.discussion.artefacts.sync_all_active_to_disk()
        
        # Verify .cir file exists in workspace
        cir_path = self.workspace_dir / "rlc_circuit.cir"
        self.assertTrue(cir_path.exists())
        
        # Verify content matches
        loaded_content = cir_path.read_text()
        self.assertEqual(loaded_content, cir_content)
    
    def test_workspace_path_verification(self):
        """Test the workspace path verification diagnostic."""
        # Add an artifact
        self.discussion.artefacts.add(
            title="test_file.txt",
            artefact_type="document",
            content="Test content",
            active=True
        )
        
        # Sync
        self.discussion.artefacts.sync_all_active_to_disk()
        
        # Verify workspace structure
        self.assertTrue(self.workspace_dir.exists())
        self.assertTrue((self.workspace_dir / "test_file.txt").exists())
        
        # List all files (mimicking the diagnostic in execute_python_data_query)
        all_files = list(self.workspace_dir.rglob("*"))
        file_count = sum(1 for f in all_files if f.is_file())
        self.assertGreaterEqual(file_count, 1)
    
    def test_artifact_build_then_use_pattern(self):
        """Test the workflow: LLM builds artifact, then tool uses it."""
        # Step 1: LLM creates a .cir artifact
        cir_content = "* Test Circuit\nV1 1 0 DC 5V\nR1 1 0 10k"
        art = self.discussion.artefacts.add(
            title="my_circuit.cir",
            artefact_type="document",
            content=cir_content,
            active=True
        )
        
        # Step 2: Sync makes it available to tools
        self.discussion.artefacts.sync_all_active_to_disk()
        
        # Step 3: Verify tool can access the file
        cir_path = self.workspace_dir / "my_circuit.cir"
        self.assertTrue(cir_path.exists())
        
        # Simulate what a custom tool would do
        loaded = cir_path.read_text()
        self.assertIn("V1 1 0 DC 5V", loaded)
        self.assertIn("R1 1 0 10k", loaded)
    
    def test_inactive_artifacts_not_synced(self):
        """Verify that inactive artifacts are NOT synced to workspace."""
        # Add active and inactive artifacts
        self.discussion.artefacts.add(
            title="active_file.txt",
            artefact_type="document",
            content="Active content",
            active=True
        )
        
        self.discussion.artefacts.add(
            title="inactive_file.txt",
            artefact_type="document",
            content="Inactive content",
            active=False
        )
        
        # Sync only active
        self.discussion.artefacts.sync_all_active_to_disk()
        
        # Verify only active file exists
        self.assertTrue((self.workspace_dir / "active_file.txt").exists())
        self.assertFalse((self.workspace_dir / "inactive_file.txt").exists())


if __name__ == "__main__":
    unittest.main()
