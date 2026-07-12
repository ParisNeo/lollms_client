import unittest
import sys
import tempfile
import shutil
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_artefact import ArtefactType


class MockClient:
    """Mock client for testing artifact operations without an LLM."""
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


class TestDataArtefactCorruptionFix(unittest.TestCase):
    """
    Validates that updating a DATA artifact (e.g., SQLite .db) does not corrupt
    the physical binary file by overwriting it with the .lam logical schema text.
    """

    def setUp(self):
        self.client = MockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.tmp_dir = tempfile.mkdtemp(prefix="lollms_data_corruption_")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="data_corruption_test",
            workspace_path=self.tmp_dir,
            autosave=True
        )

        # Create a valid SQLite database in a temporary file and read its bytes
        self.tmp_db_path = Path(self.tmp_dir) / "source_test.db"
        self.conn = sqlite3.connect(str(self.tmp_db_path))
        self.conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        self.conn.execute("INSERT INTO users (name) VALUES ('Alice')")
        self.conn.commit()
        self.conn.close()
        self.db_bytes = self.tmp_db_path.read_bytes()

        # The expected valid SQLite header
        self.valid_header = b"SQLite format 3\x00"

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_update_preserves_sqlite_binary_integrity(self):
        """
        Scenario: A SQLite database is imported as a DATA artifact. The LLM updates
        the artifact's logical content (.lam schema).
        Expected: The physical .db file on disk must remain a valid SQLite database.
                  It must NOT be overwritten with the .lam schema text.
        """
        title = "test_db.db"
        lam_content = "# SQLite Database: test_db.db\n## Table: users\nColumns: id (INTEGER), name (TEXT)"

        # 1. Add the data artifact with physical bytes
        self.discussion.artefacts.add(
            title=title,
            artefact_type=ArtefactType.DATA,
            content=lam_content,
            logical_content=lam_content,
            physical_data=self.db_bytes,
            file_ext=".db",
            active=True
        )
        self.discussion.commit()

        # Verify initial state is valid
        ws_data_dir = Path(self.discussion.workspace_data_path)
        db_path = ws_data_dir / title
        self.assertTrue(db_path.exists(), "Physical .db file should exist after add()")
        self.assertEqual(db_path.read_bytes()[:16], self.valid_header, "Initial DB should have valid SQLite header")

        # 2. Update the artifact's logical content (simulates LLM editing the .lam)
        updated_lam = "# SQLite Database: test_db.db\n## Table: users\nColumns: id (INTEGER), name (TEXT), email (TEXT)"
        self.discussion.artefacts.update(
            title=title,
            new_content=updated_lam,
            logical_content=updated_lam,
            bump_version=True
        )
        self.discussion.commit()

        # 3. CRITICAL ASSERTION: The physical file must still be a valid SQLite database
        self.assertTrue(db_path.exists(), "Physical .db file should still exist after update()")
        header = db_path.read_bytes()[:16]
        self.assertEqual(header, self.valid_header,
                         f"Database corrupted after update! Header is '{header}' instead of '{self.valid_header}'. "
                         f"The .lam schema text was likely written to the .db file.")

        # 4. Verify we can still query the database
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = 1")
            row = cursor.fetchone()
            self.assertIsNotNone(row, "Database query should return a row")
            self.assertEqual(row[0], "Alice", "Database content should be preserved")
            conn.close()
        except sqlite3.DatabaseError as e:
            self.fail(f"Failed to query database after update: {e}. File may be corrupted.")

    def test_sync_to_disk_refuses_text_for_binary_db(self):
        """
        Scenario: _sync_to_disk_workspace is called directly with string content
        but no physical_data for a .db file.
        Expected: The method must refuse to write the string content to the .db file
                  to prevent silent corruption.
        """
        title = "protected_db.db"
        lam_content = "# This is schema text, not binary data"

        # Call the internal sync method directly with no physical_data
        self.discussion.artefacts._sync_to_disk_workspace(
            title=title,
            content=lam_content,
            version=1,
            atype=ArtefactType.DATA,
            file_ext=".db",
            physical_data=None,
            logical_content=lam_content
        )

        ws_data_dir = Path(self.discussion.workspace_data_path)
        db_path = ws_data_dir / title

        # The .db file should either NOT exist, or if it does, it must NOT contain the schema text
        if db_path.exists():
            content = db_path.read_bytes()
            self.assertNotIn(lam_content.encode('utf-8'), content,
                             "Schema text was written to the .db file! Binary corruption occurred.")
        # If it doesn't exist, the defense-in-depth worked by refusing to write.

    def test_update_preserves_csv_physical_bytes(self):
        """
        Scenario: A CSV file is imported as a DATA artifact. The LLM updates the schema.
        Expected: The physical .csv file on disk must contain the original CSV rows,
                  not the .lam schema text.
        """
        title = "data.csv"
        csv_bytes = b"id,name\n1,Alice\n2,Bob\n"
        lam_content = "# CSV Schema: data.csv\nColumns: id (int), name (str)"

        # 1. Add the data artifact
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

        # 2. Update the logical schema
        updated_lam = "# CSV Schema: data.csv\nColumns: id (int), name (str), email (str)"
        self.discussion.artefacts.update(
            title=title,
            new_content=updated_lam,
            logical_content=updated_lam,
            bump_version=True
        )
        self.discussion.commit()

        # 3. Verify physical file still contains CSV data
        ws_data_dir = Path(self.discussion.workspace_data_path)
        csv_path = ws_data_dir / title
        self.assertTrue(csv_path.exists(), "Physical .csv file should exist after update()")

        content = csv_path.read_bytes()
        self.assertEqual(content, csv_bytes,
                         "Physical CSV bytes should be exactly preserved after update().")
        self.assertNotIn(b"# CSV Schema", content,
                         "The .lam schema text was written to the physical .csv file! Corruption detected.")


if __name__ == "__main__":
    unittest.main()
