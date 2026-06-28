import unittest
import tempfile
import sqlite3
from pathlib import Path
import pandas as pd
import shutil

from lollms_client.lollms_artefact.data_files import _parse_data_file

class TestLollmsDataFiles(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to act as a workspace
        self.test_dir = Path(tempfile.mkdtemp())
        self.old_workspace_dir = None
        
        # Monkeypatch APP_WORKSPACE_DIR in lollms_client.app.server if imported
        try:
            import lollms_client.app.server as server
            self.old_workspace_dir = server.APP_WORKSPACE_DIR
            server.APP_WORKSPACE_DIR = self.test_dir
        except ImportError:
            pass

    def tearDown(self):
        # Restore old workspace directory if patched
        try:
            import lollms_client.app.server as server
            server.APP_WORKSPACE_DIR = self.old_workspace_dir
        except ImportError:
            pass
            
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parse_csv_file(self):
        # Create a mock CSV file
        csv_path = self.test_dir / "test_data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 88.0, 92.5]
        })
        df.to_csv(csv_path, index=False)

        # Run parse function
        schema, _ = _parse_data_file(csv_path, "test_data", version=1)

        # Verify schema results
        self.assertIn("# Data Interface: test_data", schema)
        self.assertIn("Format: CSV (.csv)", schema)
        self.assertIn("Total Rows: 3", schema)
        self.assertIn("id (int64)", schema)
        self.assertIn("name (str)", schema)
        self.assertIn("score (float64)", schema)
        self.assertIn("Alice", schema)

        # Verify correct versioned & active files are written to the workspace
        versioned_dest = self.test_dir / "test_data_v1.csv"
        active_dest = self.test_dir / "test_data.csv"
        self.assertTrue(versioned_dest.exists())
        self.assertTrue(active_dest.exists())

    def test_parse_excel_file(self):
        # Create a mock Excel file
        excel_path = self.test_dir / "test_data.xlsx"
        df1 = pd.DataFrame({
            "product": ["Widget", "Gizmo"],
            "price": [10.99, 15.49]
        })
        df2 = pd.DataFrame({
            "location": ["Store A", "Store B"],
            "stock": [100, 150]
        })
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)

        # Run parse function
        schema, _ = _parse_data_file(excel_path, "test_excel", version=2)

        # Verify schema results
        self.assertIn("# Data Interface: test_excel", schema)
        self.assertIn("Format: Excel (.xlsx) | Total Sheets: 2", schema)
        self.assertIn("## Sheet: Sheet1", schema)
        self.assertIn("## Sheet: Sheet2", schema)
        self.assertIn("Widget", schema)
        self.assertIn("Store A", schema)

        # Verify correct files are written to the workspace
        versioned_dest = self.test_dir / "test_excel_v2.xlsx"
        active_dest = self.test_dir / "test_excel.xlsx"
        self.assertTrue(versioned_dest.exists())
        self.assertTrue(active_dest.exists())

    def test_parse_sqlite_file(self):
        # Create a mock SQLite database file
        db_path = self.test_dir / "test_data.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create a mock table
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, age INTEGER)")
        cursor.execute("INSERT INTO users (username, age) VALUES ('admin', 30)")
        cursor.execute("INSERT INTO users (username, age) VALUES ('guest', 25)")
        conn.commit()
        conn.close()

        # Run parse function
        schema, _ = _parse_data_file(db_path, "test_db", version=1)

        # Verify schema results
        self.assertIn("# Data Interface: test_db", schema)
        self.assertIn("Format: SQLite Relational Database (.db) | Total Tables: 1", schema)
        self.assertIn("## Table: users", schema)
        self.assertIn("id (INTEGER)", schema)
        self.assertIn("username (TEXT)", schema)
        self.assertIn("admin", schema)

        # Verify correct files are written to the workspace
        versioned_dest = self.test_dir / "test_db_v1.db"
        active_dest = self.test_dir / "test_db.db"
        self.assertTrue(versioned_dest.exists())
        self.assertTrue(active_dest.exists())

if __name__ == "__main__":
    unittest.main()
