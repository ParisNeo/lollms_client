import unittest
from pathlib import Path
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager, ArtefactType
from lollms_client.lollms_core import LollmsClient
from lollms_client.lollms_types import LCPResult

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

class TestLollmsArtifactTools(unittest.TestCase):
    def setUp(self):
        # In-memory db manager for test isolation
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = DummyClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_session",
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()

    def test_lcp_result_serialization(self):
        # Test basic dataclass serialization
        res = LCPResult(
            success=True,
            output="Done",
            prompt_injection="Injected Prompt",
            images=["img1_b64"],
            code_blocks=[{"content": "print('hello')", "language": "python"}],
            paths=["data_workspace/file.csv"],
            sources=[{"title": "Doc", "content": "Text", "source": "http"}]
        )
        data = res.to_dict()
        self.assertTrue(data["success"])
        self.assertEqual(data["output"], "Done")
        self.assertEqual(data["prompt_injection"], "Injected Prompt")
        self.assertEqual(data["images"][0], "img1_b64")
        self.assertEqual(data["code_blocks"][0]["language"], "python")
        self.assertEqual(data["paths"][0], "data_workspace/file.csv")
        self.assertEqual(data["sources"][0]["title"], "Doc")

        # Test from_dict
        restored = LCPResult.from_dict(data)
        self.assertEqual(restored.output, "Done")
        self.assertEqual(restored.sources[0]["title"], "Doc")

    def test_ephemeral_artifact_and_promotion(self):
        # Create an ephemeral artifact
        art = self.discussion.artefacts.add(
            title="temp_config",
            artefact_type=ArtefactType.DOCUMENT,
            content="temp content",
            ephemeral=True
        )
        self.assertTrue(art.get("ephemeral"))
        
        # Verify it is in database but flagged
        db_art = self.discussion.artefacts.get("temp_config")
        self.assertTrue(db_art.get("ephemeral"))

        # In server.py, we exclude ephemeral files from list. Let's verify list filtering logic:
        latest = [db_art]
        filtered = [a for a in latest if not a.get("ephemeral")]
        self.assertEqual(len(filtered), 0) # Hidden from sidebar list!

        # Promote artifact to persistent using promote_artifact
        # First, ensure the tool is registered in the discussion
        active_tools = {}
        # Simulate registration
        def _promote_artifact_impl(title: str):
            target = self.discussion.artefacts.get(title)
            if not target:
                return {"success": False, "error": "Not found"}
            self.discussion.artefacts.update(
                title=title,
                active=True,
                ephemeral=False,
                bump_version=False
            )
            self.discussion.commit()
            return {"success": True}

        # Run promotion
        res = _promote_artifact_impl("temp_config")
        self.assertTrue(res["success"])

        # Verify it is now persistent / visible in list
        promoted_art = self.discussion.artefacts.get("temp_config")
        self.assertFalse(promoted_art.get("ephemeral"))
        
        latest_after = [promoted_art]
        filtered_after = [a for a in latest_after if not a.get("ephemeral")]
        self.assertEqual(len(filtered_after), 1) # Now visible!

    def test_read_only_flag_and_metadata_rendering(self):
        # Create a fresh isolated discussion to avoid pollution from other tests
        import tempfile
        tmp_workspace = tempfile.mkdtemp(prefix="lollms_readonly_test_")
        isolated_disc = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=LollmsDataManager("sqlite:///:memory:"),
            id="test_readonly_session",
            autosave=True,
            workspace_path=tmp_workspace
        )

        # Create a read-only artifact
        art = isolated_disc.artefacts.add(
            title="sales_data",
            artefact_type="data",
            content="data schema description",
            file_ext=".xlsx",
            read_only=True
        )
        self.assertTrue(art.get("read_only"))

        # Verify the context builder omits version string and appends extension for data artifacts
        isolated_disc.artefacts.activate("sales_data", version=1)
        isolated_disc.commit()
        zone = isolated_disc.artefacts.build_artefacts_context_zone()

        # Displays as: sales_data.xlsx instead of sales_data, and omits v1/version
        self.assertIn("sales_data.xlsx", zone)
        self.assertNotIn("v1", zone)

        # Test toggle endpoint logic on the isolated discussion before cleanup
        existing = isolated_disc.artefacts.get("sales_data")
        new_state = not existing.get("read_only", False)
        isolated_disc.artefacts.update(
            title="sales_data",
            read_only=new_state,
            bump_version=False
        )
        isolated_disc.commit()
        toggled = isolated_disc.artefacts.get("sales_data")
        self.assertFalse(toggled.get("read_only"))

        # Cleanup
        isolated_disc.close()
        import shutil
        shutil.rmtree(tmp_workspace, ignore_errors=True)

    def test_in_memory_sqlite_sql_query_operations(self):
        # Create a mock data file (CSV)
        import pandas as pd
        workspace_dir = Path(self.discussion.workspace_data_path)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        csv_path = workspace_dir / "user_salaries.csv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [120000, 95000, 110000]
        })
        df.to_csv(csv_path, index=False)

        art = self.discussion.artefacts.add(
            title="user_salaries",
            artefact_type="data",
            content="Mock schema",
            file_ext=".csv",
            version=1,
            read_only=False
        )

        # Re-fetch local tool executor
        import sqlite3
        conn = sqlite3.connect(":memory:")
        df.to_sql("user_salaries", conn, index=False, if_exists="replace")
        
        # Test SELECT query execution on memory SQL
        df_res = pd.read_sql_query("SELECT * FROM user_salaries WHERE salary > 100000", conn)
        self.assertEqual(len(df_res), 2)
        self.assertIn("Alice", df_res["name"].values)
        
        # Test INSERT write query execution on memory SQL
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_salaries (name, salary) VALUES ('David', 130000)")
        conn.commit()
        
        df_res_after = pd.read_sql_query("SELECT * FROM user_salaries", conn)
        self.assertEqual(len(df_res_after), 4)
        self.assertIn("David", df_res_after["name"].values)
        conn.close()

        # Clean up mock file
        if csv_path.exists():
            csv_path.unlink()
        # Clean up any versioned files
        for f in workspace_dir.glob("user_salaries*.csv"):
            f.unlink()

    def test_external_sql_connection_operations(self):
        # 1. Create a dummy SQLite DB on disk to simulate the "external" database
        import sqlite3
        import pandas as pd
        workspace_dir = Path(self.discussion.workspace_data_path)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        db_path = workspace_dir / "external_test_db.sqlite"
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, role TEXT)")
        cursor.execute("INSERT INTO employees (name, role) VALUES ('Eve', 'Architect')")
        cursor.execute("INSERT INTO employees (name, role) VALUES ('Frank', 'Auditor')")
        conn.commit()
        conn.close()

        # 2. Write a '.sqlconn' connection file in the workspace
        sqlconn_path = workspace_dir / "my_external_db.sqlconn"
        conn_info = {
            "type": "sql_connection",
            "dialect": "sqlite",
            "database": str(db_path.resolve())
        }
        import json
        with open(sqlconn_path, "w", encoding="utf-8") as f:
            json.dump(conn_info, f, indent=2)

        # 3. Test schema parsing on .sqlconn file
        from lollms_client.lollms_artefact.data_files import _parse_data_file
        schema, _, raw_physical_bytes = _parse_data_file(sqlconn_path, "my_external_db", version=1)

        self.assertIn("Format: Remote Relational Database (sqlite)", schema)
        self.assertIn("## Table: employees", schema)
        self.assertIn("Eve", schema)

        # 4. Register as active session artifact with physical_data
        art = self.discussion.artefacts.add(
            title="my_external_db",
            artefact_type="data",
            content=schema,
            file_ext=".sqlconn",
            version=1,
            read_only=False,
            physical_data=raw_physical_bytes
        )
        self.discussion.commit()

        # 5. Test LCP SQL Query on .sqlconn
        import os
        original_cwd = os.getcwd()
        os.chdir(self.discussion.workspace_data_path)
        from lollms_client.tools_bindings.lcp.default_tools.execute_sql_query.execute_sql_query import tool_execute_sql_query
        sql_res = tool_execute_sql_query(
            sql_query="SELECT name FROM employees WHERE role = 'Architect'",
            file_name="my_external_db.sqlconn"
        )
        os.chdir(original_cwd)
        self.assertTrue(sql_res["success"], f"SQL tool failed: {sql_res.get('error', 'Unknown error')}")
        self.assertIn("Eve", sql_res["output"])

        # 6. Test LCP Python Sandbox on .sqlconn
        from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query.execute_python_data_query import tool_execute_python_data_query

        py_code = (
            "import pandas as pd\n"
            "df = pd.read_sql_query('SELECT * FROM employees', conn)\n"
            "print(f'Total employees: {len(df)}')\n"
        )

        os.chdir(self.discussion.workspace_data_path)
        python_res = tool_execute_python_data_query(
            code=py_code
        )
        os.chdir(original_cwd)
        self.assertTrue(python_res["success"])
        self.assertIn("Total employees: 2", python_res["output"])

        # Clean up files
        for f in (db_path, sqlconn_path, Path(self.discussion.workspace_data_path) / "my_external_db_v1.sqlconn", Path(self.discussion.workspace_data_path) / "my_external_db.sqlconn"):
            if f.exists():
                f.unlink()


    def test_lcp_sql_and_python_query_tools(self):
        # 1. Create mock data file in the workspace
        import pandas as pd
        import io
        workspace_dir = Path(self.discussion.workspace_data_path)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        csv_path = workspace_dir / "user_salaries_v1.csv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [120000, 95000, 110000]
        })
        df.to_csv(csv_path, index=False)

        # Create active unversioned fallback
        df.to_csv(workspace_dir / "user_salaries.csv", index=False)

        # Read raw CSV bytes for physical_data
        raw_csv_bytes = csv_path.read_bytes()

        # 2. Register data artifact in active discussion with physical_data
        art = self.discussion.artefacts.add(
            title="user_salaries",
            artefact_type="data",
            content="Mock schema",
            file_ext=".csv",
            version=1,
            read_only=False,
            physical_data=raw_csv_bytes
        )

        # Set up a temporary active branch message so discussion_instance.active_branch_id is valid
        user_msg = self.discussion.add_message(sender="user", content="Query data")
        ai_msg = self.discussion.add_message(sender="assistant", content="Analyzing")

        # 3. Test LCP SQL Query Tool
        from lollms_client.tools_bindings.lcp.default_tools.execute_sql_query.execute_sql_query import tool_execute_sql_query

        import os
        original_cwd = os.getcwd()
        os.chdir(self.discussion.workspace_data_path)
        sql_res = tool_execute_sql_query(
            sql_query="SELECT name FROM user_salaries WHERE salary > 100000",
            file_name="user_salaries.csv"
        )
        os.chdir(original_cwd)
        self.assertTrue(sql_res["success"])
        self.assertIn("Alice", sql_res["output"])
        self.assertIn("Charlie", sql_res["output"])

        # 4. Test LCP Python Data Query Tool
        from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query.execute_python_data_query import tool_execute_python_data_query

        original_cwd = os.getcwd()
        os.chdir(self.discussion.workspace_data_path)
        python_res = tool_execute_python_data_query(
            code="df['salary'] = df['salary'] + 5000",
            discussion_instance=self.discussion
        )
        os.chdir(original_cwd)
        self.assertTrue(python_res["success"])

        # Verify the file was updated and a new version is registered
        updated_art = self.discussion.artefacts.get("user_salaries")
        self.assertEqual(updated_art["version"], 2)

        # Clean up files
        for f in (Path(self.discussion.workspace_data_path) / "user_salaries_v1.csv", Path(self.discussion.workspace_data_path) / "user_salaries_v2.csv", Path(self.discussion.workspace_data_path) / "user_salaries.csv"):
            if f.exists():
                f.unlink()

    def test_token_count_caching(self):
        # Create a mock client with a tracking counter for count_tokens calls
        class MockLLM:
            def __init__(self):
                self.calls = 0
            def count_tokens(self, text):
                self.calls += 1
                return len(text) // 4

        # Wrap in LollmsClient shell
        test_client = LollmsClient()
        test_client.llm = MockLLM()

        # Test first call (uncached)
        c1 = test_client.count_tokens("This is a long string to verify correct MD5 caching.")
        self.assertEqual(test_client.llm.calls, 1)

        # Test second identical call (cached)
        c2 = test_client.count_tokens("This is a long string to verify correct MD5 caching.")
        self.assertEqual(test_client.llm.calls, 1) # Still 1!
        self.assertEqual(c1, c2)

        # Test different call (uncached)
        c3 = test_client.count_tokens("Different string.")
        self.assertEqual(test_client.llm.calls, 2)

    def test_latex_to_pdf_export_detection(self):
        # Create a mock LaTeX artifact
        art = self.discussion.artefacts.add(
            title="receipt",
            artefact_type="document",
            content="\\documentclass{article}\\begin{document}Receipt\\end{document}",
            language="latex"
        )
        # Verify it is detected as latex
        content = art.get("content", "").strip()
        is_latex = (
            art.get("language") in ("latex", "tex") or
            content.startswith("\\documentclass")
        )
        self.assertTrue(is_latex)

    def test_multi_version_export_and_import(self):
        # 1. Create multiple versions of an artifact
        self.discussion.artefacts.add(
            title="script.py",
            artefact_type="code",
            content="print('v1')",
            language="python",
            version=1,
            commit_message="First release"
        )
        self.discussion.artefacts.update(
            title="script.py",
            new_content="print('v2')",
            language="python",
            bump_version=True,
            commit_message="Second release"
        )
        self.discussion.artefacts.update(
            title="script.py",
            new_content="print('v3')",
            language="python",
            bump_version=True,
            commit_message="Third release"
        )

        # 2. Export the artifact with all its versions
        exported_data = self.discussion.export_artefact("script.py")
        self.assertIsNotNone(exported_data)
        self.assertEqual(exported_data["title"], "script.py")
        self.assertEqual(len(exported_data["versions"]), 3)
        self.assertEqual(exported_data["versions"][0]["content"], "print('v1')")
        self.assertEqual(exported_data["versions"][1]["content"], "print('v2')")
        self.assertEqual(exported_data["versions"][2]["content"], "print('v3')")

        # 3. Create a second fresh discussion
        another_discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_session_2",
            autosave=True
        )

        # 4. Import the artifact object into the second discussion
        imported_art = another_discussion.import_artefact(exported_data, activate=True)
        self.assertIsNotNone(imported_art)
        self.assertEqual(imported_art["title"], "script.py")
        self.assertEqual(imported_art["version"], 3)
        self.assertEqual(imported_art["content"], "print('v3')")

        # Verify the entire version history is present in the second discussion
        history = another_discussion.artefacts.get_version_history("script.py")
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["version"], 1)
        self.assertEqual(history[1]["version"], 2)
        self.assertEqual(history[2]["version"], 3)
        self.assertTrue(history[2]["is_active"])

        another_discussion.close()

if __name__ == "__main__":
    unittest.main()
