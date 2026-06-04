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
        # Create a read-only artifact
        art = self.discussion.artefacts.add(
            title="sales_data",
            artefact_type="data",
            content="data schema description",
            file_ext=".xlsx",
            read_only=True
        )
        self.assertTrue(art.get("read_only"))

        # Verify the context builder omits version string and appends extension for data artifacts
        self.discussion.artefacts.activate("sales_data", version=1)
        self.discussion.commit()
        zone = self.discussion.artefacts.build_artefacts_context_zone()

        # Displays as: sales_data.xlsx instead of sales_data, and omits v1/version
        self.assertIn("sales_data.xlsx", zone)
        self.assertNotIn("v1", zone)

        # Test toggle endpoint logic
        existing = self.discussion.artefacts.get("sales_data")
        new_state = not existing.get("read_only", False)
        self.discussion.artefacts.update(
            title="sales_data",
            read_only=new_state,
            bump_version=False
        )
        self.discussion.commit()
        toggled = self.discussion.artefacts.get("sales_data")
        self.assertFalse(toggled.get("read_only"))

    def test_in_memory_sqlite_sql_query_operations(self):
        # Create a mock data file (CSV)
        import pandas as pd
        workspace_dir = Path("./data_workspace")
        workspace_dir.mkdir(exist_ok=True)
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

    def test_lcp_sql_and_python_query_tools(self):
        # 1. Create mock data file in the workspace
        import pandas as pd
        workspace_dir = Path("./data_workspace")
        workspace_dir.mkdir(exist_ok=True)
        csv_path = workspace_dir / "user_salaries_v1.csv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [120000, 95000, 110000]
        })
        df.to_csv(csv_path, index=False)

        # Create active unversioned fallback
        df.to_csv(workspace_dir / "user_salaries.csv", index=False)

        # 2. Register data artifact in active discussion
        art = self.discussion.artefacts.add(
            title="user_salaries",
            artefact_type="data",
            content="Mock schema",
            file_ext=".csv",
            version=1,
            read_only=False
        )

        # Set up a temporary active branch message so discussion_instance.active_branch_id is valid
        user_msg = self.discussion.add_message(sender="user", content="Query data")
        ai_msg = self.discussion.add_message(sender="assistant", content="Analyzing")

        # 3. Test LCP SQL Query Tool
        from lollms_client.tools_bindings.lcp.default_tools.execute_sql_query.execute_sql_query import tool_execute_sql_query
        
        sql_res = tool_execute_sql_query(
            sql_query="SELECT name FROM user_salaries WHERE salary > 100000",
            discussion_instance=self.discussion
        )
        self.assertTrue(sql_res["success"])
        self.assertIn("Alice", sql_res["output"])
        self.assertIn("Charlie", sql_res["output"])

        # 4. Test LCP Python Data Query Tool
        from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query.execute_python_data_query import tool_execute_python_data_query
        
        python_res = tool_execute_python_data_query(
            code="df['salary'] = df['salary'] + 5000",
            discussion_instance=self.discussion
        )
        self.assertTrue(python_res["success"])
        
        # Verify the file was updated and a new version is registered
        updated_art = self.discussion.artefacts.get("user_salaries")
        self.assertEqual(updated_art["version"], 2)

        # Clean up files
        for f in (workspace_dir / "user_salaries_v1.csv", workspace_dir / "user_salaries_v2.csv", workspace_dir / "user_salaries.csv"):
            if f.exists():
                f.unlink()


    def test_lcp_sql_and_python_query_tools(self):
        # 1. Create mock data file in the workspace
        import pandas as pd
        workspace_dir = Path("./data_workspace")
        workspace_dir.mkdir(exist_ok=True)
        csv_path = workspace_dir / "user_salaries_v1.csv"
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [120000, 95000, 110000]
        })
        df.to_csv(csv_path, index=False)

        # Create active unversioned fallback
        df.to_csv(workspace_dir / "user_salaries.csv", index=False)

        # 2. Register data artifact in active discussion
        art = self.discussion.artefacts.add(
            title="user_salaries",
            artefact_type="data",
            content="Mock schema",
            file_ext=".csv",
            version=1,
            read_only=False
        )

        # Set up a temporary active branch message so discussion_instance.active_branch_id is valid
        user_msg = self.discussion.add_message(sender="user", content="Query data")
        ai_msg = self.discussion.add_message(sender="assistant", content="Analyzing")

        # 3. Test LCP SQL Query Tool
        from lollms_client.tools_bindings.lcp.default_tools.execute_sql_query.execute_sql_query import tool_execute_sql_query

        sql_res = tool_execute_sql_query(
            sql_query="SELECT name FROM user_salaries WHERE salary > 100000",
            discussion_instance=self.discussion
        )
        self.assertTrue(sql_res["success"])
        self.assertIn("Alice", sql_res["output"])
        self.assertIn("Charlie", sql_res["output"])

        # 4. Test LCP Python Data Query Tool
        from lollms_client.tools_bindings.lcp.default_tools.execute_python_data_query.execute_python_data_query import tool_execute_python_data_query

        python_res = tool_execute_python_data_query(
            code="df['salary'] = df['salary'] + 5000",
            discussion_instance=self.discussion
        )
        self.assertTrue(python_res["success"])

        # Verify the file was updated and a new version is registered
        updated_art = self.discussion.artefacts.get("user_salaries")
        self.assertEqual(updated_art["version"], 2)

        # Clean up files
        for f in (workspace_dir / "user_salaries_v1.csv", workspace_dir / "user_salaries_v2.csv", workspace_dir / "user_salaries.csv"):
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
