import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from unittest.mock import MagicMock

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE


class ScriptedMockClient:
    """Yields deterministic token chunks per reasoning round."""
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock"
        self.binding_name = "mock"
        self.round_scripts = []
        self.current_round = 0

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 0

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def reset_cancel(self):
        pass

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated text generation."

    def generate_from_messages(self, messages: list, **kwargs) -> str:
        callback = kwargs.get("streaming_callback")
        script = self.round_scripts[self.current_round] if self.current_round < len(self.round_scripts) else []
        self.current_round += 1
        
        if callback:
            for chunk in script:
                callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK, {})
        return ""


class TestAgenticToolOrchestration(unittest.TestCase):
    """Tests phantom tool interception, loop blockers, and CWD management."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_orchestration_")
        self.client = ScriptedMockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_orchestration",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_phantom_tool_interception(self):
        """Verify hallucinated tool calls are intercepted and corrected."""
        self.client.round_scripts = [
            ['<tool>{"name": "tool_does_not_exist", "parameters": {}}</tool>'],
            ["Sorry, I cannot do that."]
        ]

        res = self.discussion.chat(
            user_message="Use a non-existent tool",
            max_reasoning_steps=3
        )
        
        content = res["ai_message"].content
        self.assertIn("Tool call blocked", content)
        self.assertIn("not available", content)

    def test_success_loop_blocker(self):
        """Verify the system blocks identical successful tool calls to prevent infinite loops."""
        def dummy_tool(**kwargs):
            return {"success": True, "output": "Done"}
            
        self.discussion.lollmsClient.tools = None
        active_tools = {
            "tool_dummy": {
                "name": "tool_dummy",
                "description": "Dummy",
                "parameters": [],
                "callable": dummy_tool
            }
        }

        self.client.round_scripts = [
            ['<tool>{"name": "tool_dummy", "parameters": {}}</tool>'],
            ['<tool>{"name": "tool_dummy", "parameters": {}}</tool>'],
            ["Finished."]
        ]

        res = self.discussion.chat(
            user_message="Run dummy tool twice",
            tools=active_tools,
            max_reasoning_steps=5
        )

        content = res["ai_message"].content
        self.assertIn("Repetitive successful tool call blocked", content)

    def test_lcp_tool_cwd_isolation(self):
        """Verify CWD is switched to workspace for LCP tools and restored."""
        original_cwd = os.getcwd()
        captured_cwd = []
        
        def mock_lcp_execute(tool_name, params, **kwargs):
            captured_cwd.append(os.getcwd())
            return {"success": True, "output": "ok"}

        mock_lcp = MagicMock()
        mock_lcp.execute_tool = mock_lcp_execute
        mock_lcp.to_chat_tool_specs = MagicMock(return_value={})
        
        self.discussion.lollmsClient.tools = mock_lcp
        
        active_tools = {
            "tool_lcp_test": {
                "name": "tool_lcp_test",
                "description": "LCP test",
                "parameters": []
            }
        }

        self.client.round_scripts = [
            ['<tool>{"name": "tool_lcp_test", "parameters": {}}</tool>'],
            ["Done."]
        ]

        self.discussion.chat(
            user_message="Run LCP tool",
            tools=active_tools,
            max_reasoning_steps=3
        )

        self.assertEqual(len(captured_cwd), 1)
        expected_cwd = Path(self.discussion.workspace_data_path).resolve()
        self.assertEqual(Path(captured_cwd[0]).resolve(), expected_cwd)
        self.assertEqual(Path(os.getcwd()).resolve(), Path(original_cwd).resolve())


if __name__ == "__main__":
    unittest.main()
