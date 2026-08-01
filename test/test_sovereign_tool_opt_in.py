import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors


class TelemetryMockClient:
    """
    Mock client that intercepts the system prompt passed to the LLM to extract
    the exact tools registered for the turn.
    """
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock-telemetry"
        self.binding_name = "mock"
        self.tools = None
        self.captured_system_prompt = ""
        self.captured_messages = []

    def count_tokens(self, text): return len(text) // 4
    def count_image_tokens(self, img): return 0
    def remove_thinking_blocks(self, text): return text
    def get_ctx_size(self, model_name=None): return 8192
    def reset_cancel(self): pass

    def generate_from_messages(self, messages, **kwargs):
        self.captured_messages = messages
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            self.captured_system_prompt = part.get("text", "")
                            break
                else:
                    self.captured_system_prompt = str(content)
                break
        
        cb = kwargs.get("streaming_callback")
        if cb:
            cb("Test complete.<done/>", MSG_TYPE.MSG_TYPE_CHUNK, {})
        return "Test complete.<done/>"

    def generate_text(self, prompt, **kwargs):
        return "ok"


class TestSovereignToolOptIn(unittest.TestCase):
    """Verifies that chat() strictly adheres to the Sovereign Opt-In Doctrine for tools."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lollms_tool_test_")
        self.client = TelemetryMockClient()
        self.db = LollmsDataManager("sqlite:///:memory:")
        self.disc = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db,
            id="test_tool_opt_in",
            workspace_path=self.tmp,
            autosave=True
        )

    def tearDown(self):
        self.disc.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _extract_active_tools_from_prompt(self) -> list:
        """Parses the captured system prompt to extract the list of active tools."""
        prompt = self.client.captured_system_prompt
        if "=== TOOLS AVAILABLE ===" not in prompt:
            return []
        
        tools_section = prompt.split("=== TOOLS AVAILABLE ===")[1]
        if "=== END TOOLS ===" in tools_section:
            tools_section = tools_section.split("=== END TOOLS ===")[0]
        
        tools = []
        for line in tools_section.strip().splitlines():
            line = line.strip()
            if line.startswith("- "):
                # Format: "- tool_name(params): description"
                tool_name = line[2:].split("(")[0].strip()
                tools.append(tool_name)
        
        return tools

    def test_1_no_tools_by_default(self):
        """Verify that NO default LCP tools are injected if tools=None and no data files exist."""
        ASCIIColors.cyan("\n▶ Test 1: No tools activated by default")
        
        self.disc.chat(
            user_message="Hello",
            tools=None,
            enable_data_tools=True,
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertEqual(active_tools, [], f"Expected no tools, but found: {active_tools}")

    def test_2_data_tools_auto_mount_with_data_files(self):
        """Verify semantic_data_engineer is mounted if data files exist and enable_data_tools=True."""
        ASCIIColors.cyan("\n▶ Test 2: Data tools auto-mount with data files")
        
        # Create a dummy data file in the workspace
        ws_data = Path(self.disc.workspace_data_path)
        ws_data.mkdir(parents=True, exist_ok=True)
        (ws_data / "data.csv").write_text("col1,col2\n1,2")
        
        self.disc.chat(
            user_message="Analyze this data",
            tools=None,
            enable_data_tools=True,
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertIn("tool_execute_python_data_query", active_tools, 
                      "semantic_data_engineer tool should be auto-mounted")

    def test_3_data_tools_disabled_flag(self):
        """Verify semantic_data_engineer is NOT mounted if enable_data_tools=False."""
        ASCIIColors.cyan("\n▶ Test 3: Data tools disabled via flag")
        
        ws_data = Path(self.disc.workspace_data_path)
        ws_data.mkdir(parents=True, exist_ok=True)
        (ws_data / "data.csv").write_text("col1,col2\n1,2")
        
        self.disc.chat(
            user_message="Analyze this data",
            tools=None,
            enable_data_tools=False,
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertEqual(active_tools, [], 
                         f"Data tools should not be mounted when disabled, but found: {active_tools}")

    def test_4_explicit_tool_names_resolution(self):
        """Verify that passing a list of tool names resolves to the correct tools."""
        ASCIIColors.cyan("\n▶ Test 4: Explicit tool name resolution")
        
        # Create mock LCP binding with pre-registered tools
        mock_lcp = MagicMock()
        mock_lcp.to_chat_tool_specs.return_value = {
            "tool_internet_search": {
                "name": "tool_internet_search",
                "description": "Searches the internet.",
                "parameters": [{"name": "query", "type": "str"}],
                "callable": lambda query: {"success": True}
            },
            "tool_calculator": {
                "name": "tool_calculator",
                "description": "Performs calculations.",
                "parameters": [{"name": "expression", "type": "str"}],
                "callable": lambda expression: {"success": True}
            },
            "tool_unused": {
                "name": "tool_unused",
                "description": "Should not be activated.",
                "parameters": [],
                "callable": lambda: None
            }
        }
        self.client.tools = mock_lcp
        
        self.disc.chat(
            user_message="Search for AI news",
            tools=["tool_internet_search", "tool_calculator"],
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertIn("tool_internet_search", active_tools)
        self.assertIn("tool_calculator", active_tools)
        self.assertNotIn("tool_unused", active_tools, "Unrequested tools should not be activated")

    def test_5_unknown_tool_name_handling(self):
        """Verify that requesting unknown tool names does not crash and warns gracefully."""
        ASCIIColors.cyan("\n▶ Test 5: Unknown tool name handling")
        
        mock_lcp = MagicMock()
        mock_lcp.to_chat_tool_specs.return_value = {
            "tool_known": {
                "name": "tool_known",
                "description": "A known tool.",
                "parameters": [],
                "callable": lambda: None
            }
        }
        self.client.tools = mock_lcp
        
        # Should not raise an exception
        self.disc.chat(
            user_message="Do something",
            tools=["tool_known", "tool_nonexistent"],
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertIn("tool_known", active_tools)
        self.assertNotIn("tool_nonexistent", active_tools)

    def test_6_custom_callable_dict(self):
        """Verify that passing a dict of callables directly registers them correctly."""
        ASCIIColors.cyan("\n▶ Test 6: Custom callable dictionary injection")
        
        def my_custom_tool(param1: str) -> dict:
            return {"success": True, "output": f"Processed {param1}"}
        
        custom_tools = {
            "tool_my_custom": {
                "name": "tool_my_custom",
                "description": "A custom user-supplied tool.",
                "parameters": [{"name": "param1", "type": "str", "description": "Input value"}],
                "callable": my_custom_tool
            }
        }
        
        self.disc.chat(
            user_message="Run custom tool",
            tools=custom_tools,
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertIn("tool_my_custom", active_tools)

    def test_7_code_execution_flag(self):
        """Verify that enable_code_execution=True registers the python execution tool."""
        ASCIIColors.cyan("\n▶ Test 7: Code execution flag")
        
        # Create mock LCP binding
        mock_lcp = MagicMock()
        mock_lcp.to_chat_tool_specs.return_value = {
            "tool_execute_python_code": {
                "name": "tool_execute_python_code",
                "description": "Executes Python code.",
                "parameters": [{"name": "code", "type": "str"}],
                "callable": lambda code: {"success": True}
            }
        }
        self.client.tools = mock_lcp
        
        self.disc.chat(
            user_message="Run this code",
            enable_code_execution=True,
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertIn("tool_execute_python_code", active_tools, 
                      "tool_execute_python_code should be registered when enable_code_execution=True")

    def test_8_code_execution_flag_disabled(self):
        """Verify that enable_code_execution=False does NOT register the python execution tool."""
        ASCIIColors.cyan("\n▶ Test 8: Code execution flag disabled")
        
        mock_lcp = MagicMock()
        mock_lcp.to_chat_tool_specs.return_value = {
            "tool_execute_python_code": {
                "name": "tool_execute_python_code",
                "description": "Executes Python code.",
                "parameters": [{"name": "code", "type": "str"}],
                "callable": lambda code: {"success": True}
            }
        }
        self.client.tools = mock_lcp
        
        self.disc.chat(
            user_message="Run this code",
            enable_code_execution=False,
            max_reasoning_steps=1
        )
        
        active_tools = self._extract_active_tools_from_prompt()
        self.assertNotIn("tool_execute_python_code", active_tools,
                         "tool_execute_python_code should NOT be registered when enable_code_execution=False")


if __name__ == "__main__":
    unittest.main()