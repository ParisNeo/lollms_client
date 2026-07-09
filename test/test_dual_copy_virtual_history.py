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


class ScriptedToolClient:
    """
    Mock client that deterministically streams a tool call followed by a final answer.
    This guarantees the creation of virtual_history in the ChatMixin.
    """
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


class TestDualCopyVirtualHistory(unittest.TestCase):
    """Tests the Dual-Copy Virtual History Persistence and Splicing protocol."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_dual_copy_")
        self.client = ScriptedToolClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_dual_copy",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_virtual_history_persistence_and_splicing(self):
        """
        Verify that:
        1. An agentic turn persists virtual_history to ai_msg.metadata.
        2. export() splices the raw virtual_history for the preceding turn.
        3. export() does NOT splice for the active turn (where virtual_history is passed as a parameter).
        """
        # --- Turn 1: Agentic Tool Execution ---
        def dummy_tool(**kwargs):
            return {"success": True, "output": "Tool executed successfully."}

        self.discussion.lollmsClient.tools = None
        active_tools = {
            "tool_dummy": {
                "name": "tool_dummy",
                "description": "Dummy tool for testing.",
                "parameters": [],
                "callable": dummy_tool
            }
        }

        # Script: Round 1 streams a tool call, Round 2 streams the final answer
        self.client.round_scripts = [
            ['<tool>{"name": "tool_dummy", "parameters": {}}</tool>'],
            ["The tool executed successfully. The result is 42."]
        ]

        res_turn1 = self.discussion.chat(
            user_message="Please run the dummy tool and tell me the result.",
            tools=active_tools,
            max_reasoning_steps=5
        )

        ai_msg_turn1 = res_turn1["ai_message"]

        # 1. Verify virtual_history was persisted in metadata
        self.assertIn("virtual_history", ai_msg_turn1.metadata, 
                      "virtual_history must be persisted in ai_msg.metadata after an agentic turn.")
        
        persisted_vh = ai_msg_turn1.metadata["virtual_history"]
        self.assertIsInstance(persisted_vh, list)
        self.assertGreater(len(persisted_vh), 0, "Persisted virtual_history should not be empty.")
        
        # Verify the persisted VH contains the raw tool tag and tool result
        vh_contents = [entry["content"] for entry in persisted_vh]
        self.assertTrue(
            any("<tool>" in c for c in vh_contents),
            "Persisted virtual_history must contain the raw <tool> tag."
        )
        self.assertTrue(
            any("<tool_result" in c for c in vh_contents),
            "Persisted virtual_history must contain the <tool_result> payload."
        )

        # --- Turn 2: Standard Chat (No Tools) ---
        # We need a second turn to verify export() splices Turn 1's history correctly.
        self.client.round_scripts = [
            ["I understand. Let me know if you need anything else."]
        ]

        # Export context for Turn 2 WITHOUT active virtual_history
        # This simulates the standard context assembly before generation begins.
        exported_messages = self.discussion.export(
            format_type="openai_chat",
            virtual_history=None 
        )

        # 2. Verify export() spliced the raw virtual_history for Turn 1
        # The exported context should contain the raw <tool> tag from Turn 1's virtual_history,
        # NOT the sanitized [SYSTEM_ARTIFACT_CREATED:...] or stripped processing blocks.
        exported_text = json.dumps(exported_messages)
        
        self.assertIn(
            "<tool>", exported_text,
            "export() must splice the raw virtual_history (containing <tool>) for the preceding turn."
        )
        self.assertIn(
            "<tool_result", exported_text,
            "export() must splice the raw tool_result payload for the preceding turn."
        )
        self.assertNotIn(
            "[SYSTEM_ARTIFACT_CREATED:", exported_text,
            "Sanitized system markers should NOT appear if virtual_history was spliced correctly."
        )

        # 3. Verify export() does NOT splice if active virtual_history is passed (simulating mid-turn)
        # We pass a dummy active virtual_history to simulate being mid-way through Turn 2.
        dummy_active_vh = [
            SimpleNamespace(sender_type="assistant", content="<tool>{\"name\": \"tool_active\", \"parameters\": {}}</tool>"),
            SimpleNamespace(sender_type="user", content="<tool_result name=\"tool_active\">Active result</tool_result>")
        ]

        exported_active = self.discussion.export(
            format_type="openai_chat",
            virtual_history=dummy_active_vh
        )
        exported_active_text = json.dumps(exported_active)

        # The active VH should be present
        self.assertIn("tool_active", exported_active_text, "Active virtual_history must be appended to export.")
        
        # The historical Turn 1 VH should STILL be spliced because it is a historical message
        self.assertIn("tool_dummy", exported_active_text, "Historical virtual_history must still be spliced even during an active turn.")

    def test_no_virtual_history_does_not_break_export(self):
        """
        Verify that export() functions normally when no virtual_history is present
        in historical messages or passed as a parameter.
        """
        # Add a simple user message and AI message without any tool calls
        self.discussion.add_message(sender="user", content="Hello")
        self.discussion.add_message(sender="assistant", content="Hi there!")

        # Add another user message to serve as the branch tip
        self.discussion.add_message(sender="user", content="How are you?")

        try:
            exported_messages = self.discussion.export(
                format_type="openai_chat",
                virtual_history=None
            )
            self.assertIsInstance(exported_messages, list)
            self.assertGreater(len(exported_messages), 0)
            
            # Ensure no splicing artifacts are present
            exported_text = json.dumps(exported_messages)
            self.assertNotIn("<tool>", exported_text)
            self.assertNotIn("<tool_result", exported_text)
        except Exception as e:
            self.fail(f"export() raised an exception unexpectedly when no virtual_history was present: {e}")


if __name__ == "__main__":
    unittest.main()
