import unittest
import tempfile
import shutil
from pathlib import Path
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE

class DocMockClient:
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock-doc"
        self.binding_name = "mock"
        self.round = 0
        self.scripts = [
            [
                "Here are the tools:\n",
                "```xml\n",
                "<artifact name=\"example.py\" type=\"code\">\nprint('hello')\n</artifact>\n",
                "```\n",
                "Done."
            ]
        ]

    def count_tokens(self, text): return len(text) // 4
    def count_image_tokens(self, img): return 0
    def remove_thinking_blocks(self, text): return text
    def generate_text(self, prompt, **kwargs): return "ok"
    def reset_cancel(self): pass

    def generate_from_messages(self, messages, **kwargs):
        cb = kwargs.get("streaming_callback")
        script = self.scripts[self.round] if self.round < len(self.scripts) else []
        self.round += 1
        if cb:
            for chunk in script:
                cb(chunk, MSG_TYPE.MSG_TYPE_CHUNK, {})
        return ""

class TestCodeFenceProtection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lollms_fence_")
        self.client = DocMockClient()
        self.db = LollmsDataManager("sqlite:///:memory:")
        self.disc = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db,
            id="test_fence",
            workspace_path=self.tmp,
            autosave=True
        )

    def tearDown(self):
        self.disc.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_artifact_tag_inside_code_fence_is_ignored(self):
        res = self.disc.chat(user_message="List tools", max_reasoning_steps=2)

        content = res["ai_message"].content
        self.assertIn("```xml", content)
        self.assertIn("<artifact name=\"example.py\" type=\"code\">", content)

        # Verify no processing block was injected
        self.assertNotIn("<processing type=\"artefact\"", content)

        # Verify no artifact was actually created in the database
        art = self.disc.artefacts.get("example.py")
        self.assertIsNone(art, "Artifact should NOT have been created from inside a code fence.")

    def test_inline_artifact_tag_is_ignored(self):
        """Verify that an artifact tag embedded inline in prose is ignored."""
        self.client.scripts = [
            [
                "You can create a file using <artifact name=\"inline.py\">print('inline')</artifact> as shown.\n",
                "Done."
            ]
        ]
        res = self.disc.chat(user_message="Explain inline", max_reasoning_steps=2)

        content = res["ai_message"].content
        self.assertIn("<artifact name=\"inline.py\">", content)
        self.assertNotIn("<processing type=\"artefact\"", content)

        art = self.disc.artefacts.get("inline.py")
        self.assertIsNone(art, "Inline artifact should NOT have been created.")

    def test_inline_tool_tag_is_ignored(self):
        """Verify that a tool tag embedded inline in prose is ignored."""
        self.client.scripts = [
            [
                "To run code, you use <tool>{\"name\": \"tool_execute_python_code\", \"parameters\": {\"code\": \"print(1)\"}}</tool> like this.\n",
                "Done."
            ]
        ]
        res = self.disc.chat(user_message="Explain tool", max_reasoning_steps=2)

        content = res["ai_message"].content
        self.assertIn("<tool>", content)
        self.assertNotIn("<processing type=\"tool\"", content)

        # Ensure the tool was not executed (no tool_calls in metadata)
        self.assertEqual(len(res["ai_message"].metadata.get("tool_calls", [])), 0)

    def test_inline_unlock_file_tag_is_ignored(self):
        """Verify that an unlock_file tag embedded inline in prose is ignored."""
        self.client.scripts = [
            [
                "You can unlock files with <unlock_file>data.csv</unlock_file> if needed.\n",
                "Done."
            ]
        ]
        res = self.disc.chat(user_message="Explain unlock", max_reasoning_steps=2)

        content = res["ai_message"].content
        self.assertIn("<unlock_file>", content)
        self.assertNotIn("<processing type=\"context_update\"", content)

if __name__ == "__main__":
    unittest.main()
