import unittest
import tempfile
import shutil
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from unittest.mock import MagicMock

from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE


class ScriptedLoopMockClient:
    """
    Simulates an LLM that falls into an artifact recreation loop.
    It emits the exact same <artifact> tag three times in a row.
    """
    def __init__(self):
        self.llm = self
        self.ai_name = "Assistant"
        self.model_name = "mock-loop"
        self.binding_name = "mock"
        self.current_round = 0
        
        artifact_payload = (
            '<artifact name="rlc_lowpass_filter.asc" type="code" language="asc">\n'
            "Version 4\n"
            "V1 1 0 AC 1\n"
            "R1 1 2 50\n"
            "L1 2 3 1mH\n"
            "C1 3 0 1uF\n"
            ".ac dec 100 10 100k\n"
            ".tran 0 10m\n"
            ".backanno\n"
            ".end\n"
            "</artifact>"
        )
        
        # Round 1: Emit artifact. Round 2: Emit identical artifact. Round 3: Emit final answer.
        self.round_scripts = [
            [artifact_payload],
            [artifact_payload],
            ["The RLC filter has been created successfully."]
        ]

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


class TestArtifactRecreationLoop(unittest.TestCase):
    """Validates that the ChatMixin breaks infinite artifact recreation loops."""

    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_loop_test_")
        self.client = ScriptedLoopMockClient()
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_artifact_loop",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_artifact_recreation_loop_is_broken(self):
        """
        Scenario:
        1. LLM emits <artifact name="rlc.asc">...</artifact>
        2. System dispatches it and asks for continuation.
        3. LLM emits the EXACT SAME <artifact> tag again.
        4. System MUST intercept this loop, inject a hard stop, and force the final answer.
        5. LLM emits the final conversational answer.
        
        Assertion: The loop must terminate after 3 rounds, and the final message must contain
        the conversational answer, NOT an infinite stream of artifacts.
        """
        res = self.discussion.chat(
            user_message="create an RLC filter and show me the baud diagram",
            max_reasoning_steps=5,
            enable_artefacts=True
        )

        # The mock client has 3 rounds. If the loop guard failed, it would hit max_reasoning_steps (5)
        # and the client would run out of scripts, potentially causing errors or empty responses.
        self.assertEqual(
            self.client.current_round, 
            3, 
            f"Expected exactly 3 reasoning rounds (Artifact -> Loop Intercept -> Final Answer), got {self.client.current_round}"
        )

        final_content = res["ai_message"].content
        
        # Verify the final answer is present
        self.assertIn("The RLC filter has been created successfully.", final_content)
        
        # Verify the artifact was only created once in the database
        art = self.discussion.artefacts.get("rlc_lowpass_filter.asc")
        self.assertIsNotNone(art, "Artifact should exist in the database.")
        self.assertEqual(art["version"], 1, "Artifact should not have been updated to v2 because the second creation was blocked.")
        
        # Verify the force_final_answer flag was triggered
        self.assertTrue(
            getattr(self.discussion, "_force_final_answer", False),
            "The _force_final_answer flag should have been set to True by the loop interceptor."
        )


if __name__ == "__main__":
    unittest.main()
