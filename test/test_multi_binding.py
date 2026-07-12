import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_core import LollmsClient
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from lollms_client.lollms_tts_binding import LollmsTTSBinding


class TestMultiBindingSupport(unittest.TestCase):
    """Validates the multi-binding mounting and switching logic."""

    def setUp(self):
        self.mock_llm_master = MagicMock(spec=LollmsLLMBinding)
        self.mock_llm_extra = MagicMock(spec=LollmsLLMBinding)
        self.mock_tti_master = MagicMock(spec=LollmsTTIBinding)
        self.mock_tts_master = MagicMock(spec=LollmsTTSBinding)

        self.patcher_llm = patch(
            "lollms_client.lollms_llm_binding.LollmsLLMBindingManager.create_binding",
            side_effect=self._mock_llm_factory
        )
        self.patcher_tti = patch(
            "lollms_client.lollms_tti_binding.LollmsTTIBindingManager.create_binding",
            side_effect=self._mock_tti_factory
        )
        self.patcher_tts = patch(
            "lollms_client.lollms_tts_binding.LollmsTTSBindingManager.create_binding",
            side_effect=self._mock_tts_factory
        )

        self.patcher_llm.start()
        self.patcher_tti.start()
        self.patcher_tts.start()
        self.addCleanup(self.patcher_llm.stop)
        self.addCleanup(self.patcher_tti.stop)
        self.addCleanup(self.patcher_tts.stop)

    def _mock_llm_factory(self, binding_name, **kwargs):
        if binding_name == "openai":
            self.mock_llm_master.binding_name = "openai"
            return self.mock_llm_master
        elif binding_name == "ollama":
            self.mock_llm_extra.binding_name = "ollama"
            return self.mock_llm_extra
        return None

    def _mock_tti_factory(self, binding_name, **kwargs):
        if binding_name == "openai":
            self.mock_tti_master.binding_name = "openai"
            return self.mock_tti_master
        return None

    def _mock_tts_factory(self, binding_name, **kwargs):
        if binding_name == "openai":
            self.mock_tts_master.binding_name = "openai"
            return self.mock_tts_master
        return None

    def test_master_binding_registration(self):
        """Ensure the primary binding is automatically registered as 'master'."""
        lc = LollmsClient(
            llm_binding_name="openai",
            tti_binding_name="openai",
            tts_binding_name="openai",
        )

        self.assertIn("master", lc.llms)
        self.assertEqual(lc.llm, lc.llms["master"])
        self.assertEqual(lc._active_llm_alias, "master")

        self.assertIn("master", lc.ttis)
        self.assertEqual(lc.tti, lc.ttis["master"])

        self.assertIn("master", lc.tts_bindings)
        self.assertEqual(lc.tts, lc.tts_bindings["master"])

    def test_extra_llm_mounting_and_switching(self):
        """Verify extra LLMs are mounted and can be swapped with the master."""
        lc = LollmsClient(
            llm_binding_name="openai",
            extra_llms={
                "local_model": {
                    "binding_name": "ollama",
                    "binding_config": {"host_address": "http://localhost:11434"}
                }
            },
        )

        self.assertIn("local_model", lc.llms)
        self.assertEqual(lc.llms["local_model"], self.mock_llm_extra)

        lc.mount_llm("local_model")
        self.assertEqual(lc.llm, self.mock_llm_extra)
        self.assertEqual(lc._active_llm_alias, "local_model")

        lc.mount_llm("master")
        self.assertEqual(lc.llm, self.mock_llm_master)
        self.assertEqual(lc._active_llm_alias, "master")

    def test_invalid_alias_mount(self):
        """Ensure mounting a non-existent alias fails gracefully and returns False."""
        lc = LollmsClient(llm_binding_name="openai")
        
        success = lc.mount_llm("non_existent_alias")
        self.assertFalse(success)
        self.assertEqual(lc._active_llm_alias, "master")
        self.assertEqual(lc.llm, self.mock_llm_master)


if __name__ == "__main__":
    unittest.main()
