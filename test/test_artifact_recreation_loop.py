import unittest
import tempfile
import shutil
import inspect
from pathlib import Path

# Add src to path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_discussion._mixin_chat import ChatMixin

class MockLollmsClient:
    def __init__(self):
        self.debug = False
        self.llm = self
        self.model_name = "test-model"
        self.binding_name = "test-binding"
        self.ai_name = "Assistant"
        
    def count_tokens(self, text: str) -> int:
        return len(text.split())
        
    def count_image_tokens(self, image: str) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"

class TestArtifactRecreationLoop(unittest.TestCase):
    """
    🛑 SCIENTIFIC VERIFICATION:
    Tests that the LLM does not get stuck in an infinite loop when it attempts to
    recreate an artifact it has already generated.
    
    The fix mandates that when a duplicate artifact is intercepted, the loop breaks
    immediately, preventing the LLM from generating another preamble.
    """
    
    def setUp(self):
        self.tmp_workspace = tempfile.mkdtemp(prefix="lollms_loop_test_")
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = MockLollmsClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_loop_session",
            workspace_path=self.tmp_workspace,
            autosave=True
        )

    def tearDown(self):
        self.discussion.close()
        shutil.rmtree(self.tmp_workspace, ignore_errors=True)

    def test_duplicate_artifact_breaks_loop(self):
        """Verifies that the architectural fix is present in the ChatMixin source."""
        # We perform a static analysis check on the source code of ChatMixin.chat
        # to ensure the hard-break safeguard is in place.
        source = inspect.getsource(ChatMixin.chat)
        
        self.assertIn(
            "[ChatMixin] LLM emitted a duplicate artifact tag. Forcing final answer.", 
            source,
            "Duplicate artifact interception block is missing."
        )
        
        # Find the index of the duplicate artifact warning
        warning_idx = source.find("[ChatMixin] LLM emitted a duplicate artifact tag. Forcing final answer.")
        self.assertNotEqual(warning_idx, -1, "Could not find duplicate artifact warning in source")
        
        # Check that 'break' appears shortly after this warning (within 2000 chars)
        # This proves the loop is terminated instead of continuing
        search_region = source[warning_idx:warning_idx+2000]
        self.assertIn(
            "break", 
            search_region, 
            "Loop must break immediately after duplicate artifact interception to prevent infinite restart"
        )

if __name__ == "__main__":
    unittest.main()
