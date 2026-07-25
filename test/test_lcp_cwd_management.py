import os
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion._db import LollmsDataManager
from lollms_client.lollms_discussion._message import LollmsMessage
from lollms_client.lollms_types import MSG_TYPE


class MockLollmsClient:
    def __init__(self):
        self.debug = False
        self.llm = self
        self.model_name = "test-model"
        self.binding_name = "test-binding"
        self.ai_name = "Assistant"
        self.tools = None
        
    def count_tokens(self, text: str) -> int:
        return len(text.split())
        
    def count_image_tokens(self, image: str) -> int:
        return 256

    def remove_thinking_blocks(self, text: str) -> str:
        return text

    def generate_text(self, prompt: str, **kwargs) -> str:
        return "Simulated response"

    def generate_from_messages(self, messages, **kwargs):
        pass


class FakeLCPBinding:
    """A mock LCP binding that records the CWD during execution."""
    def __init__(self):
        self.executed = False
        self.cwd_during_exec = None
        self.raise_exception = False

    def execute_tool(self, tool_name, params, lollms_client_instance=None, discussion_instance=None):
        self.executed = True
        self.cwd_during_exec = Path(os.getcwd()).resolve()

        if self.raise_exception:
            raise RuntimeError("Simulated tool crash!")

        return {"success": True, "output": "Executed successfully"}

    def to_chat_tool_specs(self, **kwargs):
        return {}


@pytest.fixture
def isolated_discussion(tmp_path, monkeypatch):
    """Creates a LollmsDiscussion instance with an isolated workspace."""
    mock_client = MockLollmsClient()
    
    db_file_path = tmp_path / "discussion.db"
    db_manager = LollmsDataManager(f"sqlite:///{db_file_path}")

    disc = LollmsDiscussion.create_new(
        lollms_client=mock_client,
        db_manager=db_manager,
        id="test_cwd_disc",
        workspace_path=str(tmp_path),
        autosave=False
    )
    
    yield disc
    
    disc.close()


def _make_streaming_mock(tool_name: str, tool_params: dict):
    """
    Creates a mock generate_from_messages that accurately simulates
    the LLM streaming a <tool> tag.
    """
    tool_json = json.dumps({"name": tool_name, "parameters": tool_params})
    full_payload = f"<tool>{tool_json}</tool>"

    def mock_generate(*args, **kwargs):
        callback = kwargs.get('streaming_callback')
        if callback:
            # Send the entire payload in one chunk to avoid partial-tag 
            # state machine fragmentation in _StreamState.
            callback(full_payload, MSG_TYPE.MSG_TYPE_CHUNK, {})
        return ""

    return mock_generate


def _setup_discussion_mocks(discussion, msg_id="msg_1"):
    """Helper to setup the necessary mocks for the chat loop to execute tools."""
    dummy_msg = SimpleNamespace(
        id=msg_id, 
        sender="user", 
        sender_type="user", 
        content="test",
        parent_id=None, 
        discussion_id=discussion.id, 
        images=[],
        active_images=[], 
        metadata={}, 
        created_at=datetime.utcnow(),
        tokens=10,
        raw_content="test",
        thoughts=None,
        scratchpad=None,
        binding_name="test",
        model_name="test",
        generation_speed=10.0
    )
    
    discussion._message_index = {msg_id: dummy_msg}
    discussion.active_branch_id = msg_id
    
    discussion.get_branch = MagicMock(return_value=[LollmsMessage(discussion, dummy_msg)])
    
    # Mock add_message to return a fake AI message wrapper to prevent DB errors
    def fake_add_message(**kwargs):
        ai_dummy = SimpleNamespace(
            id="ai_msg_1",
            sender="assistant",
            sender_type="assistant",
            content="",
            parent_id=msg_id,
            discussion_id=discussion.id,
            images=[],
            active_images=[],
            metadata={},
            created_at=datetime.utcnow(),
            tokens=0,
            raw_content="",
            thoughts=None,
            scratchpad=None,
            binding_name="test",
            model_name="test",
            generation_speed=10.0
        )
        discussion._message_index["ai_msg_1"] = ai_dummy
        return LollmsMessage(discussion, ai_dummy)
        
    discussion.add_message = MagicMock(side_effect=fake_add_message)


def test_lcp_dispatch_sets_and_restores_cwd(isolated_discussion, tmp_path):
    """
    Verifies that the ChatMixin orchestrator sets the CWD to the isolated
    workspace_data directory before LCP tool execution and restores it.

    Uses 'tool_dummy_cwd_test' — a dummy tool that reads a file from the CWD
    to verify workspace isolation.
    """
    # Arrange
    original_cwd = Path(os.getcwd()).resolve()
    fake_lcp = FakeLCPBinding()

    isolated_discussion.lollmsClient.tools = fake_lcp

    workspace_data_dir = Path(isolated_discussion.workspace_data_path)
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    (workspace_data_dir / "dummy.txt").write_text("TEST_CONTENT")

    active_tools = {
        "tool_dummy_cwd_test": {
            "name": "tool_dummy_cwd_test",
            "description": "Dummy tool that reads a file from the workspace CWD for testing.",
            "parameters": [],
        }
    }
    
    _setup_discussion_mocks(isolated_discussion)
    
    isolated_discussion.lollmsClient.generate_from_messages = _make_streaming_mock(
        "tool_dummy_cwd_test", 
        {"file_name": "dummy.txt"}
    )
    
    # Act
    isolated_discussion.chat(
        user_message="Run dummy tool",
        tools=active_tools,
        branch_tip_id="msg_1",
        add_user_message=False,
        max_reasoning_steps=1
    )
    
    # Assert
    assert fake_lcp.executed, "LCP binding execute_tool was never called"
    expected_cwd = Path(isolated_discussion.workspace_data_path).resolve()
    assert fake_lcp.cwd_during_exec == expected_cwd, "CWD was not set to workspace_data during LCP execution"
    
    final_cwd = Path(os.getcwd()).resolve()
    assert final_cwd == original_cwd, "CWD was not restored after LCP execution"


def test_lcp_dispatch_restores_cwd_on_crash(isolated_discussion, tmp_path):
    """
    Verifies that the CWD is restored even if the LCP tool raises an unhandled exception.

    Uses 'tool_dummy_crash_test' — a dummy tool that intentionally raises
    RuntimeError to test CWD restoration on exceptions.
    """
    # Arrange
    original_cwd = Path(os.getcwd()).resolve()
    fake_lcp = FakeLCPBinding()
    fake_lcp.raise_exception = True

    isolated_discussion.lollmsClient.tools = fake_lcp

    active_tools = {
        "tool_dummy_crash_test": {
            "name": "tool_dummy_crash_test",
            "description": "Dummy tool that intentionally crashes to test CWD restoration.",
            "parameters": [],
        }
    }
    
    _setup_discussion_mocks(isolated_discussion)
    
    isolated_discussion.lollmsClient.generate_from_messages = _make_streaming_mock(
        "tool_dummy_crash_test", 
        {}
    )
    
    # Act
    isolated_discussion.chat(
        user_message="Run crashing dummy tool",
        tools=active_tools,
        branch_tip_id="msg_1",
        add_user_message=False,
        max_reasoning_steps=1
    )
    
    # Assert
    assert fake_lcp.executed, "LCP binding execute_tool was never called"
    
    final_cwd = Path(os.getcwd()).resolve()
    assert final_cwd == original_cwd, "CWD was NOT restored after LCP tool crash!"


def test_direct_callable_path_sets_and_restores_cwd(isolated_discussion, tmp_path):
    """
    Verifies that the standard direct callable path (with 'callable' key) 
    also correctly manages CWD.
    """
    # Arrange
    original_cwd = Path(os.getcwd()).resolve()
    cwd_during_exec = []

    def mock_tool_callable(discussion_instance=None, **kwargs):
        cwd_during_exec.append(Path(os.getcwd()).resolve())
        return {"success": True, "output": "Direct call success"}

    isolated_discussion.lollmsClient.tools = None

    active_tools = {
        "tool_direct_test": {
            "name": "tool_direct_test",
            "description": "Direct callable test",
            "parameters": [],
            "callable": mock_tool_callable
        }
    }
    
    _setup_discussion_mocks(isolated_discussion)
    
    isolated_discussion.lollmsClient.generate_from_messages = _make_streaming_mock(
        "tool_direct_test", 
        {}
    )
    
    # Act
    isolated_discussion.chat(
        user_message="Run direct tool",
        tools=active_tools,
        branch_tip_id="msg_1",
        add_user_message=False,
        max_reasoning_steps=1
    )
    
    # Assert
    assert len(cwd_during_exec) > 0, "Direct callable was never executed"
    expected_cwd = Path(isolated_discussion.workspace_data_path).resolve()
    assert cwd_during_exec[0] == expected_cwd, "CWD was not set to workspace_data during direct execution"
    
    final_cwd = Path(os.getcwd()).resolve()
    assert final_cwd == original_cwd, "CWD was not restored after direct execution"
