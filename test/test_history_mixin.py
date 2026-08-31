import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lollms_client.lollms_discussion._mixin_history import HistoryMixin
from lollms_client.lollms_discussion._mixin_core import CoreMixin
from lollms_client.lollms_discussion._mixin_utils import UtilsMixin


class MockLollmsClient:
    def __init__(self):
        self.ai_name = "Assistant"
        self.count_tokens = MagicMock(return_value=10)
        self.count_image_tokens = MagicMock(return_value=20)


class TestHistoryMixin:
    """Unit tests for the HistoryMixin export and normalization logic."""

    def setup_method(self):
        """Set up a minimal mock discussion environment for testing."""
        self.client = MockLollmsClient()
        
        # We instantiate a dummy class that inherits just the mixin to test it in isolation
        class TestDiscussion(HistoryMixin, UtilsMixin, CoreMixin):
            def __init__(self, client):
                object.__setattr__(self, 'lollmsClient', client)
                object.__setattr__(self, '_system_prompt', "System Prompt")
                object.__setattr__(self, 'scratchpad', "")
                object.__setattr__(self, 'pruning_summary', None)
                object.__setattr__(self, 'pruning_point_id', None)
                object.__setattr__(self, 'active_branch_id', "msg3")
                object.__setattr__(self, 'memory_manager', None)
                
                # Mock messages
                self.msg1 = SimpleNamespace(id="msg1", sender="user", sender_type="user", content="Hello", parent_id=None, metadata={}, images=[], active_images=[], get_active_images=lambda: [])
                self.msg2 = SimpleNamespace(id="msg2", sender=self.client.ai_name, sender_type="assistant", content="<tool>run</tool>", parent_id="msg1", metadata={}, images=[], active_images=[], get_active_images=lambda: [])
                self.msg3 = SimpleNamespace(id="msg3", sender="user", sender_type="user", content="Thanks", parent_id="msg2", metadata={}, images=[], active_images=[], get_active_images=lambda: [])
                
                self._message_index = {
                    "msg1": self.msg1,
                    "msg2": self.msg2,
                    "msg3": self.msg3
                }

            def get_branch(self, leaf_id):
                return [self.msg1, self.msg2, self.msg3]

            def get_full_data_zone(self):
                return "Data Zone"

            def get_discussion_images(self):
                return []

            def _inject_memory_into_messages(self, messages, *args, **kwargs):
                return messages

        self.discussion = TestDiscussion(self.client)

    def test_export_openai_chat_basic(self):
        """Test basic OpenAI chat export without virtual history."""
        messages = self.discussion.export("openai_chat")
        
        # Should have system, user, assistant, user
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        
        # Check functional tag preservation (recent quota)
        assert "<tool>" in messages[2]["content"]

    def test_export_openai_chat_with_virtual_history(self):
        """Test OpenAI chat export with injected virtual history."""
        v_history = [
            SimpleNamespace(sender_type="assistant", content="Agentic step 1"),
            SimpleNamespace(sender_type="user", content="Tool result 1")
        ]
        
        messages = self.discussion.export("openai_chat", virtual_history=v_history)
        
        # Original 4 + 2 virtual = 6
        assert len(messages) == 6
        assert messages[4]["role"] == "assistant"
        assert messages[4]["content"] == "Agentic step 1"
        assert messages[5]["role"] == "user"
        assert messages[5]["content"] == "Tool result 1"

    def test_export_markdown_basic(self):
        """Test markdown export format."""
        md_output = self.discussion.export("markdown")
        
        assert "system: System Prompt" in md_output
        assert "**User**: Hello" in md_output
        assert "**Assistant**: <tool>run</tool>" in md_output
        assert "**User**: Thanks" in md_output

    def test_normalize_openai_messages_merges_consecutive(self):
        """Test that consecutive same-role messages are merged."""
        raw_messages = [
            {"role": "system", "content": "Sys1"},
            {"role": "user", "content": "U1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A1"}
        ]
        
        normalized = self.discussion._normalize_openai_messages(raw_messages)
        
        # Should merge U1 and U2
        assert len(normalized) == 3
        assert normalized[0]["role"] == "system"
        assert normalized[1]["role"] == "user"
        assert "U1" in normalized[1]["content"]
        assert "U2" in normalized[1]["content"]
        assert normalized[2]["role"] == "assistant"

    def test_normalize_openai_messages_prepends_user_if_assistant_first(self):
        """Test that if the first message is from assistant, a 'Continue.' user msg is prepended."""
        raw_messages = [
            {"role": "system", "content": "Sys"},
            {"role": "assistant", "content": "Hello there."}
        ]
        
        normalized = self.discussion._normalize_openai_messages(raw_messages)
        
        assert len(normalized) == 3
        assert normalized[0]["role"] == "system"
        assert normalized[1]["role"] == "user"
        assert normalized[1]["content"] == "Continue."
        assert normalized[2]["role"] == "assistant"