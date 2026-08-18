import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from lollms_client.lollms_personality import LollmsPersonality

@pytest.fixture
def setup_personality(tmp_path):
    mock_client = MagicMock()
    mock_client.get_ctx_size.return_value = 8192
    mock_client.count_tokens.return_value = 10
    
    workspace = tmp_path / "ws"
    workspace.mkdir()
    
    pers = LollmsPersonality(
        name="TestAgent",
        author="Test",
        category="test",
        description="Test",
        system_prompt="Test prompt",
        lollms_client=mock_client,
        workspace_path=workspace
    )
    
    pers._init_artefact_system()
    return pers, mock_client, workspace

def test_done_with_artifact_executes_immediately(setup_personality):
    pers, mock_client, workspace = setup_personality

    llm_output = (
        "I have created the file.\n"
        "<artifact name=\"story.md\" language=\"markdown\">\n"
        "# The End\n"
        "</artifact>\n"
        "<done/>"
    )

    def fake_generate_from_messages(messages, stream, streaming_callback, **kwargs):
        for char in llm_output:
            streaming_callback(char, None, None)
        return ""

    mock_client.generate_from_messages.side_effect = fake_generate_from_messages

    result = pers.chat(
        prompt="Write a story",
        streaming_callback=lambda *args: True,
        event_mode=0,
        use_internal_history=False
    )

    story_path = workspace / "story.md"
    assert story_path.exists(), "Artifact was not written to disk"
    assert story_path.read_text(encoding="utf-8").strip() == "# The End"
    
    assert result["response"].strip() == "I have created the file."
    assert result["rounds"] == 1, "Loop should have terminated in 1 round"