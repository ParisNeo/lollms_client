import pytest
from pathlib import Path
from unittest.mock import MagicMock
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
    
    # Initialize artefact system and create a dummy file
    pers._init_artefact_system()
    (workspace / "main.py").write_text("print('initial')")
    pers._sync_artefact_index_with_disk()
    
    return pers, workspace

def test_rolling_compaction_evicts_oldest(setup_personality):
    pers, workspace = setup_personality
    
    # Simulate 5 artifact rounds in virtual_history
    virtual_history = []
    for i in range(5):
        virtual_history.append(SimpleNamespace(
            sender_type="assistant",
            content=f"<artifact name='main.py'># Version {i}</artifact>"
        ))
        virtual_history.append(SimpleNamespace(
            sender_type="user",
            content=f"Tool result for round {i}"
        ))
    
    base_conversation = [{
        "role": "user",
        "content": "Initial prompt.\n\n=== CURRENT WORKSPACE CONTEXT ===\n## Workspace Directory Tree Index\n  workspace/\n  ├── main.py  [U]\n=== END CURRENT WORKSPACE CONTEXT ===\n\nTask: Write code."
    }]
    
    # Apply compaction
    surviving_history = pers._apply_rolling_artifact_compaction(virtual_history, base_conversation)
    
    # We should have evicted the oldest assistant artifact and its corresponding user result
    assert len(surviving_history) == 8, "Should have evicted the oldest artifact pair (2 items)"
    
    # The oldest artifact content should no longer be in virtual_history
    assert "# Version 0" not in surviving_history[0].content
    
    # The base context should now contain the full file content (synced from disk)
    assert "[C]" in base_conversation[0]["content"], "Base context should now show main.py as fully loaded [C]"
    assert "print('initial')" in base_conversation[0]["content"], "Base context should contain the actual file content"