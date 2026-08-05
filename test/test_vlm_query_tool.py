import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the tool directly
sys.path.insert(0, str(PROJECT_ROOT / "src" / "lollms_client" / "tools_bindings" / "lcp" / "default_tools" / "vlm_query"))
from vlm_query import tool_vlm_query

def test_vlm_query_no_context():
    """Tests failure when discussion or client instances are missing."""
    result = tool_vlm_query(image_index=0, query="test")
    assert result["success"] is False
    assert "System context not available" in result["error"]

def test_vlm_query_no_vlm_mounted():
    """Tests failure when no VLM is found on the client."""
    mock_client = MagicMock()
    mock_client.llm.child_bindings = {}  # No child bindings
    mock_client.llm.vision_enabled = False  # Master doesn't support vision
    
    result = tool_vlm_query(image_index=0, query="test", discussion_instance=MagicMock(), lollms_client_instance=mock_client)
    assert result["success"] is False
    assert "No Vision-Language Model (VLM) is mounted" in result["error"]

def test_vlm_query_finds_child_vlm():
    """Tests that the tool finds a VLM among SmartRouter child bindings."""
    mock_client = MagicMock()
    
    # Mock SmartRouter structure
    mock_text_binding = MagicMock()
    mock_text_binding.vision_enabled = False
    
    mock_vlm_binding = MagicMock()
    mock_vlm_binding.vision_enabled = True
    mock_vlm_binding.generate_from_messages.return_value = "Image shows a cat."
    
    mock_client.llm.child_bindings = {
        "text": mock_text_binding,
        "vlm": mock_vlm_binding
    }
    mock_client.llm.vision_enabled = False  # Master is router, not VLM itself
    
    # Mock discussion branch
    mock_discussion = MagicMock()
    mock_msg = MagicMock()
    mock_msg.sender_type = "user"
    mock_msg.images = ["base64_cat_image"]
    mock_discussion.get_branch.return_value = [mock_msg]
    
    result = tool_vlm_query(
        image_index=0, 
        query="What is in the image?", 
        discussion_instance=mock_discussion, 
        lollms_client_instance=mock_client
    )
    
    assert result["success"] is True
    assert "Image shows a cat." in result["output"]
    mock_vlm_binding.generate_from_messages.assert_called_once()

def test_vlm_query_finds_master_vlm():
    """Tests that the tool falls back to master if it supports vision and no children exist."""
    mock_client = MagicMock()
    mock_client.llm.child_bindings = {}  # No children
    mock_client.llm.vision_enabled = True  # Master supports vision
    mock_client.llm.generate_from_messages.return_value = "Image shows a dog."
    
    mock_discussion = MagicMock()
    mock_msg = MagicMock()
    mock_msg.sender_type = "user"
    mock_msg.images = ["base64_dog_image"]
    mock_discussion.get_branch.return_value = [mock_msg]
    
    result = tool_vlm_query(
        image_index=0, 
        query="What is in the image?", 
        discussion_instance=mock_discussion, 
        lollms_client_instance=mock_client
    )
    
    assert result["success"] is True
    assert "Image shows a dog." in result["output"]
    mock_client.llm.generate_from_messages.assert_called_once()

def test_vlm_query_invalid_image_index():
    """Tests out-of-bounds image index."""
    mock_client = MagicMock()
    mock_vlm_binding = MagicMock()
    mock_vlm_binding.vision_enabled = True
    mock_client.llm.child_bindings = {"vlm": mock_vlm_binding}
    mock_client.llm.vision_enabled = False
    
    mock_discussion = MagicMock()
    mock_msg = MagicMock()
    mock_msg.sender_type = "user"
    mock_msg.images = ["only_one_image"]
    mock_discussion.get_branch.return_value = [mock_msg]
    
    result = tool_vlm_query(
        image_index=5,  # Out of bounds
        query="test", 
        discussion_instance=mock_discussion, 
        lollms_client_instance=mock_client
    )
    
    assert result["success"] is False
    assert "Invalid image_index" in result["error"]
    assert "Contains 1 image(s)" in result["error"]