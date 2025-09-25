# test_lollms_discussion.py

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import base64
import json

# Assuming your class definitions are in a file named 'lollms_discussion_module.py'
# Adjust the import path as necessary.
from lollms_client import LollmsDataManager, LollmsDiscussion

# --- Test Fixtures and Helpers ---

@pytest.fixture
def mock_lollms_client():
    """Creates a mock LollmsClient that simulates its core functions."""
    client = MagicMock()
    # Simulate token counting with a simple word count
    client.count_tokens.side_effect = lambda text: len(str(text).split())
    # Simulate image token counting with a fixed value
    client.count_image_tokens.return_value = 75
    # Simulate structured content generation
    client.generate_structured_content.return_value = {"title": "A Mocked Title"}
    # Simulate text generation for summary and memory
    client.generate_text.return_value = "This is a mock summary."
    # Simulate chat generation
    client.chat.return_value = "This is a mock chat response."
    client.remove_thinking_blocks.return_value = "This is a mock chat response."
    return client

@pytest.fixture
def db_manager(tmp_path):
    """Creates a LollmsDataManager with a temporary database."""
    db_file = tmp_path / "test_discussions.db"
    return LollmsDataManager(f'sqlite:///{db_file}')

def create_dummy_image_b64(text="Test"):
    """Generates a valid base64 encoded dummy PNG image string."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (60, 30), color = 'red')
        d = ImageDraw.Draw(img)
        d.text((10,10), text, fill='white')
        
        # In-memory saving to avoid disk I/O in test
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except ImportError:
        # Fallback if Pillow is not installed
        # This is a 1x1 transparent pixel
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

@pytest.fixture
def dummy_image_b64_1():
    return create_dummy_image_b64("Image 1")

@pytest.fixture
def dummy_image_b64_2():
    return create_dummy_image_b64("Image 2")


# --- Test Class ---

class TestLollmsDiscussion:

    def test_creation_in_memory(self, mock_lollms_client):
        """Tests that an in-memory discussion can be created."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        assert disc.id is not None
        assert not disc._is_db_backed

    def test_creation_db_backed(self, mock_lollms_client, db_manager):
        """Tests that a database-backed discussion can be created."""
        metadata = {"title": "DB Test"}
        disc = LollmsDiscussion.create_new(
            lollms_client=mock_lollms_client,
            db_manager=db_manager,
            discussion_metadata=metadata
        )
        assert disc._is_db_backed
        assert disc.metadata["title"] == "DB Test"

    def test_add_and_get_message(self, mock_lollms_client):
        """Tests basic message addition and retrieval."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        disc.add_message(sender="user", content="Hello, world!")
        
        assert disc.active_branch_id is not None
        messages = disc.get_messages()
        assert len(messages) == 1
        assert messages[0].sender == "user"
        assert messages[0].content == "Hello, world!"

    def test_data_zones_and_export(self, mock_lollms_client):
        """Tests that data zones are correctly included in exports."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        disc.system_prompt = "You are a tester."
        disc.memory = "Remember this."
        disc.user_data_zone = "User is a dev."
        disc.add_message(sender="user", content="Test prompt.")

        exported_md = disc.export("markdown")
        assert "You are a tester." in exported_md
        assert "-- Memory --" in exported_md
        assert "Remember this." in exported_md
        assert "-- User Data Zone --" in exported_md
        assert "User is a dev." in exported_md
        assert "**User**: Test prompt." in exported_md

    def test_message_image_activation(self, mock_lollms_client, dummy_image_b64_1, dummy_image_b64_2):
        """Tests activating and deactivating images on a specific message."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        msg = disc.add_message(sender="user", content="Look", images=[dummy_image_b64_1, dummy_image_b64_2])

        # Initially, both images should be active
        assert len(msg.get_active_images()) == 2
        
        # Deactivate the first image
        msg.toggle_image_activation(0, active=False)
        assert len(msg.get_active_images()) == 1
        all_imgs = msg.get_all_images()
        assert not all_imgs[0]["active"]
        assert all_imgs[1]["active"]

        # Toggle the first image back on
        msg.toggle_image_activation(0)
        assert len(msg.get_active_images()) == 2
        assert msg.get_all_images()[0]["active"]

    def test_discussion_image_management(self, mock_lollms_client, dummy_image_b64_1, dummy_image_b64_2):
        """Tests adding, toggling, and retrieving discussion-level images."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        
        # Add two images
        disc.add_discussion_image(dummy_image_b64_1)
        disc.add_discussion_image(dummy_image_b64_2)

        # Check initial state
        assert len(disc.get_discussion_images()) == 2
        assert disc.get_discussion_images()[0]["active"]
        assert disc.get_discussion_images()[1]["active"]

        # Deactivate the second image
        disc.toggle_discussion_image_activation(1, active=False)
        
        # Verify changes
        all_disc_imgs = disc.get_discussion_images()
        assert len(all_disc_imgs) == 2
        assert all_disc_imgs[0]["active"]
        assert not all_disc_imgs[1]["active"]

    def test_export_with_multimodal_system_prompt(self, mock_lollms_client, dummy_image_b64_1):
        """Tests that active discussion images are included in the system prompt for export."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        disc.system_prompt = "Analyze this image."
        disc.add_discussion_image(dummy_image_b64_1) # This one is active
        disc.add_discussion_image(create_dummy_image_b64("Inactive"))
        disc.toggle_discussion_image_activation(1, active=False) # This one is inactive

        disc.add_message(sender="user", content="What do you see?")
        
        openai_export = disc.export("openai_chat")
        
        system_message = openai_export[0]
        assert system_message["role"] == "system"
        
        content_parts = system_message["content"]
        assert len(content_parts) == 2 
        
        text_part = next(p for p in content_parts if p["type"] == "text")
        image_part = next(p for p in content_parts if p["type"] == "image_url")

        assert text_part["text"] == "Analyze this image."
        assert image_part["image_url"]["url"].endswith(dummy_image_b64_1)

    def test_auto_title_updates_metadata(self, mock_lollms_client, db_manager):
        """Tests that auto_title calls the LLM and updates metadata."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client, db_manager=db_manager)
        disc.add_message(sender="user", content="This is a test to generate a title.")

        title = disc.auto_title()
        
        mock_lollms_client.generate_structured_content.assert_called_once()
        assert title == "A Mocked Title"
        assert disc.metadata['title'] == "A Mocked Title"

    def test_delete_branch(self, mock_lollms_client, db_manager):
        """Tests deleting a message and its descendants."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client, db_manager=db_manager)
        
        msg_a = disc.add_message(sender="user", content="A")
        msg_b = disc.add_message(sender="user", content="B", parent_id=msg_a.id)
        msg_c = disc.add_message(sender="user", content="C", parent_id=msg_b.id)
        
        assert disc.active_branch_id == msg_c.id
        assert len(disc.get_messages()) == 3
        
        disc.delete_branch(msg_b.id)

        assert disc.active_branch_id == msg_a.id
        
        remaining_messages = disc.get_messages()
        assert len(remaining_messages) == 1
        assert remaining_messages[0].id == msg_a.id

    def test_branching_and_switching(self, mock_lollms_client):
        """Tests creating and switching between conversational branches."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        
        root_msg = disc.add_message(sender="user", content="Hello")
        ai_msg_1 = disc.add_message(sender="assistant", content="Hi there!")
        
        disc.switch_to_branch(root_msg.id)
        ai_msg_2 = disc.add_message(sender="assistant", content="Greetings!")

        assert len(disc._message_index) == 3
        assert disc.active_branch_id == ai_msg_2.id

        branch_1 = disc.get_branch(ai_msg_1.id)
        assert [m.id for m in branch_1] == [root_msg.id, ai_msg_1.id]

        branch_2 = disc.get_branch(ai_msg_2.id)
        assert [m.id for m in branch_2] == [root_msg.id, ai_msg_2.id]
        
    def test_regenerate_branch(self, mock_lollms_client, db_manager):
        """Tests that regeneration deletes the old AI message and creates a new one."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client, db_manager=db_manager)
        
        user_msg = disc.add_message(sender="user", content="Tell me a joke.")
        ai_msg = disc.add_message(sender="assistant", content="Why did the scarecrow win an award?")

        original_ai_id = ai_msg.id
        
        result = disc.regenerate_branch()
        new_ai_msg = result["ai_message"]
        
        assert new_ai_msg.id != original_ai_id
        assert new_ai_msg.content == "This is a mock chat response."
        
        messages = disc.get_messages(new_ai_msg.id)
        assert len(messages) == 2
        assert original_ai_id not in [m.id for m in messages]
        
        assert original_ai_id in disc._messages_to_delete_from_db

    def test_summarize_and_prune(self, mock_lollms_client):
        """Tests that the discussion is pruned when the token limit is exceeded."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        disc.max_context_size = 20
        
        disc.add_message(sender="user", content="This is the first long message to establish history.")
        disc.add_message(sender="assistant", content="I see. This is the second long message.")
        m3 = disc.add_message(sender="user", content="Third message.")
        disc.add_message(sender="assistant", content="Fourth message.")
        disc.add_message(sender="user", content="Fifth message.")
        disc.add_message(sender="assistant", content="Sixth message.")

        disc.summarize_and_prune(max_tokens=20, preserve_last_n=4)
        
        mock_lollms_client.generate_text.assert_called_once()
        assert "This is a mock summary." in disc.pruning_summary
        assert disc.pruning_point_id == m3.id
        
    def test_memorize_updates_memory_zone(self, mock_lollms_client):
        """Tests that memorize calls the LLM and appends to the memory data zone."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        disc.add_message(sender="user", content="My favorite color is blue.")
        
        disc.memorize()
        
        mock_lollms_client.generate_text.assert_called_once()
        assert "This is a mock summary." in disc.memory
        assert "Memory entry from" in disc.memory
        
    def test_get_context_status(self, mock_lollms_client, dummy_image_b64_1, dummy_image_b64_2):
        """Tests the token calculation in get_context_status."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        disc.system_prompt = "You are a helpful AI."
        disc.add_discussion_image(dummy_image_b64_1) # 1 active discussion image
        
        msg = disc.add_message(sender="user", content="Hello there!", images=[dummy_image_b64_1, dummy_image_b64_2])
        msg.toggle_image_activation(1, active=False) # 1 active message image
        
        status = disc.get_context_status()

        # System context: "!@>system:\nYou are a helpful AI.\n" -> 6 tokens
        system_tokens = status["zones"]["system_context"]["tokens"]
        assert system_tokens == 6

        # Discussion Images: 1 active discussion image -> 75 tokens
        discussion_image_zone = status["zones"]["discussion_images"]
        assert discussion_image_zone["tokens"] == 75

        # History:
        #   - Text: "!@>user:\nHello there!\n(2 image(s) attached)\n" -> 6 tokens
        #   - Image: 1 active message image -> 75 tokens
        history_zone = status["zones"]["message_history"]
        assert history_zone["breakdown"]["text_tokens"] == 6
        assert history_zone["breakdown"]["image_tokens"] == 75
        assert history_zone["tokens"] == 81 # 6 + 75
        
        # Total should be sum of system, discussion images, and history
        assert status["current_tokens"] == 6 + 75 + 81 # 162

    # --- NEW PERSISTENCE AND SEPARATION TESTS ---

    def test_image_separation_and_context(self, mock_lollms_client, dummy_image_b64_1, dummy_image_b64_2):
        """Tests that discussion and message images are distinct but aggregated correctly."""
        disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client)
        
        # Add one image to the discussion, one to the message
        disc.add_discussion_image(dummy_image_b64_1)
        msg = disc.add_message(sender="user", content="Check these out", images=[dummy_image_b64_2])

        # Test separation
        assert len(disc.get_discussion_images()) == 1
        assert disc.get_discussion_images()[0]['data'] == dummy_image_b64_1
        assert len(msg.get_all_images()) == 1
        assert msg.get_all_images()[0]['data'] == dummy_image_b64_2

        # Test aggregation for context
        all_active_images = disc.get_active_images()
        assert len(all_active_images) == 2
        assert dummy_image_b64_1 in all_active_images
        assert dummy_image_b64_2 in all_active_images

    def test_full_multimodal_persistence(self, mock_lollms_client, db_manager, dummy_image_b64_1, dummy_image_b64_2):
        """End-to-end test for persisting both discussion and message images and their states."""
        disc_id = None
        
        # --- Scope 1: Create and Save ---
        with db_manager.get_session():
            disc = LollmsDiscussion.create_new(lollms_client=mock_lollms_client, db_manager=db_manager)
            disc_id = disc.id
            
            # Add and deactivate a discussion image
            disc.add_discussion_image(dummy_image_b64_1)
            disc.toggle_discussion_image_activation(0, active=False)
            
            # Add and deactivate a message image
            msg = disc.add_message(sender="user", content="Multimodal test", images=[dummy_image_b64_2])
            msg.toggle_image_activation(0, active=False)
            
            disc.commit()
            disc.close()

        # --- Scope 2: Reload and Verify ---
        reloaded_disc = db_manager.get_discussion(mock_lollms_client, disc_id)
        assert reloaded_disc is not None
        
        # Verify discussion image state
        reloaded_disc_imgs = reloaded_disc.get_discussion_images()
        assert len(reloaded_disc_imgs) == 1
        assert reloaded_disc_imgs[0]['data'] == dummy_image_b64_1
        assert not reloaded_disc_imgs[0]['active'] # Must be inactive

        # Verify message image state
        reloaded_msg = reloaded_disc.get_messages()[0]
        reloaded_msg_imgs = reloaded_msg.get_all_images()
        assert len(reloaded_msg_imgs) == 1
        assert reloaded_msg_imgs[0]['data'] == dummy_image_b64_2
        assert not reloaded_msg_imgs[0]['active'] # Must be inactive

        # Verify that get_active_images returns nothing
        assert len(reloaded_disc.get_active_images()) == 0