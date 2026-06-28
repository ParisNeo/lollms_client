import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from lollms_client.lollms_memory.lollms_memory import LollmsMemoryManager, MemoryConfig
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager

class DummyClient:
    def __init__(self):
        self.debug = True
        self.llm = self
        self.model_name = "unknown"
        self.binding_name = "unknown"
        self.user_name = "user"
        self.ai_name = "assistant"
    def count_tokens(self, text):
        return len(text) // 4
    def count_image_tokens(self, img):
        return 256
    def remove_thinking_blocks(self, text):
        return text
    def generate_text(self, prompt, **kwargs):
        return "Simulated response"
    def chat(self, discussion, **kwargs):
        return "Chat response"

class TestLollmsEpisodicMemory(unittest.TestCase):
    def setUp(self):
        # Setup clean in-memory databases for tests
        self.memory_manager = LollmsMemoryManager(
            db_path="sqlite:///:memory:",
            owner_id="test_owner",
            config=MemoryConfig()
        )
        self.db_manager = LollmsDataManager("sqlite:///:memory:")
        self.client = DummyClient()
        self.discussion = LollmsDiscussion.create_new(
            lollms_client=self.client,
            db_manager=self.db_manager,
            id="test_session",
            autosave=True
        )
        self.discussion._init_memory(self.memory_manager)

    def tearDown(self):
        self.discussion.close()

    def test_episodic_status_and_timestamps(self):
        # 1. Zone should be empty initially
        zone_init = self.memory_manager.build_working_zone()
        self.assertEqual(zone_init, "")

        # 2. Add some working memories with chronological spacing
        m1 = self.memory_manager.add(
            content="Memory A",
            importance=0.8,
            tags=["a"],
            level=1
        )
        m2 = self.memory_manager.add(
            content="Memory B",
            importance=0.9,
            tags=["b"],
            level=1
        )

        # Space them out explicitly in the SQLite database to guarantee order
        with self.memory_manager._session() as s:
            from lollms_client.lollms_memory.lollms_memory import _MemoryRecord
            rec1 = s.query(_MemoryRecord).filter_by(id=m1["id"]).one()
            rec2 = s.query(_MemoryRecord).filter_by(id=m2["id"]).one()
            rec1.created_at = datetime.utcnow() - timedelta(seconds=5)
            rec2.created_at = datetime.utcnow()

        # 3. Zone should be ordered chronologically (Memory A first, then Memory B)
        zone_populated = self.memory_manager.build_working_zone()
        self.assertIn("=== WORKING MEMORY ===", zone_populated)

        # Verify the timestamps are prepended
        self.assertIn("Memory A", zone_populated)
        self.assertIn("Memory B", zone_populated)

        # Verify correct chronological order (m1 was added before m2)
        idx1 = zone_populated.index("Memory A")
        idx2 = zone_populated.index("Memory B")
        self.assertTrue(idx1 < idx2, "Working memories are not in chronological order")

    def test_automatic_episodic_saving_to_working_memory(self):
        # Run a simulated saving of a conversation turn
        user_text = "What is the capital of France?"
        ai_text = "<processing type='thinking'>Thinking...</processing>The capital of France is Paris."
        
        self.discussion._save_episodic_memory_turn(user_text, ai_text, self.memory_manager)
        
        # Verify the episode is recorded as level 1 Working Memory
        all_mems = self.discussion.list_all_memories(level=1)
        mems = all_mems.get("memories", [])
        self.assertEqual(len(mems), 1)
        
        episode = mems[0]
        self.assertEqual(episode["level"], 1)
        self.assertIn("User asked: \"What is the capital of France?\"", episode["content"])
        self.assertIn("AI responded: \"The capital of France is Paris.\"", episode["content"])
        # Ensure temporary <processing> tags were stripped out successfully
        self.assertNotIn("<processing", episode["content"])

if __name__ == "__main__":
    unittest.main()
