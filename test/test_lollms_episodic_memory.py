import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager, MemoryConfig
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

    def test_episodic_memory_creation_and_zone(self):
        # 1. Zone should be empty initially
        zone_init = self.memory_manager.build_episodic_zone()
        self.assertEqual(zone_init, "")

        # 2. Add level 4 (Episodic) memory
        mem = self.memory_manager.add(
            content="User discussed building a rocket to Mars.",
            importance=0.9,
            tags=["mars", "rocket"],
            level=4
        )
        self.assertEqual(mem["level"], 4)
        self.assertEqual(mem["importance"], 0.9)

        # 3. Zone should now be populated with episodic header and stub
        zone_populated = self.memory_manager.build_episodic_zone()
        self.assertIn("=== EPISODIC MEMORY ===", zone_populated)
        self.assertIn("User discussed building a rocket to Mars.", zone_populated)

    def test_episodic_memory_no_decay(self):
        # 1. Add level 1 and level 4 memories
        m_working = self.memory_manager.add("Regular working fact", importance=0.8, level=1)
        m_episodic = self.memory_manager.add("Past conversation episode", importance=0.8, level=4)

        # Set creation/update times back to simulate passage of time
        past_time = datetime.utcnow() - timedelta(days=10)
        
        with self.memory_manager._session() as s:
            from lollms_client.lollms_discussion.lollms_memory import _MemoryRecord
            rec_w = s.query(_MemoryRecord).filter_by(id=m_working["id"]).one()
            rec_e = s.query(_MemoryRecord).filter_by(id=m_episodic["id"]).one()
            rec_w.updated_at = past_time
            rec_e.updated_at = past_time

        # 2. Apply decay (only records with level <= 3 should decay)
        decay_count = self.memory_manager.apply_decay()
        
        # Working memory should decay, Episodic memory should be completely unaffected
        self.assertTrue(decay_count >= 1)
        
        m_working_after = self.memory_manager.get(m_working["id"])
        m_episodic_after = self.memory_manager.get(m_episodic["id"])
        
        self.assertTrue(m_working_after["importance"] < 0.8)
        self.assertEqual(m_episodic_after["importance"], 0.8)  # Unchanged!

    def test_automatic_episodic_saving(self):
        # Run a simulated saving of a conversation turn
        user_text = "What is the capital of France?"
        ai_text = "<processing type='thinking'>Thinking...</processing>The capital of France is Paris."
        
        self.discussion._save_episodic_memory_turn(user_text, ai_text, self.memory_manager)
        
        # Verify the episode is recorded
        all_mems = self.discussion.list_all_memories(level=4)
        mems = all_mems.get("memories", [])
        self.assertEqual(len(mems), 1)
        
        episode = mems[0]
        self.assertEqual(episode["level"], 4)
        self.assertIn("User asked: \"What is the capital of France?\"", episode["content"])
        self.assertIn("AI responded: \"The capital of France is Paris.\"", episode["content"])
        # Ensure temporary <processing> tags were stripped out successfully
        self.assertNotIn("<processing", episode["content"])

if __name__ == "__main__":
    unittest.main()
