#!/usr/bin/env python3
"""
test_lollms_memory.py
=====================
Descriptive verification suite for the human-brain-inspired two-layer persistent memory system.
Tests:
1. Basic CRUD: memory creation, updates, and direct tagging.
2. Demotion cycles: Working Memory (Level 1) -> Deep Memory (Level 2) -> Archive (Level 3).
3. Handle generation and reloading to working memory.
4. Active usage reinforcement during dream cycles.
5. AI-Assisted selective forgetting via the "Dreamer" LLM interface.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure correct workspace imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client.lollms_memory.lollms_memory import LollmsMemoryManager, MemoryConfig, _MemoryRecord
from ascii_colors import ASCIIColors


class MockLollmsClient:
    """Mock client mimicking LollmsClient for structured dreaming responses."""
    def __init__(self, keep_decision: bool = True, reason: str = "Retained by request"):
        self.keep_decision = keep_decision
        self.reason = reason

    def generate_structured_content(self, prompt: str, schema: Dict, temperature: float = 0.1) -> Dict[str, Any]:
        return {
            "keep": self.keep_decision,
            "reason": self.reason
        }


class TestLollmsMemorySystem(unittest.TestCase):
    def setUp(self):
        # In-memory SQLite context for rapid, clean isolation with small token budget to trigger demotion
        self.config = MemoryConfig(
            working_token_budget=20,   # Sized down to trigger demotions easily during tests
            decay_rate_per_day=0.1,    # High decay rate for testing transitions
            demotion_threshold=0.4,
            archive_threshold=0.1,
            forget_threshold=0.03,
            dream_min_interval_hours=0  # Allow immediate sequential dreaming
        )
        self.manager = LollmsMemoryManager(
            db_path="sqlite:///:memory:",
            owner_id="test_user_owner",
            config=self.config
        )

    def test_basic_crud_operations(self):
        ASCIIColors.cyan("\n--- Test 1: CRUD Operations ---")
        # 1. Addition
        m = self.manager.add(
            content="Sovereign Workspace rule is highly active.",
            importance=0.8,
            tags=["architecture", "sovereign"]
        )
        self.assertIsNotNone(m["id"])
        self.assertEqual(m["level"], 1)
        self.assertEqual(m["importance"], 0.8)
        self.assertEqual(m["tags"], "architecture,sovereign")

        # 2. Retrieval
        retrieved = self.manager.get(m["id"])
        self.assertEqual(retrieved["content"], m["content"])

        # 3. Update (triggers boost)
        updated = self.manager.update(m["id"], "Sovereign Workspace rules are verified and strictly active.")
        self.assertTrue(updated["importance"] > 0.8)
        self.assertEqual(updated["content"], "Sovereign Workspace rules are verified and strictly active.")

        # 4. Tagging (Retrieval Boost)
        tagged = self.manager.tag(m["id"])
        self.assertEqual(tagged["use_count"], 1)

        # 5. Deletion
        self.manager.delete(m["id"])
        self.assertIsNone(self.manager.get(m["id"]))

    def test_demotion_to_deep_memory_on_token_overflow(self):
        ASCIIColors.cyan("\n--- Test 2: Token Budget Demotion ---")
        # Add multiple memories that exceed the 20 token budget
        m1 = self.manager.add("First critical memory regarding backend configurations.", importance=0.9)
        m2 = self.manager.add("Second critical memory explaining neural networks.", importance=0.8)
        m3 = self.manager.add("Third critical memory tracking agent behavior patterns.", importance=0.7)
        m4 = self.manager.add("Fourth less critical memory covering styling options.", importance=0.3)

        # Enforce budget
        demoted_count = self.manager.enforce_budget()
        ASCIIColors.info(f"Demoted {demoted_count} memory/memories to deep layer due to budget pressure.")
        
        # Lower importance memory (m4) should have been demoted to Level 2
        rec_m4 = self.manager.get(m4["id"])
        self.assertEqual(rec_m4["level"], 2)

        # Higher importance ones remain in Level 1
        rec_m1 = self.manager.get(m1["id"])
        self.assertEqual(rec_m1["level"], 1)

    def test_time_based_decay_and_archive(self):
        ASCIIColors.cyan("\n--- Test 3: Aging Decay & Archival ---")
        m = self.manager.add("An ancient memory that hasn't been accessed for weeks.", importance=0.5)

        # Manipulate last_used_at and updated_at to simulate aging (e.g. 3 days ago)
        with self.manager._session() as s:
            rec = s.get(_MemoryRecord, m["id"])
            rec.last_used_at = datetime.utcnow() - timedelta(days=3)
            rec.updated_at = datetime.utcnow() - timedelta(days=3)

        # Apply aging decay
        decay_count = self.manager.apply_decay()
        self.assertEqual(decay_count, 1)

        # Decay was 0.1 per day. After 3 days, importance should drop from 0.5 to ~0.2
        decayed_m = self.manager.get(m["id"])
        self.assertLess(decayed_m["importance"], 0.3)
        # Should have demoted below demotion_threshold (0.4) to Level 2 (Deep Memory)
        self.assertEqual(decayed_m["level"], 2)

    def test_deep_memory_handle_retrieval_and_reload(self):
        ASCIIColors.cyan("\n--- Test 4: Handles & Reloading ---")
        # Explicitly configure as Level 2 (Deep Memory) to isolate and test handle retrieval and reloading
        m = self.manager.add("Deep knowledge regarding SQL architecture.", importance=0.3, level=2)

        # Build handles zone (Deep Memory list)
        handles_zone = self.manager.build_handles_zone()
        self.assertIn("Deep knowledge regarding SQL", handles_zone)
        self.assertIn(m["id"][:8], handles_zone)

        # Load back to working memory
        reloaded = self.manager.load_to_working(m["id"])
        self.assertEqual(reloaded["level"], 1)
        self.assertGreaterEqual(reloaded["importance"], self.config.demotion_threshold)

    def test_dream_consolidation_and_assisted_forgetting(self):
        ASCIIColors.cyan("\n--- Test 5: Dreaming, Reinforcement & Forgetting ---")
        
        # 1. Test reinforcement of used memories
        m_active = self.manager.add("Frequently used facts about API design.", importance=0.6)
        # Simulate active retrieval tagging
        self.manager.tag(m_active["id"])
        self.manager.tag(m_active["id"])

        # 2. Test faded memory eligible for forgetting
        # Directly insert into the Level 3 Archive with a low importance to streamline state consistency,
        # and back-date the last_used_at and updated_at timestamps to avoid the 24h recently-used promotion filter in dream()
        two_days_ago = datetime.utcnow() - timedelta(days=2)
        m_faded_keep = self.manager.add("Important custom user instruction.", importance=0.01, level=3)
        m_faded_forget = self.manager.add("Trivial noise from scratchpad.", importance=0.01, level=3)

        with self.manager._session() as s:
            rec_keep = s.get(_MemoryRecord, m_faded_keep["id"])
            rec_keep.last_used_at = two_days_ago
            rec_keep.updated_at = two_days_ago

            rec_forget = s.get(_MemoryRecord, m_faded_forget["id"])
            rec_forget.last_used_at = two_days_ago
            rec_forget.updated_at = two_days_ago

        # Build a mock dreamer that wants to keep the important instruction but discard the noise
        class DreamerClient:
            def generate_structured_content(self, prompt: str, schema: dict, temperature: float = 0.1):
                if "custom user instruction" in prompt:
                    return {"keep": True, "reason": "Saves critical user preference"}
                return {"keep": False, "reason": "Redundant/unimportant details"}

        mock_llm = DreamerClient()

        # Trigger Consolidation Dream
        report = self.manager.dream(lollms_client=mock_llm)
        
        # Verify active memory got reinforced
        reinforced_m = self.manager.get(m_active["id"])
        self.assertGreater(reinforced_m["importance"], 0.6)

        # Verify dreamer decisions
        self.assertEqual(report["retained_by_dreamer"], 1)
        self.assertEqual(report["forgotten"], 1)

        # Important memory should be saved and restored to safe level
        self.assertIsNotNone(self.manager.get(m_faded_keep["id"]))
        self.assertEqual(self.manager.get(m_faded_keep["id"])["level"], 1)

        # Redundant memory should have been forgotten (permanently pruned)
        self.assertIsNone(self.manager.get(m_faded_forget["id"]))

        ASCIIColors.success("✨ All memory operations and dream cycles validated successfully.")

    def test_ontological_triples_and_spreading_activation(self):
        ASCIIColors.cyan("\n--- Test 6: Ontological Triples & Spreading Activation ---")
        # 1. Add memories with explicit Subject-Predicate-Object relations
        m1 = self.manager.add(
            content="ParisNeo prefers Rust for system-level programming.",
            subject="ParisNeo",
            predicate="PREFERS",
            obj="Rust",
            importance=0.9
        )
        m2 = self.manager.add(
            content="Rust uses a borrow checker to guarantee memory safety.",
            subject="Rust",
            predicate="USES",
            obj="borrow_checker",
            importance=0.8,
            level=2  # Deep Memory
        )

        self.assertEqual(m1["subject"], "parisneo")
        self.assertEqual(m1["predicate"], "PREFERS")
        self.assertEqual(m1["object"], "rust")

        # 2. Tag/retrieve m1, which should trigger Spreading Activation to m2 (linked via 'Rust')
        tagged = self.manager.tag(m1["id"])
        self.assertIsNotNone(tagged)

        # 3. Verify m2 has been pre-warmed/promoted due to spreading activation
        m2_updated = self.manager.get(m2["id"])
        self.assertGreater(m2_updated["activation"], 0.0)
        self.assertGreater(m2_updated["importance"], 0.8)


if __name__ == "__main__":
    unittest.main()
