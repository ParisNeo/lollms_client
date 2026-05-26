#!/usr/bin/env python3
"""
lollms_memory_gemma4_example.py
=============================================================================
A comprehensive interactive example showcasing the multi-level persistent
memory system utilizing the Ollama binding with gemma4:e2b.

This script demonstrates:
  1. Adding, tag-retrieving, and updating memories.
  2. Forcing time-based decay of experiences to witness memory demotion.
  3. Generating and injecting deep memory handle stubs into the LLM context.
  4. Executing a "dream cycle" where the model actively evaluates which faded
     archived memories to keep or forget.
=============================================================================
"""

import sys
import time
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Ensure correct workspace import resolution
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager, MemoryConfig, _MemoryRecord
from ascii_colors import ASCIIColors


class SimulatedLollmsClient:
    """Fallback client to simulate Ollama behavior if offline."""
    def __init__(self):
        self.debug = True

    def generate_text(self, prompt, **kwargs):
        return "I have updated my memories with your preferences."

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def generate_structured_content(self, prompt: str, schema: dict, temperature: float = 0.1):
        # Deterministic simulation matching expected Dreamer behavior
        if "rust" in prompt.lower() or "preference" in prompt.lower():
            return {
                "keep": True,
                "reason": "Retaining critical programming preference for user ParisNeo."
            }
        return {
            "keep": False,
            "reason": "Purging transient/unimportant conversational noise."
        }


def check_ollama_status(host: str, model_name: str) -> bool:
    """Verifies if the local Ollama instance is online with the model pulled."""
    try:
        res = requests.get(f"{host}/api/tags", timeout=2)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            if model_name in models or any(m.startswith(model_name) for m in models):
                return True
            else:
                ASCIIColors.warning(f"Ollama is online but the model '{model_name}' is not pulled.")
                ASCIIColors.info(f"Please run: 'ollama pull {model_name}' to execute live.")
        return False
    except Exception:
        return False


def main():
    print("=" * 80)
    print("🧠 LOLLMS PERSISTENT TWO-LAYER MEMORY SYSTEM — GEMMA4:E2B EXAMPLE")
    print("=" * 80)

    LLM_MODEL_NAME = "gemma4:e2b"
    host_address = "http://localhost:11434"

    # Verify Ollama status
    is_online = check_ollama_status(host_address, LLM_MODEL_NAME)

    if is_online:
        ASCIIColors.green(f"⚡ Live connection detected! Initializing LollmsClient on {LLM_MODEL_NAME}...")
        llm_client = LollmsClient(
            llm_binding_name="ollama",
            llm_binding_config={
                "model_name": LLM_MODEL_NAME,
                "host_address": host_address
            },
            user_name="ParisNeo",
            ai_name="Lollms",
            cooperative_vram_management=True,
            debug=True
        )
    else:
        ASCIIColors.yellow("⚠️  Ollama server is offline or 'gemma4:e2b' is not installed locally.")
        ASCIIColors.info("Running in SIMULATION MODE with Mock LLM behavior to demonstrate full pipeline.")
        llm_client = SimulatedLollmsClient()

    # ── 1. CONFIGURE AND INITIALIZE MEMORY MANAGER ─────────────────────────
    # We choose highly sensitive thresholds so we can demonstrate all transitions
    # within a single execution script.
    memory_config = MemoryConfig(
        working_token_budget=512,  # Compact budget to see demotions clearly
        handles_token_budget=256,
        decay_rate_per_day=0.1,    # Rapid decay (10% per day) for testing
        demotion_threshold=0.4,    # Move to Level 2 (Deep Memory) when below 0.4
        archive_threshold=0.1,     # Move to Level 3 (Archive) when below 0.1
        forget_threshold=0.03,     # Purge/Evaluate when below 0.03
        dream_min_interval_hours=0 # Allow immediate dreaming passes
    )

    db_file = "examples_memory.db"
    # Remove previous test database if present to ensure clean run
    if Path(db_file).exists():
        Path(db_file).unlink()

    memory_manager = LollmsMemoryManager(
        db_path=f"sqlite:///{db_file}",
        owner_id="parisneo_test_session",
        lollms_client=llm_client,
        config=memory_config
    )

    # ── 2. CREATE A CONVERSATION DISCUSSION ───────────────────────────────
    discussion = LollmsDiscussion(
        lollmsClient=llm_client,
        discussion_id="memory_demo_chat",
        memory_manager=memory_manager
    )

    # ── 3. INGEST INITIAL MEMORIES (WORKING LAYER - LEVEL 1) ───────────────
    print("\n[Step 1] Ingesting initial system experiences into Working Memory...")
    
    # Let's add three distinct memories
    m_user_pref = memory_manager.add(
        content="User ParisNeo strictly prefers Rust for system-level programming.",
        importance=0.90,
        tags=["preference", "rust", "programming"]
    )
    m_project_goal = memory_manager.add(
        content="The current project is building a custom lightweight RPC agent.",
        importance=0.80,
        tags=["rpc", "architecture"]
    )
    m_trivia = memory_manager.add(
        content="ParisNeo mentioned they drank an excellent espresso during the code review.",
        importance=0.45,
        tags=["trivia", "espresso"]
    )

    print("\n--- Current Working Memory (Level 1) ---")
    for wm in memory_manager.list_working():
        print(f"  • [{wm['id'][:8]}] (Imp: {wm['importance']:.0%}) {wm['content']} #{wm['tags']}")

    # ── 4. RETRIEVAL & ENFORCEMENT STRENGTH ───────────────────────────────
    print("\n[Step 2] Retrieving/Using memories (Tagging) to boost retrieval strength...")
    # ParisNeo's preference is used; this increments its use_count
    memory_manager.tag(m_user_pref["id"])
    memory_manager.tag(m_user_pref["id"])
    
    updated_pref = memory_manager.get(m_user_pref["id"])
    print(f"  • Preferred memory use_count is now: {updated_pref['use_count']}")

    # ── 5. TIME-BASED AGING DECAY ─────────────────────────────────────────
    print("\n[Step 3] Simulating the passage of 5 days of absolute silence...")
    # Artificially shift last_used_at timestamps 5 days backward in the SQLite DB
    five_days_ago = datetime.utcnow() - timedelta(days=5)
    with memory_manager._session() as s:
        for rec in s.query(_MemoryRecord).all():
            if rec.id != m_user_pref["id"]:  # Let's keep the user preference active
                rec.last_used_at = five_days_ago

    # Apply aging decay
    decayed_count = memory_manager.apply_decay()
    print(f"  • Faded {decayed_count} unused memories according to the decay factor (10% loss/day).")

    # Let's see our memory status now
    print("\n--- Memory Status After 5 Days of Aging ---")
    with memory_manager._session() as s:
        for rec in s.query(_MemoryRecord).all():
            print(f"  • [{rec.id[:8]}] (Imp: {rec.importance:.0%}) Level {rec.level} | Content: {rec.content[:50]}...")

    # Notice how m_trivia has demoted past the demotion_threshold (0.40) into Level 2 (Deep Memory)
    # and even past archive_threshold (0.10) into Level 3 (Archived Memory)!

    # ── 6. DEEP HANDLE zone INJECTION ─────────────────────────────────────
    print("\n[Step 4] Checking deep memory handles context generation...")
    # Any memory residing in Level 2 will have its summary handle injected automatically
    handles_zone = memory_manager.build_handles_zone()
    print("-" * 60)
    print(handles_zone.strip())
    print("-" * 60)

    # ── 7. THE CONSOLIDATION DREAM CYCLE ──────────────────────────────────
    print("\n[Step 5] Triggering a Dream Cycle (AI Subconscious Consolidation)...")
    print("  During this phase:")
    print("    - Active memories with positive use counts receive a log-reinforcement boost.")
    print("    - Memories falling below forget_threshold are evaluated by the Dreamer.")
    print("    - The Dreamer decides to retain critical preferences or purge transient details.")
    print("-" * 60)

    dream_report = memory_manager.dream(lollms_client=llm_client)

    print("-" * 60)
    print("📊 DREAM REPORT SUMMARY:")
    print(f"  • Decayed records:      {dream_report.get('decayed', 0)}")
    print(f"  • Reinforced records:   {dream_report.get('reinforced', 0)} (Active ones got boosted!)")
    print(f"  • Forgotten records:    {dream_report.get('forgotten', 0)} (Noise was permanently purged)")
    print(f"  • Retained by Dreamer:  {dream_report.get('retained_by_dreamer', 0)}")
    print(f"  • Duration:             {dream_report.get('duration_seconds', 0.0):.4f} seconds")

    # ── 8. VERIFY CONSOLIDATION STATE ─────────────────────────────────────
    print("\n[Step 6] Final database inventory after the Dream:")
    print("-" * 60)
    with memory_manager._session() as s:
        for rec in s.query(_MemoryRecord).all():
            print(f"  • [{rec.id[:8]}] (Imp: {rec.importance:.1%}) Level {rec.level} | Content: {rec.content}")
    print("-" * 60)
    print("Notice how:")
    print("  1. The 'Rust programming' preference is safely preserved at a high importance level.")
    print("  2. The 'espresso trivia' was completely and permanently forgotten, freeing database slots.")
    print("-" * 60)

    # Cleanup database file at the end
    if Path(db_file).exists():
        try:
            # Dispose of the engine connection pool so Windows can release file locks
            memory_manager._engine.dispose()
            Path(db_file).unlink()
            print("🧹 Cleaned up temporary database file.")
        except Exception as e:
            print(f"⚠️  Could not delete temporary database file: {e}")

    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE PERSISTENT MEMORY CYCLE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
