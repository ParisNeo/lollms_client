#!/usr/bin/env python3
"""
episodic_memory_example.py
==========================
This example demonstrates how to set up and use the multi-level memory system,
focusing on the newly added Level 4 (Episodic) Memory.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_memory.lollms_memory import LollmsMemoryManager, MemoryConfig
from ascii_colors import ASCIIColors

def main():
    ASCIIColors.cyan("=========================================================")
    ASCIIColors.green("🧠 LoLLMS Tiered & Episodic Memory System Example")
    ASCIIColors.cyan("=========================================================\n")

    # 1. Initialize Mock Lollms Client
    client = LollmsClient()
    
    # 2. Initialize in-memory SQLite database managers for discussion and memory
    db_manager = LollmsDataManager("sqlite:///:memory:")
    memory_manager = LollmsMemoryManager(
        db_path="sqlite:///:memory:",
        owner_id="example_session",
        config=MemoryConfig(working_token_budget=1024)
    )

    # 3. Create active conversation discussion
    discussion = LollmsDiscussion.create_new(
        lollms_client=client,
        db_manager=db_manager,
        id="example_session",
        autosave=True
    )
    
    # Attach persistent memory manager to discussion
    discussion._init_memory(memory_manager)
    ASCIIColors.success("✓ Tiered Memory Manager attached to active conversation successfully.")

    # 4. Ingest baseline project rules into Level 1 (Working Memory)
    ASCIIColors.info("\nAdding critical architectural rules to Working Memory (Level 1)...")
    memory_manager.add(
        content="Project frontend MUST strictly use Tailwind CSS for all UI layouts.",
        importance=0.9,
        tags=["frontend", "css", "rule"],
        level=1
    )
    memory_manager.add(
        content="Backend APIs should be completely stateless and use JWT tokens.",
        importance=0.85,
        tags=["api", "auth", "stateless"],
        level=1
    )

    # 5. Simulate conversational turns with automatic Working Memory (Level 1) generation with episodic status!
    # When a conversation turn finishes, the system automatically logs the exchange as an active working memory record!
    ASCIIColors.info("\nSimulating conversation exchange turns...")
    
    turn_1_user = "Can we use Bootstrap instead of Tailwind?"
    turn_1_ai = "<processing type='thinking'>Recalling rule...</processing>No, according to our project rules, the frontend must strictly use Tailwind CSS."
    
    # Log turn 1
    discussion._save_episodic_memory_turn(turn_1_user, turn_1_ai, memory_manager)
    ASCIIColors.success("✓ Turn 1 logged as a timestamped working memory.")

    turn_2_user = "What database are we using?"
    turn_2_ai = "We are using SQLite for local development and PostgreSQL for production environments."
    
    # Log turn 2
    discussion._save_episodic_memory_turn(turn_2_user, turn_2_ai, memory_manager)
    ASCIIColors.success("✓ Turn 2 logged as a timestamped working memory.")

    # 6. Retrieve and Print the fully formatted context memory block
    # This represents exactly what is injected into the LLM system prompt context!
    ASCIIColors.info("\nAssembling prompt context memory blocks...")
    context_block = discussion._build_memory_context_block(memory_manager)
    
    print("\n--- INJECTED CONTEXT MEMORY BLOCK ---")
    print(context_block)
    print("-------------------------------------\n")

    # 7. Print total memory count and breakdown
    all_memories = memory_manager.list_all(page_size=100)
    memories = all_memories.get("memories", [])
    
    ASCIIColors.cyan("=== Persistent Memories Breakdown ===")
    for m in memories:
        level_label = {1: "Working", 2: "Deep", 3: "Archived"}.get(m["level"], "Unknown")
        print(f"• [{level_label}] ID: {m['id'][:8]} (Imp: {m['importance']:.0%}): {m['content'][:90]}...")
    print("=====================================\n")
if __name__ == "__main__":
    main()
