#!/usr/bin/env python3
"""
🎓 TUTORIAL: LOLLMS MULTI-LEVEL PERSISTENT COGNITIVE MEMORY SYSTEM
=============================================================================
This integration tutorial provides an in-depth, multi-turn look into the
human-brain-inspired, tiered memory architecture of the LoLLMS framework.

🧠 THE 4-LEVEL MEMORY ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────────
  Level 0 — Volatile Scratchpad (Volatile, turn-specific)
    • Unpersisted, fast-access workspace for raw execution logs and steps.
    
  Level 1 — Working Memory (SQLite, always injected into active context)
    • High-importance / highly-recent memories within the token budget.
    • Verbatims are injected so the LLM has zero-latency recall.

  Level 2 — Deep Memory (SQLite, accessed via Compact Handles or Auto-Pull)
    • Older, decayed, or lower-importance memories.
    • To preserve token budget, verbatims are NOT injected. Only compact
      8-character handle stubs (e.g., [a1b2c3d4]) or grouped labels are present.
    • RECALL METHOD A (Reactive): LLM emits `<mem_load id="a1b2c3d4" />` on seeing a handle.
    • RECALL METHOD B (Proactive): Framework auto-greps keywords and pulls before generation.

  Level 3 — Archived Memory (SQLite, subconscious/dream evaluation)
    • Cold memories. Auto-dream consolidation either purges them permanently
      or promotes them back if the LLM Dreamer considers them critical.

FLOW DEMONSTRATION IN THIS TUTORIAL:
─────────────────────────────────────────────────────────────────────────────
  Turn 1  : User introduces high-density facts (favorite language & project).
            The LLM returns XML memory tags, creating Level 1 records in SQLite.
  Turn 2  : User queries the AI. The AI accesses Level 1 Working Memory instantly.
  Decay   : We simulate 6 days of silence. Memories decay and demote to Level 2.
  Turn 3  : User asks about their favorite language in a brand-new, separate discussion.
            Since it's in Deep Memory (Level 2), we disable Auto-Pulling to 
            demonstrate manual/reactive `<mem_load>` recall and cross-discussion persistence.
  Dream   : Auto-dream consolidation runs, reinforcing used memories and purging noise.
=============================================================================
"""

import sys
import os
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta

# Ensure correct workspace import resolution
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_memory.lollms_memory import LollmsMemoryManager, MemoryConfig, _MemoryRecord
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors


class MockGemma4Client:
    """Mock client simulating gemma4:e2b cognitive replies if offline."""
    def __init__(self):
        self.debug = True
        self.llm = self
        self.model_name = "gemma4:e2b"
        self.binding_name = "ollama"
        self._turn_index = 0

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def count_image_tokens(self, img) -> int:
        return 256

    def generate_text(self, prompt, **kwargs):
        # Simulated responses mimicking true gemma4 memory tags emissions
        self._turn_index += 1
        if self._turn_index == 1:
            return "Got it! I will remember that you prefer Rust for systems and are building an RPC agent."
        elif self._turn_index == 2:
            return "Yes, your current goal is building a custom lightweight RPC agent."
        else:
            return "I recall your preference. You love Rust!"

    def chat(self, discussion, **kwargs):
        self._turn_index += 1
        # Extract messages
        history = discussion.get_branch(discussion.active_branch_id)
        last_user = history[-1].content if history else ""
        
        callback = kwargs.get("streaming_callback")
        
        if "favorite programming language" in last_user.lower() and "do you remember" in last_user.lower():
            # AI detects the deep handle in the handles zone, and emits <mem_load> to re-load it!
            working_zone = discussion.memory_manager.build_working_zone()
            handles_zone = discussion.memory_manager.build_handles_zone()
            
            # Find the handle id for the Rust memory in the simulated database
            rust_handle_id = "unknown"
            with discussion.memory_manager._session() as s:
                r = discussion.memory_manager._q(s).filter(_MemoryRecord.content.like("%Rust%")).first()
                if r:
                    rust_handle_id = r.id[:8]
            
            tag = f'<mem_load id="{rust_handle_id}" />'
            reply = f"Let me load that fact. {tag}\nYour favorite programming language is Rust!"
            
            if callback:
                callback(reply, MSG_TYPE.MSG_TYPE_CHUNK)
            return reply

        # Default turns
        if self._turn_index == 1:
            reply = (
                "Got it! I am storing those facts into my long-term memory system.\n"
                "<mem_new importance=\"0.95\">ParisNeo prefers Rust for system-level programming.</mem_new>\n"
                "<mem_new importance=\"0.85\">ParisNeo is building a custom lightweight RPC agent.</mem_new>"
            )
        elif self._turn_index == 2:
            # Find the ID of the RPC project memory in the database to emit the correct tag
            rpc_id = "unknown"
            with discussion.memory_manager._session() as s:
                r = discussion.memory_manager._q(s).filter(_MemoryRecord.content.like("%RPC%")).first()
                if r:
                    rpc_id = r.id[:8]
            reply = f'<mem_tag id="{rpc_id}" /> Yes, your current goal is building a custom lightweight RPC agent.'
        else:
            reply = "You prefer Rust."

        if callback:
            callback(reply, MSG_TYPE.MSG_TYPE_CHUNK)
        return reply

    def generate_structured_content(self, prompt, schema, **kwargs):
        if "custom user instruction" in prompt or "Rust" in prompt:
            return {"keep": True, "reason": "Saves critical user language preference"}
        return {"keep": False, "reason": "Purging trivial details."}


def check_ollama(host: str, model: str) -> bool:
    try:
        res = requests.get(f"{host}/api/tags", timeout=1.5)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            return model in models or any(m.startswith(model) for m in models)
        return False
    except Exception:
        return False


_current_proc = {"title": None, "statuses": []}


def print_event(chunk: str, msg_type: MSG_TYPE, meta: dict = None):
    """Callback to print streaming tokens and memory manager events in real time."""
    global _current_proc
    if not meta:
        meta = {}

    event_type = meta.get("type")

    # Intercept and pretty-render processing tags
    if event_type == "processing_open":
        title = meta.get("title") or meta.get("tool") or "Task"
        _current_proc["title"] = title.replace("_", " ").title()
        _current_proc["statuses"] = []
        print(f"\n⚙️  [PROCESSING] {_current_proc['title']} initialized...", flush=True)
        return True

    elif event_type == "processing_status":
        status = meta.get("status", "").strip()
        if status:
            _current_proc["statuses"].append(status)
            print(f"   ⤷ ⏳ {status}", flush=True)
        return True

    elif event_type == "processing_close":
        if _current_proc["title"]:
            title_text = f"⚙️  {_current_proc['title']} (Execution Complete)"
            content_text = "\n".join(f" ✓ {s}" for s in _current_proc["statuses"]) if _current_proc["statuses"] else " ✓ Task finished."
            print()  # spacing newline
            ASCIIColors.panel(content_text, title_text)
            _current_proc = {"title": None, "statuses": []}
        return True

    # Standard message chunks
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        if meta.get("type") in ("processing_open", "processing_status", "processing_close"):
            # Suppress raw XML / status text chunks from printing to stdout (handled by pretty-render above)
            return True
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_INFO and meta.get("type") == "memory_update":
        report = meta.get("report", {})
        if any(report.values()):
            ASCIIColors.panel(json.dumps(report, indent=2), "SQLite Persistent Memory Updated")
    elif msg_type == MSG_TYPE.MSG_TYPE_INFO and meta.get("type") == "memory_dream":
        report = meta.get("report", {})
        ASCIIColors.panel(json.dumps(report, indent=2), "CALLBACK RECEIVED: Subconscious Memory Dream")


def print_memory_database_state(memory_manager: LollmsMemoryManager):
    """Prints a beautiful summary of all stored memories grouped by Level."""
    print("\n" + "📊 CURRENT SQLITE DATABASE INVENTORY:")
    print("─" * 60)
    with memory_manager._session() as s:
        mems = memory_manager._q(s).order_by(_MemoryRecord.level.asc(), _MemoryRecord.importance.desc()).all()
        if not mems:
            print("  (Database is completely empty)")
        else:
            for m in mems:
                level_names = {1: "Level 1 (Working)", 2: "Level 2 (Deep)", 3: "Level 3 (Archived)"}
                lvl_name = level_names.get(m.level, f"Level {m.level}")
                print(f"  • [{m.id[:8]}] [{lvl_name}] (Imp: {m.importance:.1%}, Uses: {m.use_count}) {m.content}")
    print("─" * 60 + "\n")


def main():
    print("=" * 80)
    print("🔬 LOLLMS COGNITIVE PERSISTENT TWO-LAYER MEMORY TUTORIAL")
    print("=" * 80)

    LLM_MODEL_NAME = "gemma4:e2b"
    host_address = "http://localhost:11434"

    is_online = check_ollama(host_address, LLM_MODEL_NAME)

    if is_online:
        ASCIIColors.green(f"⚡ Live connection detected! Connecting to LollmsClient with {LLM_MODEL_NAME}...")
        client = LollmsClient(
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
        ASCIIColors.yellow("⚠️  Ollama server is offline or 'gemma4:e2b' is not installed.")
        ASCIIColors.info("Running in FULL INTEGRATION SIMULATION MODE.")
        client = MockGemma4Client()

    # ── 1. INITIALIZE MEMORY SUBSYSTEM ────────────────────────────────────
    # MemoryConfig allows fine-tuning how memory acts over time.
    # We establish tight constraints here to demonstrate Level 1 -> Level 2
    # transitions inside a single short conversational test.
    memory_config = MemoryConfig(
        working_token_budget=512,  # Max tokens allocated to active Level 1 memories in prompt
        handles_token_budget=256,  # Max tokens allocated to deep memory handle stubs
        decay_rate_per_day=0.1,    # High decay rate (10% per simulated day) to show fading
        demotion_threshold=0.4,    # Below 40% importance -> demoted to Level 2 (Deep)
        archive_threshold=0.1,     # Below 10% importance -> demoted to Level 3 (Archived)
        forget_threshold=0.03,     # Below 3% importance -> eligible for permanent forgetting
        dream_min_interval_hours=0 # Set to 0 so we can trigger dreams on-demand
    )

    db_file = "integration_test_memory.db"
    if Path(db_file).exists():
        Path(db_file).unlink()

    # Create the sqlite-backed LollmsMemoryManager scoped to the unique user
    memory_manager = LollmsMemoryManager(
        db_path=f"sqlite:///{db_file}",
        owner_id="user_ParisNeo",
        lollms_client=client,
        config=memory_config
    )

    # ── 2. INITIALIZE STATEFUL DISCUSSION 1 ───────────────────────────────
    discussion = LollmsDiscussion(
        lollmsClient=client,
        discussion_id="integration_chat_session_1",
        memory_manager=memory_manager,
        autosave=True
    )
    discussion.system_prompt = (
        "You are Lollms, a highly intelligent persistent developer assistant.\n"
        "CRITICAL PROTOCOLS:\n"
        "1. MEMORY INGESTION: When the user shares new personal facts, preferences, or goals, "
        "you MUST save them using the `<mem_new importance=\"...\">content</mem_new>` tag.\n"
        "2. MEMORY RECALL: When you retrieve or refer to any active memory [ID] from "
        "the WORKING MEMORY zone to answer a question, you MUST include `<mem_tag id=\"ID\" />` "
        "at the very beginning of your response stream. This is required for the memory manager to "
        "track memory usage strength."
    )

    # ── TURN 1: INTRODUCE FACTS ───────────────────────────────────────────
    # 📝 TUTORIAL CONCEPT: MEMORY INGESTION
    # When the user shares new facts, the AI realizes their technical density.
    # Instead of just replying, it outputs XML memory tags:
    #   <mem_new importance="0.95">ParisNeo prefers Rust...</mem_new>
    #
    # The Lollms Communication Protocol (LCP) intercepts these tags on-the-fly,
    # strips them from the visible chat output, and saves them into SQLite as Level 1.
    print("\n" + "=" * 50)
    print("💬 TURN 1: User Introduces High-Density Facts")
    print("=" * 50)
    user_msg_1 = (
        "Hello Lollms! For my systems programming, I strictly prefer Rust, "
        "and my current project is building a custom lightweight RPC agent."
    )
    ASCIIColors.yellow(f"> ParisNeo: {user_msg_1}\n")

    ASCIIColors.green(f"< Lollms: ")
    res_1 = discussion.chat(
        user_message=user_msg_1,
        streaming_callback=print_event,
        enable_memory=True,
        enable_auto_dream=False # Do not dream yet
    )
    print()

    # Show database state after Turn 1
    print_memory_database_state(memory_manager)

    # ── Fallback Seeding (To support non-deterministic or weaker local models) ──
    # If the LLM didn't emit the tags, we programmatically seed the memories
    # so the rest of the tutorial (decay, demotion, cross-discussion loading) can run.
    working_mems = memory_manager.list_working()
    if not working_mems:
        ASCIIColors.warning("\n⚠️  [Fallback Seeding] Model did not emit XML memory tags. Programmatically seeding memories to continue the tutorial...")
        memory_manager.add(
            content="ParisNeo strictly prefers Rust for system-level programming.",
            importance=0.95,
            tags=["preference", "rust", "programming"]
        )
        memory_manager.add(
            content="ParisNeo is building a custom lightweight RPC agent.",
            importance=0.85,
            tags=["rpc", "architecture"]
        )
        # Re-display state after seeding
        print_memory_database_state(memory_manager)

    # ── TURN 2: RECALL WORKING MEMORY ──────────────────────────────────────
    # 📝 TUTORIAL CONCEPT: ZERO-LATENCY RECALL
    # Since these memories were just created, their importance remains high
    # (above the 40% demotion threshold). They reside in Level 1 (Working Memory).
    #
    # Lollms automatically formats these and injects them verbatim in the system
    # block. The LLM answers instantly, without making any database tool queries!
    print("\n" + "=" * 50)
    print("💬 TURN 2: Verifying Immediate Working Memory Recall")
    print("=" * 50)
    user_msg_2 = "Do you remember what my current project goal is?"
    ASCIIColors.yellow(f"> ParisNeo: {user_msg_2}\n")

    ASCIIColors.green(f"< Lollms: ")
    res_2 = discussion.chat(
        user_message=user_msg_2,
        streaming_callback=print_event,
        enable_memory=True,
        enable_auto_dream=False
    )
    print()

    # Show database state after Turn 2
    print_memory_database_state(memory_manager)

    # ── TIME-BASED AGING DECAY ─────────────────────────────────────────────
    # 📝 TUTORIAL CONCEPT: EXPERIENCE DECAY AND DEMOTION
    # In real life, memories fade if not regularly accessed. To demonstrate this,
    # we artificially shift the 'last_used_at' timestamp 6 days into the past,
    # then trigger the mathematical decay algorithm.
    #
    # • Unused memories lose 10% importance per day.
    # • The 'RPC project' and 'Rust preference' drop below 40% (demotion_threshold).
    # • They are demoted from Level 1 (Working) down to Level 2 (Deep Memory).
    # • This keeps the working context pristine and frees precious token space.
    print("\n" + "=" * 50)
    print("⏳ SIMULATION: 6 Days of Inactivity Pass (Applying Decay)...")
    print("=" * 50)
    # Artificially age the last_used_at timestamps of our memories
    six_days_ago = datetime.utcnow() - timedelta(days=6)
    with memory_manager._session() as s:
        for rec in s.query(_MemoryRecord).all():
            rec.last_used_at = six_days_ago

    # Decay move memories Working (Level 1) -> Deep (Level 2)
    decayed_count = memory_manager.apply_decay()
    print(f"  • Decayed {decayed_count} memory records.")
    print("  • Unused memories have demoted to Level 2 (Deep Memory).")

    # Show demoted database state
    print_memory_database_state(memory_manager)

    # ── TURN 3: VERIFY KNOWLEDGE BETWEEN DISCUSSIONS (CROSS-SESSION) ───────
    # 📝 TUTORIAL CONCEPT: PROACTIVE PULL vs REACTIVE RELOAD & CROSS-DISCUSSION PERSISTENCE
    # We now close the first discussion and create an entirely independent
    # discussion (representing a separate chat interface/history).
    #
    # By default, LoLLMS features "Proactive Pulling" (enable_deep_memory_pulling=True)
    # which automatically scans user messages for keywords and loads matching
    # deep memories into Level 1 before generation starts (zero-latency).
    #
    # Here, we explicitly disable Proactive Pulling (enable_deep_memory_pulling=False)
    # to demonstrate the alternative "Reactive Loading" flow:
    #   1. Only the compact handle stub [abc123de] is injected into prompt context.
    #   2. The LLM notices it has a handle relating to "favorite programming language".
    #   3. The LLM decides to fetch it by emitting a `<mem_load id="abc123de" />` tag.
    #   4. The system intercepts the tag, loads the verbatim from SQLite back to Level 1,
    #      and completes the recall turn successfully.
    print("\n" + "=" * 50)
    print("💬 TURN 3: Cross-Discussion Verification (New Session) & Reactive Reload")
    print("=" * 50)
    
    # Close old session
    discussion.close()

    # Create completely fresh discussion
    discussion2 = LollmsDiscussion(
        lollmsClient=client,
        discussion_id="integration_chat_session_2",
        memory_manager=memory_manager,
        autosave=True
    )
    discussion2.system_prompt = (
        "You are Lollms, a highly intelligent persistent developer assistant.\n"
        "CRITICAL PROTOCOLS:\n"
        "1. MEMORY INGESTION: When the user shares new personal facts, preferences, or goals, "
        "you MUST save them using the `<mem_new importance=\"...\">content</mem_new>` tag.\n"
        "2. MEMORY RECALL: When you retrieve or refer to any active memory [ID] from "
        "the WORKING MEMORY zone to answer a question, you MUST include `<mem_tag id=\"ID\" />` "
        "at the very beginning of your response stream. This is required for the memory manager to "
        "track memory usage strength."
    )

    user_msg_3 = "By the way, do you remember my favorite programming language?"
    ASCIIColors.yellow(f"> ParisNeo: {user_msg_3}\n")

    ASCIIColors.green(f"< Lollms: ")
    res_3 = discussion2.chat(
        user_message=user_msg_3,
        streaming_callback=print_event,
        enable_memory=True,
        enable_auto_dream=True, # Auto-Dream is now enabled at the end of the turn!
        enable_deep_memory_pulling=False
    )
    print()

    # Show database state after Turn 3 (Memory should be promoted back to Level 1)
    print_memory_database_state(memory_manager)

    # ── CLEANUP ───────────────────────────────────────────────────────────
    # Disposing of the SQLite database engine frees connection pools and file locks
    # so the OS can clean up without leaving orphan session handlers on disk.
    discussion2.close()
    if Path(db_file).exists():
        try:
            memory_manager._engine.dispose()
            Path(db_file).unlink()
            print("\n")
        except Exception as e:
            print(f"\n⚠️  Could not delete temporary database file: {e}")

    print("\n" + "=" * 80)
    print("🎉 FULL MEMORY SYSTEM INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
