---
name: Lollms Cognitive Persistent Memory
description: Teaches the model about the persistent tiered memory system (Level 1 Working, Level 2 Deep, Level 3 Archive), proactive keyword pulling, and the required tagging protocols.
author: ParisNeo
version: 1.0.0
category: lollms_client/memory
created: 2026-05-24
---

# Lollms Cognitive Persistent Memory

This skill explains how to utilize and interact with the tiered long-term memory system of the LoLLMS framework.

## 1. The Multi-Tiered Memory Architecture
The system prevents context bloating by categorizing information into distinct tiers:
- **Level 1 — Working Memory**: Active, high-importance facts injected verbatim into the system prompt context.
- **Level 2 — Deep Memory**: Inactive, faded, or lower-importance memories. Verbatims are withheld; only compact 8-character handle stubs (e.g. `[7c8855bf]`) are injected to save token space.
- **Level 3 — Archived Memory**: Cold memories slated for permanent forgetting or evaluation during a Dream Consolidation cycle.

## 2. Cognitive Memory Tags (XML Protocol)
The model interacts with the memory manager in real-time by emitting XML tags:
- `<mem_new importance="0.95">content</mem_new>` — Record a new fact.
- `<mem_tag id="8-CHAR-ID" />` — Signal that an active memory has been recognized and used in the reply. This increments `use_count` and boosts its importance score.
- `<mem_load id="8-CHAR-ID" />` — Manually load an inactive Deep Memory (Level 2) back into active Working Memory (Level 1).

## 3. Proactive Pulling (Grep / TF-IDF Search)
When `enable_deep_memory_pulling=True` is active, the memory manager automatically tokenizes incoming user messages, performs a fast on-the-fly TF-IDF query matching against Level 2 memories, and promotions them back to Level 1 before generation starts. This ensures zero-latency recall for matching keywords without wasting context.

## 4. Memory Decay and Subconscious Dreams
- **Decay**: Memories that are not accessed lose a portion of their importance over time (defined by `decay_rate_per_day`).
- **Dream Cycle**: Triggered programmatically or automatically. Active memories with a positive `use_count` are reinforced, after which their usage count is decremented back to `0` to allow decay again if left unused. Archived memories below `forget_threshold` are permanently forgotten.

```python
from lollms_client.lollms_discussion.lollms_memory import LollmsMemoryManager, MemoryConfig

memory_config = MemoryConfig(
    working_token_budget=512,
    decay_rate_per_day=0.05,
    demotion_threshold=0.40,
    archive_threshold=0.10
)

memory_manager = LollmsMemoryManager(
    db_path="sqlite:///memories.db",
    owner_id="user_ParisNeo",
    config=memory_config
)

# Perform a subconscious dream pass
dream_report = memory_manager.dream(lollms_client=client)
print(f"Purged: {dream_report['forgotten']} | Reinforced: {dream_report['reinforced']}")
```