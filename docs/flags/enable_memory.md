# `enable_memory`

```python
enable_memory: bool = False
```

**Category**: Memory
**Parent flags**: none (master switch)
**Gate**: a memory manager must resolve (`personality.memory_manager` →
`discussion._get_memory_manager()` → explicit `memory_manager` arg →
`self.memory_manager`)

## What it does

Master switch for the memory subsystem. When `True`, the turn:
- Injects memory system instructions into the system prompt (`build_system_instructions`).
- Applies importance decay (`apply_decay`).
- Enforces the memory budget (`enforce_budget`).
- Processes `<mem_*>` tags in the LLM output (`<mem_new>`, `<mem_update>`,
  `<mem_tag>`, `<mem_load>`, `<mem_delete>`, `<mem_search>`, `<mem_rel>`).
- Injects the Active Memories context block.

When `False` (the default), none of this happens — no memory reads, no memory
writes, no `<mem_*>` processing.

> **Default changed**: older versions defaulted `enable_memory` to `True`. The
> current signature defaults to `False` so that a bare `chat()` call does NOT
> silently persist conversation content. Enable explicitly when you want
> persistent agent memory.

## What to expect

- **Manager resolution**: the manager is resolved at turn start. If no manager
  can be resolved, memory instructions are silently skipped (a warning is
  logged) even with `enable_memory=True`.
- **Sub-flags** (`enable_episodic_memory`, `enable_auto_dream`,
  `enable_deep_memory_pulling`) are ALL inert when this flag is `False` — they
  need the master.
- **Scoping**: if a personality with its own `memory_manager` is active, that
  manager owns the memory ops (independent life). Otherwise the discussion's
  manager is used (system-managed life).

## Usage

```python
# Explicit opt-in to persistent memory
result = discussion.chat(
    "Remember my daughter's exam results",
    enable_memory=True,
    enable_episodic_memory=True,     # save the turn as episodic memory (non-trivial only)
    enable_deep_memory_pulling=True, # pre-generation retrieval
)

# Privacy session — no memory persistence (the default)
result = discussion.chat(
    "Summarize this private note",
    enable_memory=False,
)
```

## Related flags

- `enable_episodic_memory` — save non-trivial turns as episodic memory.
- `enable_auto_dream` — run a post-turn memory consolidation cycle.
- `enable_deep_memory_pulling` — pre-generation retrieval of deep memories.
- `wipe_all_memories()` — deletes all episodic + associative memories from
  the DB (see `ChatMixin.wipe_all_memories`).

## Recovering generated stuff

The turn result carries `memory_report` (dict of memory ops performed —
inserts, updates, loads) and `dream_report` (when `enable_auto_dream` is on).
Memory records are queryable via the manager's `query(text, top_k, level)`
method (levels 1=Working, 2=Deep, 3=Archived).