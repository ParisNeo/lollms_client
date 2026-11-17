# `enable_episodic_memory`

```python
enable_episodic_memory: bool = False
```

**Category**: Memory
**Parent flags**: `enable_memory` (required — this flag is inert when the
master switch is off)

## What it does

Saves the conversation turn as episodic memory — but only for non-trivial
exchanges. Trivial detection:

- **Substantial content**: turn total length > 200 chars, OR
- **Tools were used** (`tool_calls_this_turn` non-empty), OR
- **Artifacts were created** (the stream state's `affected_artefacts`
  non-empty).

Greetings ("hi", "hello", "thanks", "ok"), yes/no exchanges, and other trivial
patterns are detected and skipped even when the length threshold is met.

## What to expect

- **Inert without `enable_memory=True`** — the master switch gates everything.
- **Non-trivial-only**: greetings and yes/no exchanges are never persisted.
- **No performance impact** when the exchange is trivial (early-exit check
  before the DB write).
- **Failures are logged as warnings** — a memory DB error never aborts the
  turn.

## Usage

```python
# Persist conversation turns (requires enable_memory=True)
result = discussion.chat(
    "Help me plan a 3-week trip to Japan",
    enable_memory=True,
    enable_episodic_memory=True,
)

# Skip saving a one-off answer even inside a memory-enabled session
result = discussion.chat(
    "What time is it in Tokyo?",
    enable_memory=True,
    enable_episodic_memory=False,
)
```

## Related flags

- `enable_memory` — required master switch.
- `enable_auto_dream` — post-turn consolidation of episodic memories.
- `enable_deep_memory_pulling` — pre-generation retrieval.

## Recovering generated stuff

Episodic memories are queryable via the memory manager:

```python
mm = discussion._get_memory_manager()
results = mm.query("trip to Japan", top_k=5)
```

Each result carries `id`, `content`, `importance`, `tags`, `level`
(1=Working, 2=Deep, 3=Archived).