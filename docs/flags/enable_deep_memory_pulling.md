# `enable_deep_memory_pulling`

```python
enable_deep_memory_pulling: bool = False
```

**Category**: Memory
**Parent flags**: `enable_memory` (required — this flag is inert when the
master switch is off)

## What it does

Automatically pulls relevant deep memories into working memory based on the
user message, before generation begins. This is the pre-generation retrieval
side of the memory subsystem.

## What to expect

- **Inert without `enable_memory=True`** — the master switch gates everything.
- **Extra retrieval latency** per turn (typically sub-second for reasonable
  memory volume; scales with the store size).
- **Failures are logged as warnings** — retrieval errors never abort the turn.
- **Deduplication**: the same memory is not pulled twice in a turn.

## Usage

```python
# Enable deep-memory retrieval before each turn
result = discussion.chat(
    "Do you remember my daughter's exam?",
    enable_memory=True,
    enable_deep_memory_pulling=True,
)
# Relevant deep memories are pulled into working memory before generation,
# so the model's response can reference them naturally.
```

## Related flags

- `enable_memory` — required master switch.
- `enable_episodic_memory` — populates the memory store the puller reads.
- `enable_auto_dream` — consolidates memories into deeper tiers.

## Recovering generated stuff

Nothing new is produced; the retrieved memories are injected into the system
prompt's Active Memories zone. To inspect manually:

```python
mm = discussion._get_memory_manager()
pulled = mm.query(user_message, top_k=5, level=2)  # Deep memory
```