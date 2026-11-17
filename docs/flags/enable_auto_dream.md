# `enable_auto_dream`

```python
enable_auto_dream: bool = False
```

**Category**: Memory
**Parent flags**: `enable_memory` (required — this flag is inert when the
master switch is off)

## What it does

Runs a memory consolidation ("dream") cycle after the turn completes. The
cycle reorganizes episodic memories into higher-level abstractions, prunes
redundant items, and updates importance scores.

## What to expect

- **Inert without `enable_memory=True`** — the master switch gates everything.
- **Post-turn**: the dream cycle runs after the final assistant message is
  committed. It does NOT delay the visible response.
- **Extra LLM calls**: consolidation may invoke the LLM to generate summaries
  — expect additional latency proportional to memory volume.
- **Failures are logged as warnings** — a dream failure never aborts the turn.
- **No-op when no manager resolves** or when memory volume is too small to
  benefit.

## Usage

```python
# Full memory lifecycle — persistence + consolidation
result = discussion.chat(
    "Let's plan the full quarter",
    enable_memory=True,
    enable_episodic_memory=True,
    enable_auto_dream=True,
)
# The turn completes normally; the dream cycle runs in the background.
# Inspect the report:
report = result.get("dream_report")
```

## Related flags

- `enable_memory` — required master switch.
- `enable_episodic_memory` — produces the memories the dream cycle consolidates.
- `enable_deep_memory_pulling` — pre-generation retrieval.

## Recovering generated stuff

The `dream_report` key in the turn result carries the consolidation report
(when enabled and the manager exists). Example structure:

```python
{
    "merged": [...],       # memory IDs merged
    "pruned": [...],       # memory IDs pruned
    "promoted": [...],     # memory IDs promoted to a higher tier
}
```