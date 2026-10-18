# `enable_auto_dream`

```python
enable_auto_dream: bool = True
```

**Category**: Memory

## What it does

Runs a memory consolidation ("dream") cycle after the turn completes.

## Recovering generated stuff (report)

The chat result's `"dream_report"` key carries the report (when enabled and the
manager exists).

## Usage

```python
result = discussion.chat(msg, enable_auto_dream=True)
report = result.get("dream_report")
```

## Related flags

- `enable_memory` — master switch.