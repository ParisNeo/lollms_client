# `dynamic_effort`

```python
dynamic_effort: bool = False
```

**Category**: Core loop

## What it does

Enables the `<effort level="..."/>` interception so the model can adjust its own
reasoning effort between agentic rounds.

## What to expect

- Round 1 uses the explicitly passed `reasoning_effort` (or `"none"` if unset).
- When the model emits `<effort level="high"/>`, subsequent rounds run with
  `reasoning_effort="high"` and `think=True`. Level `"none"/"off"` disables
  reasoning for subsequent rounds.
- Requires a reasoning-capable binding to have any visible effect.

## Usage

```python
discussion.chat(task, dynamic_effort=True, reasoning_effort="medium")
```

## Related flags

- `think`, `reasoning_effort` (non-boolean companions).