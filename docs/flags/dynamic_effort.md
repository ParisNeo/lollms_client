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

## Where the effort levels come from

`dynamic_effort` switches *whether* the model is allowed to pick its own
effort level. Which levels exist is declared on the model profile via
`supported_reasoning_efforts` (default `low, medium, high`) and whether the
model can think at all is `thinking_enabled`. A profile with
`thinking_enabled=False` effectively disables this flag.

## Related flags

- `think`, `reasoning_effort` (non-boolean companions).