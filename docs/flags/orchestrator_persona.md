# `orchestrator_persona`

```python
orchestrator_persona: bool = False
```

**Category**: Personas & Diagnostics

## What it does

Runs the turn as the delegation-only **Orchestrator persona**: plain-language
worker catalogue, **no tool grammar**, and artifact writes plus tool execution
are structurally refused at the dispatcher (`<tool>` dispatch → refusal
envelope; `enable_artefacts` forced off in `_StreamState`).

## What to expect

- `<delegate>` / `<agent>` are the only action paths. Attempted `<tool>` calls
  are refused with a delegation-correction envelope.

## Usage

```python
discussion.chat(task, orchestrator_persona=True)
```

## Related flags

- `orchestrator_mode` — the runner-based variant.