# `orchestrator_mode`

```python
orchestrator_mode: bool = False
```

**Category**: Personas & Diagnostics

## What it does

Routes the entire turn through the `AgenticRunner` orchestrator, which spawns
specialist worker sub-agents and collects sealed reports.

## What to expect

- The orchestrator never sees tool-call grammar; workers execute.
- The returned structure comes from `AgenticRunner.run()` (not the standard chat
  result dict shape).

## Usage

```python
result = discussion.chat(complex_task, orchestrator_mode=True)
```

## Related flags

- `orchestrator_persona` — the delegation-only persona variant.