# `enforce_end_tag`

```python
enforce_end_tag: bool = False
```

**Category**: Core loop

## What it does

Controls whether a first-round, action-free, `<done/>`-less response may finish
the turn.

## What to expect

- `False` (default): a pure conversational round 1 (no tools, no artifacts, no
  prior actions) ends the turn normally — ideal for chat.
- `True`: the loop injects a termination protocol and keeps generating until the
  model explicitly emits `<done/>` — intended for strict agentic sessions where
  prose-only answers are rejected.

## Usage

```python
discussion.chat(task, enforce_end_tag=True)
```

## Related flags

- `max_nb_rounds` (non-boolean) — bounds the loop even with `enforce_end_tag`.