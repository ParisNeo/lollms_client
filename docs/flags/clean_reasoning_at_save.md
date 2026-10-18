# `clean_reasoning_at_save`

```python
clean_reasoning_at_save: bool = True
```

**Category**: Core loop

## What it does

When `True`, the discussion strips ```reasoning
console.log("live");
```
 output from the final saved assistant message content.

## What to expect

- `True` (default): the persisted `ai_message.content` contains only the
  conversational answer; reasoning traces are dropped at save time (they remain
  visible in `ai_message.thoughts` during streaming).
- `False`: raw reasoning markup stays in the saved content.

## Usage

```python
discussion.chat(msg, clean_reasoning_at_save=False)
```

## Related flags

- `stream` — reasoning is captured separately during streaming.