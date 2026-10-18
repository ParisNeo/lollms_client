# `stream`

```python
stream: bool = False
```

**Category**: Core loop

## What it does

Selects the callback mode for the turn: when `True`, the chat loop passes a
streaming callback to the LLM binding so tokens are relayed live; when `False`,
generation runs in blocking mode and the full response is returned at once.

## What to expect

- `True`: register your callback via `discussion.events_manager` (or pass a
  `callback` kwarg) to receive incremental tokens. The returned
  `ai_message.content` still contains the complete text.
- `False`: no incremental events; the turn returns only when generation
  finishes.

## Usage

```python
discussion.chat("Tell me a story", stream=True)
```

## Related flags

- `enable_in_message_status` — status lines are only useful in streaming mode.