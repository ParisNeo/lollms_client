# `enable_deep_memory_pulling`

```python
enable_deep_memory_pulling: bool = True
```

**Category**: Memory

## What it does

Automatically pulls relevant deep memories into working memory based on the user
message, before generation.

## What to expect

- Extra retrieval latency per turn; failures are logged as warnings and never
  abort the turn.

## Usage

```python
discussion.chat(msg, enable_deep_memory_pulling=True)  # default
```

## Related flags

- `enable_memory` — master switch.