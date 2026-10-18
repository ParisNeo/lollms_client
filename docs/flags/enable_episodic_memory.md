# `enable_episodic_memory`

```python
enable_episodic_memory: bool = True
```

**Category**: Memory

## What it does

Saves this conversation turn as episodic memory — but only for **non-trivial
exchanges**:

- Total length > 200 chars, OR tools were used, OR artifacts were created.
- Greetings ("hi", "hello"), thanks, and yes/no exchanges are detected and
  skipped.

## What to expect

- Set `False` for privacy-sensitive sessions or when you want manual control
  over memory persistence.

## Usage

```python
discussion.chat(secret_message, enable_episodic_memory=False)
```

## Related flags

- `enable_memory` — master switch.