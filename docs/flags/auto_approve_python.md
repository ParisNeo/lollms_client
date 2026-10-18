# `auto_approve_python`

```python
auto_approve_python: bool = False
```

**Category**: Tools & Security

## What it does

Python execution runs **without confirmation prompts**.

## What to expect

- Use only in trusted, sandboxed environments. When `False`, the
  `confirm_handler` (if provided) is consulted before dangerous executions.

## Usage

```python
discussion.chat(
    task,
    enable_code_execution=True,
    auto_approve_python=True,   # trusted sandbox only!
)
```

## Related flags

- `enable_code_execution` — the flag this one modifies.