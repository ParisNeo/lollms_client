# `debug`

```python
debug: bool = False
```

**Category**: Personas & Diagnostics

## What it does

Mounts the `debug_toolset` (`tool_dump_context`) and prints the active tool
registry to the console.

## What to expect

- Combine with `discussion._debug_mode = True` for persistent per-discussion
  dumps. Zero-registry turns print an explicit warning.

## Usage

```python
discussion._debug_mode = True
discussion.chat(task, debug=True)
```

## Related flags

- `debug_export` — file-based dumps.