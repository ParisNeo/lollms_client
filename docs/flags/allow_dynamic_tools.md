# `allow_dynamic_tools`

```python
allow_dynamic_tools: bool = False
```

**Category**: Tools & Security

## What it does

Security gate for tool-authored tools. With `False`, artifacts of `type="tool"`
are **not** registered as executable LCP tools.

## What to expect

- Enable only if you explicitly trust the model to author and run its own tools.

## Usage

```python
discussion.chat("Build a tool that fetches weather", allow_dynamic_tools=True)
```

## Related flags

- `enable_artefacts` — tool artifacts are created through the artifact system.