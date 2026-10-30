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

## Implementation Reference

See `examples/simple/personality_with_tti_tool.py` for a concrete example of
attaching a tool to an `AIPersonality` that accesses the injected
`LollmsClient` instance (reserved parameter `lollms_client_instance`) to call
the active TTI binding, never constructing its own client.

Dynamically created/registered tools follow the same LCP signature rules: they
may declare `lollms_client_instance=None` and `discussion_instance=None` to
receive the system-injected client/discussion at call time. A dynamic tool
that omits these reserved parameters simply will not receive the instances —
it still works as a plain function.