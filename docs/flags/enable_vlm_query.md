# `enable_vlm_query`

```python
enable_vlm_query: bool = False
```

**Category**: Tools & Security

## What it does

Mounts `tool_vlm_query` as a **vision fallback** — only when the active LLM
lacks vision AND a vision-capable binding exists elsewhere on the client
(`client.llms` registry).

## What to expect

- With a native vision model active, this mounts nothing (the fallback is
  unnecessary).
- With a text-only model active and a vision binding available, the model can
  delegate image questions to the vision binding.

## Usage

```python
discussion.chat("What's in this image?", enable_vlm_query=True)
```

## Related flags

- `enable_computer_use` — the primary vision-gated toolset.
- `suppress_images` — withholds pixels entirely.