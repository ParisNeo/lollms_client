# `list_available_models`

```python
list_available_models: bool = False
```

**Category**: Core loop

## What it does

Mounts the models toolset so the model can list available LLM bindings/models
and switch between them mid-conversation (useful for model-router
personalities).

## What to expect

- `True`: `tool_list_available_models`, `tool_switch_model` (and friends) are
  registered in the tool registry. Supports all modalities including tools.
- `False`: no model-switching tools; the model uses the active binding.

## Usage

```python
discussion.chat("Which models are available?", list_available_models=True)
```

## Related flags

- `enable_computer_use` — another tool-mounting flag, but vision-gated.

## Interaction with native tool calling

`list_available_models` **does not** toggle native tool calling. Whether the
active model uses the binding's native `tools=` path or falls back to the XML
`<tool>` protocol is decided **per model profile** by the binding itself:

- The model profile declares `native_tools_enabled: True` (a capability hint,
  same mechanism as `vision_enabled`).
- The binding combines that hint with its own backend capability probe and
  exposes the resolved value as `binding.native_tools_enabled`.
- The chat mixin reads `binding.native_tools_enabled` to decide whether to
  pass `tools=` / `tool_choice="auto"` to `generate_from_messages`.

See [`docs/UNIVERSAL_MODEL_PROFILES.md` §2.4](../UNIVERSAL_MODEL_PROFILES.md#24-native-tool-calling-native_tools_enabled)
for the full resolution rule. Enabling `list_available_models` on a profile
whose `native_tools_enabled` is `False` simply means the model-switching tools
are advertised via the XML `<tool>` protocol instead of native `tool_calls` —
the tools themselves work identically.