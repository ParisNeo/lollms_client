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