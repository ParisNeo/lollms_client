# `enable_inline_widgets`

```python
enable_inline_widgets: bool = False
```

**Category**: Artifacts

## What it does

Enables `<lollms_inline>` widgets — host-UI rendered components.

## What to expect

- Presentation-layer only; requires `enable_artefacts=True` and a host UI that
  renders the widget markup.
- No persisted content beyond the raw XML passed through to the message content.

## Usage

```python
discussion.chat("Show a button widget", enable_inline_widgets=True)
```

## Related flags

- `enable_artefacts` — master switch.