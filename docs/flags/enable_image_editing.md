# `enable_image_editing`

```python
enable_image_editing: bool = True
```

**Category**: Artifacts

## What it does

Enables the `<edit_image>` tag. **Capability-gated**: instructions are only
injected when the client has a TTI binding. A tag emitted without an engine
fails gracefully with a `status:failure` processing block.

## What to expect

- Requires a source image (existing artifact pixels or the last active image)
  AND non-empty instructions; otherwise it fails cleanly.
- Edited images are stored as image artifacts (new version of the source, or a
  new artifact).

## Dimension Handling

`width` and `height` are **optional** attributes on `<edit_image>`.

- **Omitted** → forwarded as absent to the TTI engine → the edit runs at the
  **source image's native dimensions** (pass-through contract). This is the
  default and the correct choice for content-only edits (restyle, recolor,
  add/remove elements) where the output should match the input size.
- **Explicitly provided** → clamped to the engine's supported range (64–4096)
  and honored as the target output size. Use this only when you must force a
  specific size or aspect ratio.
- **Malformed value** (non-numeric) → treated as unspecified (`None`), never
  silently coerced to a default that would resize the source.

This matches the backend semantics: the edit uses the input image's dimensions
when none are supplied.

## Recovering generated stuff (images)

Same as `enable_image_generation`: `discussion.artefacts.get(title)` →
`art["images"][0]`, message anchor `<artefact_image id="...::0" />`, physical
file via `sync_to_workspace`.

## Usage

```python
result = discussion.chat(
    "Make the cat wear sunglasses",
    enable_image_editing=True,
)
```

## Related flags

- `enable_image_generation` — companion flag.
- `enable_artefacts` — master switch.