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