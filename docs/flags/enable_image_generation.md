# `enable_image_generation`

```python
enable_image_generation: bool = True
```

**Category**: Artifacts

## What it does

Enables the `<generate_image>` tag. **Capability-gated**: instructions are only
injected when the client has a TTI binding (`client.tti` or a TTI profile
registry); a tag emitted without an engine fails gracefully with a
`status:failure` processing block.

## What to expect

- Generated images are stored as image artifacts with their pixels (base64 PNG)
  plus a logical markdown twin.
- The message is anchored with `<artefact_image id="<title>::0" />`.

## Recovering generated stuff (images)

- **Artifact pixels**: `art = discussion.artefacts.get(title)` →
  `art["images"][0]` (base64 PNG).
- **Message anchor**: `<artefact_image id="<title>::0" />` appended to
  `ai_msg.content`.
- **Physical file**: sync via `discussion.artefacts.sync_to_workspace(title)` or
  read the workspace twin directly.
- Also listed in the returned `"artefacts"` array of the chat result.

## Usage

```python
result = discussion.chat("Generate an image of a cat", enable_image_generation=True)
cat_art = discussion.artefacts.get("cat.png")
```

## Related flags

- `enable_image_editing` — the companion editing tag.
- `enable_artefacts` — master switch.
- `suppress_images` — withholds pixels from the model (not from you).