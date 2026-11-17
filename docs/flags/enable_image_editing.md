# `enable_image_editing`

```python
enable_image_editing: bool = True
```

**Category**: Media Generation
**Parent flags**: `enable_artefacts` (master switch)
**Gate**: TTI with a callable `edit_image` on the live binding, OR a non-empty
`tti_model_profiles_registry`

## What it does

Enables the `<edit_image>` XML tag — the model emits it to modify an existing
image (resize, crop, restyle, upscale, recolor, extend, add elements, combine
multiple sources). Editing is a TAG capability, NOT a tool.

## What to expect

- **Capability-gated** on a TTI binding whose `edit_image` method is callable,
  OR on a TTI profile registry. (Tighter than `enable_image_generation`, which
  only needs `generate_image`.)
- **Source resolution**: the tag accepts a `sources` attribute (comma-separated
  workspace image paths) for multi-image edits, or the `name` attribute (single
  image), or — failing both — falls back to the last active image in the chat.
  Missing source fails cleanly (no generation, no crash).
- **Decision rule (EDIT vs GENERATE)**: when the user references an existing
  image (filename, artifact title, "this image", "the image you made", or any
  transform verb), the model MUST use `<edit_image>` — attempting `<generate_image>`
  destroys the user's original content (the doctrine explicitly forbids this).
- **Dimension control**: `width`/`height` are OPTIONAL. When omitted, the
  engine edits at the SOURCE image's native dimensions. When provided, values
  are clamped to 64–4096.
- **Aspect-ratio doctrine**: aspect-ratio requests ("make it 16:9") MUST be
  translated into explicit `width`/`height` attributes — describing the ratio
  inside the prompt text does nothing (the engine renders at the tag's
  dimensions).
- **Default visibility `SINGLE_ROUND`**: the edited image is visible to the LLM
  on the round immediately following creation, then auto-demoted to reduce
  context bloat.

## Gate

Requires either:
- `client.tti` with a callable `edit_image`, OR
- a non-empty `client.tti_model_profiles_registry`.

When the gate is unmet, the model is NOT taught the editing protocol; a tag
emitted anyway dispatches through the normal path (which surfaces a
`status:failure` block).

A separate gate in the dispatcher checks that `edit_image` is callable —
`enable_image_generation` alone does NOT enable editing. Both flags are needed
for a full image workflow.

## Usage

```python
# Editing works out of the box (both image flags default to True)
result = discussion.chat(
    "Resize my_image.png to be 16:9",
)
# The model emits:
#   <edit_image name="my_image_16x9.png" sources="my_image.png"
#               width="1920" height="1080">
#     Preserve the content, extend to the wider frame.
#   </edit_image>

# Combine multiple source images
result = discussion.chat(
    "Blend these two photos: photo_a.png as the base, photo_b.png as the style",
)
# The model emits:
#   <edit_image name="blended.png" sources="photo_a.png, photo_b.png">...</edit_image>
```

## Related flags

- `enable_image_generation` — enables `<generate_image>` (new images).
- `enable_artefacts` — master switch; must be `True` for the edited image to
  be registered as a persistent artifact.
- `suppress_images` — negative polarity; withholds image pixels from context.

## Recovering generated stuff

```python
art = discussion.artefacts.get("<output_name>")    # artifact record
img_b64 = art["images"][0]                         # base64 PNG
discussion.artefacts.sync_to_workspace("<output_name>")
```

Also listed in `result["artefacts"]` and referenced in message content via
`<artefact_image id="<output_name>::0" />`.