# `enable_image_generation`

```python
enable_image_generation: bool = True
```

**Category**: Media Generation
**Parent flags**: `enable_artefacts` (master switch)
**Gate**: TTI (requires a live `client.tti` binding with `generate_image`, OR a
non-empty `tti_model_profiles_registry`)

## What it does

Enables the `<generate_image>` XML tag — the model emits it to create a brand-
new image from a text prompt via the Text-to-Image engine. Image generation is a
TAG capability, NOT a tool: attempts to call an image-related tool name
(`generate_image`, `create_image`, `draw_image`, ...) are rejected as phantom
tools with a redirect to the correct tag syntax.

## What to expect

- **Capability-gated**: instructions are only injected when `_has_tti` is True
  (live binding OR profile registry). A `<generate_image>` tag emitted without
  an engine fails with a `status:failure` processing block and a console
  warning — the model is told to check the engine configuration.
- **Same-response mandate**: the tag MUST be emitted in the same response as
  the intent statement. Only emitting prose ("I will generate...") performs
  nothing.
- **Naming doctrine**: pick a descriptive kebab-case `name` attribute; subfolder
  paths are allowed (`name="images/white_rabbit.png"` — subfolders are created
  automatically). Never use `generated_image_xxxx` placeholders.
- **Dimension pass-through**: `width`/`height` are optional (defaults applied
  by the TTI engine). Valid values are clamped to the engine range (64–4096).
- Generated images are stored as image artifacts with their pixels (base64 PNG)
  plus a logical markdown twin.
- **Default visibility `SINGLE_ROUND`**: the LLM sees the image on the round
  immediately following creation (to verify the result and decide whether to
  edit), then the artifact is auto-demoted to `TREE_UNLOCKABLE` and the base64
  payload is freed from context. This is the primary mitigation against image
  context bloat.

## Gate

Requires either:
- `client.tti` with a callable `generate_image`, OR
- a non-empty `client.tti_model_profiles_registry` (user-declared intent;
  lazily-mountable binding).

The gate governs **instruction injection only**. The dispatcher is
authoritative at execution time — a lazily-created binding that fails to
initialize degrades to a warning, never a crash.

When the gate is unmet, the model is NOT told the tag exists; a tag emitted
anyway is dispatched through the normal path (which surfaces a `status:failure`
block).

## Usage

```python
# Image generation is on by default — just ask
result = discussion.chat(
    "Draw a cyberpunk cat with neon eyes",
)
# Disk: image artifact with base64 PNG + workspace twin
# Chat: [image_card] with <artefact_image id="<title>::0" /> anchor appended
```

To disable, pass `enable_image_generation=False` — the doctrine is withheld
and the tag becomes inert.

## Related flags

- `enable_image_editing` — enables `<edit_image>` (modify an existing image).
- `enable_artefacts` — master switch; must be `True` for the resulting image
  to be registered as a persistent artifact.
- `suppress_images` — negative polarity; withholds ALL image pixels (including
  generated ones) from the model's context.

## Recovering generated stuff

```python
art = discussion.artefacts.get("<title>")           # artifact record
img_b64 = art["images"][0]                          # base64 PNG
# or sync to workspace:
discussion.artefacts.sync_to_workspace("<title>")
```

The image is also referenced in the returned `result["artefacts"]` array and in
the AI message content via an `<artefact_image id="<title>::0" />` anchor.