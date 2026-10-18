# `suppress_images`

```python
suppress_images: bool = False
```

**Category**: Tools & Security

## What it does

Withholds **all** image pixels from the model's context this turn —
user-supplied images, generated plots, unlocked image artifacts.

**Negative-polarity flag**: `True` *disables* vision hydration rather than
enabling a feature.

## What to expect

- Vision hydration is skipped entirely; a console diagnostic
  (`[VisionDiag] suppress_images=True`) is emitted.
- Use with text-only models or to cut token cost. Unlocked images produce
  guidance telling the model pixels are unavailable (and it must say so rather
  than pretend to see).

## Usage

```python
discussion.chat(msg_with_images, suppress_images=True)
```

## Related flags

- `enable_computer_use` — relies on vision hydration; expect degraded behavior
  if combined.
- `enable_image_generation` — pixels are still stored in artifacts; only the
  model's view is suppressed.