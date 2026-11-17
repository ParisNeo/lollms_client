# Media Generation Flags — `enable_generate_video`, `enable_edit_video`, `enable_generate_music`, `enable_edit_music`, `enable_synthesize_speech`

**Category**: Media Generation
**Parent flags**: none (each is an independent per-modality opt-in)
**Capability gates**: TTV (video), TTM (music), TTS (speech) — each requires a
live binding callable OR a matching `*_model_profiles_registry` entry on the
client.

This page consolidates the five opt-in media flags. The image flags
(`enable_image_generation` / `enable_image_editing`) default to `True` and are
documented separately in [enable_image_generation.md](enable_image_generation.md)
and [enable_image_editing.md](enable_image_editing.md).

## Signatures

```python
enable_generate_video: bool = False
enable_edit_video:     bool = False
enable_generate_music: bool = False
enable_edit_music:     bool = False
enable_synthesize_speech: bool = False
```

## What they do

Each flag activates ONE text-to-X modality through dedicated XML tags. These
tags are NOT tools: they cannot be wrapped in `<tool>{...}</tool>` — such calls
are rejected as phantom tools. The chat layer teaches the model the correct
grammar only when the flag is set AND the corresponding engine is configured
(live binding or profile registry).

| Flag | Tag(s) enabled | Doctrine block |
|---|---|---|
| `enable_generate_video` | `<generate_video>` | `VIDEO GENERATION PROTOCOL` |
| `enable_edit_video` | `<edit_video>` | `VIDEO EDITING PROTOCOL` |
| `enable_generate_music` | `<generate_music>` | `MUSIC GENERATION PROTOCOL` |
| `enable_edit_music` | `<edit_music>` | `MUSIC EDITING PROTOCOL` |
| `enable_synthesize_speech` | `<generate_speech>` (+ `<edit_speech>` where bound) | `SPEECH SYNTHESIS PROTOCOL` |

Both the streaming dispatcher (`_StreamState`) and the non-streaming
post-processor (`_post_process_llm_response` via `_execute_media_tag`) handle
these tags identically: parse attributes → invoke the engine → decode the
payload → write to the workspace → register a metadata-card artifact → inject a
markdown anchor in the message content.

## What to expect

- All five flags are **`False` by default** — media generation is expensive
  (seconds to minutes per clip) and requires explicit opt-in.
- **Capability-gated per modality**. Missing binding + missing profile registry
  = the doctrine block is silently withheld and the tag fails with a
  `status:failure` processing block if the model emits it anyway.
- **Duration/params via attributes**: numeric parameters (`duration`, `width`,
  `height`, `fps`, `seed`, `negative_prompt`) belong in the tag attributes.
  Describing them inside the tag body has no effect. Malformed values are
  dropped (never silently coerced to a default that changes user intent).
- **Edit tags require `source` / `sources`** — a workspace file name for the
  target clip/track. Missing source surfaces a structured error to the model.
- **Output defaults**: `.mp4` for video, `.mp3` for music, `.wav` for speech.
  An extension is appended when the model omits it.
- **Files land in the discussion workspace** (`workspace_data_path`) and are
  registered as metadata-card artifacts (`type="video"` or `type="audio"`).
  The artifact carries a summary card, NOT the bytes — the physical file is
  the source of truth (as-is doctrine).
- **`enable_artefacts=False` still writes the physical file** but suppresses
  the artifact card (the media dispatcher logs a warning and returns the file
  path anyway).

### Edit-speech caveat

`edit_speech` is only supported when the active TTS binding exposes an edit
callable. As of this writing, no shipped TTS binding does — the dispatcher
logs an explicit warning and skips the tag, and the non-streaming post-processor
mirrors the same behavior. Use `<generate_speech>` to produce a fresh audio
file.

## Gate

Each flag requires the corresponding engine on the client:

| Modality | Live binding attribute | Required callable | Profile registry |
|---|---|---|---|
| Video | `client.ttv` | `generate_video` (and `edit_video` for editing) | `client.ttv_model_profiles_registry` |
| Music | `client.ttm` | `generate_music` (and `edit_music` for editing) | `client.ttm_model_profiles_registry` |
| Speech | `client.tts` | `generate_speech` (and `edit_speech` when available) | `client.tts_model_profiles_registry` |

The pipeline treats EITHER signal as sufficient: a live binding with the
required callable (strongest), OR a non-empty profile registry (user-declared
intent, lazily-mountable binding). The dispatcher remains authoritative at
execution time — a lazily-created binding that fails to initialize degrades to
a warning, never a crash.

## Usage

**Generate a short video clip**:
```python
result = discussion.chat(
    "Generate a 5-second timelapse of a sunset over the ocean",
    enable_generate_video=True,
)
# Disk: ./sunset_timelapse.mp4
# Artifact card: discussion.artefacts.get("sunset_timelapse.mp4")
# Message: [🎬 sunset_timelapse](./sunset_timelapse.mp4)
```

**Edit an existing video**:
```python
result = discussion.chat(
    "Trim sunset_timelapse.mp4 to the first 3 seconds and slow it down",
    enable_edit_video=True,
)
# Disk: ./sunset_timelapse_edited.mp4 (source is NOT overwritten)
```

**Compose a music track**:
```python
result = discussion.chat(
    "Compose a 30s lofi background track, instrumental only",
    enable_generate_music=True,
)
# Disk: ./lofi_track.mp3
```

**Narrate a script**:
```python
result = discussion.chat(
    "Speak this intro: 'Welcome to the show. Today we talk about AI.'",
    enable_synthesize_speech=True,
)
# Disk: ./intro.wav
```

**Compose + narrate + timeline in one turn (all three modalities)**:
```python
result = discussion.chat(
    "Compose a 20s lofi intro, then narrate 'Welcome to the show', then "
    "generate a 3-second animated logo clip",
    enable_generate_music=True,
    enable_synthesize_speech=True,
    enable_generate_video=True,
)
# Disk: ./intro.mp3, ./narration.wav, ./logo.mp4
# All three metadata cards appear in result["artefacts"].
```

## Related flags

- `enable_artefacts` — master switch for artifact registration. When off, the
  physical file is still written; only the DB card is skipped.
- `enable_image_generation` / `enable_image_editing` — the equivalent image-side
  flags (documented separately; default `True`, TTI-gated).
- `enable_books` — books/presentations may reference media artifacts via
  `<artefact_image>` anchors; combine carefully in multi-modal turns.

## Recovering generated stuff

Each modality produces a physical file + a metadata-card artifact. Recover
either way:

| Where | How |
|---|---|
| Disk | `./<name>.mp4` / `.mp3` / `.wav` in the discussion workspace (`workspace_data_path`) |
| Artifact API | `discussion.artefacts.get("<name>.<ext>")` → returns the metadata card dict |
| Turn result | `result["artefacts"]` array lists every card produced this turn |
| Message anchor | `[🎬 …](…mp4)` / `[🎵 …](…mp3)` / `[🗣️ …](…wav)` replaces the tag in the message body |

Unlike image artifacts, media cards do NOT carry pixel/byte payloads — the
file on disk is the sole source of truth (as-is doctrine). Use the physical
file for playback, upload, or further processing.