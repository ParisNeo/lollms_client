# `enable_vlm_query`

```python
enable_vlm_query: bool = False
```

**Category**: Tools & Security

## What it does

Mounts `tool_vlm_query` as a **vision fallback** — only when the active LLM
lacks vision AND a vision-capable binding exists elsewhere on the client
(`client.llms` registry).

## What to expect

- With a native vision model active, this mounts nothing (the fallback is
  unnecessary).
- With a text-only model active and a vision binding available, the model can
  delegate image questions to the vision binding.

## Usage

```python
discussion.chat("What's in this image?", enable_vlm_query=True)
```

## Related tools & flags

- **`tool_render_pdf_pages`** (from `as_is_document_tools`) — CREATES
  images from a PDF (scanned pages, curves, charts) and attaches them as
  ephemeral vision content for the next round only. `tool_vlm_query`
  COMPLEMENTS it: use `tool_render_pdf_pages` when you have a PDF to look
  at, `tool_vlm_query` when you have an image and need a _different_
  (vision-capable) model to describe it. If the active model has native
  vision, `tool_render_pdf_pages` alone is enough.
- `enable_computer_use` — the primary vision-gated toolset.
- `suppress_images` — withholds pixels entirely.
- **`as_is_document_tools` auto-mount** — the document toolset (home of
  `tool_render_pdf_pages`) is admitted whenever a `.pdf`, `.docx`, `.xlsx`,
  or `.pptx` file is present in the workspace (file-type detection — see
  `TOOL_LIBRARY_FILE_TYPES`). `enable_office_files` remains a backward-
  compatible escape hatch.