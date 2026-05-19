# File Import Mixin

## Overview

The `FileImportMixin` adds `import_file()` to `LollmsDiscussion`, enabling files of many formats to be imported into the artefact system. This bridges the gap between external documents and the LLM's context window, supporting text extraction, image rendering, and even OCR via vision-capable models.

## Import Modes

Four modes control how files are processed:

| Mode | Description | Creates Text Artefact | Creates Image Artefact |
|------|-------------|----------------------|------------------------|
| `text` | Extract plain text only | ✅ | ❌ |
| `text_images` | Extract text + embedded/rendered images | ✅ | ✅ (deactivated) |
| `images_only` | Rasterise/collect images only | ❌ | ✅ (deactivated) |
| `ocr` | Render to image → LLM vision OCR → text | ✅ | ❌ |

### Mode Details

**`text`** — The default fallback. Extracts all readable text from the document. For PDFs, each page is prefixed with `## Page N`. For spreadsheets, each sheet becomes a section. No images are extracted.

**`text_images`** — Extracts text and simultaneously renders pages/slides as images. The text artefact contains `<artefact_image id="TITLE::N" />` anchors that correlate prose with pixel data. Images are stored in a separate deactivated image artefact (title `TITLE::images`) to avoid bloating the LLM context unless explicitly needed.

**`images_only`** — Skips text extraction entirely. Useful for:
- Scanned PDFs where text extraction would be noisy
- Pure image files (PNG, JPG, etc.)
- Slide decks where visual layout matters more than text

**`ocr`** — Renders each page to an image, then passes it to the LLM vision API with a transcription prompt. The per-page transcripts are stitched into a single text artefact. Requires `lollmsClient.llm` to support vision input.

## Supported Formats

### Text & Code (text extraction only)
- `.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.xml`, `.html`, `.htm`
- `.rst`, `.log`, `.tex`, `.ini`, `.toml`, `.cfg`, `.conf`
- Source code: `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.sh`, `.bat`, `.ps1`, `.sql`, `.r`, `.lua`, `.erl`, `.hs`, `.clj`, `.scala`, `.dart`, `.zig`, `.nim`

### Rich Documents (text + images)
| Format | Text Extraction | Image Extraction | Notes |
|--------|----------------|------------------|-------|
| PDF | ✅ (pymupdf / pypdf) | ✅ (page rendering) | Best with `pymupdf` |
| DOCX | ✅ (python-docx) | ✅ (embedded media) | — |
| PPTX | ✅ (python-pptx) | ✅ (slide thumbnails) | — |
| XLSX | ✅ (openpyxl) | ❌ | Sheets → sections |

### Pure Images
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`, `.tiff`, `.tif`, `.svg`

## Image Artefact Convention

When `text_images` or `images_only` mode creates images, they follow a strict convention:

1. **Image artefact is DEACTIVATED by default** — it is not injected into the LLM context automatically
2. **Text artefact contains anchors** — `<artefact_image id="TITLE::N" />` where `N` is the 0-based image index
3. **The UI or application must activate** the image artefact if raw pixel data is needed in context

This prevents large image payloads from consuming the token budget unexpectedly.

## Usage Examples

### Basic Text Import
```python
result = discussion.import_file(
    path="/path/to/report.pdf",
    mode="text",
    title="Q4_Report",  # optional, defaults to filename stem
    activate=True,      # activate text artefact immediately
)
# result["text_artefact"]  → the created/updated artefact dict
# result["image_artefact"] → None (text mode)
```

### Text + Images (Recommended for Documents)
```python
result = discussion.import_file(
    path="/path/to/manual.pdf",
    mode="text_images",
)
# Text artefact contains content with <artefact_image id="manual::0" /> anchors
# Image artefact "manual::images" holds the rendered pages, deactivated
```

### OCR for Scanned Documents
```python
result = discussion.import_file(
    path="/path/to/scanned_contract.pdf",
    mode="ocr",
)
# Requires vision-capable LLM. Each page is rendered, then transcribed.
```

### Images Only
```python
result = discussion.import_file(
    path="/path/to/slides.pptx",
    mode="images_only",
)
# Creates "slides::images" artefact with slide thumbnails
```

## Return Value

`import_file()` returns a dictionary:

```python
{
    "text_artefact":  dict | None,   # Created/updated text artefact
    "image_artefact": dict | None,   # Created image artefact (deactivated)
    "mode":            str,           # The import mode used
    "page_count":      int,           # Number of pages/slides processed
    "image_count":     int,           # Number of images extracted
    "warnings":        [str],         # Any non-fatal issues encountered
}
```

## Dependencies

All dependencies are optional — they are **automatically installed on first use** via `pipmaster`. You can also pre-install them manually:

| Package | Purpose | Install |
|---------|---------|---------|
| `pipmaster` | Automatic package installer | `pip install pipmaster` |
| `pymupdf` | PDF text + image rendering | `pip install pymupdf` |
| `pypdf` | PDF text fallback | `pip install pypdf` |
| `python-docx` | DOCX parsing | `pip install python-docx` |
| `python-pptx` | PPTX parsing | `pip install python-pptx` |
| `Pillow` | Image resize/format conversion | `pip install pillow` |
| `openpyxl` | XLSX text extraction | `pip install openpyxl` |
| `pdf2image` | PDF rendering fallback | `pip install pdf2image` |

If `pipmaster` is not installed, the mixin falls back to standard `ImportError` behavior and the caller must install packages manually.

## Technical Notes

### Image Resizing
All extracted images are resized so their longest side ≤ 1024 pixels (configurable via `_MAX_LLM_IMAGE_DIM`). This matches typical vision model requirements. JPEG quality is set to 85.

### PDF Rendering
- Primary: `pymupdf` (fitz) at 150 DPI
- Fallback: `pdf2image` + Pillow

### OCR Prompt
The OCR mode uses this system prompt for each page:
> "You are an OCR engine. Transcribe ALL text visible in this image exactly as it appears, preserving paragraph breaks. Output only the transcribed text — no commentary, no markdown code fences, no preamble."

### Error Handling
- Unsupported formats with valid UTF-8 content are treated as plain text with a warning
- Binary files that cannot be decoded raise `RuntimeError`
- Missing optional dependencies trigger warnings but do not crash

## Integration with Artefact System

Imported files become first-class artefacts:
- Versioned on re-import (same title → new version)
- Type-detected automatically (`CODE` for source files, `DOCUMENT` for prose, `IMAGE` for image artefacts)
- Language hint set for code files (e.g., `python`, `javascript`)
- Participate in the full artefact lifecycle: activation, deactivation, update, revert, diff