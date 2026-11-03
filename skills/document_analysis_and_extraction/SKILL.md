---
name: document_analysis_and_extraction
title: Document Analysis, Extraction and Annotation
category: document_processing
tags: [documents, pdf, docx, xlsx, pptx, extraction, proofreading, annotations]
description: In-depth workflow for extracting text, tables, schemas, and applying corrections or comments to PDF, Word, Excel, and PowerPoint documents.
---

# Document Analysis and Extraction Skill

## Document Ingestion Protocol
1. **Direct Native Reading**: Use `<unlock_file>path/to/document.pdf</unlock_file>` to load document text into context. LoLLMS automatically extracts text from PDF, DOCX, XLSX, and PPTX natively.
2. **Inspecting Metadata**: For large documents (>50 pages), use `tool_inspect_document` to inspect page count, sheet names, or slide layouts.
3. **Batched Extraction**: For deep analysis of long documents, use `tool_read_document_content` specifying page ranges (`page_or_sheet="1-10"`).
4. **Locating keywords**: Use `tool_grep_document` to find a fragment of text without loading the whole document — returns match locations (page numbers, paragraph indices) BEFORE rendering or reading.

## Visual Inspection Protocol (scanned pages, curves, charts, diagrams)
Use `tool_render_pdf_pages` when the page resists text extraction:

- **Scanned pages** with no selectable text.
- **Mathematical curves, plots, spectrograms, heat-maps**.
- **Diagrams, flowcharts, schematics, engineering drawings**.
- **Signatures, stamps, forms**, or layout where "picture = meaning".

### Workflow

1. Call `tool_inspect_document` first to learn the total page count.
2. Call `tool_render_pdf_pages` with a **narrow page selector** — `pages="7"` for one page, `pages="5-7"` for a batch. **Never** dump a whole large document in one call; the hard cap is 12 pages.
3. The tool returns base64 images attached to **this round only** (`ephemeral=True`).
4. **Inspect the images with your own vision** — read scanned text, interpret axes/series, describe diagrams.
5. **Write every useful finding into the scratchpad or an artifact BEFORE the round ends.** Rendered images are NOT persisted across rounds; anything not saved is lost.
6. For plot-heavy pages, raise `dpi=220` for legible axis labels; set `include_embedded_figures=true` to also harvest the pristine figure rasters.

### Chunked-walk pattern (context-safe)

```
tool_render_pdf_pages("scan.pdf", pages="1-10")   # one round, ~10 images
→ persist findings → round ends → images discarded
tool_render_pdf_pages("scan.pdf", pages="11-20")  # next chunk
→ persist findings → ...
```

## Delegation to an external VLM (when you lack vision)

If the active model is **text-only** and a vision-capable binding is
available on the client, mount `enable_vlm_query=True` in the chat call.
`tool_vlm_query` then delegates image questions ("what does this chart
show?") to that vision model — useful when `tool_render_pdf_pages` returns
an image but you cannot see it yourself. Both tools can be combined: render
the PDF page first, then ask `tool_vlm_query` to describe it.

## Proofreading & Annotation Protocol
When asked to proofread, correct, or review a document:
- Step 1: Read the document text batch by batch (or render scanned pages with `tool_render_pdf_pages` and save the findings to the scratchpad).
- Step 2: Note exact verbatim quotes for issues (grammar, clarity, errors).
- Step 3: Call `tool_annotate_document` (for comments/highlights) or `tool_edit_document_text` (for surgical in-place text replacement).
- Step 4: When using `tool_annotate_document`, set `commenter_name` to the current OS username.