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

## Rewrite / Transform Workflow (Anti-Loop Contract)

When the user asks to **rewrite, translate, restyle, or transform** a document (e.g. "rewrite this course without the word X"), follow this contract strictly. It exists because agents routinely burn tokens re-querying the same document and then confabulate tool failures.

### 1. Sufficiency rule — stop when you have the data

- **One targeted `tool_grep_document`** for the token(s) to replace (e.g. `(?i)claude`) returns **every occurrence with line numbers**. That is *sufficient* to plan the rewrite.
- **Do NOT** follow up with `pattern="."` or `max_matches=200` "to get the full text". Large greps get summarized by the runtime and yield nothing new.
- **Do NOT** re-call `tool_inspect_document` after it already returned paragraph/heading counts.
- **Loop budget**: at most **2 exploratory calls** per document (1 inspect + 1 grep, or 2 greps). After that, commit to producing output.

### 2. No phantom tools — only call what exists

The document toolset is exactly: `tool_inspect_document`, `tool_read_document_content`, `tool_grep_document`, `tool_render_pdf_pages`, `tool_modify_docx`, `tool_modify_excel`, `tool_modify_pptx_slide`, `tool_modify_pdf_annotation`, `tool_annotate_document`, `tool_edit_document_text`.

- **Never invent** `tool_build_word`, `tool_read_word`, `tool_write_docx`, or any other name. If a tool is not in the list above, it does not exist — do not call it, do not "diagnose" it as broken.
- If you genuinely need a capability that is missing, say so **once** and propose the closest existing tool.

### 3. Format fidelity — deliver the same format the user gave you

- Source is `.docx` → deliver `.docx` via `tool_modify_docx` (full-document replace) or `tool_edit_document_text` (surgical).
- Source is `.pdf` → deliver annotated `.pdf` via `tool_modify_pdf_annotation`, or a new `.docx`/`.md` **only if the user asked for a different format**.
- **Never silently downgrade** a `.docx` request to a `.md` artifact and blame a tool. If `tool_modify_docx` fails, report the exact error and ask the user how to proceed — do not substitute a different format unilaterally.

### 4. Rewrite execution pattern

```
1. tool_grep_document(pattern="(?i)<token>")   # get all occurrences — ONE call
2. Plan the replacements from the grep output (no more reads needed)
3. tool_modify_docx(file, new_content=...)      # full rewrite, OR
   tool_edit_document_text(file, edits=[...])   # surgical replacements
4. Report what changed; emit <done/>
```

If the document is large and the grep output is truncated, use `tool_read_document_content` with **narrow page ranges** (`page_or_sheet="1-10"`) — never a whole-document read.

### 5. Anti-confabulation checklist (self-audit before emitting `<done/>`)

- [ ] Did I call any tool not in the list above? → If yes, that call was a hallucination; discard it.
- [ ] Did I re-query the same document more than twice? → If yes, I violated the loop budget.
- [ ] Did I deliver the same format the user provided? → If no, justify it explicitly or fix it.
- [ ] Did I claim a tool is "broken" without an actual error message from that tool? → If yes, that is confabulation.

## Proofreading & Annotation Protocol
When asked to proofread, correct, or review a document:
- Step 1: Read the document text batch by batch (or render scanned pages with `tool_render_pdf_pages` and save the findings to the scratchpad).
- Step 2: Note exact verbatim quotes for issues (grammar, clarity, errors).
- Step 3: Call `tool_annotate_document` (for comments/highlights) or `tool_edit_document_text` (for surgical in-place text replacement).
- Step 4: When using `tool_annotate_document`, set `commenter_name` to the current OS username.