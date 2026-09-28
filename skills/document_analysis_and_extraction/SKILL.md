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

## Proofreading & Annotation Protocol
When asked to proofread, correct, or review a document:
- Step 1: Read the document text batch by batch.
- Step 2: Note exact verbatim quotes for issues (grammar, clarity, errors).
- Step 3: Call `tool_annotate_document` (for comments/highlights) or `tool_edit_document_text` (for surgical in-place text replacement).
- Step 4: When using `tool_annotate_document`, set `commenter_name` to the current OS username.