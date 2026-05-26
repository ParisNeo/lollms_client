---
name: Lollms Multimodal Document and Image Parsing
description: Teaches how to import, parse, and process PDF, DOCX, PPTX, and image files using anchor-mapped and OCR import modes in LollmsDiscussion.
author: ParisNeo
version: 1.0.0
category: lollms_client/parsing
created: 2026-05-24
---

# Lollms Multimodal Document and Image Parsing

This skill teaches how to import rich documents (PDF, Word, Slides) and image files into the LoLLMS system using appropriate import modes and anchor configurations.

## 1. Import Modes Overview
The file import subsystem supports four distinct operational modalities:
- `"text"`: Extracts plain text and Markdown only. No images are rendered.
- `"text_images"`: Extracts the text content and simultaneously renders pages/embedded images.
- `"images_only"`: Bypasses text extraction and converts the document strictly to high-resolution JPEG pages (great for scanned documents).
- `"ocr"`: Renders pages and passes them to a local/remote vision-capable LLM to transcribe the visual text, stitching results into a unified document.

## 2. Importing a Document with Anchors
To import a file, use the `import_file` method. For `text_images` mode, a companion image artifact (`<title>::images`) is created automatically. The text is populated with inline anchor tags:

```python
# Import a PDF report
result = discussion.import_file(
    path="quarterly_report.pdf",
    mode="text_images",
    title="q2_report",
    activate=True
)

print(f"Text artifact title: {result['text_artefact']['title']}")
print(f"Images artifact title: {result['image_artefact']['title']}")
print(f"Total pages rendered: {result['page_count']}")
```

The parsed text is injected into the prompt, containing inline reference anchors like:
`<artefact_image id="q2_report::images::0" />`

## 3. How the LLM Processes Anchors
- **Vision Map**: When the chat turn executes, the active companion images are appended to the model's vision input array.
- **Correlation**: The model reads the text anchors and correlates them with the image indexes in the vision input (e.g. image index #1 corresponds to the anchor `q2_report::images::0`).
- **Answering**: The model can then output the anchor verbatim in its response to reference specific figures, e.g.:
  `"As we can observe in Figure 1, represented by <artefact_image id=\"q2_report::images::0\" />, the revenue increased..."`

## 4. Reconstructing and Bundling
All imported multimodal artifacts can be exported as a unified, self-contained JSON-serializable bundle containing both the text content and base64-encoded image files.

```python
# Export a bundle
bundle = discussion.artefacts.export_artefact_bundle("q2_report")

# Import the bundle back into an entirely different discussion session (restoring all sub-images)
restored_main = separate_discussion.artefacts.import_artefact_bundle(bundle, activate=True)
```