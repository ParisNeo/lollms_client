# `enable_office_files` — DEPRECATED

**Category**: Tools & Security

> ⚠️ **This flag is deprecated.** It is no longer a signature parameter of
> `chat()` — it is accepted as a keyword argument for backward compatibility
> only.

## What it does now

Passing `enable_office_files=True` enables **all three per-software office
flags** and mounts the monolithic legacy `office_files_builder` library:

- `enable_powerpoint = True`
- `enable_word = True`
- `enable_excel = True`

A one-line deprecation notice is logged to the console.

## Migrate to the per-software flags

The canonical API is now three dedicated flags, each mounting exactly one
toolset with its own doctrine:

- [`enable_powerpoint`](enable_powerpoint.md) — 5 tools: build/edit/read +
  `tool_pptx_info` (template awareness) + `tool_view_slides` (≤4 transient
  slide images per call).
- [`enable_word`](enable_word.md) — 3 tools: build/edit/read with
  comment/highlight/section/field helpers.
- [`enable_excel`](enable_excel.md) — 3 tools: build/edit/read with the
  formulas-only doctrine.

## Usage (legacy)

```python
# Legacy — still works, logs a deprecation notice:
result = discussion.chat(
    "Build a quarterly report: a .docx summary and an .xlsx KPI workbook",
    enable_office_files=True,
)

# Preferred — per-software:
result = discussion.chat(
    "Build a quarterly report: a .docx summary and an .xlsx KPI workbook",
    enable_word=True,
    enable_excel=True,
)
```

## Related flags

- [`enable_data_tools`](enable_data_tools.md) — contextual data/document
  tools (mount only when matching files exist in the workspace).