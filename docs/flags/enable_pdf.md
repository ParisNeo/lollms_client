# `enable_pdf`

```python
enable_pdf: bool = False
```

**Category**: Tools & Security
**Parent flags**: none
**Gate**: LCP tools binding required (or auto-bootstrapped when the flag is set)
**Auto-admission**: also admitted when a `.pdf` file is present in the
workspace or the discussion's artifacts — see below.

## What it does

Mounts the `pdf_tools` LCP library and admits only its tool family. This
provides the agent with a PDF read/inspect/annotate surface: text extraction
from a PDF, page-level inspection, and annotation writes.

The exact admitted set follows the same per-family isolation contract used by
`enable_powerpoint` / `enable_word` / `enable_excel`: `enable_pdf=True` alone
never exposes PowerPoint, Word, or Excel tools.

## What to expect

- **Per-flag isolation**: this flag admits ONLY the PDF family.
- **Auto-admission**: this is the ONLY office-family flag that is auto-admitted
  when a matching file is detected. A `.pdf` file in the workspace (or as a
  discussion artefact) causes the toolset to auto-mount without `enable_pdf=True`.
  Passing `enable_pdf=True` forces the mount even with no `.pdf` file present.
- **Session-flag propagation**: the LCP gate is fail-closed and does not read
  `chat()` kwargs. The chat layer pre-authorizes the office-family session flags
  on the binding before the mount, so the auto-admission path always succeeds.
- **Silent no-op when no LCP binding**: warning logged; PDF doctrine is withheld
  (the model is not told the tools exist).
- **First-use install**: invokes `pypdf` via pipmaster on the first call —
  expect a slower first invocation.
- **Sandboxing**: annotation writes go through the standard office-save
  integrity-verification pipeline (validate → reparse → rollback on corruption).

## Usage

```python
# Explicit mount — even without any .pdf file present
result = discussion.chat(
    "Read this PDF and extract the section headings",
    enable_pdf=True,
)

# Auto-admission — no flag needed if a .pdf is in the workspace
result = discussion.chat(
    "Summarize report.pdf",
)
# The chat layer detects .pdf in the workspace → mounts pdf_tools automatically.
```

## Related flags

- `enable_powerpoint` / `enable_word` / `enable_excel` — sibling per-format
  OOXML toolsets (each is opt-in; none auto-admit).
- `enable_data_tools` — governs contextual document-tool admission for non-PDF
  files; independent of `enable_pdf`.
- `enable_office_files` (DEPRECATED) — legacy umbrella that enables all four
  per-format flags plus the monolithic `office_files_builder`.

## Recovering generated stuff

- Tools write into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file`, `file_name` (sanitized).
- Annotated files are auto-registered as artifacts by `_sync_tool_artifacts` —
  recover via `discussion.artefacts.get(...)` or open `workspace_file` directly
  on disk.
- The turn result's `"artefacts"` array also lists them.