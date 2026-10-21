# `enable_office_files: bool = False`

**Category**: Tools & Security

## What it does

Mounts the `office_files_builder` LCP toolset — read, create and edit real
Office files via sandboxed recipe tools:

- `tool_office_read` — inspect an existing `.pptx` / `.docx` / `.xlsx` and
  get a structured markdown inventory (slides, paragraphs, sheets, tables).
- `tool_office_build` — create a brand-new Office file from a sandboxed
  recipe (python-pptx / python-docx / openpyxl code).
- `tool_office_edit` — modify an existing Office file with a sandboxed
  recipe operating on the pre-loaded document object.

Supported formats: `.pptx`, `.pptm`, `.docx`, `.docm`, `.xlsx`, `.xlsm`.

## What to expect — NO vision gate

The mount is **unconditional** when `True` (text-only workflow: recipe code
in → markdown inventory out). Recipe code is sandboxed: **no `import`
statements are allowed** — the libraries are pre-injected; violations are
rejected before execution.

**Integrity-verified saves**: every save passes a 3-layer validation
(ZIP integrity, strict XML parse, library re-open). If a save would corrupt
the file, it is automatically **rolled back** to the pre-save snapshot and
the tool returns a machine-readable failure code — the model fixes the
recipe and retries. First invocation auto-installs `python-pptx` /
`python-docx` / `openpyxl` via pipmaster — expect a slower first call.

## Recovering generated stuff

- Tools write into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's
  `workspace_data_path`).
- Tool result keys: `workspace_file` (absolute path), `context_files` (list
  containing the path), `file_name` (sanitized).
- Files are auto-registered as artifacts by `_sync_tool_artifacts`
  (workspace snapshot diffing) — recover via `discussion.artefacts.get(...)`
  or open `workspace_file` directly on disk.
- The turn result's `"artefacts"` array also lists them.

## Usage

```python
result = discussion.chat(
    "Build a quarterly report: a .docx summary and an .xlsx KPI workbook",
    enable_office_files=True,
)
```

## Related flags

- [`enable_pptx_builder`](enable_pptx_builder.md) — the presentation-focused
  sibling: mounts the `pptx_builder` toolset plus keyword-gated presentation
  artifact instructions.
- [`enable_data_tools`](enable_data_tools.md) — contextual data/document
  tools (mount only when matching files exist in the workspace).