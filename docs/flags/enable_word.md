# `enable_word: bool = False`

**Category**: Tools & Security

## What it does

Mounts the `word_tools` LCP toolset — build, edit and read Word documents
(`.docx` / `.docm`) via sandboxed python-docx recipe code. Three tools:

- `tool_build_word(file, recipe)` — create a brand-new document from a
  sandboxed recipe.
- `tool_edit_word(file, recipe)` — modify an existing document; the pre-edit
  bytes are restored if validation fails.
- `tool_read_word(file)` — validate, then return a markdown inventory
  (paragraphs, tables, structure).

## What to expect

- **Edit freedom + hints**: the injected doctrine is hints, not restrictions —
  the model is told to "use your own coding skills" with any python-docx
  technique, plus a high-leverage vocabulary:
  - `add_comment(doc, paragraph, text, author)` — real OOXML comment (falls
    back to yellow highlighting when the comment part is unavailable).
  - `run.font.highlight_color = WD_COLOR_INDEX.YELLOW` — native highlights.
  - Styles, fonts, alignment, spacing — free reformatting.
  - `doc.add_section(WD_SECTION.NEW_PAGE)` — sections and page breaks.
  - Links & fields via OXML: `qn('w:fldSimple')` with an instruction
    attribute; Word updates fields on open.
  - Tables & images via `media('name.png')` — workspace media resolution.
- **NO vision gate**: the mount is unconditional when the flag is set
  (text-only workflow: recipe code in → markdown inventory out).
- **Sandbox**: imports allowed ONLY from the `docx.*` whitelist plus safe
  stdlib modules; `qn` and all `WD_*` enums are pre-imported; the root `doc`
  is pre-bound. Forbidden tokens are rejected before execution. Recipes capped
  at 60,000 characters.
- **No `doc.save()`**: the harness saves and integrity-verifies automatically.
  Every save passes 3-layer validation (ZIP + required OOXML parts, strict XML
  parse, python-docx re-open). Corruption → automatic rollback + failure code
  `office_file_failed_validation`.
- **First call is slower**: pipmaster auto-installs `python-docx`.
- **Fail-closed LCP gate**: the `word_tools` library is office-family and
  never auto-discovers — it appears only when this flag (or the legacy kwarg)
  is set.

## Recovering generated stuff

- Tools write into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file`, `file_name` (sanitized).
- Files are auto-registered as artifacts by `_sync_tool_artifacts` (workspace
  snapshot diffing) — recover via `discussion.artefacts.get("report.docx")` or
  open `workspace_file` directly on disk.
- The turn result's `"artefacts"` array also lists them.

## Usage

```python
result = discussion.chat(
    "Build a quarterly report as a .docx with an executive summary and KPI table",
    enable_word=True,
)
# Recover: discussion.artefacts.get("quarterly_report.docx")
```

## Related flags

- [`enable_powerpoint`](enable_powerpoint.md) / [`enable_excel`](enable_excel.md)
  — the sibling per-software toolsets.
- [`enable_office_files`](enable_office_files.md) — deprecated legacy alias.