# `enable_pptx_builder: bool = False`

**Category**: Artifacts (Workspace Files)

> Replaces the former `enable_presentations` flag (removed).

## What it does

Injects presentation-building instructions (keyword-gated: "presentation",
"slide", "slideshow", "deck", "diaporama") into the system prompt AND mounts
the `pptx_builder` LCP toolset: `tool_pptx_build`, `tool_pptx_edit`,
`tool_pptx_read`.

## What to expect — NO vision gate

The mount is **unconditional** when `True` (text-only workflow: recipe code
in → markdown inventory out). The model writes sandboxed python-pptx
"recipe" code — no `import` statements allowed; violations are rejected
before execution. First invocation auto-installs `python-pptx` via pipmaster
— expect a slower first call.

## Recovering generated stuff (.pptx)

- The tool writes the deck into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's
  `workspace_data_path`).
- Tool result keys: `workspace_file` (absolute path), `context_files` (list
  containing the path), `file_name` (sanitized; default `presentation.pptx`).
- The file is auto-registered as a **document artifact** by
  `_sync_tool_artifacts` (workspace snapshot diffing) — recover via
  `discussion.artefacts.get("presentation.pptx")` or open `workspace_file`
  directly on disk.
- The turn result's `"artefacts"` array also lists it.

## Usage

```python
result = discussion.chat(
    "Create a 5-slide presentation about photosynthesis",
    enable_pptx_builder=True,
)
```

## Related flags

- [`enable_office_files`](enable_office_files.md) — the generalized sibling:
  mounts the `office_files_builder` toolset for reading, creating and editing
  `.pptx` / `.docx` / `.xlsx` (and `.xlsm` / `.docm` / `.pptm`) files with
  integrity-verified saves and automatic rollback.
- [`enable_artefacts`](enable_artefacts.md) — master switch for the artifact
  subsystem (the keyword-gated presentation instructions are a sub-feature).