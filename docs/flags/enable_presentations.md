# `enable_presentations`

```python
enable_presentations: bool = False
```

**Category**: Artifacts

## What it does

Injects presentation-building instructions (keyword-gated: "presentation",
"slide", "slideshow", "deck", "diaporama") AND mounts the `pptx_builder` LCP
toolset: `tool_pptx_build`, `tool_pptx_edit`, `tool_pptx_read`.

## What to expect — NO vision gate

The mount is **unconditional** when `True` (text-only workflow: recipe code in →
markdown inventory out). The model writes sandboxed python-pptx "recipe" code
(no `import` statements allowed; violations are rejected before execution). First
invocation auto-installs `python-pptx` via pipmaster — expect a slower first
call.

## Recovering generated stuff (.pptx)

- The tool writes the deck into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's `workspace_data_path`).
- Result keys: `workspace_file` (absolute path), `context_files` (list
  containing the path), `file_name` (sanitized; default `presentation.pptx`).
- The file is auto-registered as a **document artifact** by `_sync_tool_artifacts`
  (workspace snapshot diffing) — recover via
  `discussion.artefacts.get("presentation.pptx")` or open `workspace_file`
  directly on disk.

## Usage

```python
result = discussion.chat(
    "Create a 5-slide presentation about photosynthesis",
    enable_presentations=True,
)
deck = discussion.artefacts.get("presentation.pptx")
```

## Related flags

- `enable_artefacts` — master switch (instructions layer; the pptx toolset mount
  itself is independent).