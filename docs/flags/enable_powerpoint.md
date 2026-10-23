# `enable_powerpoint: bool = False`

**Category**: Tools & Security

## What it does

Mounts the `powerpoint_tools` LCP toolset — build, edit, read and visually
inspect PowerPoint decks (`.pptx` / `.pptm`) via sandboxed python-pptx recipe
code. Five tools:

- `tool_build_powerpoint(file, recipe)` — create a brand-new deck from a
  sandboxed recipe.
- `tool_edit_powerpoint(file, recipe)` — modify an existing deck; the pre-edit
  bytes are restored if validation fails.
- `tool_read_powerpoint(file)` — validate, then return a markdown inventory
  (per slide: title + shape texts).
- `tool_pptx_info(file)` — **template/size awareness**: slide size (inches +
  EMU + aspect), slide count, per-layout placeholder table (idx + type), theme
  fonts (best-effort). Call this FIRST.
- `tool_view_slides(file, slides="1-4")` — **transient vision**: renders up to
  4 slides per call via LibreOffice (`soffice`) + `pypdfium2`; the images are
  attached to the CURRENT generation round only as vision content and are
  **never stored in the workspace**. Accepts ranges (`"1-4"`), lists
  (`"2,5,7"`) or a single slide (`"3"`). Falls back to a structural text
  inventory when the renderers are unavailable.

## What to expect

- **16:9 geometry**: every new deck is created at 16:9 — 13.333in × 7.5in
  (EMU 12192000 × 6858000). The root `prs` is pre-bound; position shapes
  within these bounds.
- **Template awareness first**: the injected doctrine directs the model to
  call `tool_pptx_info` BEFORE building, then add slides via
  `prs.slide_layouts[i]` using the discovered layout index.
- **NO vision gate on the mount**: mounting is unconditional when the flag is
  set (recipe code in → markdown inventory out). `tool_view_slides` renders
  best-effort and degrades to text when LibreOffice/pypdfium2 are missing.
- **Sandbox**: imports allowed ONLY from the `pptx.*` whitelist plus safe
  stdlib modules; core design symbols (`Presentation`, `Inches`, `Pt`, `Emu`,
  `RGBColor`, `MSO_SHAPE`, `PP_ALIGN`, `qn`, ...) are pre-imported. Forbidden
  tokens are rejected before execution. Recipes capped at 60,000 characters.
- **No `prs.save()`**: the harness saves and integrity-verifies automatically.
  Every save passes 3-layer validation (ZIP + required OOXML parts, strict XML
  parse, python-pptx re-open requiring ≥1 slide). Corruption → automatic
  rollback + failure code `office_file_failed_validation`.
- **First call is slower**: pipmaster auto-installs `python-pptx` (and
  best-effort `pypdfium2`).
- **Fail-closed LCP gate**: the `powerpoint_tools` library is office-family
  and never auto-discovers — it appears only when this flag (or the legacy
  kwarg) is set.

## Recovering generated stuff

- Tools write into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file`, `file_name` (sanitized; default
  `presentation.pptx`).
- Files are auto-registered as artifacts by `_sync_tool_artifacts` (workspace
  snapshot diffing) — recover via `discussion.artefacts.get("presentation.pptx")`
  or open `workspace_file` directly on disk.
- The turn result's `"artefacts"` array also lists them.
- `tool_view_slides` / `tool_pptx_info` produce **no workspace writes** — no
  artifacts, by design (transient vision only).

## Usage

```python
result = discussion.chat(
    "Create a 5-slide presentation (.pptx) about photosynthesis",
    enable_powerpoint=True,
)
# Model flow: tool_pptx_info → tool_build_powerpoint → tool_view_slides
# Recover: discussion.artefacts.get("presentation.pptx")
```

## Related flags

- [`enable_word`](enable_word.md) / [`enable_excel`](enable_excel.md) — the
  sibling per-software toolsets.
- [`enable_office_files`](enable_office_files.md) — deprecated legacy alias.