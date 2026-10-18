# `enable_artefacts`

```python
enable_artefacts: bool = True
```

**Category**: Artifacts — **MASTER SWITCH**

## What it does

Enables the entire artifact subsystem: `<artifact>`/`<artefact>` creation and
SEARCH/REPLACE patching, scratchpad writes, context-visibility tags
(`<unlock_file>`, `<lock_file>`, `<hide_file>`), the Active Artifacts context
zone, artifact instructions in the system prompt, and image tag dispatch.

## What to expect

- `False`: artifact tags are ignored at dispatch; the model cannot create or
  modify workspace files through tags. All sub-flags (notes, skills, widgets,
  books, presentations instructions) are inert.
- `True`: every artifact write is **immediately materialized on disk**
  (versioned storage under `.versions/<discussion_id>/`).

## Recovering generated stuff

- **API**: `discussion.artefacts.get(title)`, `.list()`,
  `.get_version_history(title)`, `.diff_versions(title, v1, v2)`.
- **Disk**: every created/patched artifact is materialized under the discussion's
  workspace directory.
- **Turn result**: the returned dict's `"artefacts"` key lists affected artifacts
  (title, type, version); message metadata `artefacts_modified` lists titles.

## Usage

```python
result = discussion.chat("Create report.md with a summary", enable_artefacts=True)
report = discussion.artefacts.get("report.md")
```

## Related flags

- `auto_activate_artefacts`, `enable_notes`, `enable_skills`,
  `enable_inline_widgets`, `enable_forms`, `enable_books`,
  `enable_presentations`, `enable_image_generation`, `enable_image_editing` —
  all sub-flags force-disabled when this master is `False`.