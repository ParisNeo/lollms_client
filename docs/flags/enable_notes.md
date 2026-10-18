# `enable_notes`

```python
enable_notes: bool = True
```

**Category**: Artifacts

## What it does

Enables the `<note title="...">...</note>` tag — quick markdown notes saved as
note-type artifacts (patchable via SEARCH/REPLACE).

## What to expect

- Requires `enable_artefacts=True` (sub-flag of the master switch).
- Notes created with `enable_notes=True` are patchable via the same
  SEARCH/REPLACE flow as documents.

## Recovering generated stuff

- **API**: `discussion.artefacts.get("<note title>")`.
- **Disk**: the note's workspace file under the discussion directory.

## Usage

```python
discussion.chat("Remember: deploy on Friday", enable_notes=True)
```

## Related flags

- `enable_artefacts` — master switch.