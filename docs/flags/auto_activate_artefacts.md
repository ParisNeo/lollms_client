# `auto_activate_artefacts`

```python
auto_activate_artefacts: bool = True
```

**Category**: Artifacts

## What it does

Newly created or patched artifacts become **active** immediately — their content
is loaded into the model's context (Active Artifacts zone).

## What to expect

- `True` (default): after creation/patching, the model can read the artifact's
  content without an explicit `<unlock_file>`.
- `False`: artifacts are created with `TREE_UNLOCKABLE` visibility — the model
  must `<unlock_file>` them to read content. Useful for large files to protect
  the context budget.

## Usage

```python
discussion.chat(
    "Write chapter 1 of my novel",
    auto_activate_artefacts=False,
)
```

## Related flags

- `enable_artefacts` — master switch (must be `True`).