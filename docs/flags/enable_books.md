# `enable_books`

```python
enable_books: bool = False
```

**Category**: Artifacts

## What it does

Injects book/chapter-building instructions into the system prompt.

## What to expect

- Keyword-gated: instructions appear only when the user message mentions "book",
  "tome", "novel", "chapitre".
- Requires `enable_artefacts=True`. The model then writes chapters as regular
  document artifacts (recovered like any artifact).

## Recovering generated stuff

Chapters are document artifacts: `discussion.artefacts.get("<chapter title>")`
or the workspace files in the discussion directory.

## Usage

```python
discussion.chat("Write a book about dragons", enable_books=True)
```

## Related flags

- `enable_artefacts` — master switch.