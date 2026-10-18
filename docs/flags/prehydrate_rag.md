# `prehydrate_rag`

```python
prehydrate_rag: bool = True
```

**Category**: Knowledge & Search

## What it does

Queries RAG before generation and injects retrieved context into the system
prompt, capped at **50,000 characters** to prevent context bloat. Source order:
personality data first (`personality.query_data`), else the client's RAG binding
(`client.query_rag`).

## What to expect

- Retrieved sources appear in the returned `"sources"` list with `type: "rag"`.
- RAG tools built by the personality are also registered in the tool registry.

## Usage

```python
result = discussion.chat("What does the handbook say about PTO?", prehydrate_rag=True)
for src in result.get("sources", []):
    print(src["type"], src.get("title"))
```

## Related flags

- `web_search` — the web-search analog.