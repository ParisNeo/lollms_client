# `web_search` / `internet_search`

```python
web_search: bool = False
internet_search: bool = False
```

**Category**: Knowledge & Search

> These two parameters are **aliases** — enabling either has the same effect.
> The kwargs `enable_web_search` / `enable_internet_search` also trigger it.

## What it does

Pre-hydrates up to **5 web search results** into the system prompt before
generation.

## What to expect

- Extra network latency per turn; sources appear in `"sources"` with
  `type: "web"`.
- Failures are logged as warnings and do not abort the turn.

## Usage

```python
result = discussion.chat("Latest news on quantum computing?", web_search=True)
```

## Related flags

- `prehydrate_rag` — the document-corpus analog.