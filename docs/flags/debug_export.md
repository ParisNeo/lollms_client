# `debug_export`

```python
debug_export: bool = False
```

**Category**: Personas & Diagnostics

## What it does

Writes per-round prompt dumps, raw LLM stream logs, and an end-of-turn JSON
context dump.

## Recovering generated stuff (diagnostics)

Files land in `<workspace>/_debug_dumps/`:

- `full_prompt_round_<N>.log` — the exact messages sent to the LLM.
- `prompt_dump_round_<N>_shortened.md` — head/tail shortened view.
- `raw_llm_output_round_<N>.log` — the raw streamed tokens.
- `turn_dump_<timestamp>.json` — virtual history, message content, and metadata.

## Usage

```python
result = discussion.chat(task, debug_export=True)
# inspect <workspace>/_debug_dumps/
```

## Related flags

- `debug` — console-based counterpart.