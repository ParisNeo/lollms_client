# `enable_in_message_status`

```python
enable_in_message_status: bool = False
```

**Category**: Artifacts

## What it does

Emits in-message status/progress lines inside `<processing>` blocks while the
model works.

## What to expect

- UI-verbosity flag. Keep `False` for clean transcripts; the processing blocks
  are scrubbed from LLM context either way (`scrub_processing_and_status_blocks`).

## Usage

```python
discussion.chat(task, enable_in_message_status=True, stream=True)
```

## Related flags

- `stream` — status lines stream live.
- `enable_artefacts` — master switch.