# `forward_artefact_chunks`

```python
forward_artefact_chunks: bool = False
```

**Category**: Artifacts

## What it does

Forwards live artifact body chunks to the streaming callback as
`MSG_TYPE_ARTEFACT_CHUNK` events while the artifact streams.

## What to expect

- UI-verbosity flag. No generated content to recover — the artifact itself is
  recovered via the master-switch path.
- Requires `stream=True` to be meaningful.

## Usage

```python
discussion.chat("Write a long report", stream=True, forward_artefact_chunks=True)
```

## Related flags

- `stream` — chunk events only fire in streaming mode.
- `enable_artefacts` — master switch.