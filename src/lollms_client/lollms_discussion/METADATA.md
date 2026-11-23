
# Metadata in Lollms Discussions

This document describes how **metadata** works at both the **discussion** level and the
**message** level: the two storage columns, the public property/setter API, the helper
methods, the reserved keys, and how metadata round-trips through export/import.

Metadata is the extension point for anything that does not deserve its own typed column
(generation telemetry, image groupings, token caches, workspace pointers, replayable
virtual history, arbitrary caller annotations). It is stored as a **JSON column** in the
database and surfaced as a plain Python `dict`.

---

## 1. Storage Model

Metadata is persisted in the SQLAlchemy ORM through two nullable `JSON` columns defined in
`_db.py`:

| ORM model        | Column                 | Default    |
|------------------|------------------------|------------|
| `DiscussionBase` | `discussion_metadata`  | `{}`       |
| `MessageBase`    | `message_metadata`     | `{}`       |

Both columns are `nullable=True, default=dict`, so an unset value reads back as `{}`.

You almost never touch these columns directly — you use the `metadata` property on the
`LollmsDiscussion` / `LollmsMessage` wrappers, which coerce the ORM value to a `dict` and
mark the row dirty via `touch()`.

---

## 2. Discussion-Level Metadata

### 2.1 Public API

`LollmsDiscussion.metadata` reads/writes the underlying `discussion_metadata` column:

```python
discussion.metadata            # -> dict (never None)
discussion.metadata = {...}    # replaces the whole dict, calls touch()
discussion.set_metadata_item("title", "My chat")   # convenience single-key setter
```

`set_metadata_item` (in `_mixin_utils.py`) copies the current dict, sets one key, writes
it back and commits:

```python
discussion.set_metadata_item("workspace_path", "/path/to/agentic/workspace")
```

### 2.2 Reserved Discussion Metadata Keys

These keys are consumed by the framework itself. Do not repurpose them.

| Key                          | Written by                                   | Purpose |
|------------------------------|----------------------------------------------|---------|
| `workspace_path`             | `CoreMixin.set_workspace_path()` / `__init__`| Custom agentic workspace root; presence of a custom path enables **agentic mode** (`disable_artefact_versioning=True`). Read by `_mixin_core.py`, `_mixin_utils.py`, and `clone_without_messages`. |
| `title`                      | `UtilsMixin.auto_title()`                    | Auto-generated discussion title. |
| `_token_cache`               | `UtilsMixin.get_context_status()`            | Per-zone token-count cache: `{zone_key: {"hash": md5(text), "tokens": n}}`. Reused until the hashed text changes. Dropped by `clone_without_messages()`. |
| `last_synopsis_fingerprint`  | `UtilsMixin.summarize_and_prune()`           | SHA-256 fingerprint of the branch; prevents regenerating an identical pruning synopsis. |

### 2.3 Helper Interactions

- **`clone_without_messages()`** copies the discussion configuration into a fresh
  discussion, copying `discussion_metadata` **minus** `_token_cache` (stale by definition
  for a new empty discussion).
- **`export_to_json_str()`** embeds the full `discussion_metadata` dict under the
  `"discussion_metadata"` key of the JSON payload.
- **`import_from_json_str()`** accepts that same payload and passes it straight into
  `create_new(..., discussion_metadata=...)`, so a JSON round-trip preserves all metadata.
- **`get_context_status()`** uses `_token_cache` to avoid re-tokenizing unchanged zones
  on every status call; when it writes back a refreshed cache it sets
  `self.metadata = meta` then `touch()` + `commit()`.

---

## 3. Message-Level Metadata

### 3.1 Public API

`LollmsMessage.metadata` reads/writes the underlying `message_metadata` column (in
`_message.py`):

```python
msg.metadata                        # -> dict (never None)
msg.metadata = {...}                # replaces the dict, calls discussion.touch()
msg.set_metadata_item("k", v, discussion)   # single-key setter + commit
msg.generation_info                 # derived view: msg.metadata.get("generation", {})
```

The property implementation (source of truth):

```python
@property
def metadata(self) -> dict:
    return getattr(self._db_message, 'message_metadata', {}) or {}

@metadata.setter
def metadata(self, value: Dict[str, Any]) -> None:
    setattr(self._db_message, 'message_metadata', value)
    self._discussion.touch()
```

### 3.2 `set_metadata_item`

Convenience setter that creates a modified copy, assigns it, and commits the discussion:

```python
msg.set_metadata_item("custom_flag", True, discussion)
```

Useful when you want a single key change persisted immediately without manually copying
the dict.

### 3.3 Generation Telemetry: `generation_info`

The LLM generation loop writes a `"generation"` sub-dict into `message_metadata`. It is
surfaced through the **read-only** `generation_info` property:

```python
msg.metadata["generation"] == msg.generation_info
```

Typical keys (see `_generation_telemetry.py`, which merges the profile into
`ai_msg.metadata`):

| Key                       | Meaning |
|---------------------------|---------|
| `personality_name`        | Personality used for the turn. |
| `model_name`              | Model that answered. |
| `profile_name`            | Universal model profile name. |
| `binding_name`            | Binding backend (e.g. `openai`, `llama_cpp_server`). |
| `reasoning_effort`        | Requested reasoning effort (if supported). |
| `time_to_first_token_s`   | Wall time until first streamed token. |
| `generation_time_s`       | Total generation wall time. |
| `generated_tokens`        | Token count of the produced content. |
| `mean_tokens_per_second`  | Throughput average. |

`generation_info` is empty (`{}`) for user messages or messages created outside the
generation loop.

### 3.4 Image Groupings

Message metadata carries the **authoritative image grouping state**. Two keys are used
interchangeably by `_sync_active_images_flags()`:

- `image_groups` — image packs added via `add_image_pack()` (usually `type="generated"`)
  and ad-hoc single-image groups created by `toggle_image_pack_activation()`.
- `image_generation_groups` — alternate key recognized when present (legacy/other
  producers).

Each group is a dict:

```json
{
  "id": "<uuid>",
  "type": "generated | upload",
  "indices": [0, 1, 2],
  "created_at": "<iso8601>",
  "main_image_index": 0,
  "is_active": true,
  "title": "optional",
  "prompt": "optional"
}
```

The relationship between metadata and the flat `active_images` list:

- `msg.images` is the **flat list** of all base64 images on the message.
- `msg.active_images` is a **parallel boolean list** (`len == len(images)`) telling the
  LLM export which pixels are included.
- `_sync_active_images_flags()` rebuilds `active_images` from `image_groups` +
  `image_generation_groups`:
  - For each **active** group, exactly the `main_image_index` is marked `True`
    (falling back to the first index of the group if the main index is missing/out of
    range).
  - Any image **not covered by a group** defaults to `True`.
  - Images in an **inactive** group are all `False`.

Mutators that keep the two in sync automatically:

| Method                             | Effect |
|------------------------------------|--------|
| `add_image(img)`                   | Append one image, re-sync active flags. |
| `add_images(imgs)`                 | Append many images, re-sync. |
| `add_image_pack(imgs, ...)`        | Append a pack and register a new group entry. |
| `toggle_image_activation(i, a)`    | Toggle one image flag directly (no group rewrite). |
| `toggle_image_pack_activation(i,a)`| Create or mutate the group containing image `i`, then re-sync. |

All of these call `_sync_active_images_flags()` and `discussion.commit()` when the
discussion is DB-backed.

### 3.5 Persisted Agentic Replay: `virtual_history`

When an agentic turn ends, the active `virtual_history` (tool rounds, corrections, etc.)
is serialized into the assistant message's metadata under the `"virtual_history"` key.
On export (`HistoryManager.export` in `lollms_history.py`), any assistant message whose
metadata contains a non-empty `virtual_history` is **replayed** inline:

- The persisted entries are emitted as user/assistant turns in place of the plain
  content.
- This allows a later turn to re-hydrate the exact agentic tool trajectory without a
  live `virtual_history` argument.
- If an assistant message is empty **and** has no persisted `virtual_history`, it is
  dropped from the export (a "building" message that never completed).

---

## 4. Reading and Writing Metadata Safely

Because metadata is stored as `JSON`, only **JSON-serializable** values survive the
round-trip: strings, numbers, booleans, `None`, lists, and dicts. Never store Python
objects (ORM rows, datetimes without ISO formatting, file handles, bytes).

Recommended patterns:

```python
# Read a sub-dict without mutating it
info = dict(msg.metadata.get("generation") or {})

# Copy-modify-write (atomic from the ORM's perspective)
meta = dict(msg.metadata or {})
meta["my_key"] = value
msg.metadata = meta

# Single-key helper (message-level)
msg.set_metadata_item("my_key", value, discussion)

# Single-key helper (discussion-level, commits for you)
discussion.set_metadata_item("my_key", value)
```

Do **not** mutate the dict returned by `msg.metadata` / `discussion.metadata` in place
and expect persistence — the ORM won't notice a nested mutation. Always reassign the
whole dict (or use the helpers, which reassign internally).

`_MessageWrapper` in `lollms_history.py` defensively coerces any non-dict `metadata`
value to `{}` before use, so a malformed metadata payload can never abort a history
export.

---

## 5. Export / Import Round-Trip

`discussion.export_to_json_str()` emits message metadata under `"message_metadata"` and
discussion metadata under `"discussion_metadata"`. `LollmsDiscussion.import_from_json_str`
reconstructs both. Practical consequence:

- Any metadata you set through the public API survives a JSON backup/restore cycle.
- Derived caches (`_token_cache`) survive too but are harmless — they simply invalidate
  when the hashed text changes.
- Image groupings survive, but the exported `"active_images"` top-level key is popped on
  import and recomputed from the persisted groups, so flags stay consistent with
  `image_groups`.

---

## 6. Quick Reference

| Level      | Public accessor        | ORM column             | Convenience setter |
|------------|------------------------|------------------------|--------------------|
| Discussion | `discussion.metadata`  | `discussion_metadata`  | `set_metadata_item(name, value)` |
| Message    | `msg.metadata`         | `message_metadata`     | `set_metadata_item(name, value, discussion)` |

| Reserved message key     | Owner                              |
|--------------------------|------------------------------------|
| `generation`             | `_generation_telemetry.py` (LLM loop) |
| `generation_info`        | read-only derived property         |
| `image_groups`           | image pack / activation helpers    |
| `image_generation_groups`| alt grouping key (recognized)      |
| `virtual_history`        | agentic turn persistence, replayed on export |

| Reserved discussion key       | Owner                            |
|-------------------------------|----------------------------------|
| `workspace_path`              | agentic workspace mode           |
| `title`                       | `auto_title()`                   |
| `_token_cache`                | `get_context_status()`           |
| `last_synopsis_fingerprint`   | `summarize_and_prune()`          |
