# 06 — Chat-Mixin Integration (`lollms_discussion`)

> **Purpose**: How LCP tools live inside `lollms_discussion`'s chat flow — the
> two files `_mixin_chat.py` (execution, dispatch, mounts, recovery) and
> `_mixin_prompt.py` (LLM-visible doctrine). This is the document that answers
> "what actually happens when the LLM calls a tool during a chat?".
> Prerequisites: [01_LCP_BINDING_CORE.md](./01_LCP_BINDING_CORE.md),
> [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md).

---

## 1. The two halves of the integration

```
┌────────────────────────────────────────────────────────────────┐
│                     lollms_discussion                          │
│                                                                │
│  _mixin_prompt.py                    _mixin_chat.py            │
│  ───────────────────────────        ─────────────────────────  │
│  ADVERTISEMENT (before the round)   EXECUTION (during/after)   │
│  _build_tool_prompt                 tag/dispatch machinery     │
│  _build_word_instructions           execute_tool routing       │
│  _build_excel_instructions          office-mount seam          │
│  _build_powerpoint_instructions     media-tag handlers         │
│  _build_image_generation_           mimicry/empty recovery     │
│    instructions                     staged context assembly    │
└────────────────────────────────────────────────────────────────┘
```

- **Advertisement** (outbound): discovered tool specs become system-prompt
  doctrine so the LLM knows what exists and the exact calling format.
- **Execution** (inbound): the LLM's streamed reply is intercepted; tool calls
  and media tags are dispatched; results and artefacts are attached for the
  next round.

---

## 2. The advertisement path (`_mixin_prompt.py`)

### 2.1 `_build_tool_prompt`

The central builder that turns the mounted/discovered toolset into LLM-visible
doctrine. It emits:

- the list of available tools with their names, descriptions, and parameter
  schemas (from the spec shapes documented in
  [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md) §3),
- the **tool-calling format** the LLM must use (the tag-based call syntax the
  dispatch machinery in §3 is built to parse),
- per-capability doctrine blocks (below).

### 2.2 Per-capability instruction builders

Each office capability has a dedicated doctrine builder:

| Builder | Doctrine for | Gated on |
|---|---|---|
| `_build_word_instructions` | Word document tools (`word_tools`) | `enable_word` |
| `_build_excel_instructions` | Excel tools (`excel_tools`) | `enable_excel` |
| `_build_powerpoint_instructions` | PowerPoint tools (`powerpoint_tools`) | `enable_powerpoint` |
| `_build_image_generation_instructions` | Media-tag generation (image/music/speech/video) | TTI registry probes |

The media-tag doctrine is **capability-probed, not flag-gated alone**: the
builder emits image-generation instructions only when the TTI (text-to-image)
binding registry actually has a usable binding — the LLM is never taught a
tag it cannot execute. The same probe pattern applies to the other media
modalities (TTM/TTS/TTV registries for music/speech/video).

### 2.3 What the LLM is taught (the calling format)

The LLM is instructed to emit tool calls as **tags inside its streamed text**
(e.g. a `<tool_call>`-style construct carrying the tool name and JSON
params). This is why the execution path is a *stream interceptor*: the call
arrives embedded in the model's reply, and the mixin must recognize, extract,
and dispatch it mid-stream. (Exact tag spellings are defined in
`_mixin_prompt.py`; consult it for the literal syntax taught to the model.)

---

## 3. The execution path (`_mixin_chat.py`)

### 3.1 Tag interception and dispatch

```
LLM stream
   │
   ▼
mixin's tag/dispatch machinery intercepts the stream
   │
   ├── tool-call tag ──► routed to LCPBinding.execute_tool(tool_name, params, discussion_instance)
   │                        │
   │                        ▼
   │                     result dict {success, error, output, file_name, context_files, ...}
   │                        │
   │                        ▼
   │                     normalization (see 05_TOOL_EXECUTION_SEAM.md §3)
   │                        │
   │                        ▼
   │                     artefact attachment (USER-role only)
   │
   └── media tag ──────► _dispatch_closed_tag → _handle_media_tag (see §3.3)
```

Tool calls are dispatched through the binding's `execute_tool` — the single
execution seam documented in [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md).
The result dict's `context_files` are resolved to absolute existing paths and
feed the artefact/attachment pipeline: workspace artefacts produced by the
tool become next-round context, attached as **USER-role messages only**.

### 3.2 The office-mount seam

The chat mixin can mount office toolsets on demand. The seam:

- **`_office_mount_requests`** — the queue of pending mount requests (e.g.
  raised when the discussion needs Word/Excel/PowerPoint capability).
- **Admission sets** — the exact tool-name sets that define each family:
  - `_OFFICE_PPTX_TOOL_NAMES` — 5 PowerPoint tools,
  - `_OFFICE_WORD_TOOL_NAMES` — 3 Word tools,
  - `_OFFICE_XLSX_TOOL_NAMES` — 4 Excel tools (including
    `tool_excel_from_markdown`).
- **`_legacy_office` shim** — compatibility layer for older call paths.
- **`set_session_flags`** — writes the admitting flags (e.g. `enable_word`)
  into the session-flag chain (see
  [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md) §3), which the
  binding's admission gates then read.

Flow: a mount request → admission-set check → `set_session_flags` → the
binding admits the library → the toolset is mounted → the log family
`[ChatMixin] Mounted 'word_tools' toolset (4 tools)` appears.

### 3.3 Media-tag handlers

Media generation is tag-driven, not tool-call-driven:

- `_dispatch_closed_tag` — the general closed-tag router; media tags land in
  `_handle_media_tag`.
- `_MEDIA_TAG_CONFIG` — the registry mapping each media tag to its modality
  and generation parameters: generate/edit for **image**, **music**, **speech**,
  **video**.
- `_StreamState` — the per-round state object that accumulates generated
  bytes/pixels so they can be attached after the stream closes:
  `generated_image_pixels`, `generated_music_bytes`,
  `generated_speech_bytes`, `generated_video_bytes`.

**The USER-role-only attachment invariant** (hard-won law): generated media
is attached to the conversation as USER-role messages, never system-role.
The historical incident: attaching media to system-role messages produced the
provider 400 error `"System message cannot contain images"`. The fix
established both the invariant and a **400-signature recovery**: when that
error signature is detected, the offending media is sanitized out of the
system message and rerouted to a USER-role message, then the round is
retried.

### 3.4 Empty-response recovery and mimicry

When a round produces an empty response (0 tokens) or an artefact-only round
with no text, the mixin engages recovery machinery:

- `_previous_round_was_artifact_only_or_empty` — the flag marking that the
  previous round ended without usable text.
- `_inject_empty_round_mimicry_correction` — injects a corrective instruction
  so the model produces text on the next attempt.
- `_mimicry_attempt_counts[0] == 2` — the test-locked marker for the mimicry
  attempt counter (the count at which the correction machinery is verified to
  fire; locked by `test/test_mimicry_interception.py`).

This machinery exists because tool-heavy rounds can end with the model
emitting only a tool call and no prose — the recovery loop teaches it back to
producing text.

### 3.5 Staged context assembly

The context sent to the LLM is assembled in **fixed stages** by
`_assemble_context_stages`:

```
1. base prompt
2. personality
3. discussion data
4. sensations
5. RAG
6. user message
```

- **`TAGS` registry** — the tag vocabulary for stage control, including
  `[SNSC]` (sensations) and `[DISCUSSION_DATA]` (discussion data).
- **Per-stage `include_*` flags + `*_max_tokens` budgets** — each stage can be
  included/excluded and budgeted independently.
- **Drop-never-truncate semantics**: when a stage exceeds its budget, the
  stage is **dropped entirely**, never truncated mid-content. A half-cut
  document is worse than an absent one.
- **`parse_response_tags` + `_route_parsed_tags`** — parse the model's
  response tags and route them. Condition semantics: `true` / `false` /
  `test`. Tags requesting stages that are **disabled** are ignored (the model
  cannot re-enable a stage the host disabled).
- **`_count_msg_tokens_cached`** — cached token counting for messages, used
  for budget enforcement without recounting.

---

## 4. Sequence diagrams

### (a) A full chat round with one office tool call

```
 user message ──► _assemble_context_stages (base→personality→discussion_data
                  →sensations→RAG→user; budgets; drop-never-truncate)
                        │
                        ▼
                  _build_tool_prompt + per-capability builders
                  (word/excel/powerpoint/media doctrine)
                        │
                        ▼
                     LLM stream
                        │  tool-call tag intercepted mid-stream
                        ▼
                  dispatch → LCPBinding.execute_tool(tool_name, params,
                                                     discussion_instance)
                        │
                        ▼
                  library callable runs (workspace CWD, restored after)
                        │
                        ▼
                  result dict → normalization → context_files resolved
                        │
                        ▼
                  artefact attachment (USER-role only)
                        │
                        ▼
                  next round (or empty-response recovery if no text)
```

### (b) A media-tag generation round

```
 LLM stream emits media tag (e.g. image generation)
        │
        ▼
 _dispatch_closed_tag ──► _handle_media_tag
        │                   (_MEDIA_TAG_CONFIG: modality + params)
        ▼
 TTI/TTM/TTS/TTV binding generates
        │
        ▼
 _StreamState.generated_image_pixels / generated_music_bytes /
 generated_speech_bytes / generated_video_bytes accumulate
        │
        ▼
 attachment as USER-role message (NEVER system-role)
   └─ on provider 400 "System message cannot contain images":
      sanitize + reroute to USER-role + retry (400-signature recovery)
```

### (c) Staged assembly + response-tag parsing round

```
 host config: include_* flags + *_max_tokens budgets per stage
        │
        ▼
 _assemble_context_stages
   fixed order: base → personality → discussion data → sensations
                → RAG → user message
   budget law: drop-never-truncate
        │
        ▼
 LLM reply contains response tags
        │
        ▼
 parse_response_tags ──► _route_parsed_tags
   conditions: true / false / test
   disabled-stage tags from the model are IGNORED
        │
        ▼
 next round's stage composition updated
```

---

## 5. Key invariants

- The chat integration has two halves: **advertisement** (`_mixin_prompt.py`,
  before the round) and **execution** (`_mixin_chat.py`, during/after).
- `_build_tool_prompt` is the central doctrine builder; per-capability builders
  (`_build_word_instructions`, `_build_excel_instructions`,
  `_build_powerpoint_instructions`) add family-specific doctrine.
- `_build_image_generation_instructions` is gated on **TTI registry probes** —
  the LLM is never taught a media tag it cannot execute.
- Tool calls arrive as **tags inside the streamed reply**; the mixin
  intercepts and dispatches them to `LCPBinding.execute_tool`.
- The office-mount seam: `_office_mount_requests` → admission sets
  (`_OFFICE_PPTX_TOOL_NAMES` 5 / `_OFFICE_WORD_TOOL_NAMES` 3 /
  `_OFFICE_XLSX_TOOL_NAMES` 4 incl. `tool_excel_from_markdown`) →
  `set_session_flags` → binding admission → `Mounted 'X' toolset (N tools)`.
- Media tags route through `_dispatch_closed_tag` → `_handle_media_tag` with
  `_MEDIA_TAG_CONFIG`; generated bytes/pixels accumulate in `_StreamState`.
- **USER-role-only attachment**: generated media and artefacts attach as
  USER-role messages; the 400-signature recovery sanitizes and reroutes when
  a provider rejects system-role media.
- Empty/artefact-only rounds trigger mimicry recovery:
  `_previous_round_was_artifact_only_or_empty` →
  `_inject_empty_round_mimicry_correction`; the
  `_mimicry_attempt_counts[0] == 2` marker is test-locked.
- Staged assembly is fixed-order (base → personality → discussion data →
  sensations → RAG → user) with per-stage `include_*` flags and
  `*_max_tokens` budgets, and **drop-never-truncate** budget semantics.
- Response tags are parsed by `parse_response_tags` + `_route_parsed_tags`
  (conditions true/false/test); **disabled-stage tags from the model are
  ignored** — the model cannot override host configuration.
- `_count_msg_tokens_cached` provides budget enforcement without recounting.