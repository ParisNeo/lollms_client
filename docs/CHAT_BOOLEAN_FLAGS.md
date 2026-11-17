# `LollmsDiscussion.chat()` — Boolean Flags Reference

This guide documents every **boolean parameter** of `LollmsDiscussion.chat()`
(`src/lollms_client/lollms_discussion/_mixin_chat.py`). For each flag you get:

- **What it does** — the precise subsystem it enables or disables.
- **What to expect** — runtime behavior, gates (e.g. vision checks), latency,
  first-use installs, silent skips, and how failures surface.
- **Recovering generated stuff** — when the flag causes the model to *produce*
  files, artifacts, images, or reports, this tells you exactly where they land
  and how to retrieve them (API keys, disk paths, result-dict entries).

All flags are passed as keyword arguments. All default to a safe value, so a bare
`discussion.chat("Hello")` works out of the box.

---

## Mental Model: Three Layers

Before the flag table, understand the three layers a flag can act on:

1. **Instructions** — text injected into the system prompt teaching the model a
   tag grammar (`<artifact>`, `<note>`, `<generate_image>`, ...). Without these,
   the model doesn't know the feature exists.
2. **Dispatch** — the streaming parser intercepting those tags and executing them
   (creating artifacts, calling tools). A flag that adds instructions but not
   dispatch means the model can write the tag but nothing happens.
3. **Context hydration** — content loaded *into* the model's context (active
   artifacts, memories, RAG excerpts, web results, image pixels). 

Most enable-flags act on all three layers coherently. A few are
**capability-gated**: the instructions are only injected when the client has the
required binding (vision model, TTI engine, ...), otherwise the feature is
silently skipped (with a console warning) and the model truthfully reports it
has no such capability.

### Tool runtime context (`env`)

Tool and code-shell execution always receives a single `env` object providing
`llm`, `tti`, their `*_model_profiles`, and `*_switch` functions — independent
of which chat system is active (see
[`lcp/00_OVERVIEW.md`](lcp/00_OVERVIEW.md) §6–§7). Flags like
`enable_image_generation` gate *whether TTI instructions are injected*, while
`env.tti is None` expresses *whether a TTI backend is configured at all*.
Both must be consulted.

---

## Message Schema Cross-Reference

The flags below operate on messages — but this document does not define the
`LollmsMessage` entity itself. For its structure (`id`, `parent_id`, `sender`,
`sender_type`, `content`, `images`, `active_images`, `metadata`, `created_at`),
see **`BRANCHING.md` §3**.

Note: `active_images` is a **message-level field** (a per-image list of
activation flags living on `LollmsMessage`), not a `chat()` parameter. It is
controlled at the message level, not per turn, and is therefore not listed in
the table below.

---

## Quick Reference Table

| Flag | Default | Category | Gate / Note |
|---|---|---|---|
| `stream` | `False` | Core loop | streaming callback mode |
| `list_available_models` | `False` | Core loop | model picker tool |
| `enforce_end_tag` | `False` | Core loop | strict `<done/>` protocol |
| `dynamic_effort` | `False` | Core loop | `<effort>` interception |
| `clean_reasoning_at_save` | `True` | Core loop | scrub `midt
` before saving |
| `enable_artefacts` | `True` | Artifacts | **master switch** |
| `auto_activate_artefacts` | `True` | Artifacts | active-context loading |
| `enable_notes` | `True` | Artifacts | `<note>` tag |
| `enable_skills` | `True` | Artifacts | `<skill>` tag |
| `enable_inline_widgets` | `False` | Artifacts | `<lollms_inline>` widgets |
| `enable_forms` | `False` | Artifacts | `<lollms_form>` interactivity |
| `enable_books` | `False` | Artifacts | keyword-gated instructions |
| `enable_image_generation` | `True` | Artifacts | TTI capability-gated |
| `enable_image_editing` | `True` | Artifacts | TTI capability-gated |
| `forward_artefact_chunks` | `False` | Artifacts | UI streaming events |
| `enable_in_message_status` | `False` | Artifacts | UI progress lines |
| `enable_memory` | `False` | Memory | **master switch** |
| `enable_episodic_memory` | `False` | Memory | skips trivial exchanges |
| `enable_auto_dream` | `False` | Memory | post-turn consolidation |
| `enable_deep_memory_pulling` | `False` | Memory | pre-generation retrieval |
| `prehydrate_rag` | `True` | Knowledge | 50,000-char cap |
| `web_search` / `internet_search` | `False` | Knowledge | max 5 hits prehydrated |
| `enable_data_tools` | `True` | Tools | contextual workspace scan |
| `compact_workspace_tree` | `False` | Tools | inline tree → on-demand `tool_show_workspace_tree` |
| `enable_code_execution` | `False` | Tools | security-sensitive |
| `auto_approve_python` | `False` | Tools | security-sensitive |
| `allow_dynamic_tools` | `False` | Tools | security-sensitive |
| `enable_vlm_query` | `False` | Tools | vision fallback only |
| `enable_computer_use` | `False` | Tools | **vision-gated** |
| `enable_powerpoint` | `False` | Tools | mounts PowerPoint tools (build/edit/read + info + view_slides) |
| `enable_word` | `False` | Tools | mounts Word tools (build/edit/read) |
| `enable_excel` | `False` | Tools | mounts Excel tools (build/edit/read, formulas doctrine) |
| `enable_pdf` | `False` | Tools | mounts PDF tools; **auto-admitted on `.pdf` presence** |
| `enable_toolset_editing` | `False` | Tools | mounts `toolset_staging`; needs LCP binding |
| `enable_skill_editing` | `False` | Tools | mounts `skill_editor`; needs LCP binding |
| `enable_generate_video` | `False` | Media | **TTV**-gated `<generate_video>` |
| `enable_edit_video` | `False` | Media | **TTV**-gated `<edit_video>` |
| `enable_generate_music` | `False` | Media | **TTM**-gated `<generate_music>` |
| `enable_edit_music` | `False` | Media | **TTM**-gated `<edit_music>` |
| `enable_synthesize_speech` | `False` | Media | **TTS**-gated `<generate_speech>` |
| `suppress_images` | `False` | Tools | negative polarity |
| `orchestrator_mode` | `False` | Personas | AgenticRunner route |
| `orchestrator_persona` | `False` | Personas | delegation-only persona |
| `debug` | `False` | Diagnostics | tool registry dump |
| `debug_export` | `False` | Diagnostics | prompt/stream dumps to disk |
| `verbose` | `False` | Diagnostics | per-round telemetry (independent of `debug`) |

> Non-boolean parameters (`max_nb_rounds`, `tolerance_level`, `shell_autonomy_level`,
> `python_autonomy_level`, `context_compaction_threshold`, `event_mode`, `think`,
> `reasoning_effort`, `reasoning_summary`, `confirm_handler`, `allow_computer_use`
> — an `Optional[bool]` capability override — `toolset_editing_paths`,
> `skill_editing_paths`, `builtin_tools_activation_list`, `fast_artefact_replicas`)
> are out of scope here. Kwarg-only RSI flags `enable_toolsmith` / `enable_skillmaker`
> are documented in their own section below.

---

## Core Loop

### `stream: bool = False`

**What it does**: Enables streaming mode — LLM tokens are forwarded to a callback
as they arrive instead of being returned only at the end.

**What to expect**:
- With `True`, pass a `callback` (or use `event_mode`); you receive
  `MSG_TYPE.MSG_TYPE_FULL_INSPECTOR`/chunk events as generation progresses, plus
  tool/artifact/status events emitted by the agentic loop.
- With `False` (default), the method blocks until the final result dict is ready.
- Cancellation (`discussion.cancel_generation()`) is honored in both modes at
  token and round boundaries; a cancelled turn persists partial content and is
  resumable via `discussion.resume_turn()`.

**Recovering generated stuff**: nothing new — the standard result dict is still
returned; the callback is purely a live view.

### `list_available_models: bool = False`

**What it does**: Registers a `list_available_models` tool so the model can
enumerate the client's loaded models/child bindings on demand.

**What to expect**: with `False`, the model has no way to discover other models
and will not propose switching. With `True`, the tool appears in the per-turn tool
registry.

### `clean_reasoning_at_save: bool = True`

**What it does**: Strips `midthink` / `<|thinking|>` / `<reasoning>` / `<output>`
reasoning blocks from the final saved assistant message.

**What to expect**:
- `True`: the persisted `ai_message.content` contains only the conversational
  answer; reasoning traces are dropped at save time (they remain visible in
  `ai_message.thoughts` during streaming).
- `False`: raw reasoning markup stays in the saved content.

### `dynamic_effort: bool = False`

**What it does**: Enables the `<effort level="..."/>` interception so the model
can adapt its own reasoning effort between agentic rounds.

**What to expect**:
- Round 1 uses the explicitly passed `reasoning_effort` (or `"none"` if none
  given).
- When the model emits `<effort level="high"/>`, subsequent rounds run with that
  effort; level `"none"`/`"off"` disables reasoning for later rounds.
- Requires a reasoning-capable binding to have any visible effect.

### `enforce_end_tag: bool = False`

**What it does**: Controls whether a first-round, action-free, `<done/>`-less
response may finish the turn.

**What to expect**:
- `False` (default): a pure conversational round 1 (no tools, no artifacts, no
  prior actions) ends the turn normally — ideal for chat.
- `True`: the loop injects a termination protocol and keeps generating until the
  model explicitly emits `<done/>` — intended for strict agentic sessions where
  prose-only answers are rejected.

---

## Artifacts (Workspace Files)

### `enable_artefacts: bool = True` — MASTER SWITCH

**What it does**: Enables the entire artifact subsystem: `<artifact>`/`<artefact>`
creation and SEARCH/REPLACE patching, scratchpad writes, context-visibility tags
(`<unlock_file>`, `<lock_file>`, `<hide_file>`), the Active Artifacts context
zone, artifact instructions in the system prompt, and image tag dispatch.

**What to expect**:
- `False`: artifact tags are ignored at dispatch; the model cannot create or
  modify workspace files through tags. All sub-flags (notes, skills, widgets,
  books, presentations instructions) are inert.
- `True`: every artifact write is immediately materialized on disk
  (versioned storage under `.versions/<discussion_id>/` in the discussion
  workspace).

**Recovering generated stuff**:
- API: `discussion.artefacts.get(title)`, `discussion.artefacts.list()`,
  `discussion.artefacts.get_version_history(title)`,
  `discussion.artefacts.diff_versions(title, v1, v2)`.
- Disk: the discussion's workspace directory (workspace twin files).
- Turn result: the returned dict's `"artefacts"` key lists affected artifacts
  (title, type, version); message metadata `artefacts_modified` lists titles.

### `auto_activate_artefacts: bool = True`

**What it does**: Newly created or patched artifacts become active immediately —
their content is loaded into the model's context (Active Artifacts zone) on the
very next round. This is the default "the model can immediately see what it
just wrote" behavior.

**What to expect**:
- **Sub-flag of `enable_artefacts`** (master). Inert when the master switch is
  off.
- `True` (default): after creation/patching, the model can read the artifact's
  content directly — no `<unlock_file>` needed. This is what makes
  iterative editing (`SEARCH/REPLACE` hunks against a file the model just
  authored) work smoothly.
- `False`: artifacts are created with `TREE_UNLOCKABLE` visibility — the
  model must `<unlock_file>` them explicitly to read content. Use this for
  large files where context budget is at risk of being blown by the model
  re-seeding its own output into the prompt.
- **Interaction with `active_images`**: images embedded in artefacts obey their
  own default `SINGLE_ROUND` visibility (see `enable_image_generation`), NOT
  this flag. `auto_activate_artefacts=False` does *not* prevent the pixel
  payload of a freshly created image from being seen on the next round — it
  only withholds *text* content. See `BRANCHING.md` §3 for `active_images`
  as a **message-level** field, not a `chat()` parameter.
- **Cost / latency**: `True` materially increases context size on successive
  rounds as artefacts accumulate. With `True` and no compaction, expect
  per-round prompt growth proportional to the sum of active artefact sizes up
  to `context_compaction_threshold`. Turn it off (per-file `<lock_file>` tags
  or `auto_activate_artefacts=False` globally) when the workspace contains
  larger artefacts than the model will actually edit.
- Per-message vs. per-turn: `auto_activate_artefacts` is **per-turn** (a
  `chat()` parameter). It cannot be toggled mid-turn. Use per-file
  `<unlock_file>` / `<lock_file>` tags or `active_images` dispositions if you
  need finer granularity on individual artefacts.

**Recovering generated stuff**: fetched via `discussion.artefacts.get(title)`
either way — the flag only governs whether the model *sees* content during the
turn, not whether the artefact exists. See
[`flags/auto_activate_artefacts.md`](flags/auto_activate_artefacts.md) for the
full semantics and worked examples.

### `enable_notes: bool = True`

**What it does**: Enables the `<note title="...">...</note>` tag — quick markdown
notes saved as note-type artifacts (patchable via SEARCH/REPLACE).

**What to expect**: requires `enable_artefacts=True` (sub-flag of the master
switch).

**Recovering generated stuff**: `discussion.artefacts.get("<note title>")`, or
the workspace file under the discussion directory.

### `enable_skills: bool = True`

**What it does**: Enables the `<skill>` tag. Routing is personality-aware:
- **Handbag personality** (has `skills_manager`): skills are written directly
  into the handbag's skills folder (`create_skill`/`update_skill`). Read-only
  skills are refused with a clear failure block.
- **Stateless/manual personality**: skills are stored as discussion artifacts
  (`<title>.md`, skill type).

**What to expect**: requires `enable_artefacts=True`.

**Recovering generated stuff**:
- Handbag → `personality.skills_manager.skills` (persisted across discussions).
- Stateless → `discussion.artefacts.get("<title>.md")`.

### `enable_inline_widgets: bool = False`

**What it does**: Enables `<lollms_inline>` widgets — host-UI rendered
components.

**What to expect**: presentation-layer only; requires `enable_artefacts=True`
and a host UI that renders the widget markup. No persisted content beyond the
raw XML passed through to the message content.

### `enable_forms: bool = False`

**What it does**: Enables `<lollms_form>` — the model can emit an interactive
form descriptor (fields, ratings, textareas, sections) that the host UI renders
natively.

**What to expect**:
- Form instructions are injected **only when trigger keywords** appear in the
  user message ("form", "formulaire", "survey", "questionnaire").
- The parsed descriptor is parked in `discussion._get_pending_forms()[form_id]`
  and emitted as `MSG_TYPE_FORM_READY`.

**Recovering generated stuff (answers)**: call
`discussion.submit_form_response(form_id, answers)` — answers are formatted and
injected into the conversation as a user message, and a
`MSG_TYPE_FORM_SUBMITTED` event fires.

### `enable_books: bool = False`

**What it does**: Injects book/chapter-building instructions into the system
prompt.

**What to expect**: keyword-gated — instructions appear only when the user
message mentions "book", "tome", "novel", "chapitre". Requires
`enable_artefacts=True`. The model then writes chapters as regular document
artifacts (recovered like any artifact).

### `enable_pptx_builder: bool = False`

**What it does**: Injects presentation-building instructions (keyword-gated:
"presentation", "slide", "slideshow", "deck", "diaporama") AND mounts the
`pptx_builder` LCP toolset: `tool_pptx_build`, `tool_pptx_edit`,
`tool_pptx_read`.

**What to expect — NO vision gate**: the mount is **unconditional** when `True`
(text-only workflow: recipe code in → markdown inventory out). The model writes
sandboxed python-pptx "recipe" code — no `import` statements allowed; violations
are rejected before execution. First invocation auto-installs `python-pptx` via
pipmaster — expect a slower first call.

**Recovering generated stuff (.pptx)**:
- The tool writes the deck into the discussion working directory
  (`tool_context["working_dir"]`, i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file` (absolute path), `context_files` (list
  containing the path), `file_name` (sanitized; default `presentation.pptx`).
- The file is auto-registered as a **document artifact** by `_sync_tool_artifacts`
  (workspace snapshot diffing) — recover via
  `discussion.artefacts.get("presentation.pptx")` or open `workspace_file`
  directly on disk.
- The turn result's `"artefacts"` array also lists it.

### `enable_image_generation: bool = True` / `enable_image_editing: bool = True`

**What it does**: Enables the `<generate_image>` / `<edit_image>` tags.
**Capability-gated**: instructions are only injected when the client has a TTI
binding (`client.tti` or a TTI profile registry). A tag emitted without an
engine fails gracefully with a `status:failure` processing block.

**What to expect**:
- `edit_image` requires a source image (existing artifact pixels or the last
  active image) AND non-empty instructions; otherwise it fails cleanly.
- **Dimension pass-through**: `width`/`height` are OPTIONAL. When the model
  omits them, they are forwarded as absent to the TTI engine, which edits at
  the SOURCE image's native dimensions. Explicit (valid) values are clamped to
  the engine's supported range (64–4096); a malformed value is treated as
  unspecified (never silently coerced to a default that would resize the
  source).
- Generated images are stored as image artifacts with their pixels (base64 PNG)
  plus a logical markdown twin.
- **Default visibility is `SINGLE_ROUND`**: the LLM sees the image on the round
  immediately following creation (to verify the result and decide whether to
  edit it), then the artefact is auto-demoted to `TREE_UNLOCKABLE` and the
  base64 payload is freed from context. This is the primary mitigation against
  image context bloat over long sessions. To keep an image permanently loaded,
  the LLM can emit `<unlock_file>` (which sets `FULL`) or pass
  `visibility="full"` on a subsequent write.

**Recovering generated stuff (images)**:
- Artifact pixels: `art = discussion.artefacts.get(title)` →
  `art["images"][0]` (base64 PNG).
- Message anchor: `<artefact_image id="<title>::0" />` is appended to
  `ai_msg.content`.
- Physical file: sync via `discussion.artefacts.sync_to_workspace(title)` or
  read the workspace twin directly.
- Also listed in the returned `"artefacts"` array of the chat result.

### `forward_artefact_chunks: bool = False`

**What it does**: Forwards live artifact body chunks to the streaming callback
as `MSG_TYPE_ARTEFACT_CHUNK` events while the artifact streams.

**What to expect**: UI-verbosity flag. No generated content to recover — the
artifact itself is recovered via the master-switch path above.

### `enable_in_message_status: bool = False`

**What it does**: Emits in-message status/progress lines inside `<processing>`
blocks while the model works.

**What to expect**: UI-verbosity flag. Keep `False` for clean transcripts; the
processing blocks are scrubbed from LLM context either way.

---

## Memory

### `enable_memory: bool = False` — MASTER SWITCH

**What it does**: Enables the memory subsystem for the turn: memory system
instructions, decay, budget enforcement, `<mem_*>` tag processing (`<mem_new>`,
`<mem_update>`, `<mem_tag>`, `<mem_load>`, `<mem_delete>`, `<mem_search>`,
`<mem_rel>`), and the Active Memories context block.

**Manager resolution order**: personality's own `memory_manager` → discussion's
`_get_memory_manager()` → explicit `memory_manager` argument →
`self.memory_manager`.

**What to expect**: with `False` (the default), no memory operations occur and
memory tags are not processed. Enable explicitly to opt into persistent agent
memory.

### `enable_episodic_memory: bool = False`

**What it does**: Saves this conversation turn as episodic memory — but only for
**non-trivial exchanges**:
- Total length > 200 chars, OR tools were used, OR artifacts were created.
- Greetings ("hi", "hello"), thanks, and yes/no exchanges are detected and
  skipped.

**What to expect**: set `False` for privacy-sensitive sessions or when you want
manual control over memory persistence.

### `enable_auto_dream: bool = False`

**What it does**: Runs a memory consolidation ("dream") cycle after the turn
completes.

**Recovering generated stuff (report)**: the chat result's `"dream_report"` key
carries the report (when enabled and the manager exists).

### `enable_deep_memory_pulling: bool = False`

**What it does**: Automatically pulls relevant deep memories into working memory
based on the user message, before generation.

**What to expect**: extra retrieval latency per turn; failures are logged as
warnings and never abort the turn.

---

## Knowledge & Search

### `prehydrate_rag: bool = True`

**What it does**: Queries RAG before generation and injects retrieved context
into the system prompt, capped at **50,000 characters** to prevent context
bloat. Source order: personality data first (`personality.query_data`), else the
client's RAG binding (`client.query_rag`).

**What to expect**: retrieved sources appear in the returned `"sources"` list
with `type: "rag"`. RAG tools built by the personality are also registered in
the tool registry.

### `web_search: bool = False` / `internet_search: bool = False`

**What it does**: Pre-hydrates up to **5 web search results** into the system
prompt before generation. Both are aliases; kwargs `web_search` /
`internet_search` / `enable_web_search` / `enable_internet_search` also trigger
it.

**What to expect**: extra network latency per turn; sources appear in
`"sources"` with `type: "web"`. Failures are logged as warnings and do not abort
the turn.

---

## Tools & Security

### `enable_data_tools: bool = True`

**What it does**: Enables data tools **contextually**:
- SQL/pandas query tools (`tool_execute_python_data_query`,
  `tool_get_table_schema`, `tool_query_database_sql`) mount only when **data
  files exist** in the workspace (`.csv`, `.db`, `.sqlite`, `.xlsx`, `.parquet`,
  ...).
- Document tools (`tool_inspect_document`, `tool_read_document_content`,
  `tool_annotate_document`, `tool_edit_document_text`, and other docx/excel/pptx
  modifiers) mount only when **document files exist** (`.pdf`, `.docx`, `.pptx`,
  `.md`, ...).

**What to expect**: no matching files → no data tools in the registry, even with
the flag on. The workspace scan is cached per write-revision for performance.

### `compact_workspace_tree: bool = False`

**What it does**: Swaps the **inline recursive workspace tree** (injected into
the system prompt via the artefact context zone) for an **on-demand discovery
protocol**. The model receives a compact one-line stub describing the workspace
root and the number of top-level items, plus a doctrine block teaching it to
call `tool_show_workspace_tree(folder, depth, max_items)` — never to hallucinate
paths it has not seen.

**What to expect**:
- `False` (default): the full recursive tree is inlined every round. This is
  ideal for small sessions (a handful of files) where the tree is genuinely
  useful for orientation and costs almost nothing.
- `True`:
  - The inline tree is suppressed. The discussion mirrors the flag onto
    `self.artefacts.compact_workspace_tree` and, as defence-in-depth, post-strips
    any residual `=== WORKSPACE TREE === … === END WORKSPACE TREE ===` block from
    the artefact context zone (so old artefact-manager versions that ignore the
    flag are still silenced).
  - A `=== WORKSPACE (COMPACT TREE) ===` stub is injected instead, stating the
    top-level item count and pointing at the tool.
  - The `workspace_tools` LCP library is auto-mounted (even when
    `enable_workspace_tools`-style flags are off) and admits the on-demand
    inspection family: `tool_show_workspace_tree`, `tool_list_files`,
    `tool_find_files`, `tool_grep_files`, `tool_read_file`, `tool_read_lines`,
    `tool_read_chars`, `tool_grep_file`.
  - A system-prompt **doctrine** block instructs the model to (a) call
    `tool_show_workspace_tree` before referencing/reading/editing any workspace
    file, (b) never invent filenames or folder structures, and (c) drill down
    incrementally (root first, then pass a `folder`).

**When to use it**: large projects, repositories with a `.git` directory or
versioned storage (`.versions/<id>/` bloat), o1-flavoured models that derail on
long noisy contexts, and any session where context budget is tight. This is the
primary mitigation against "workspace tree trashes the context window and
confuses the model" — the exact failure mode observed on big monorepos.

**Recovering generated stuff**: no output is generated by this flag itself. It
only changes *how* the model discovers files. The tool returns the tree text
under the `"output"` key on success (`{"success": True, "output": "..."}`) or
`{"success": False, "error": "..."}` for invalid folders/depths. Once the model
sees the tree, it uses the standard read toolset (`tool_read_file`,
`tool_read_lines`, ...) to fetch content.

**Example**:
```python
result = discussion.chat(
    "Refactor the auth module",
    compact_workspace_tree=True,   # big repo — don't inline 4000 files
    enable_code_execution=True,
)
# The model first calls:
#   <tool>{"name": "tool_show_workspace_tree",
#          "parameters": {"folder": "", "depth": 2}}</tool>
# then drills into the relevant subdirectory before reading files.
```

### `enable_code_execution: bool = False`

**What it does**: Mounts the Python execution toolset: `tool_execute_python_code`
(inline code, max 2000 chars), `tool_execute_python_file` (runs an existing
workspace script), plus file-reading helpers (`tool_read_lines`,
`tool_read_chars`, `tool_grep_file`). **Security-sensitive — off by default.**

**What to expect**: pair with `python_autonomy_level` (default `"safe"`),
`auto_approve_python`, and a `confirm_handler` to control execution approval.
Execution runs with the discussion workspace as CWD.

**Recovering generated stuff**: scripts run with the working directory as CWD;
any files they write are auto-registered as artifacts by the workspace snapshot
diffing (`_sync_tool_artifacts`) — recovered via `discussion.artefacts` like any
artifact.

### `auto_approve_python: bool = False`

**What it does**: Python execution runs **without confirmation prompts**.

**What to expect**: use only in trusted, sandboxed environments. When `False`,
the `confirm_handler` (if provided) is consulted before dangerous executions.

### `allow_dynamic_tools: bool = False`

**What it does**: Security gate for tool-authored tools. With `False`, artifacts
of `type="tool"` are **not** registered as executable LCP tools. With `True`,
the model may author a `.py` artifact declaring `tool_*` functions, and that
file becomes an LCP-registrable Python toolset on subsequent rounds.

**What to expect**:
- **Security-sensitive.** This is a code-execution surface — the model writes
  arbitrary Python that the runtime will import. It MUST be paired with
  `enable_code_execution=True` and a `confirm_handler` in any non-isolated
  deployment. Without the flag, tool artifacts still exist as filesystem
  content — they just never register into the tool registry.
- **Tool runtime context.** Dynamic `tool_*` functions run with the standard
  LCP `env` (`env.llm`, `env.tti`, their `*_model_profiles`, `*_switch`
  functions). They may declare reserved parameters `lollms_client_instance` /
  `discussion_instance` to receive injected instances at call time.
- Companion to (but independent from) `enable_toolsmith`. `enable_toolsmith`
  gates the **act-phase** of editing a *registered* toolset (read/write/test/
  commit via `tool_maker`). `allow_dynamic_tools` gates registering a **brand-
  new tool** authored as an artifact. Neither implies the other.
- Silent skip: if the LCP binding cannot import the artifact (syntax error,
  violation of the tool signature contract), registration is skipped with a
  console warning and the tool-creation attempt is reported back to the model.
  The turn is **not** aborted.

**Recovering generated stuff**: the tool artifact lives at
`discussion.artefacts.get("<title>")`. Once registered, it appears in
`client.tools.discovered_tools` and is callable exactly like any other LCP
tool. See [`flags/allow_dynamic_tools.md`](flags/allow_dynamic_tools.md) for
the full enablement recipe, the enable_code_execution interaction matrix, and
failure-mode table.

### `enable_vlm_query: bool = False`

**What it does**: Mounts `tool_vlm_query` as a **vision fallback** — only when
the active LLM lacks vision AND a vision-capable binding exists elsewhere on the
client.

**What to expect**: with a native vision model active, this mounts nothing (the
fallback is unnecessary).

### `enable_toolsmith: bool = False` / `enable_skillmaker: bool = False`

**What they do**: Activate the **heavy RSI act-phase** of the `tool_maker` and
`skill_maker` toolsets respectively. The basic manipulation surface (read /
write / test / revert / discard / list_versions) for toolsets and skills is
**always available** and independent of these flags — the flags gate ONLY the
act-phase tools and the promotion tool:

| Flag | Enables (LLM-visible) | Basic tier (always available) |
|---|---|---|
| `enable_toolsmith` | `tool_run_test_agent`, `tool_commit_toolset` | `tool_read_tool_source`, `tool_write_tool_source`, `tool_test_tool_source`, `tool_list_tool_versions`, `tool_revert_tool_source`, `tool_discard_changes` |
| `enable_skillmaker` | `tool_run_skill_test_agent`, `tool_commit_skill` | `tool_read_skill_source`, `tool_write_skill_source`, `tool_test_skill_source`, `tool_list_skill_versions`, `tool_revert_skill`, `tool_discard_skill_changes` |

**What to expect**:
- Both flags are **fail-closed** and **independent**: `enable_toolsmith` does
  not enable skill RSI and vice versa. `LCPBinding.discover_tools()` chains
  `_filter_toolsmith_visibility()` then `_filter_skillmaker_visibility()`.
- When a flag is off, its two act-phase tools are **hidden from auto-discovery**
  — the LLM's spec list does not contain them. Direct `execute_tool` calls are
  still possible for tests / GUI paths, but each act-phase tool body re-checks
  the flag and returns `{"success": False, "requires_flag": "..."}` when it is
  off (defence in depth).
- The basic tier is **never** hidden — a session without these flags can still
  edit toolsets and skills. Only the heavy subprocess-swap-and-commit cycle is
  gated.
- The `code_execution` confirmation modal still fires on every `tool_maker` /
  `skill_maker` tool regardless of these flags — the flags govern **exposure**,
  not **approval**.
- `enable_toolsmith` spawns a constrained child LLM as part of
  `tool_run_test_agent` — expect latency and additional model calls. `commit_*`
  records a file-system release manifest under `.rsi/commits/<epoch>.json`
  (no git required) and, when git is available, opportunistically runs
  `git add/commit` scoped to the toolset/skill folder.

**Recovering generated stuff**: releases are recoverable on disk under
`<toolset_or_skill_dir>/.rsi/commits/<epoch>.json` (embedded `source`);
rollback snapshots under `.rsi_backup/`. Both are surfaced to the model via
`tool_list_tool_versions` / `tool_list_skill_versions`.

### `enable_computer_use: bool = False`

**What it does**: Mounts the desktop automation toolset:
`tool_computer_desktop_info`, `tool_computer_screenshot`, `tool_computer_click`,
`tool_computer_move_cursor`, `tool_computer_mouse_down`,
`tool_computer_mouse_up`, `tool_computer_drag`, `tool_computer_type`,
`tool_computer_key`, `tool_computer_scroll`, `tool_computer_wait`,
`tool_computer_cursor_position`.

**What to expect — VISION GATE (hard structural gate)**: the toolset is mounted
**only if the active model has vision capability**
(`client.has_vision_capability()`, or the active binding's `vision_enabled`, or a
vision-capable child binding). With a blind model, the mount is **silently
skipped with a console warning** and the model truthfully reports it has no such
tools. First use auto-installs `pyautogui` / `pillow` via pipmaster — expect a
slower first call.

The capability override `allow_computer_use` (Optional[bool], also read from
`personality.capabilities`) ORs with this flag to request the mount.

**Recovering generated stuff (screenshots)**:
- Default file: `computer_use_screenshot.png` in the working directory —
  **overwritten on every recapture** (each screenshot shows the CURRENT screen
  state). Pass `file_name` to `tool_computer_screenshot` to keep a specific
  frame.
- Tool result keys: `screenshot_file`, `image_filename`, `context_files`,
  `image_b64`/`screenshot_b64` (also attached in-conversation so the vision
  model can verify), `screen_width`, `screen_height`, `cursor_x`, `cursor_y`.

### Per-Software Office Toolsets: `enable_powerpoint` / `enable_word` / `enable_excel`

**What they do**: Each flag mounts exactly ONE dedicated LCP toolset and admits
ONLY its own tool family (per-flag isolation — `enable_excel=True` alone never
exposes PowerPoint or Word tools):

| Flag | Library | Tools |
|---|---|---|
| `enable_powerpoint` | `powerpoint_tools` | `tool_build_powerpoint`, `tool_edit_powerpoint`, `tool_read_powerpoint`, `tool_pptx_info`, `tool_view_slides` |
| `enable_word` | `word_tools` | `tool_build_word`, `tool_edit_word`, `tool_read_word` |
| `enable_excel` | `excel_tools` | `tool_build_excel`, `tool_edit_excel`, `tool_read_excel` |

Supported formats: `.pptx`, `.pptm` (PowerPoint); `.docx`, `.docm` (Word);
`.xlsx`, `.xlsm` (Excel).

**Per-software doctrine** (injected into the system prompt when the flag is on):
- **PowerPoint**: new decks are **16:9** (13.333in × 7.5in, EMU 12192000×6858000);
  call `tool_pptx_info` FIRST to discover slide layouts and theme fonts; verify
  visually with `tool_view_slides` — **max 4 slides per call**, images attached
  to the CURRENT ROUND ONLY, never stored in the workspace; never call
  `prs.save()` — auto-saved and validated.
- **Word**: full edit freedom with python-docx ("use your own coding skills")
  plus high-leverage hints: `add_comment(doc, paragraph, text, author)` (real
  OOXML comment with yellow-highlight fallback), `run.font.highlight_color =
  WD_COLOR_INDEX.YELLOW`, styles/fonts/alignment reformatting,
  `doc.add_section(WD_SECTION.NEW_PAGE)`, links/fields via OXML
  `qn('w:fldSimple')` (Word updates on open), tables/images via `media()`.
- **Excel**: **FORMULAS ONLY for computed values — NEVER write calculation
  results manually**. Excel computes them on open:
  `ws["F2"] = "=SUM(B2:E2)"`, `ws["G2"] = "=AVERAGE(B2:E2)"`,
  `ws["H2"] = '=IF(F2>100,"High","Low")'`, `=VLOOKUP(...)`, `=INDEX/MATCH`.
  Literals are ONLY for input data. Chart-recalc caveat: charts referencing
  formula cells may appear empty until Excel recalculates on open — formulas
  are still the correct choice.

**What to expect — NO vision gate**: mounts are **unconditional** when their
flag is set (text-only workflow: recipe code in → markdown inventory out).
Recipe code is sandboxed: imports are allowed ONLY from the per-format
whitelist (`pptx.*` / `docx.*` / `openpyxl.*` plus safe stdlib modules) —
core design symbols are pre-imported; forbidden tokens (`eval`, `exec`,
`open`, `os.`, `sys.`, subprocess, dunder access, ...) are rejected before
execution. Recipes are capped at 60,000 characters.

**Integrity-verified saves**: every save passes a 3-layer validation (ZIP
integrity + required OOXML parts, strict XML parse, library re-open). If a
save would corrupt the file, it is automatically **rolled back** to the
pre-save snapshot and the tool returns the machine-readable failure code
`office_file_failed_validation` — the model fixes the recipe and retries.
First invocation auto-installs the per-format library (`python-pptx` /
`python-docx` / `openpyxl`) via pipmaster — expect a slower first call.

**LCP fail-closed gate**: the three toolset libraries are office-family and
are **excluded from discovery unless their flag is set** — they never
auto-appear in a session (the same leak class fixed earlier for as-is tools).

**Recovering generated stuff**:
- Tools write into the discussion working directory (`tool_context["working_dir"]`,
  i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file`, `file_name` (sanitized).
- Files are auto-registered as artifacts by `_sync_tool_artifacts` (workspace
  snapshot diffing) — recover via `discussion.artefacts.get(...)` or open
  `workspace_file` directly on disk.
- The turn result's `"artefacts"` array also lists them.

### Per-turn built-in tool activation (non-LCP): `builtin_tools_activation_list`

**Not a boolean flag** — a parameter (`Optional[List[str]]`, default `None`) —
documented here because it gates tool exposure per turn. It carries a list of
native built-in tool names (discoverable via `discussion.list_builtin_tools()`)
to force-load **when the client has no tools binding configured**:

```python
result = discussion.chat(
    "Inspect this document",
    builtin_tools_activation_list=["inspect_text", "read_text_file"],
)
```

| Property | Value |
|---|---|
| Type | `Optional[List[str]]` |
| Default | `None` |
| Effect | When `lcp_binding is None`: each name is looked up in the builtin inventory and registered directly with its callable into the turn's tool registry — executed via the direct-callable path, no LCP infrastructure required |

**Constraints:**
- Non-empty **strings only** — validated at `chat()` entry; non-list input and
  invalid entries are warned and skipped, never raised.
- Duplicates are deduplicated order-preserved (`dict.fromkeys`).
- Unknown names and uncallable specs are warn-and-skip.
- No schema validation at activation (deferred to call time).
- Result normalization is **identical** to LCP-dispatched tools
  (`{success, error, output, file_name, context_files, ...}`) and feeds the
  same artefact pipeline.
- Per-turn only: nothing is persisted onto `self.lollmsClient.tools`.

**Interaction matrix:**

| `lcp_binding` state | `builtin_tools_activation_list` effect |
|---|---|
| Valid binding present | Ignored (debug log: LCP takes precedence) |
| Bootstrap succeeded (flags active) | Ignored |
| Bootstrap failed / no flags / `lcp_binding is None` | Direct-call registration; tools dispatch normally |

### `enable_pdf: bool = False`

**What it does**: Mounts the `pdf_tools` LCP library and admits only its tool
family (PDF read / inspect / annotate). Per-flag isolation applies — no
PowerPoint/Word/Excel tools leak in.

**What to expect**: unlike the other office-family flags, `enable_pdf` is
**auto-admitted** when a `.pdf` file is present in the workspace or the
discussion's artifacts, even without the flag being passed. Passing
`enable_pdf=True` forces the mount even when no PDF exists yet.

**Recovering generated stuff**: annotated PDFs are written to the discussion
working directory and auto-registered as artifacts by `_sync_tool_artifacts` —
recover via `discussion.artefacts.get(...)` or open the `workspace_file` path
directly. First invocation auto-installs `pypdf` via pipmaster.

### Per-Toolset Editing Flags: `enable_toolset_editing` / `enable_skill_editing`

**What they do**: Mount the callable editing surfaces for LCP toolsets and
`SKILL.md` files respectively — inside the turn.

| Flag | Library | Admitted tools |
|---|---|---|
| `enable_toolset_editing` | `toolset_staging` | `tool_stage_list_stageable`, `tool_stage`, `tool_stage_patch`, `tool_stage_rewrite`, `tool_stage_test`, `tool_stage_unstage`, `tool_stage_status` |
| `enable_skill_editing` | `skill_editor` | `tool_edit_skill_list/activate/read/patch/rewrite/reload/test/load_status/close` |

Companions `toolset_editing_paths` / `skill_editing_paths` (Optional[List[str]],
non-boolean) extend the editable set to folders OUTSIDE `workspace_path`. Both
are validated once per turn (`expanduser().resolve()`, directory-checked),
deduplicated, and pushed into the loaded module via `set_staging_paths(...)` /
`set_skill_paths(...)` after the mount. Invalid entries are warned and skipped —
never raised.

**What to expect**:
- **Require a live LCP tools binding.** With none configured, the mount is
  refused with an actionable diagnostic; the turn proceeds with whatever tools
  it has.
- **`toolset_staging`**: one focused toolset at a time (multi-staging refused);
  only folder-layout toolsets (`<name>/TOOLS.py`) are editable. Patches are
  applied atomically (`ast.parse` + a top-level `def tool_*` requirement); a
  broken source never reaches disk. `execute_python`, `tool_write_file`, and
  `<artifact>` are FORBIDDEN writers against a staged toolset.
- **`skill_editor`**: auto-activates against a sole candidate, allowing an agent
  to update a skill it already holds without an explicit activation step. A
  skill that declares `modifiable: false`, `SKILL_LIBRARY_READONLY: true`, or
  `SKILL_LIBRARY_READONLY = True` refuses all writes (report, do not bypass).
- Both mounts pre-authorize their fail-closed session flag on the binding
  (`tools_binding_config` + `host_tool_configs[stem]` + `set_session_flags`)
  before mounting, so discovery succeeds in normal operation.
- On success, a matching doctrine block (`TOOLSET EDITING PROTOCOL` /
  `SKILL EDITING PROTOCOL`) is injected into the system prompt for the turn.

**Recovering generated stuff**: the edited `TOOLS.py` / `SKILL.md` lives on disk
in its folder — the file IS the deliverable. See
[`TOOLSET_STAGING_AND_SANDBOX.md`](TOOLSET_STAGING_AND_SANDBOX.md) and
[`SKILL_EDITOR_AGENT.md`](SKILL_EDITOR_AGENT.md).

### Media Generation Flags: `enable_generate_video` / `enable_edit_video` / `enable_generate_music` / `enable_edit_music` / `enable_synthesize_speech`

**What they do**: Each activates ONE text-to-X modality through dedicated XML
tags (NOT tools — wrapping them in `<tool>{}` is rejected as phantom). All five
default to `False` (media generation is expensive). Image equivalents
(`enable_image_generation` / `enable_image_editing`) default to `True` and are
documented above.

| Flag | Tag | Engine gate |
|---|---|---|
| `enable_generate_video` | `<generate_video>` | **TTV** |
| `enable_edit_video` | `<edit_video>` | **TTV** (`edit_video`) |
| `enable_generate_music` | `<generate_music>` | **TTM** |
| `enable_edit_music` | `<edit_music>` | **TTM** (`edit_music`) |
| `enable_synthesize_speech` | `<synthesize_speech>` | **TTS** |

**What to expect**:
- **Capability-gated per modality**: a live binding with the required callable
  OR a non-empty `*_model_profiles_registry`. Missing both → the doctrine block
  is withheld and any emitted tag surfaces a `status:failure` block.
- Numeric parameters (`duration`, `width`, `height`, `fps`, `seed`,
  `negative_prompt`) belong in the tag ATTRIBUTES; describing them in the body
  has no effect. Malformed values are dropped.
- Edit tags require `source` (exact workspace file name). The source is NOT
  overwritten — output defaults to `<stem>_edited.<ext>` unless `name` is given.
- Output extensions default to `.mp4` (video), `.mp3` (music), `.wav`
  (speech); subfolder paths in `name` are created automatically.

**Recovering generated stuff**: the physical file lands in the discussion
workspace (`workspace_data_path`) and a metadata-card artifact (`type="video"`
or `"audio"`) is registered. The card carries a summary, NOT the bytes — the
file on disk is the source of truth (as-is doctrine). Recover via
`discussion.artefacts.get("<name>.<ext>")` or open the file directly; the turn
result's `"artefacts"` array lists every card produced. Message anchors use
markdown links (`[🎬 …](…mp4)` / `[🎵 …](…mp3)` / `[🗣️ …](…wav)`).

### Legacy flag — `enable_office_files` (DEPRECATED)**: passing
`enable_office_files=True` as a kwarg enables all four per-software flags AND
mounts the monolithic legacy `office_files_builder` library (12 tools:
`tool_office_build/edit/read` plus nine per-format aliases). A one-line
deprecation notice is logged. Prefer the per-software flags.

### `suppress_images: bool = False` — NEGATIVE POLARITY

**What it does**: Withholds **all** image pixels from the model's context this
turn — user-supplied images, generated plots, unlocked image artifacts.

**What to expect**: vision hydration is skipped entirely; a console diagnostic is
emitted. Use with text-only models or to cut token cost. Unlocked images produce
guidance telling the model pixels are unavailable (and it must say so rather
than pretend to see).

---

## Personas & Diagnostics

### `orchestrator_mode: bool = False`

**What it does**: Routes the entire turn through the `AgenticRunner`
orchestrator, which spawns specialist worker sub-agents and collects sealed
reports.

**What to expect**: the orchestrator never sees tool-call grammar; workers
execute. The returned structure comes from `AgenticRunner.run()` (not the
standard chat result dict shape).

### `orchestrator_persona: bool = False`

**What it does**: Runs the turn as the delegation-only **Orchestrator persona**:
plain-language worker catalogue, **no tool grammar**, and artifact writes plus
tool execution are structurally refused at the dispatcher (`<tool>` dispatch →
refusal envelope).

**What to expect**: `<delegate>` / `<agent>` are the only action paths. Attempted
`<tool>` calls are refused with a delegation-correction envelope.

### `debug: bool = False`

**What it does**: Mounts the `debug_toolset` (`tool_dump_context`) and prints
the active tool registry to the console.

**What to expect**: combine with `discussion._debug_mode = True` for persistent
per-discussion dumps. Zero-registry turns print an explicit warning.

### `verbose: bool = False`

**What it does**: Enables per-round telemetry — trace timings, token fill
percentage, time-to-first-token (TTFT), round bookkeeping, and vision
diagnostics. Independent of `debug` (which mounts the debug toolset and logs the
tool registry).

**What to expect**: no behavioural change to the turn — purely console output.
The CLI stays clean unless `verbose=True`. Typically enabled during performance
profiling or agentic-loop debugging.

### `debug_export: bool = False`

**What it does**: Writes per-round prompt dumps, raw LLM stream logs, and an
end-of-turn JSON context dump.

**Recovering generated stuff (diagnostics)**: files land in
`<workspace>/_debug_dumps/`:
- `full_prompt_round_<N>.log` — the exact messages sent to the LLM.
- `raw_llm_output_round_<N>.log` — the raw streamed tokens.
- `turn_dump_<timestamp>.json` — virtual history, message content, and metadata.

---

## Cross-Flag Interactions & Gotchas

1. **Master-switch hierarchy**: `enable_artefacts=False` disables notes, skills,
   widgets, books instructions, and image-tag dispatch even when their own
   flags are `True`. (Office toolset instructions and tools are governed solely
   by `enable_powerpoint` / `enable_word` / `enable_excel` / `enable_pdf` — or
   the deprecated `enable_office_files` kwarg. Media flags are independent of
   `enable_artefacts` for PHYSICAL file writing — only the artifact card
   registration is suppressed.)
2. **Discovery ≠ exposure**: LCP libraries are AST-discovered at client
   construction, but the model only sees a library when the per-turn chat flag
   mounts it. A model claiming "I have no such tools" almost always means the
   flag wasn't passed (or vision was missing for `enable_computer_use`).
3. **Persistent mounts**: mounts live on `client.tools`
   (`LCPBinding.discovered_tools`) at client level — there is **no
   per-discussion tool registry and no unmount-after-turn cleanup**. Mounts are
   idempotent (`mount_tool_library_if_absent`); re-enabling a flag on later
   turns is a cheap no-op.
4. **Keyword-gated instructions**: `enable_books` and `enable_forms` inject
   their instructions only when trigger keywords appear in the user message,
   even with the flag enabled. Office toolset instructions are injected
   unconditionally per-flag when the flag is on.
5. **Per-flag isolation**: each office flag exposes ONLY its own tool family —
   `enable_excel=True` alone never exposes PowerPoint or Word tools. `enable_pdf`
   is the only office-family flag auto-admitted on file presence (a `.pdf` file
   in the workspace/artifacts triggers a mount even with the flag off). When
   both a per-software flag and the legacy `enable_office_files` kwarg are set,
   per-library mounting selects the winner and the later mount wins (identical
   tool names are deduplicated by the LCP registry).
6. **Negative-polarity flags**: `suppress_images` is inverted — `True` *disables*
   vision hydration rather than enabling a feature.
6. **Memory sub-flags require the master**: `enable_episodic_memory`,
   `enable_auto_dream`, and `enable_deep_memory_pulling` are inert when
   `enable_memory=False` or no manager resolves.
7. **Cancellation is sovereign**: `discussion.cancel_generation()` is checked at
   every round boundary and every streamed token; a cancelled turn persists
   partial content with `turn_status="cancelled"` and is resumable via
   `resume_turn()`.
8. **Compact tree requires an LCP binding**: `compact_workspace_tree=True`
   without an LCP binding logs a warning — the model will be told to call
   `tool_show_workspace_tree`, which cannot be mounted. Ensure the client has a
   tools binding configured (or a per-turn `builtin_tools_activation_list`-style
   path) before enabling this flag.
9. **Compact tree is independent of `enable_artefacts`**: the tree-stub and the
   on-demand tool mount fire even when `enable_artefacts=False`, because the
   workspace tree is a discovery concern, not an artefact concern. The stub is
   injected regardless, and the tool family is mounted regardless.
10. **RSI flags are independent of everything else**: `enable_toolsmith` and
    `enable_skillmaker` gate ONLY the heavy act-phase tools; they do not mount
    or unmount their parent toolsets. `tool_maker` and `skill_maker` remain
    mountable and their basic tier remains callable without either flag. A
    session that never sets these flags can still read, patch, test, revert,
    and manually commit toolset/skill source — only the automatic
    child-test-agent + commit cycle is hidden. Never enable either flag in a
    session whose operator has not ALSO enabled a `confirm_handler` — the
    act-phase tools spawn child models that will prompt for approval through
    the standard `code_execution` gate, and a missing handler turns each
    child-step into a silent skip.
11. **Capability gates can silently downgrade a flag**: `enable_computer_use`
    requires a vision-capable active model; `enable_image_generation` and
    `enable_image_editing` require a TTI binding. When the gate is unmet, the
    flag is effectively inert, but the model correctly reports the missing
    capability (it is never told the tools exist). This is the correct
    "instructions + dispatch + hydration" behavior of the three-layer model
    described at the top of this document. If a flag appears to do nothing,
    check the gate first — never assume the model "forgot" the tag.

---

## Quick Recipes

**Chat-only (no workspace, no tools, no memory)**:
```python
discussion.chat(
    user_message,
    enable_artefacts=False,
    enable_data_tools=False,
    enable_memory=False,
)
```

**Build a PowerPoint deck (no vision needed)**:
```python
result = discussion.chat(
    "Create a 5-slide presentation (.pptx) about photosynthesis",
    enable_powerpoint=True,
)
# The model calls tool_pptx_info first (layouts), builds via
# tool_build_powerpoint, then verifies visually with tool_view_slides
# (max 4 slides per call, transient images).
# Recover the deck:
#   → result["artefacts"] lists the pptx artifact
#   → tool_build_powerpoint's workspace_file lives in the discussion working dir
#   → discussion.artefacts.get("presentation.pptx")
```

**Build Office documents (.docx / .xlsx)**:
```python
result = discussion.chat(
    "Build a quarterly report: a .docx summary and an .xlsx KPI workbook",
    enable_word=True,
    enable_excel=True,
)
# Recover the files:
#   → result["artefacts"] lists the office artifacts
#   → tool_build_word / tool_build_excel workspace_file paths live in the
#     discussion working dir
#   → discussion.artefacts.get("quarterly_report.docx")
#   → discussion.artefacts.get("kpi_workbook.xlsx")  # formulas, not literals
```

**Desktop automation (vision model required)**:
```python
result = discussion.chat(
    "Open the browser and take a screenshot",
    enable_computer_use=True,   # silently skipped if the model lacks vision
)
# Screenshots: computer_use_screenshot.png in the working directory
```

**Generate a video clip**:
```python
result = discussion.chat(
    "Generate a 5-second timelapse of a sunset over the ocean",
    enable_generate_video=True,
)
# Disk: ./sunset_timelapse.mp4 in the workspace
# Artifact card: discussion.artefacts.get("sunset_timelapse.mp4")
# Message: [🎬 sunset_timelapse](./sunset_timelapse.mp4)
```

**Multi-modal turn (music + narration + video)**:
```python
result = discussion.chat(
    "Compose a 20s lofi intro, narrate 'Welcome to the show', then generate "
    "a 3-second animated logo clip",
    enable_generate_music=True,
    enable_synthesize_speech=True,
    enable_generate_video=True,
)
# All three physical files land in the workspace; all three cards appear in
# result["artefacts"].
```

**Fix a toolset in-place (intra-turn staging)**:
```python
result = discussion.chat(
    "The unit_converter toolset mishandles negative temperatures — fix it",
    enable_toolset_editing=True,
    toolset_editing_paths=["/path/to/handbags/toolsets"],
    enable_code_execution=True,   # needed for tool_stage_test to execute
)
# The edited TOOLS.py lives on disk in the toolset folder.
```

**Privacy session (no persistence of conversation memory)**:
```python
discussion.chat(
    user_message,
    enable_memory=False,            # master switch already off by default
    enable_episodic_memory=False,   # sub-flags inert without the master
    enable_auto_dream=False,
)
```

**Full agentic debug session**:
```python
result = discussion.chat(
    task,
    enable_code_execution=True,
    enable_data_tools=True,
    debug=True,
    debug_export=True,
    max_nb_rounds=30,
)
# Dumps: <workspace>/_debug_dumps/full_prompt_round_*.log, turn_dump_*.json
```