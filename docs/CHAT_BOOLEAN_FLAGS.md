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
| `enable_pptx_builder` | `False` | Artifacts | **also mounts pptx_builder tools** |
| `enable_image_generation` | `True` | Artifacts | TTI capability-gated |
| `enable_image_editing` | `True` | Artifacts | TTI capability-gated |
| `forward_artefact_chunks` | `False` | Artifacts | UI streaming events |
| `enable_in_message_status` | `False` | Artifacts | UI progress lines |
| `enable_memory` | `True` | Memory | **master switch** |
| `enable_episodic_memory` | `True` | Memory | skips trivial exchanges |
| `enable_auto_dream` | `True` | Memory | post-turn consolidation |
| `enable_deep_memory_pulling` | `True` | Memory | pre-generation retrieval |
| `prehydrate_rag` | `True` | Knowledge | 50,000-char cap |
| `web_search` / `internet_search` | `False` | Knowledge | max 5 hits prehydrated |
| `enable_data_tools` | `True` | Tools | contextual workspace scan |
| `enable_code_execution` | `False` | Tools | security-sensitive |
| `auto_approve_python` | `False` | Tools | security-sensitive |
| `allow_dynamic_tools` | `False` | Tools | security-sensitive |
| `enable_vlm_query` | `False` | Tools | vision fallback only |
| `enable_computer_use` | `False` | Tools | **vision-gated** |
| `enable_office_files` | `False` | Tools | mounts office_files_builder tools |
| `suppress_images` | `False` | Tools | negative polarity |
| `orchestrator_mode` | `False` | Personas | AgenticRunner route |
| `orchestrator_persona` | `False` | Personas | delegation-only persona |
| `debug` | `False` | Diagnostics | tool registry dump |
| `debug_export` | `False` | Diagnostics | prompt/stream dumps to disk |

> Non-boolean parameters (`max_nb_rounds`, `tolerance_level`, `shell_autonomy_level`,
> `python_autonomy_level`, `context_compaction_threshold`, `event_mode`, `think`,
> `reasoning_effort`, `reasoning_summary`, `confirm_handler`, `allow_computer_use`
> — an `Optional[bool]` capability override) are out of scope here.

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
their content is loaded into the model's context (Active Artifacts zone).

**What to expect**: with `False`, artifacts are created with
`TREE_UNLOCKABLE` visibility — the model must `<unlock_file>` them to read the
content. Useful for large files to protect the context budget.

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
- Generated images are stored as image artifacts with their pixels (base64 PNG)
  plus a logical markdown twin.

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

### `enable_memory: bool = True` — MASTER SWITCH

**What it does**: Enables the memory subsystem for the turn: memory system
instructions, decay, budget enforcement, `<mem_*>` tag processing (`<mem_new>`,
`<mem_update>`, `<mem_tag>`, `<mem_load>`, `<mem_delete>`, `<mem_search>`,
`<mem_rel>`), and the Active Memories context block.

**Manager resolution order**: personality's own `memory_manager` → discussion's
`_get_memory_manager()` → explicit `memory_manager` argument →
`self.memory_manager`.

**What to expect**: with `False`, no memory operations occur and memory tags are
not processed.

### `enable_episodic_memory: bool = True`

**What it does**: Saves this conversation turn as episodic memory — but only for
**non-trivial exchanges**:
- Total length > 200 chars, OR tools were used, OR artifacts were created.
- Greetings ("hi", "hello"), thanks, and yes/no exchanges are detected and
  skipped.

**What to expect**: set `False` for privacy-sensitive sessions or when you want
manual control over memory persistence.

### `enable_auto_dream: bool = True`

**What it does**: Runs a memory consolidation ("dream") cycle after the turn
completes.

**Recovering generated stuff (report)**: the chat result's `"dream_report"` key
carries the report (when enabled and the manager exists).

### `enable_deep_memory_pulling: bool = True`

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
of `type="tool"` are **not** registered as executable LCP tools.

**What to expect**: enable only if you explicitly trust the model to author and
run its own tools.

### `enable_vlm_query: bool = False`

**What it does**: Mounts `tool_vlm_query` as a **vision fallback** — only when
the active LLM lacks vision AND a vision-capable binding exists elsewhere on the
client.

**What to expect**: with a native vision model active, this mounts nothing (the
fallback is unnecessary).

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

### `enable_office_files: bool = False`

**What it does**: Mounts the `office_files_builder` LCP toolset — read, create
and edit real Office files via sandboxed recipe tools:
- `tool_office_read` — inspect an existing `.pptx` / `.docx` / `.xlsx` and get
  a structured markdown inventory (slides, paragraphs, sheets, tables).
- `tool_office_build` — create a brand-new Office file from a sandboxed recipe
  (python-pptx / python-docx / openpyxl code).
- `tool_office_edit` — modify an existing Office file with a sandboxed recipe
  operating on the pre-loaded document object.

Supported formats: `.pptx`, `.pptm`, `.docx`, `.docm`, `.xlsx`, `.xlsm`.

**What to expect — NO vision gate**: the mount is **unconditional** when `True`
(text-only workflow: recipe code in → markdown inventory out). Recipe code is
sandboxed: **no `import` statements are allowed** — the libraries are
pre-injected; violations are rejected before execution.

**Integrity-verified saves**: every save passes a 3-layer validation (ZIP
integrity, strict XML parse, library re-open). If a save would corrupt the
file, it is automatically **rolled back** to the pre-save snapshot and the tool
returns a machine-readable failure code — the model fixes the recipe and
retries. First invocation auto-installs `python-pptx` / `python-docx` /
`openpyxl` via pipmaster — expect a slower first call.

**Recovering generated stuff**:
- Tools write into the discussion working directory (`tool_context["working_dir"]`,
  i.e. the discussion's `workspace_data_path`).
- Tool result keys: `workspace_file` (absolute path), `context_files` (list
  containing the path), `file_name` (sanitized).
- Files are auto-registered as artifacts by `_sync_tool_artifacts` (workspace
  snapshot diffing) — recover via `discussion.artefacts.get(...)` or open
  `workspace_file` directly on disk.
- The turn result's `"artefacts"` array also lists them.

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
   widgets, books, presentations instructions, and image-tag dispatch even when
   their own flags are `True`.
2. **Discovery ≠ exposure**: LCP libraries are AST-discovered at client
   construction, but the model only sees a library when the per-turn chat flag
   mounts it. A model claiming "I have no such tools" almost always means the
   flag wasn't passed (or vision was missing for `enable_computer_use`).
3. **Persistent mounts**: mounts live on `client.tools`
   (`LCPBinding.discovered_tools`) at client level — there is **no
   per-discussion tool registry and no unmount-after-turn cleanup**. Mounts are
   idempotent (`mount_tool_library_if_absent`); re-enabling a flag on later
   turns is a cheap no-op.
4. **Keyword-gated instructions**: `enable_books`, `enable_pptx_builder`, and
   `enable_forms` inject their instructions only when trigger keywords appear in
   the user message, even with the flag enabled. `enable_office_files`
   instructions are injected unconditionally when the flag is on.
5. **Negative-polarity flags**: `suppress_images` is inverted — `True` *disables*
   vision hydration rather than enabling a feature.
6. **Memory sub-flags require the master**: `enable_episodic_memory`,
   `enable_auto_dream`, and `enable_deep_memory_pulling` are inert when
   `enable_memory=False` or no manager resolves.
7. **Cancellation is sovereign**: `discussion.cancel_generation()` is checked at
   every round boundary and every streamed token; a cancelled turn persists
   partial content with `turn_status="cancelled"` and is resumable via
   `resume_turn()`.

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
    "Create a 5-slide presentation about photosynthesis",
    enable_pptx_builder=True,
)
# Recover the deck:
#   → result["artefacts"] lists the pptx artifact
#   → tool_pptx_build's workspace_file lives in the discussion working dir
#   → discussion.artefacts.get("presentation.pptx")
```

**Build Office documents (.docx / .xlsx / .pptx)**:
```python
result = discussion.chat(
    "Build a quarterly report: a .docx summary and an .xlsx KPI workbook",
    enable_office_files=True,
)
# Recover the files:
#   → result["artefacts"] lists the office artifacts
#   → tool_office_build's workspace_file lives in the discussion working dir
#   → discussion.artefacts.get("quarterly_report.docx")
```

**Desktop automation (vision model required)**:
```python
result = discussion.chat(
    "Open the browser and take a screenshot",
    enable_computer_use=True,   # silently skipped if the model lacks vision
)
# Screenshots: computer_use_screenshot.png in the working directory
```

**Privacy session (no persistence of conversation memory)**:
```python
discussion.chat(
    user_message,
    enable_episodic_memory=False,
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