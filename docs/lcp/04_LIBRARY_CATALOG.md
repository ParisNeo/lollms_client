# 04 — Library Catalog (`default_tools/`)

> **Purpose**: One section per tool library: its discovery seam, gate flag(s),
> tool inventory, and notable internal architecture. This is the reference
> map; deep dives with worked examples live in
> [08_LIBRARY_DEEP_DIVES.md](./08_LIBRARY_DEEP_DIVES.md) (Step 3).
> `word_tools` is the fully-verified exemplar and the reference implementation
> for every pattern described here.

---

## 0. Cross-cutting architecture (the office doctrine)

The office-family libraries (`word_tools`, `excel_tools`, `powerpoint_tools`,
`office_files_builder`) share a security and robustness doctrine, all verified
in `word_tools.py`:

### Recipe sandbox

LLM-authored "recipe" code snippets are executed inside a hardened sandbox:

- `_FORBIDDEN_TOKENS` — tokens that may never appear in a recipe.
- `_IMPORT_PREFIX_WHITELIST` — only whitelisted import prefixes allowed
  (e.g. `("docx.",)` in `word_tools`).
- `_SAFE_STDLIB_MODULES` — a curated set of safe standard-library modules.
- `_build_safe_builtins` — constructs the restricted builtins table.
- `_check_recipe` — validates a recipe before execution (forbidden tokens,
  import whitelist).
- `_exec_recipe` — executes the checked recipe in the sandbox.
- `_annotate_traceback` — converts sandbox tracebacks into LLM-actionable
  error text.
- `_MAX_RECIPE_CHARS` — hard size cap on recipe source.

### Workspace-only media anchoring

- `_candidate_working_dirs` — computes candidate working directories with the
  **workspace as sole root** (a `Path(".")` fallback exists only when no
  workspace is set).
- `_resolve_media_path` — resolves a media reference; **rejects absolute paths
  and `..` traversal**.
- `media()` / `list_media()` / `resolve_path()` — the ONLY sanctioned ways for
  a library (and the LLM via recipes) to reference workspace assets.

**The workspace rule**: every workspace asset reference MUST use
`media()`/`resolve_path()`; bare filenames resolve against the workspace,
never the process folder; NO `os.chdir` — ever.

### Transactional save with OOXML validation

- `_save_and_verify` — snapshot → save → validate → **rollback on failure**.
- `_validate_office_file` — three-layer OOXML structural validation.
- `OOXML_VALIDATORS` — per-format validator registry.
- `OFFICE_GUARD_FAILURE_CODE` / `OFFICE_GUARD_GUIDANCE` — the failure code and
  LLM guidance emitted when validation fails, so the model can self-correct.

---

## 1. `word_tools` — the exemplar library ✅ (fully verified)

- **Seam**: B — `get_tools()` returns 4 hand-authored spec dicts.
- **Gates**: `enable_word` (per-software gate).
- **File**: `default_tools/word_tools/word_tools.py` + `description.yaml`.

### Tools

| Tool | Purpose |
|---|---|
| `tool_build_word` | Build a complete Word document from structured content. Returns a 5-key dict. |
| `tool_edit_word` | Edit an existing Word document (find/replace, structural edits). 5-key output. |
| `tool_read_word` | Read/extract content from a Word document. 5-key output. |
| `tool_word_review` | Review a document: returns a 7-key dict including `comments`, `highlights`, `inventory`. |

### Architecture highlights

- **`init_tools_library(config=None)`** — never raises; pipmaster soft-fail
  for dependencies.
- **Real Word-comment engine** (not inline text hacks):
  - `_ensure_comments_part` — creates the OPC comments part if absent:
    `Part(PackURI('/word/comments.xml'), 'application/vnd...comments+xml', parse_xml('<w:comments %s/>' % nsdecls('w')), doc.part.package)`
    then `doc.part.relate_to(part, RT.COMMENTS)`.
  - `add_comment(doc, paragraph, text, author="Assistant", run=None)` —
    run-anchored comments using `commentRangeStart`/`commentRangeEnd`,
    collision-safe ids (`max(ids)+1`), UTC timestamps, `OxmlElement`-built
    text.
  - `_list_comments`, `_list_highlights` — enumerate existing
    comments/highlights.
  - `_resolve_run_for_text` / `resolve_comment_runs` — locate the runs to
    anchor a comment against.
  - Historical note: an older implementation fell back to inline
    `[COMMENT by X: ...]` text; the current engine writes real Word comments
    via `_ensure_comments_part`.
- **Full office doctrine**: recipe sandbox, workspace anchoring,
  `_save_and_verify` transactional save — see §0.

---

## 2. `excel_tools`

- **Seam**: B — `get_tools()` spec dicts.
- **Gates**: `enable_excel`.
- **Tools** (per verified chat-side admission set): includes
  `tool_excel_from_markdown` (markdown→xlsx pipeline) and the xlsx
  build/edit/read family — 4 tools in the chat-side office admission set.
- **Architecture**: recipe sandbox (`_IMPORT_PREFIX_WHITELIST` anchored on the
  xlsx library), workspace anchoring, transactional save. Deep dive: Step 3.

## 3. `powerpoint_tools`

- **Seam**: B — `get_tools()` spec dicts.
- **Gates**: `enable_powerpoint`.
- **Tools**: the pptx build/edit/read family — 5 tools in the chat-side office
  admission set.
- **Architecture**: same office doctrine (sandbox, anchoring, transactional
  save). Deep dive: Step 3.

## 4. `office_files_builder`

- **Seam**: B — `get_tools()` spec dicts (the "spec-dict builder" library).
- **Gates**: office-family any-flag rule.
- **Purpose**: building office files from structured specs.
- **Architecture**: same office doctrine. Deep dive: Step 3.

## 5. `document_editor`

- **Seam**: A — AST-parsed top-level `tool_*` functions with docstrings (the
  canonical Seam-A example).
- **Gates**: office-family any-flag rule (no per-software gate).
- **Tools** (per verified findings): `tool_find_text_in_document`,
  `tool_read_document_comments`, `tool_get_document_info`, plus precise
  docx highlighting (`_highlight_docx_text` with run-splitting) and real PDF
  annotations (`page.add_text_annot` / `set_info`).
- **Architecture**: document-agnostic inspection/editing across formats.
  Deep dive: Step 3.

## 6. `computer_use`

- **Seam**: per verified findings — screenshot backend injection
  (`cu._BACKEND` / `_BACKEND_ERROR`), gated.
- **Gates**: `enable_computer_use` (see
  [`../flags/enable_computer_use.md`](../flags/enable_computer_use.md)).
- **Purpose**: screen interaction (screenshots, input automation).
- Deep dive: Step 3.

## 7. `execute_python` / `execute_python_data_query` / `semantic_data_engineer`

- **Seam**: code-execution family.
- **Gates**: `enable_code_execution` / `enable_data_tools` (see
  [`../flags/enable_code_execution.md`](../flags/enable_code_execution.md),
  [`../flags/enable_data_tools.md`](../flags/enable_data_tools.md)).
- **Purpose**: running Python code, querying data files, semantic data
  engineering over workspace data.
- **Note**: `semantic_data_engineer` is the library the tools README calls out
  as dynamically mountable based on workspace context (data files detected).
- Deep dive: Step 3.

## 8. `as_is_document_tools`

- **Purpose**: importing/processing documents "as-is" (verified via the
  as-is import test surface).
- Deep dive: Step 3.

## 9. `models_tools`

- **Gates**: `list_available_models` (see
  [`../flags/list_available_models.md`](../flags/list_available_models.md)).
- **Purpose**: listing/inspecting available models.
- Deep dive: Step 3.

## 10. `git_manager` / `system_shell`

- **Purpose**: repository operations and shell command execution.
- **Caution**: highest-privilege libraries; admission is strictly flag-gated.
- Deep dive: Step 3.

## 11. `ros_turtlebot`

- **Purpose**: ROS/TurtleBot robotics control.
- Deep dive: Step 3.

## 12. Other libraries on disk

The `default_tools/` folder also contains: `debug_toolset`, `dummy_test_tool`,
`execute_multi_file_service`, `execute_sparql_query`, `execute_sql_query`,
`file_organizer`, `hue`, `inspect_text`, `matter_controller`,
`tool_maker`, `user_interaction_demo`, `user_interactive_tools`, `vlm_query`,
`workspace_tools`. These follow the same two-seam discovery and the same
fail-closed gating; per-library deep dives are extended in Step 3 where
hydrated ground truth permits. *(Surface-level entries — pending deeper
scouting.)*

---

## 13. Key invariants

- Every library uses exactly one discovery seam: AST (`tool_*` + docstrings)
  or `get_tools()` spec dicts — never both, never neither.
- The office doctrine (recipe sandbox, workspace-only media anchoring,
  transactional `_save_and_verify` + `_validate_office_file`) is shared by
  `word_tools`, `excel_tools`, `powerpoint_tools`, `office_files_builder`.
- **Workspace rule**: asset references MUST use `media()`/`resolve_path()`;
  `_resolve_media_path` rejects absolute paths and `..`; no `os.chdir`.
- `word_tools`' comment engine writes **real Word comments**
  (`_ensure_comments_part` OPC part + `relate_to(RT.COMMENTS)`), with
  collision-safe ids and UTC timestamps — the inline `[COMMENT by X: ...]`
  fallback is historical.
- `init_tools_library(config=None)` in exemplar libraries never raises
  (pipmaster soft-fail).
- Libraries marked "surface-level — pending deeper scouting" are documented
  from verified findings only; no details are invented.