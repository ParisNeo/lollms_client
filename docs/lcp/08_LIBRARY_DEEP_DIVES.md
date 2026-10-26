# 08 — Library Deep Dives

> **Purpose**: One section per major library under `default_tools/` — purpose,
> discovery seam, gate flag, tool inventory, internal architecture,
> result-dict contracts, error modes, and a worked example (the exact JSON the
> LLM sends and the exact dict that comes back). `word_tools` is the
> fully-verified exemplar; libraries marked **"surface-level — pending deeper
> scouting"** are documented from verified findings only and await a future
> hydration round. Prerequisites: [04_LIBRARY_CATALOG.md](./04_LIBRARY_CATALOG.md).

---

## 1. `word_tools` — the exemplar ✅ (fully verified)

- **Purpose**: Build, edit, read, and review Word documents in the workspace.
- **Seam**: B — `get_tools()` returns 4 hand-authored spec dicts.
- **Gate**: `enable_word` ([`../flags/enable_word.md`](../flags/enable_word.md)).
- **File**: `default_tools/word_tools/word_tools.py` + `description.yaml`.

### 1.1 Tool inventory

| Tool | Purpose | Output keys |
|---|---|---|
| `tool_build_word` | Build a complete Word document from structured content. | 5 keys |
| `tool_edit_word` | Edit an existing Word document (find/replace, structural edits). | 5 keys |
| `tool_read_word` | Read/extract content from a Word document. | 5 keys |
| `tool_word_review` | Review a document: comments, highlights, inventory. | **7 keys** incl. `comments`, `highlights`, `inventory` |

### 1.2 Internal architecture

**Recipe sandbox** (LLM-authored code runs hardened):

| Component | Role |
|---|---|
| `_FORBIDDEN_TOKENS` | Tokens that may never appear in a recipe. |
| `_IMPORT_PREFIX_WHITELIST` | Only `("docx.",)` imports allowed. |
| `_SAFE_STDLIB_MODULES` | Curated safe standard-library modules. |
| `_build_safe_builtins` | Constructs the restricted builtins table. |
| `_check_recipe` | Validates a recipe before execution (forbidden tokens, import whitelist). |
| `_exec_recipe` | Executes the checked recipe in the sandbox. |
| `_annotate_traceback` | Converts sandbox tracebacks into LLM-actionable error text. |
| `_MAX_RECIPE_CHARS` | Hard size cap on recipe source. |

**Workspace-only media anchoring**:

- `_candidate_working_dirs` — workspace as **sole root** (`Path(".")` fallback
  only when no workspace is set).
- `_resolve_media_path` — **rejects absolute paths and `..` traversal**.
- `media()` / `list_media()` / `resolve_path()` — the only sanctioned asset
  references.

**Transactional save with OOXML validation**:

- `_save_and_verify` — snapshot → save → validate → **rollback on failure**.
- `_validate_office_file` — three-layer OOXML structural validation.
- `OOXML_VALIDATORS` — per-format validator registry.
- `OFFICE_GUARD_FAILURE_CODE` / `OFFICE_GUARD_GUIDANCE` — failure code +
  LLM guidance so the model can self-correct.

**The real Word-comment engine** (verified):

```python
# OPC part creation (if absent):
Part(PackURI('/word/comments.xml'),
     'application/vnd...comments+xml',
     parse_xml('<w:comments %s/>' % nsdecls('w')),
     doc.part.package)
doc.part.relate_to(part, RT.COMMENTS)

# Run-anchored comment:
add_comment(doc, paragraph, text, author="Assistant",
            run=resolve_comment_runs(doc, 'some text'))
```

- `_ensure_comments_part` — creates the comments OPC part when absent.
- `add_comment(doc, paragraph, text, author="Assistant", run=None)` —
  run-anchored via `commentRangeStart`/`commentRangeEnd`, **collision-safe
  ids** (`max(ids)+1`), **UTC timestamps**, `OxmlElement`-built text.
- `_list_comments` / `_list_highlights` — enumerate existing
  comments/highlights.
- `_resolve_run_for_text` / `resolve_comment_runs` — locate the runs to anchor
  a comment against.
- **Historical note**: an older implementation fell back to inline
  `[COMMENT by X: ...]` text; the current engine writes real Word comments.

**`init_tools_library(config=None)`** — never raises; pipmaster soft-fail for
dependencies.

### 1.3 Result-dict contract

Canonical 5-key shape (per tool, with tool-specific payload keys):

```python
{
    "success": True,
    "output": "Document built successfully: report.docx",
    "file_name": "report.docx",          # workspace-relative artefact path
    "context_files": ["report.docx"],    # resolved to absolute existing paths
    # + tool-specific keys (e.g. tool_word_review adds comments/highlights/inventory)
}
```

`tool_word_review` returns **7 keys** including `comments`, `highlights`,
`inventory`.

### 1.4 Error modes + recovery

| Error | Cause | Recovery |
|---|---|---|
| `success=False` + `error` string | Tool-level failure (bad params, missing file, validation failure) | The `error` string is surfaced to the LLM (500-style status) — the model self-corrects next round. |
| `OFFICE_GUARD_FAILURE_CODE` | `_validate_office_file` rejected the saved file | `OFFICE_GUARD_GUIDANCE` tells the model how to fix; `_save_and_verify` rolled back to the snapshot. |
| Sandbox rejection | Recipe hit `_FORBIDDEN_TOKENS` / non-whitelisted import | `_check_recipe` rejects pre-execution; `_annotate_traceback` explains. |
| Path rejection | Absolute path or `..` in a media reference | `_resolve_media_path` refuses; use `media()`/`resolve_path()`. |

### 1.5 Worked example

**LLM sends** (tool call params):

```json
{
    "title": "Quarterly Report",
    "content": "Revenue grew 12% year over year..."
}
```

**Tool returns**:

```json
{
    "success": true,
    "output": "Document built successfully: quarterly_report.docx",
    "file_name": "quarterly_report.docx",
    "context_files": ["quarterly_report.docx"]
}
```

The binding normalizes (`context_files` → absolute existing paths) and the
chat mixin attaches the artefact as a USER-role message for the next round.

---

## 2. `excel_tools` — surface-level, pending deeper scouting

- **Purpose**: Build, edit, and read Excel workbooks; markdown→xlsx pipeline.
- **Seam**: B — `get_tools()` spec dicts.
- **Gate**: `enable_excel` ([`../flags/enable_excel.md`](../flags/enable_excel.md)).
- **Tools**: the xlsx build/edit/read family — **4 tools** in the chat-side
  office admission set (`_OFFICE_XLSX_TOOL_NAMES`), including
  `tool_excel_from_markdown` (markdown→xlsx pipeline).
- **Architecture**: recipe sandbox (`_IMPORT_PREFIX_WHITELIST` anchored on the
  xlsx library), workspace anchoring, transactional `_save_and_verify` +
  OOXML validation — the shared office doctrine (see
  [04_LIBRARY_CATALOG.md](./04_LIBRARY_CATALOG.md) §0).
- **Worked example (illustrative)**:

```json
// LLM sends
{ "markdown": "| Name | Value |\n|---|---|\n| a | 1 |", "file_name": "table.xlsx" }

// Tool returns
{ "success": true, "output": "Workbook written: table.xlsx",
  "file_name": "table.xlsx", "context_files": ["table.xlsx"] }
```

*(Exact parameter names pending hydration of `excel_tools.py`.)*

---

## 3. `powerpoint_tools` — surface-level, pending deeper scouting

- **Purpose**: Build, edit, and read PowerPoint presentations.
- **Seam**: B — `get_tools()` spec dicts.
- **Gate**: `enable_powerpoint` ([`../flags/enable_powerpoint.md`](../flags/enable_powerpoint.md)).
- **Tools**: the pptx build/edit/read family — **5 tools** in the chat-side
  office admission set (`_OFFICE_PPTX_TOOL_NAMES`).
- **Architecture**: shared office doctrine (sandbox, anchoring, transactional
  save). *(Exact tool names/params pending hydration.)*

---

## 4. `office_files_builder` — surface-level, pending deeper scouting

- **Purpose**: Building office files from structured specs (the "spec-dict
  builder" library).
- **Seam**: B — `get_tools()` spec dicts.
- **Gate**: office-family any-flag rule (see
  [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md)).
- **Architecture**: shared office doctrine. *(Pending hydration.)*

---

## 5. `document_editor` — surface-level, pending deeper scouting

- **Purpose**: Document-agnostic inspection and editing across formats.
- **Seam**: A — AST-parsed top-level `tool_*` functions with docstrings (the
  canonical Seam-A example).
- **Gate**: office-family any-flag rule (no per-software gate).
- **Tools** (per verified findings): `tool_find_text_in_document`,
  `tool_read_document_comments`, `tool_get_document_info`, plus precise docx
  highlighting (`_highlight_docx_text` with run-splitting) and real PDF
  annotations (`page.add_text_annot` / `set_info`).
- **Architecture**: cross-format inspection/editing; workspace anchoring.
  *(Full inventory pending hydration.)*

---

## 6. Code-execution family — surface-level, pending deeper scouting

### 6.1 `execute_python`
- **Gate**: `enable_code_execution` ([`../flags/enable_code_execution.md`](../flags/enable_code_execution.md)).
- **Purpose**: run Python code in the workspace.

### 6.2 `execute_python_data_query`
- **Purpose**: query data files with Python.

### 6.3 `semantic_data_engineer`
- **Gate**: `enable_data_tools` ([`../flags/enable_data_tools.md`](../flags/enable_data_tools.md)).
- **Purpose**: semantic data engineering over workspace data; **dynamically
  mountable** based on workspace context (data files detected) — per the
  tools README.

---

## 7. `computer_use` — surface-level, pending deeper scouting

- **Gate**: `enable_computer_use` ([`../flags/enable_computer_use.md`](../flags/enable_computer_use.md)).
- **Purpose**: screen interaction (screenshots, input automation).
- **Verified detail**: screenshot backend injection via `cu._BACKEND` /
  `_BACKEND_ERROR` (test surface `test/test_computer_use_tools.py`).

---

## 8. `as_is_document_tools` — surface-level, pending deeper scouting

- **Purpose**: importing/processing documents "as-is" (verified via the as-is
  import test surface `test/test_as_is_import_and_tools.py`).

---

## 9. `models_tools` — surface-level, pending deeper scouting

- **Gate**: `list_available_models` ([`../flags/list_available_models.md`](../flags/list_available_models.md)).
- **Purpose**: listing/inspecting available models.

---

## 10. `git_manager` / `system_shell` — surface-level, pending deeper scouting

- **Purpose**: repository operations and shell command execution.
- **Caution**: highest-privilege libraries; strictly flag-gated.

---

## 11. `ros_turtlebot` — surface-level, pending deeper scouting

- **Purpose**: ROS/TurtleBot robotics control.

---

## 12. Key invariants

- `word_tools` is the reference implementation of every office-doctrine
  pattern; when in doubt about how a library should behave, read it first.
- The office doctrine (recipe sandbox, workspace-only anchoring, transactional
  `_save_and_verify` + `_validate_office_file`) is shared by `word_tools`,
  `excel_tools`, `powerpoint_tools`, `office_files_builder`.
- `tool_word_review` returns a **7-key** dict (`comments`, `highlights`,
  `inventory` among them); the other three `word_tools` tools return 5-key
  dicts.
- The Word-comment engine writes **real Word comments** (`_ensure_comments_part`
  OPC part + `relate_to(RT.COMMENTS)`, run-anchored, collision-safe ids,
  UTC timestamps) — the inline `[COMMENT by X: ...]` fallback is historical.
- Sections marked "surface-level — pending deeper scouting" contain only
  verified findings; no details are invented. Future hydration rounds should
  extend them.