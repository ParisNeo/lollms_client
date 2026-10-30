# 10 — LCP Toolset Authoring Guide (Reference)

> **Purpose**: The hands-on reference for **building** LCP toolsets — the
> `default_tools/<name>/<name>.py` libraries the LCP binding discovers, gates,
> mounts, and executes. While documents 01–09 explain how the LCP *works*,
> this document teaches how to **author a library that works with it**.
> Audience: library authors (human or LLM). Prerequisites: skim
> [00_OVERVIEW.md](./00_OVERVIEW.md); consult
> [09_STUDY_GUIDE.md](./09_STUDY_GUIDE.md) §(e) for the condensed tutorial.

---

## 1. What an LCP toolset is

An LCP toolset (library) is a **plain Python module** that the LCP binding
imports in-process. It has no server, no network, no framework — just:

```
src/lollms_client/tools_bindings/lcp/default_tools/
└── my_library/
    ├── my_library.py        ← the library module (REQUIRED, name must match folder)
    └── description.yaml     ← metadata + load-flag declaration (RECOMMENDED)
```

**Layout law** (enforced by `_iter_library_files`):

| Layout | Status |
|---|---|
| `default_tools/<name>/<name>.py` | ✅ Canonical (dir + primary file) |
| `default_tools/<name>.py` | ⚠️ Legacy flat (supported, not preferred) |

Sibling `.py` files inside the library folder are also scanned (as fallbacks
when the primary file is missing, and as siblings during mounts), but the
canonical pattern is **one folder, one primary file named after the folder**.

**The philosophy** (from the binding's own docstring):

- Tools are **agnostic Python scripts** — by default they do NOT know about
  discussions, artifacts, or clients. A tool that *opts in* by declaring the
  optional `lc=None` parameter DOES receive the live client (see §2 and the
  "Using the live client" section below).
- Tools operate on **files in the current working directory** (the workspace).
- The binding handles environment setup (CWD, sync) transparently.
- Host-configurable parameters are **invisible to the LLM** and enforced by
  the host app.

---

## 2. The two discovery seams — pick exactly one

Your library describes its tools through **exactly one** of two seams. The
binding detects which seam you use by what you provide.

### Seam A — AST-parsed `tool_*` functions (automatic schemas)

Write **top-level functions named `tool_*`** with Google-style docstrings. The
binding's `_parse_tool_via_ast` derives the schema automatically.

```python
def tool_find_text_in_document(file_path: str, query: str, page: int = 1):
    """Find occurrences of a text query inside a document.

    Args:
        file_path (str): workspace-relative path of the document.
        query (str): the text to search for.
        page (int, optional): 1-based page number to start from.
    """
    ...
    return {"success": True, "output": f"Found 3 matches", "matches": [...]}
```

**Schema derivation rules** (memorize these):

| Rule | Behavior |
|---|---|
| Tool name | Any top-level function starting with `tool_` |
| Description | First paragraph of the docstring |
| Param descriptions | Docstring lines matching `param (type): description` |
| `int` annotation | → `integer` |
| `bool` annotation | → `boolean` |
| `list` annotation | → `array` |
| `dict` annotation | → `object` |
| anything else | → `string` |
| No default value | → `required: true` |
| Has default value | → optional (default captured into the schema) |
| `_`-prefixed helpers | **Never** become tools — excluded |
| `args`/`kwargs`/`discussion_instance`/`lollms_client_instance`/`tool_context`/`user_interaction_handler` | Excluded from the schema (host-injected, LLM-invisible) |
| `lollms_client_instance=None` | Also **stripped** from the advertised schema; at dispatch the binding injects the live `LollmsClient` into it — the LLM never sees it (see "Using the live client" below) |
| `discussion_instance=None` | Stripped from the schema; at dispatch the binding injects the active `LollmsDiscussion` (message history, attached files, discussion metadata) |

**Best for**: libraries whose tools are plain functions with simple
parameters.

### Seam B — `get_tools()` hand-authored spec dicts (full control)

Define a module-level `get_tools()` returning explicit spec dicts. This is
what `word_tools`, `excel_tools`, `powerpoint_tools`, and
`office_files_builder` use.

```python
def get_tools():
    return [
        {
            "name": "tool_build_report",          # MUST match the callable's name
            "description": "Build a report document from structured content.",
            "parameters": [
                {"name": "title", "type": "string", "optional": False,
                 "description": "Report title."},
                {"name": "sections", "type": "array", "optional": False,
                 "description": "List of section dicts with heading/body."},
            ],
            "output": [
                {"name": "success", "type": "boolean", "description": "Whether the build succeeded."},
                {"name": "file_name", "type": "string", "description": "Workspace-relative artefact path."},
            ],
            "callable": tool_build_report,        # the actual function object
            "_source": "my_library",
            "_source_file": str(Path(__file__).resolve()),
        },
    ]
```

**Seam-B invariants** (hard rules):

1. `"name"` **MUST** match the name of its `"callable"` tool function.
2. `output` entries **MUST** match the tool's actual return-dict keys.
3. The callable must still be a `tool_*`-named function (mount verification
   via `_extract_tool_names_from_file` walks the AST for `tool_*` names — a
   library whose expected tool-name set is empty cannot mount and produces
   `⚠️ Library 'X' contains no tool definitions`).

**Best for**: libraries that want rich, hand-tuned parameter/output
documentation and exact optional flags.

### Which seam should I choose?

| Criterion | Seam A (AST) | Seam B (`get_tools`) |
|---|---|---|
| Setup effort | Minimal — docstrings are the schema | More — specs are hand-written |
| Schema quality | Derived; good with disciplined docstrings | Exact; you control everything |
| Output documentation | Not captured | Fully documented |
| Reference implementations | `document_editor` | `word_tools` (the exemplar) |

---

## Using the live client (LLM, TTI, TTV, TTS/TTM, discussion)

A toolset function can opt into the **running `LollmsClient`** by declaring the
optional parameter `lollms_client_instance=None`. At dispatch the binding
inspects the signature; if the parameter exists, it injects the live client.
The confirmed reserved names (LLM-invisible, host-injected) are:

| Reserved parameter | Injected object |
|---|---|
| `lollms_client_instance` | The active `LollmsClient` |
| `discussion_instance` | The active `LollmsDiscussion` (history, files) |
| `tool_context` | Host tool context object |
| `user_interaction_handler` | Host confirmation/Q&A handler |

The client parameter:

- is **optional** — always default it (`lollms_client_instance=None`);
- is **injected by the system**, never supplied by the LLM — it does not appear
  in `params` nor in the advertised JSON schema;
- is the **same lifecycle-managed instance** the host is using.

The same rule applies to `discussion_instance=None`: declare it to receive the
current discussion, then read `discussion_instance.messages` (history) and the
discussion's file/artefact surface for attached files. Omit it and the tool
simply receives nothing — it still works as a plain function.

> ⚠️ **The one law: never re-instantiate.** Inside a tool you must NOT call
> `LollmsClient(...)`, `LollmsLLMBindingManager().create_binding(...)`, or any
> binding constructor. A rogue instance duplicates model connections, wastes
> VRAM / API quota, and bypasses lifecycle & cancellation management.
> **Borrow, never build.**

### What the injected client gives you

With `lc = lollms_client_instance` inside the function body:

| Resource | Access pattern | Guard |
|---|---|---|
| LLM text generation | `lc.llm_binding.generate(prompt, n_predict=…)` | present when injected |
| Multimodal LLM | `lc.llm_binding.generate_with_images(prompt, images=…)` | model must support vision |
| TTI (image) | `lc.tti_binding.generate_image(prompt=…, negative_prompt=…)` | `if lc and getattr(lc, "tti_binding", None)` |
| TTV (video) | `lc.ttv_binding.generate_video(prompt=…)` | same guard |
| TTS (speech) | `lc.tts_binding.synthesize(text=…)` | same guard |
| TTM (music) | `lc.ttm_binding.generate_music(prompt=…)` | same guard |
| Discussion history | `discussion_instance.messages` | `if discussion_instance is not None` |
| Attached files | `discussion_instance` file helpers / artefact surface | same guard |

### Worked mini-example — styled image generation

```python
STYLES = {
    "photoreal": "ultra realistic photograph, 85mm lens, natural lighting",
    "anime":     "anime style, cel shaded, vibrant colors, clean lineart",
    "isometric": "isometric 3d render, low poly, soft studio light",
}

def tool_generate_image(lollms_client_instance=None, subject: str = "", style: str = "photoreal"):
    """Generate an image of `subject` in a named style.

    Args:
        subject (str): What to depict.
        style (str): One of: photoreal, anime, isometric.
    """
    lc = lollms_client_instance  # host-injected live LollmsClient (or None)
    try:
        if lc is None or getattr(lc, "tti_binding", None) is None:
            return {"success": False, "error": "No TTI binding on the client."}
        if not subject:
            return {"success": False, "error": "Parameter 'subject' is required."}
        scaffold = STYLES.get(style, STYLES["photoreal"])
        result = lc.tti_binding.generate_image(prompt=f"{scaffold}, {subject}")
        out = {"success": True, "output": f"Generated '{subject}' ({style})."}
        if isinstance(result, str):  # binding returned a saved path
            out["file_name"] = result
            out["context_files"] = [result]
        return out
    except Exception as ex:
        return {"success": False, "error": f"Image generation failed: {ex}"}
```

LLM sub-generation is the same pattern:

```python
def tool_summarize(lollms_client_instance=None, discussion_instance=None, text: str = ""):
    """Summarize a text with the host's LLM.

    Args:
        text (str): Text to summarize. If empty, summarizes the last discussion exchange.
    """
    lc = lollms_client_instance  # host-injected live LollmsClient (or None)
    if lc is None or getattr(lc, "llm_binding", None) is None:
        return {"success": False, "error": "No LLM binding available."}
    if not text and discussion_instance is not None:
        # fall back to the current discussion's messages
        text = "\n".join(m.content for m in discussion_instance.messages[-4:])
    if not text:
        return {"success": False, "error": "Nothing to summarize."}
    return {"success": True, "output": lc.llm_binding.generate(
        f"Summarize in 3 bullets:\n\n{text}", n_predict=512)}
```

Seam-B `LcpTool` subclasses receive the same injection by declaring
`lollms_client_instance=None` (and/or `discussion_instance=None`) on
`tool_function`.

> 📖 Full contract — per-modality guards, TTV surface, discussion-history
> access patterns, the "never re-instantiate" rationale, and a complete
> `image_studio` toolset with host-overridable style presets — see
> [`../lcp_tool_resource_access.md`](../lcp_tool_resource_access.md).

---

## 3. `init_tools_library(config=None)` — the initialization contract

If your library needs setup (dependency checks, workspace resolution), define:

```python
def init_tools_library(config=None):
    """Never raises: dependency installation is soft-failed via pipmaster."""
    try:
        import pipmaster as pm
        pm.install_all()          # reads requirements from the module docstring
    except Exception:
        pass                       # soft-fail: report through tool results instead
```

**The never-raise law**: `init_tools_library` MUST NOT raise. ANY raise —
from import OR from init — is caught by the binding and reported as:

```
Tool initialization failed for '<library_key>': <e>
```

and your library is **skipped entirely**. Dependency problems must surface
through tool result dicts (`success=False` + a helpful `error` string), never
through loader crashes. `word_tools.py` is the reference implementation of
this doctrine.

The `config` argument receives the host configuration (a dict) — use it to
read host-injected values like `confirm_handler`, `user_interaction_handler`,
or workspace settings.

---

## 4. Gating — making your library opt-in

### 4.1 The `TOOL_LIBRARY_LOAD_FLAG` gate

Declare a load flag at module top level:

```python
TOOL_LIBRARY_LOAD_FLAG = "enable_my_capability"
```

Resolution (fail-closed):

- No declared flag → the gate **passes** unconditionally.
- Declared flag → admitted only when the flag is truthy in
  `host_tool_configs['<library_stem>']['<flag>']` or the binding's
  constructor kwargs. String values `"true"`, `"1"`, `"yes"`, `"on"` count as
  truthy; everything else (missing, `None`, `False`) means **no**.

Also declare it in `description.yaml` so hosts can discover it.

### 4.2 The office-family session gate

If your library stem is in `_OFFICE_LIBRARY_STEMS`
(`as_is_document_tools`, `document_editor`, `pptx_builder`, `docx_mastery`,
`pptx_mastery`, `xlsx_mastery`, `powerpoint_tools`, `word_tools`,
`excel_tools`), it is additionally gated on **session flags**:

- `powerpoint_tools` → requires `enable_powerpoint`
- `word_tools` → requires `enable_word`
- `excel_tools` → requires `enable_excel`
- other office stems → admitted when **ANY** of `enable_office_files` /
  `enable_powerpoint` / `enable_word` / `enable_excel` is truthy

Absent flags are fail-closed. If you are adding a **new** office-family
library, add its stem to `_OFFICE_LIBRARY_STEMS` (and `_PER_SOFTWARE_GATES`
if it needs its own flag) in the binding source.

### 4.3 Documenting your flag

Add `docs/flags/<flag>.md` and link it from
[03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md) §5.

---

## 5. The result-dict contract

Every tool callable returns a **dict**. The binding normalizes it. The five
anchor keys:

| Key | Type | Meaning |
|---|---|---|
| `success` | `bool` | Whether the tool accomplished its purpose. |
| `error` | `str` | Human/LLM-readable error when `success=False`. |
| `output` | `str` | Primary textual result shown to the LLM. |
| `file_name` | `str` | Workspace-relative path of a produced artefact. |
| `context_files` | `list[str]` | Files whose contents feed the next round (resolved to absolute existing paths by the binding). |

**Normalization behavior you can rely on**:

1. `success=False` → 500-style status; the `error` string is surfaced to the
   LLM so it can self-correct next round. Tool failures are *information*,
   not crashes.
2. Missing `output` → auto-filled from payload keys (`result`, `data`,
   `response`, `papers`, `content`, `text`), then from the whole dict minus
   meta keys, then a generic success string. The LLM always gets readable
   output.
3. `context_files` → resolved against the current working directory to
   absolute **existing** file paths; missing entries are dropped with a debug
   log.
4. Extra keys ride along: `image_filename`, `image_b64`, `images`,
   `image_media_types`, `prompt_injection` are passed through to the final
   result. Any other custom keys stay in the dict for downstream consumers.

**Failure signals the binding treats as failure** (even without explicit
`success=False`): `status` in `("error", "failure", "failed")`, a non-zero
`return_code`, or a truthy `error` key (unless `success` is explicitly
`True`).

**Canonical success shape**:

```python
{
    "success": True,
    "output": "Document built successfully: report.docx",
    "file_name": "report.docx",
    "context_files": ["report.docx"],
}
```

**Canonical failure shape**:

```python
{
    "success": False,
    "error": "File 'report.docx' not found in workspace. Use list_media() to see available files.",
}
```

---

## 6. The workspace rule (asset anchoring)

Tools run with the **workspace as CWD for the duration of the call**; the
process CWD is transactionally restored afterward (even on crash — locked by
`test/test_lcp_cwd_management.py`). Therefore:

- **Never** call `os.chdir` permanently.
- **Never** accept bare absolute paths or `..` traversal from the LLM.
- Resolve every asset reference through sanctioned resolvers:

```python
def _candidate_working_dirs():
    """Workspace as sole root; Path('.') fallback only when no workspace set."""
    ...

def resolve_path(relative_path: str) -> Path:
    """Resolve a workspace-relative reference; reject absolute paths and '..'."""
    ...
```

The `word_tools` pattern (`_candidate_working_dirs` workspace-sole-root +
`_resolve_media_path` rejecting absolute/`..` + public `media()` /
`list_media()` / `resolve_path()` helpers) is the reference implementation.
Bare relative paths resolve against the workspace, never the process folder —
the historical `FileNotFoundError: bare relative image` incident is what this
doctrine prevents.

---

## 7. Optional doctrines for powerful libraries

### 7.1 The recipe sandbox (if you execute LLM-authored code)

If your tools accept Python code from the LLM, execute it hardened:

| Component | Role |
|---|---|
| `_FORBIDDEN_TOKENS` | Tokens that may never appear in a recipe. |
| `_IMPORT_PREFIX_WHITELIST` | Only whitelisted import prefixes (e.g. `("docx.",)`). |
| `_SAFE_STDLIB_MODULES` | Curated safe standard-library modules. |
| `_build_safe_builtins` | Constructs the restricted builtins table. |
| `_check_recipe` | Validates a recipe pre-execution (tokens + imports). |
| `_exec_recipe` | Executes the checked recipe in the sandbox. |
| `_annotate_traceback` | Converts sandbox tracebacks into LLM-actionable error text. |
| `_MAX_RECIPE_CHARS` | Hard size cap on recipe source. |

### 7.2 Transactional save with OOXML validation (if you write office files)

- `_save_and_verify` — snapshot → save → validate → **rollback on failure**.
- `_validate_office_file` — three-layer OOXML structural validation.
- `OOXML_VALIDATORS` — per-format validator registry.
- `OFFICE_GUARD_FAILURE_CODE` / `OFFICE_GUARD_GUIDANCE` — failure code +
  LLM guidance so the model can self-correct.

### 7.3 Confirmation handlers (for dangerous operations)

If your library performs irreversible actions, honor the host-injected
confirmation seam:

```python
_CONFIRM_HANDLER = None

def set_confirm_handler(handler):
    """Called by the binding when a confirm_handler is registered."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler
```

The binding propagates `confirm_handler` / `user_interaction_handler` to your
module via `set_confirm_handler` / `set_user_interaction_handler` if defined,
and passes them into `init_tools_library(config)`. Tools may also declare a
`user_interaction_handler` parameter — the binding injects it at call time
(LLM-invisible).

---

## 8. Dynamic prompt functions (optional)

A tool may have a companion `<tool_name>_prompt` function whose return value
**replaces** the tool's advertised description at prompt-build time:

```python
def tool_list_media_files():
    """List media files currently available in the workspace."""
    ...

def tool_list_media_files_prompt():
    """Dynamic description: reflects the live workspace state."""
    return f"List media files. {len(_scan_media())} files currently available."
```

The binding detects `tool_<base>_prompt` in the module AST, marks the tool
with `_has_dynamic_prompt`, and calls the function when building chat tool
specs (`to_chat_tool_specs`) and on `discover_tools`. Use this when a static
description would mislead (e.g. tool behavior depends on workspace state).

---

## 9. Worked example — a complete minimal library

`default_tools/weather_exporter/weather_exporter.py`:

```python
"""Weather data exporter: fetches nothing, formats workspace CSV data.

Requirements: none (stdlib only).
"""
from pathlib import Path
import csv

TOOL_LIBRARY_LOAD_FLAG = "enable_data_tools"

_WORKSPACE = Path(".")


def init_tools_library(config=None):
    """Never raises; nothing to initialize."""
    pass


def _resolve_workspace_path(relative_path: str) -> Path:
    cleaned = Path(str(relative_path).strip())
    if cleaned.is_absolute() or ".." in cleaned.parts:
        raise ValueError(
            "Absolute paths and '..' are forbidden. Use a workspace-relative path."
        )
    return (_WORKSPACE / cleaned).resolve()


def tool_export_summary(csv_file: str, output_name: str = "summary.txt"):
    """Summarize a workspace CSV file into a text report.

    Args:
        csv_file (str): workspace-relative path of the source CSV.
        output_name (str, optional): workspace-relative name of the output file.
    """
    try:
        src = _resolve_workspace_path(csv_file)
        if not src.is_file():
            return {
                "success": False,
                "error": f"CSV '{csv_file}' not found in the workspace.",
            }

        with src.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return {"success": False, "error": f"CSV '{csv_file}' is empty."}

        header, data_rows = rows[0], rows[1:]
        out_path = _resolve_workspace_path(output_name)
        out_path.write_text(
            f"Columns: {len(header)}\nRows: {len(data_rows)}\n",
            encoding="utf-8",
        )
        return {
            "success": True,
            "output": f"Summary written: {out_path.name}",
            "file_name": out_path.name,
            "context_files": [out_path.name],
        }
    except Exception as ex:
        return {"success": False, "error": f"Export failed: {ex}"}
```

`default_tools/weather_exporter/description.yaml`:

```yaml
name: weather_exporter
description: Summarizes workspace CSV data into text reports.
load_flag: enable_data_tools
tools:
  - tool_export_summary
```

**What the LLM sends** (tool call params):

```json
{ "csv_file": "temperatures.csv", "output_name": "temp_summary.txt" }
```

**What the tool returns**:

```json
{
  "success": true,
  "output": "Summary written: temp_summary.txt",
  "file_name": "temp_summary.txt",
  "context_files": ["temp_summary.txt"]
}
```

**What the binding returns** (after normalization): `context_files` resolved
to absolute existing paths; `status_code: 200`; the chat mixin attaches the
artefact as a **USER-role message** for the next round.

---

## 10. Testing your library — the three-stage checklist

1. **Discovery**: instantiate `LCPBinding` (with your flag enabled) and assert
   your tool appears in `binding.discover_tools()` with the expected schema
   (required params, types, description).
2. **Mount**: call `binding.mount_tool_library("<name>")` and verify the
   logged tool count matches your expected count; a mismatch means partial
   discovery (seam inconsistency or corruption).
3. **Execution**: call `binding.execute_tool("tool_x", {...})` with a worked
   example; assert the normalized dict (`success`, `output`,
   `status_code` 200/500) and that `context_files` resolve to existing
   absolute paths.

Also verify the negative paths: with the flag **disabled**, discovery must
exclude your library (fail-closed); with a missing input file, execution must
return `success=False` with an actionable `error` — never an unhandled
exception.

---

## 11. Diagnostics quick-reference (authoring view)

| Diagnostic | What it means for YOUR library | Fix |
|---|---|---|
| `AST parse failed for 'X.py': ... Attempting regex fallback.` | Your file has a `SyntaxError` (truncation/corruption). The regex fallback salvages crippled signatures — survival mode, not a feature. | Repair/rewrite the file. |
| `[LCP] Regex fallback extracted tool 'X' from 'Y.py'.` | Same — degraded specs are live (`_parsed_via_regex_fallback: True`). | Same. |
| `⚠️ Library 'X' contains no tool definitions` | Your expected tool-name set is empty: no top-level `tool_*` functions, or unparseable file. Seam-B libraries still need `tool_*` callables for mount verification. | Ensure callables are named `tool_*` at top level. |
| `Tool initialization failed for 'X'.` | Your module raised at import time OR `init_tools_library` raised. | Make init never-raise; check file integrity, then dependencies. |
| `[LCP Mount] Library 'X' is gated by TOOL_LIBRARY_LOAD_FLAG '...' which is not enabled.` | Your declared flag is off in host config. | Enable via `tools_binding_config['<flag>']=True` or `host_tool_configs['<stem>']['<flag>']=True`. |
| `[LCP Mount] Library 'X' is office-family gated and its session flag is not enabled.` | Your stem is office-family and session flags exclude it (fail-closed). | Set the admitting session flag (e.g. `enable_word`). |
| `FileNotFoundError: bare relative image` | An asset was referenced without your sanctioned resolver. | Route all assets through `media()`/`resolve_path()`; reject absolute/`..`. |
| `Personality binding 'null' yielded 0 chat tool specs` | Host misconfiguration — not your library. | Host must set `tools_binding_name="lcp"`. |

---

## 12. Authoring checklist (print this)

```
□  default_tools/<name>/<name>.py — folder + primary file, names match
□  description.yaml with metadata + load_flag
□  Exactly ONE discovery seam chosen:
     □  Seam A: top-level tool_* functions + Google-style docstrings, OR
     □  Seam B: get_tools() specs — name matches callable; output matches
        return keys; callables still named tool_*
□  init_tools_library(config=None) defined and NEVER raises
□  If the tool needs LLM/TTI/TTV/TTS/TTM/discussion access, declare optional
   lc=None and use the injected client — never construct one
□  TOOL_LIBRARY_LOAD_FLAG declared (if the library should be opt-in)
□  Flag documented in docs/flags/<flag>.md + linked from 03 §5
□  Result dicts use the 5 anchor keys (success/error/output/file_name/
   context_files); failures return success=False + actionable error
□  All asset references go through sanctioned resolvers; absolute paths
   and '..' rejected; no permanent os.chdir
□  Recipe sandbox adopted (if executing LLM-authored code)
□  Transactional _save_and_verify + validation (if writing office files)
□  set_confirm_handler honored (if performing irreversible actions)
□  Tested: discovery (schema correct) → mount (count matches) →
   execution (normalized dict + context_files resolve) → negative paths
   (flag off = excluded; bad input = success=False, no crash)
```

---

## 13. Key invariants

- A toolset is a plain Python module at
  `default_tools/<name>/<name>.py` (+ `description.yaml`); tools are agnostic
  scripts operating on workspace files.
- Exactly **one** discovery seam per library: AST `tool_*` docstring functions
  (Seam A) or `get_tools()` spec dicts (Seam B) — never both, never neither.
- Seam-B specs: `name` MUST match the callable; `output` MUST match actual
  return keys; callables MUST still be top-level `tool_*` functions.
- `init_tools_library(config=None)` MUST never raise — dependency failures
  surface through tool result dicts, not loader crashes.
- Gating is **fail-closed**: declare `TOOL_LIBRARY_LOAD_FLAG` for opt-in
  libraries; office-family stems are session-flag gated.
- Result dicts are open-keyed with five anchors (`success`, `error`,
  `output`, `file_name`, `context_files`); `success=False` surfaces the error
  to the LLM for self-correction.
- Workspace rule: sanctioned resolvers only; reject absolute paths and `..`;
  CWD changes are call-scoped and transactionally restored.
- Powerful libraries adopt the recipe sandbox and/or transactional save
  doctrines; dangerous operations honor `set_confirm_handler`.
- Test all three stages (discovery, mount, execution) plus the negative
  paths before shipping.