# 02 — The Tool Discovery Pipeline

> **Purpose**: The exhaustive anatomy of how tool libraries under
> `default_tools/` become registered, schema-described tools in the
> `discovered_tools` registry — including the two discovery seams, the AST
> parser, the regex fallback and its degraded semantics, the mount flow, and
> every diagnostic you can meet along the way.

---

## 1. The funnel at a glance

```
default_tools/  (folder on disk)
      │
      ▼
_iter_library_files()            yields default_tools/<name>/<name>.py
      │                          (+ legacy flat default_tools/<name>.py)
      ▼
_load_tool_file()                _lazy_import_tools_file() → persistent module
      │                          → module.init_tools_library(host_config)
      │                          ANY raise → "Tool initialization failed for '<key>'"
      ▼
admission gates                  _office_library_admitted() + _PER_SOFTWARE_GATES
      │                          + TOOL_LIBRARY_LOAD_FLAG   (FAIL-CLOSED)
      ▼
_parse_tool_via_ast()            top-level tool_* functions, docstrings,
      │                          annotations → schema
      │ SyntaxError
      ▼
_parse_tool_via_regex()          DEGRADED fallback: salvaged signatures,
      │                          auto "Parameter 'x'" descriptions,
      │                          _parsed_via_regex_fallback: True
      ▼
discovered_tools registry        name → {description, parameters, callable, ...}
      │
      ▼
is_library_mounted() /           _extract_tool_names_from_file() → tool-name set
mount_tool_library()             empty set → "⚠️ Library 'X' contains no tool definitions"
      │
      ▼
LLM-visible tool specs           consumed by personality _discover_tools()
                                and chat prompt builders
```

---

## 2. Step by step

### 2.1 Enumeration — `_iter_library_files(...)`

Yields candidate library files using the dir+primary pattern
(`default_tools/<name>/<name>.py`) plus legacy flat files. No import, no
gate — pure enumeration. See [01_LCP_BINDING_CORE.md](./01_LCP_BINDING_CORE.md) §5.

### 2.2 Loading — `_load_tool_file(...)`

Imports via `_lazy_import_tools_file` (persistent module
`lollms_client.tools_bindings.lcp.persistent_<library_key>`) and calls
`module.init_tools_library(host_config)` when present. Any raise →
`Tool initialization failed for '{library_key}': {e}` (~L562) and the library
is skipped. See [01_LCP_BINDING_CORE.md](./01_LCP_BINDING_CORE.md) §4.

### 2.3 Admission — fail-closed gates

Before a library's tools are extracted, the admission gates decide whether the
library participates at all. Gates are **fail-closed**: absent/false flags mean
NO admission. Full decision tree in
[03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md).

### 2.4 Schema extraction — `_parse_tool_via_ast(py_file_path)`

The primary extraction path. Contract:

- Walks the file's AST via `_iter_functions_ordered` (source-order iteration).
- Considers **only top-level functions whose name starts with `tool_`**.
  Underscore-prefixed helpers (`_ensure_comments_part`, `_exec_recipe`, ...)
  are excluded — they are library internals, not tools.
- **Description** = the first paragraph of the function's docstring.
- **Parameter descriptions** = lines matching the Google-style pattern
  `param (type): description` inside the docstring.
- **Type mapping** from Python annotations to schema types:

  | Python annotation | Schema type |
  |---|---|
  | `int` | `integer` |
  | `bool` | `boolean` |
  | `list` | `array` |
  | `dict` | `object` |
  | anything else | `string` |

- **Required vs optional**: a parameter with no default value → `required`;
  a parameter with a default → optional (the default is not captured into the
  schema by this seam).

**Example** (AST-seam style, `document_editor`-like):

```python
def tool_find_text_in_document(file_path: str, query: str, page: int = 1):
    """Find occurrences of a text query inside a document.

    Args:
        file_path (str): workspace-relative path of the document.
        query (str): the text to search for.
        page (int): 1-based page number to start from.
    """
```

yields a tool spec with description "Find occurrences of a text query inside a
document.", parameters `file_path` (string, required), `query` (string,
required), `page` (integer, optional).

### 2.5 The regex fallback — `_parse_tool_via_regex(...)`

**When it fires**: `_parse_tool_via_ast` hits a `SyntaxError` — i.e. the file
cannot be parsed as Python at all. The binding then logs:

```
AST parse failed for 'X.py': ... Attempting regex fallback.
```

**What it recovers**: function signatures are salvaged by pattern-matching
`def tool_*(...)` lines. On success it logs:

```
[LCP] Regex fallback extracted tool 'X' from 'Y.py'.
```

**Why its output is DEGRADED**:

- Parameter descriptions are auto-generated placeholders
  (`Parameter 'x'`) — no docstring-derived prose.
- No defaults are captured.
- The resulting specs are marked `_parsed_via_regex_fallback: True` so
  downstream consumers can detect degraded discovery.

**Historical case (verified)**: the `word_tools.py` truncation incident — a
truncated file produced a `SyntaxError`, the regex fallback salvaged crippled
signatures, and the system kept running with degraded tool descriptions. The
lesson: **regex-fallback extraction is a survival mode, not a feature**. If you
see the fallback log lines, the library file is broken and must be repaired.

### 2.6 Tool-name extraction for mounts — `_extract_tool_names_from_file(py_file)`

```python
_extract_tool_names_from_file(py_file) -> set
```

- Performs a bare `ast.parse` of the file and collects top-level `tool_*`
  function names.
- **Silent-except law**: on ANY parse failure it returns an empty set — no
  exception, no log from this helper itself.
- The empty set then surfaces upstream as:

```
⚠️ Library 'X' contains no tool definitions
```

**Meaning**: either the library genuinely defines no `tool_*` functions (e.g.
a pure `get_tools()`-spec library being probed by the wrong seam), or the file
is unparseable. Cross-check with the regex-fallback logs to distinguish.

### 2.7 Mount flow — `is_library_mounted(...)` / `mount_tool_library(...)`

Mounting = making a specific library's toolset active for the current
session/chat. The mount helpers use `_extract_tool_names_from_file` to compute
the **expected tool-name set** of a library, then verify/register the mount
against it. A library whose expected set is empty cannot be meaningfully
mounted (and produces the `⚠️ Library 'X' contains no tool definitions`
warning).

On the chat side, mounts surface as log lines of the family:

```
[ChatMixin] Mounted 'word_tools' toolset (4 tools)
```

(count mismatches vs the expected count are a first-class diagnostic — see
[05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md) §6.)

### 2.8 The registry — `discovered_tools`

The end product: a registry mapping tool names to their specs/callables,
accumulated across all admitted libraries. Downstream consumers:

- `LollmsPersonality._discover_tools(...)` (~L917 of `lollms_personality.py`)
  reads spec fields `callable`, `parameters`, `description`, `_source_file`.
- `lollms_tools_binding.py` (around line 159) consumes the same spec shapes.
- Chat prompt builders turn specs into LLM-visible doctrine.

---

## 3. The two discovery seams (central architectural fact)

Every LCP library uses exactly one of two seams to describe its tools:

### Seam A — AST-parsed libraries (`document_editor` style)

- Tools are **top-level `tool_*` functions** with Google-style docstrings.
- Schemas are **derived automatically** by `_parse_tool_via_ast` from
  annotations + docstrings.
- No JSON/spec duplication: the docstring IS the schema source.
- Best for: libraries whose tools are plain functions.

### Seam B — `get_tools()`-spec libraries (`word_tools` / `excel_tools` / `powerpoint_tools` / `office_files_builder` style)

- The library module defines `get_tools()` returning **explicit, hand-authored
  spec dicts**:

```python
{
    "name": "tool_build_word",                  # MUST match the callable's tool name
    "description": "Build a complete Word document from structured content.",
    "parameters": [
        {"name": "title", "type": "string", "optional": False,
         "description": "Document title."},
        ...
    ],
    "output": [
        {"name": "success", "type": "boolean", "description": "..."},
        ...
    ],
    "callable": tool_build_word,                # the actual function
    "_source": "word_tools",
    "_source_file": "<absolute path of word_tools.py>",
}
```

- Consumed at `lollms_personality.py` ~L917 (reads `callable`, `parameters`,
  `description`, `_source_file`; the `output` documentation rides along) and at
  `lollms_tools_binding.py:159`.
- Best for: libraries that want full control over the advertised schema
  (rich parameter/output documentation, exact optional flags).

**Seam invariants**:

- Spec `"name"` MUST match the name of its `"callable"` tool function.
- Spec `output` entries MUST match the tool's actual return-dict keys.
- A library declares its seam by what it provides: `tool_*` functions +
  docstrings (Seam A) or a `get_tools()` function (Seam B).

`word_tools.py` (the exemplar library, fully verified) uses Seam B: its
`get_tools()` returns exactly 4 specs — `tool_build_word`, `tool_edit_word`,
`tool_read_word`, `tool_word_review` — each with `name` matching its callable.

---

## 4. Diagnostics reference (discovery stage)

| Log / symptom | Emitted by | Root cause | Fix |
|---|---|---|---|
| `AST parse failed for 'X.py': ... Attempting regex fallback.` | `_parse_tool_via_ast` → fallback | `SyntaxError` in the library file (truncation, corruption) | Repair/rewrite the library file; fallback is survival mode only |
| `[LCP] Regex fallback extracted tool 'X' from 'Y.py'.` | `_parse_tool_via_regex` | Same as above — signatures salvaged, schemas crippled | Same as above |
| `⚠️ Library 'X' contains no tool definitions` | mount flow via `_extract_tool_names_from_file` | Empty tool-name set: no `tool_*` functions, or unparseable file (silent except) | Verify the library defines tools on the expected seam; check for corruption |
| `Tool initialization failed for 'X'.` | `_load_tool_file` (~L562) | Import raise OR `init_tools_library` raise | Check file integrity first (truncation saga), then dependencies, then init logic |
| `[ChatMixin] Registered tool_X via direct discovered_tools fallback` | chat mixin | Tool not found via mounted spec path; registered straight from `discovered_tools` | Informational — but indicates the mount path missed it; verify seam consistency |
| `Personality binding 'null' yielded 0 chat tool specs (discovered_tools count: 0)` | personality | Binding name is null AND `discovered_tools` is empty — defensive fallback engaged | Verify the tools binding is configured (`tools_binding_name="lcp"`) and discovery ran |

---

## 5. Key invariants

- Discovery is a **funnel**: enumerate → load → gate → extract → register →
  mount. Each stage has its own failure mode and its own diagnostic.
- There are exactly **two discovery seams**: AST-parsed `tool_*` docstring
  libraries (Seam A) and `get_tools()` hand-authored spec libraries (Seam B).
- `_parse_tool_via_ast` considers only **top-level `tool_*`** functions;
  `_`-prefixed helpers are never tools.
- Annotation→schema map: `int→integer`, `bool→boolean`, `list→array`,
  `dict→object`, else `string`. Non-default parameters are `required`.
- The regex fallback is **degraded by design**: auto `Parameter 'x'`
  descriptions, no defaults, `_parsed_via_regex_fallback: True`. Its presence
  in logs means the library file is broken.
- `_extract_tool_names_from_file` **silently** returns an empty set on parse
  failure; the visible symptom is `⚠️ Library 'X' contains no tool definitions`.
- Spec-shape contract (Seam B): `name` / `description` /
  `parameters[{name,type,optional?,description}]` /
  `output[{name,type,description}]` / `callable` / `_source` / `_source_file`;
  `name` MUST match the callable; `output` MUST match actual return keys.
- The `discovered_tools` registry is the single discovery product consumed by
  personality (~L917) and `lollms_tools_binding.py:159`.