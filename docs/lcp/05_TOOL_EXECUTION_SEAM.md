# 05 — The Tool Execution Seam (`execute_tool`)

> **Purpose**: What happens after the LLM emits a tool call — the
> `execute_tool` contract, the result-dict normalization rules, `context_files`
> resolution, the full diagnostics catalogue with root causes and fixes, and
> the end-to-end sequence diagram.

---

## 1. The signature

```python
execute_tool(tool_name: str, params: Dict[str, Any], discussion_instance=None, **kwargs) -> Dict[str, Any]
```

- `tool_name` — the registered tool name (e.g. `tool_build_word`). Must exist
  in the discovered/mounted toolset.
- `params` — the JSON parameters the LLM emitted, validated against the tool's
  advertised schema.
- `discussion_instance` — the live `lollms_discussion` object, forwarded so
  tools can resolve the workspace (WCD) and attach context.
- `**kwargs` — passthrough for chat-side concerns (e.g. CWD management — the
  binding sets and restores the process CWD around dispatch, verified by
  `test/test_lcp_cwd_management.py`).

---

## 2. The result-dict contract

Every library callable returns a dict. The canonical keys:

| Key | Type | Meaning |
|---|---|---|
| `success` | `bool` | Whether the tool accomplished its purpose. |
| `error` | `str` | Human/LLM-readable error description when `success=False`. |
| `output` | `str` | The primary textual result shown to the LLM. |
| `file_name` | `str` | Workspace-relative path of a produced artefact (e.g. the built `.docx`). |
| `context_files` | `list[str]` | Paths of files whose contents should feed the next round. |

Libraries may return additional keys (e.g. `tool_word_review` returns a 7-key
dict including `comments`, `highlights`, `inventory`). The contract is
open-keyed but the five above are the normalization anchors.

---

## 3. Normalization rules (applied by the binding)

1. **`success=False` ⇒ HTTP-500-style status + error surfaced.** The result
   carries a 500-equivalent `status_code` and the `error` string is presented
   to the LLM so it can self-correct on the next round. Tool failures are
   *information*, not crashes.
2. **Missing `output` ⇒ auto-filled.** If a library returns a dict without
   `output`, the binding fills it from the dict's payload keys, or from the
   whole dict — the LLM always receives readable output.
3. **`context_files` ⇒ resolved to absolute existing paths.** Relative
   entries are resolved against the workspace; the resolved list feeds the
   chat mixin's artefact/attachment pipeline (USER-role attachment only — see
   below).

---

## 4. `context_files` and the artefact pipeline

`context_files` is the bridge from a tool result into the conversation:

```
tool returns context_files: ["report.docx"]
        │
        ▼
binding resolves to absolute existing paths (workspace-anchored)
        │
        ▼
chat mixin attaches the artefacts to the next round
        │
        ▼
attachment is a USER-role message (NEVER system-role)
```

**The USER-role-only invariant**: generated media and artefacts are attached
as USER-role messages. Attaching them to system-role messages produces the
provider 400 error `"System message cannot contain images"` — a real incident
whose fix established this law. Full chat-side mechanics in
[06_CHAT_MIXIN_INTEGRATION.md](./06_CHAT_MIXIN_INTEGRATION.md) (Step 2).

---

## 5. CWD management (verified by tests)

`test/test_lcp_cwd_management.py` locks in the CWD discipline:

- `test_lcp_dispatch_sets_and_restores_cwd` — dispatching through the binding
  sets the CWD (to the workspace) and **restores** it afterward.
- `test_lcp_dispatch_restores_cwd_on_crash` — the CWD is restored **even when
  the tool crashes** (exception-safe restore).
- `test_direct_callable_path_sets_and_restores_cwd` — the same discipline
  applies on the direct-callable path (personality-mounted callables), not
  just binding dispatch.

Net law: tools run with the workspace as CWD for the duration of the call;
the process CWD is transactionally restored no matter what.

---

## 6. Diagnostics catalogue (execution stage)

| Diagnostic | Meaning | Root cause | Fix |
|---|---|---|---|
| `Tool initialization failed for '{library_key}'.` | Library could not be loaded/initialized (~L562 of the binding) | Import raise OR `init_tools_library` raise — historically file truncation/corruption (the `word_tools.py` saga: inverted spec name↔callable, orphaned tokens, dangling `{"` EOF — twice) | Verify file integrity first; then dependencies; then init logic. Full-file rewrite beats patching corruption. |
| `⚠️ Library 'X' contains no tool definitions` | Mount found an empty expected tool-name set | No `tool_*` functions on the probed seam, or unparseable file (`_extract_tool_names_from_file` silent-except) | Check the library's seam; check for corruption via the AST-fallback logs |
| `[ChatMixin] Mounted 'X' toolset (N tools)` | Informational: mount succeeded with N tools | — | Compare N against the library's expected count; mismatch ⇒ partial discovery |
| `[ChatMixin] Registered tool_X via direct discovered_tools fallback` | Tool registered straight from `discovered_tools`, bypassing the mount path | Mount path missed the tool (seam inconsistency) | Informational; verify the library's spec seam is consistent |
| `Personality binding 'null' yielded 0 chat tool specs (discovered_tools count: 0)` | Defensive fallback: binding name null AND empty registry | Tools binding not configured or discovery never ran | Set `tools_binding_name="lcp"`; check gates and load diagnostics |
| `FileNotFoundError: bare relative image` | A bare relative path escaped the workspace anchoring | Asset referenced without `media()`/`resolve_path()` | Use `media()`/`resolve_path()`; never bare filenames, never absolute paths |
| `[COMMENT by X: ...]` appearing as inline document text | The historical `word_tools` comment fallback | Old `add_comment` fallback wrote inline text instead of real comments | Fixed: current engine uses `_ensure_comments_part` + real Word comments. If seen, the document predates the fix. |

---

## 7. End-to-end sequence diagram

```
 LLM          chat mixin            LCPBinding            library callable
  │                │                      │                       │
  │  tool call in  │                      │                       │
  │  streamed text │                      │                       │
  ├───────────────►│  intercept + parse   │                       │
  │                ├─────────────────────►│ execute_tool(         │
  │                │                      │   tool_name, params,  │
  │                │                      │   discussion_instance)│
  │                │                      ├──────────────────────►│ run with workspace
  │                │                      │                       │ as CWD (restored
  │                │                      │                       │ after, even on crash)
  │                │                      │                       │
  │                │                      │   result dict         │
  │                │                      │◄──────────────────────┤
  │                │                      │ {success, error,      │
  │                │                      │  output, file_name,   │
  │                │                      │  context_files, ...}  │
  │                │  normalized dict     │                       │
  │                │◄─────────────────────┤ normalization:        │
  │                │                      │ success=False → 500   │
  │                │                      │ + error surfaced;     │
  │                │                      │ output auto-fill;     │
  │                │                      │ context_files → abs   │
  │                │                      │ existing paths        │
  │                │  attach artefacts    │                       │
  │                │  (USER-role only)    │                       │
  │  next round    │                      │                       │
  │◄───────────────┤                      │                       │
```

---

## 8. Key invariants

- `execute_tool(tool_name, params, discussion_instance, **kwargs)` is the
  single execution seam for every LCP tool, binding-mounted or
  personality-mounted.
- Result dicts are open-keyed with five normalization anchors: `success`,
  `error`, `output`, `file_name`, `context_files`.
- `success=False` ⇒ 500-style status + the `error` string surfaced to the LLM
  (failures are self-correctable information, never silent).
- Missing `output` is auto-filled from payload keys or the whole dict — the
  LLM always gets readable output.
- `context_files` are resolved to absolute existing paths and feed the
  artefact pipeline, which attaches them as **USER-role messages only**.
- CWD is transactional: set to the workspace for the call, restored afterward,
  even on crash (`test_lcp_cwd_management.py` locks this in).
- `Tool initialization failed for 'X'` covers BOTH import raises and
  `init_tools_library` raises (~L562) — check file integrity first.
- `⚠️ Library 'X' contains no tool definitions` means an empty expected
  tool-name set: wrong seam probed, or an unparseable file.