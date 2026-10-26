# 09 — Study Guide & Capstone Reference

> **Purpose**: The capstone for both audiences — a human learning path, an
> LLM-facing cheat sheet, a glossary, a troubleshooting FAQ mapping every
> diagnostic to its root cause and fix, a "how to add a new LCP library"
> tutorial, and a changelog explaining why the LCP has its current shape.
> Start here if you are new; return here when something breaks.

---

## (a) Human learning path

| Order | Document | Time | Checkpoint — you can now… |
|---|---|---|---|
| 1 | [00_OVERVIEW.md](./00_OVERVIEW.md) | 10 min | Explain the LCP's role and the three consumption surfaces. |
| 2 | [01_LCP_BINDING_CORE.md](./01_LCP_BINDING_CORE.md) | 20 min | Trace a library file from disk to a persistent module. |
| 3 | [02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md) | 30 min | Explain both discovery seams and why regex fallback is degraded. |
| 4 | [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md) | 20 min | Decide whether a library is admitted, given its flags. |
| 5 | [04_LIBRARY_CATALOG.md](./04_LIBRARY_CATALOG.md) | 25 min | Name each library's seam, gate, and doctrine. |
| 6 | [05_TOOL_EXECUTION_SEAM.md](./05_TOOL_EXECUTION_SEAM.md) | 20 min | Predict the normalized result dict for any tool outcome. |
| 7 | [06_CHAT_MIXIN_INTEGRATION.md](./06_CHAT_MIXIN_INTEGRATION.md) | 30 min | Walk a full chat round with a tool call, end to end. |
| 8 | [07_PERSONALITY_INTEGRATION.md](./07_PERSONALITY_INTEGRATION.md) | 20 min | Explain WCD, mount-vs-discover, and the client seam. |
| 9 | [08_LIBRARY_DEEP_DIVES.md](./08_LIBRARY_DEEP_DIVES.md) | 40 min | Read `word_tools` fluently and extend any deep dive. |

Total ≈ 3.5 hours for full fluency.

---

## (b) LLM cheat sheet — the key contracts

```
DISCOVERY SEAMS (exactly two):
  A) AST seam: top-level tool_* functions + Google-style docstrings.
     Schema derived: int→integer, bool→boolean, list→array, dict→object,
     else string. Non-default params → required. _-prefixed helpers excluded.
  B) get_tools() seam: hand-authored spec dicts.

SPEC SHAPE (Seam B):
  { name, description,
    parameters: [{name, type, optional?, description}],
    output:     [{name, type, description}],
    callable, _source, _source_file }
  INVARIANT: name MUST match callable; output MUST match return keys.

RESULT DICT (execute_tool normalization anchors):
  success / error / output / file_name / context_files
  success=False → 500-style status + error surfaced to the LLM.
  Missing output → auto-filled. context_files → absolute existing paths.

GATE FLAGS (fail-closed; missing/null/false = NO):
  word_tools→enable_word, excel_tools→enable_excel,
  powerpoint_tools→enable_powerpoint,
  office family no-gate libs → ANY of enable_office_files/enable_powerpoint/
  enable_word/enable_excel. TOOL_LIBRARY_LOAD_FLAG: no declared flag ⇒ passes.

WORKSPACE RULE:
  Asset references MUST use media()/resolve_path(). No bare relative paths,
  no absolute escapes, no '..'. NO permanent os.chdir (call-scoped CWD only,
  transactionally restored — test-locked).

MEDIA INVARIANT:
  Generated media attaches as USER-role messages ONLY. System-role media →
  provider 400 "System message cannot contain images" → sanitize + reroute.

REGEX FALLBACK = WARNING:
  _parsed_via_regex_fallback: True means the library file is BROKEN
  (SyntaxError). Survival mode, not a feature. Fix the file.

TWO CRITICAL DIAGNOSTICS:
  "⚠️ Library 'X' contains no tool definitions" → empty tool-name set
      (wrong seam probed or unparseable file).
  "Tool initialization failed for 'X'" → import OR init_tools_library raise
      (~L562). Check file integrity first (truncation saga).
```

---

## (c) Glossary

| Term | Meaning |
|---|---|
| **LCP** | The in-process local tool binding (`tools_bindings/lcp/`); LLM tools execute in the same Python process. |
| **Binding** | A `LollmsToolBinding` subclass (LCP or MCP) implementing `discover_tools`/`list_tools`/`execute_tool`. |
| **Library** | One `default_tools/<name>/` folder with its primary `<name>.py` (+ `description.yaml`). |
| **Toolset** | The set of tools a library exposes; "mounted" when active for a session. |
| **Tool spec** | The schema dict describing one tool (name/description/parameters/output/callable/...). |
| **Recipe sandbox** | Hardened executor for LLM-authored code (`_check_recipe`/`_exec_recipe`, whitelists, forbidden tokens). |
| **WCD / workspace** | Working-context directory; the sole root for asset resolution (`personality.wcd` → `workspace_data_path`). |
| **Artefact** | A file produced by a tool or generation, attached to the conversation (USER-role only). |
| **Session flag** | A capability toggle resolved discussion → session_flags → host config. |
| **Mount vs discover** | Mount = explicit family activation with expected counts; discover = registry-driven registration (fallback path). |
| **TTI/TTM/TTS/TTV** | Text-to-Image/Music/Speech/Video binding registries; media-tag doctrine is probe-gated on them. |
| **AST seam vs get_tools seam** | The two discovery seams (see cheat sheet). |
| **context_files** | Result-dict key listing files whose contents feed the next round (resolved to absolute existing paths). |
| **Mimicry protocol** | Empty/artefact-only round recovery: correction injection + attempt counting (test-locked marker). |

---

## (d) FAQ / Troubleshooting — every diagnostic, decoded

| Diagnostic | Root cause | Fix |
|---|---|---|
| `AST parse failed for 'X.py': ... Attempting regex fallback.` | `SyntaxError` in the library file (truncation/corruption) | Repair/rewrite the file; fallback is survival mode only. |
| `[LCP] Regex fallback extracted tool 'X' from 'Y.py'.` | Same — signatures salvaged, schemas crippled | Same. |
| `⚠️ Library 'X' contains no tool definitions` | Empty expected tool-name set: no `tool_*` functions on the probed seam, or unparseable file | Verify the library's seam; check corruption via AST-fallback logs. |
| `Tool initialization failed for 'X'.` | Import raise OR `init_tools_library` raise (~L562) | Check file integrity first (the `word_tools.py` truncation saga: inverted spec name↔callable, orphaned tokens, dangling `{"` EOF — twice), then dependencies, then init logic. |
| `Personality binding 'null' yielded 0 chat tool specs (discovered_tools count: 0)` | Binding name null AND empty registry → defensive fallback | Set `tools_binding_name="lcp"`; check gates and load diagnostics. |
| `Empty response generated by LLM (0 tokens)` + 400 `"System message cannot contain images"` | Media attached to a system-role message | Fixed by the USER-role-only invariant: sanitize + reroute to USER-role, retry. |
| `FileNotFoundError: bare relative image` | Asset referenced without `media()`/`resolve_path()` | Use `media()`/`resolve_path()`; never bare filenames or absolute paths. |
| `[COMMENT by X: ...]` appearing as inline document text | Historical `word_tools` comment fallback (inline text instead of real comments) | Fixed: current engine uses `_ensure_comments_part` + real Word comments. Documents predating the fix show the inline form. |
| `[ChatMixin] Mounted 'X' toolset (N tools)` with unexpected N | Partial discovery (seam inconsistency or corruption) | Compare N against the library's expected count; check discovery diagnostics. |
| `[ChatMixin] Registered tool_X via direct discovered_tools fallback` | Mount path missed the tool | Informational; verify seam consistency. |

---

## (e) Tutorial — how to add a new LCP library

1. **Author the file**: `default_tools/<name>/<name>.py` (dir + primary-file
   pattern). Add a `description.yaml` declaring metadata — including the
   **load flag** for the `TOOL_LIBRARY_LOAD_FLAG` gate.
2. **Choose your discovery seam**:
   - **Seam A (AST)**: write top-level `tool_*` functions with Google-style
     docstrings (first paragraph = description; `param (type): desc` lines).
     The schema is derived automatically.
   - **Seam B (`get_tools()`)**: hand-author spec dicts
     (`name`/`description`/`parameters`/`output`/`callable`/`_source`/
     `_source_file`). `name` MUST match the callable; `output` MUST match
     return keys.
3. **Implement `init_tools_library(config=None)`** so it **never raises**
   (pipmaster soft-fail pattern — copy `word_tools.py`).
4. **Follow the workspace rule**: resolve every asset via
   `media()`/`resolve_path()`; reject absolute paths and `..`; never
   `os.chdir` permanently.
5. **If your library executes LLM-authored code**: adopt the recipe-sandbox
   doctrine (`_FORBIDDEN_TOKENS`, `_IMPORT_PREFIX_WHITELIST`,
   `_SAFE_STDLIB_MODULES`, `_build_safe_builtins`, `_check_recipe`,
   `_exec_recipe`, `_annotate_traceback`, `_MAX_RECIPE_CHARS`).
6. **If your library writes office files**: adopt transactional save
   (`_save_and_verify` + `_validate_office_file` + `OOXML_VALIDATORS` +
   `OFFICE_GUARD_*`).
7. **Return the canonical result dict**: `success`/`error`/`output`/
   `file_name`/`context_files` (+ tool-specific keys).
8. **Document the flag**: add `docs/flags/<flag>.md` and link it from
   [03_SESSION_FLAGS_GATING.md](./03_SESSION_FLAGS_GATING.md) §5.
9. **Test the three stages**: discovery (spec appears), mount (expected tool
   count), execution (worked call → normalized dict → USER-role artefact).

---

## (f) Changelog — why the LCP has its current shape

| Evolution step | What changed | Why |
|---|---|---|
| **The `word_tools.py` truncation saga** | A corrupted file (dangling `{"` EOF, orphaned tokens, inverted spec name↔callable) produced `Tool initialization failed` twice; regex fallback salvaged crippled signatures | Established the error-attribution law (~L562) and the "regex fallback = broken file" doctrine; full-file rewrite beat patching |
| **The workspace-anchoring mission** | `_candidate_working_dirs` workspace-sole-root; `_resolve_media_path` rejects absolute/`..`; `media()`/`resolve_path()` doctrine | Fixed `FileNotFoundError: bare relative image` — bare paths resolved against the process folder, not the workspace |
| **The comment-engine upgrade** | `_ensure_comments_part` OPC part + `relate_to(RT.COMMENTS)`; run-anchored `add_comment`; collision-safe ids; UTC timestamps | Replaced the inline `[COMMENT by X: ...]` text fallback with real Word comments |
| **The media-tags mission** | `_dispatch_closed_tag` → `_handle_media_tag`, `_MEDIA_TAG_CONFIG`, `_StreamState` accumulators | Unified image/music/speech/video generation behind one tag-driven pipeline |
| **The staged-context mission** | `TAGS` registry, `_assemble_context_stages` fixed order, `include_*` + `*_max_tokens`, drop-never-truncate, `parse_response_tags`/`_route_parsed_tags` | Made context composition budgeted, predictable, and host-controlled (model cannot re-enable disabled stages) |
| **The 400-fix** | USER-role-only media attachment + 400-signature recovery | Providers reject system-role media (`"System message cannot contain images"`) — sanitize + reroute + retry |

---

## Key invariants (corpus-level)

- The corpus 00–09 is the single LCP documentation source; source files are
  never modified by it.
- Verified facts are grounded in hydrated source; unhydrated material is
  explicitly marked "surface-level — pending deeper scouting".
- Every diagnostic string in this codebase's logs has an entry in §(d) with
  root cause and fix.
- The cheat sheet §(b) is designed to be extracted wholesale as a
  system-prompt rules block for LLM consumers.