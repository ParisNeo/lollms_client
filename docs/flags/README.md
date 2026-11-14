# `chat()` Boolean Flags — Per-Flag Documentation

Canonical index for per-flag deep pages. Every row of the master table in
[`../CHAT_BOOLEAN_FLAGS.md`](../CHAT_BOOLEAN_FLAGS.md) links here once, and each
deep page links back to its master-table anchor.

**Single source of truth** for defaults and gates: the master table. If a deep
page disagrees with the master table, the master table wins (it is reconciled
against `_mixin_chat.py`'s signature).

## Conventions

- **Flag name = filename.** Every page is `<flag_name>.md`, lowercase
  `snake_case`. The H1 must be exactly ``# `<flag_name>` ``.
- **Defaults must match** the master table's `Default` column, verbatim.
- **Category** matches the master table's `Category` column.
- **Parent master switch** — if the flag is a sub-flag, its page's first
  paragraph names the parent master switch.
- **Gate** — if the flag is gated by a runtime capability (vision, TTI) or by
  a security precondition, the page documents it in a `## Gate` section.
- **Page template** (mirror of the master per-flag contract):
  1. Signature (` ```python flag: type = default ` ``)
  2. `**Category**`
  3. `## What it does` — one paragraph.
  4. `## What to expect` — the master's bullet set, expanded.
  5. `## Gate` — if any.
  6. `## Usage` — a minimal code example.
  7. `## Related flags` — cross-links (both parents and interactions).
  8. `## Recovering generated stuff` — only when the flag causes the model to
     *produce* files/artefacts/reports.

- **Adding a new flag page**: create the file matching the flag name, follow
  the template above, then add a row to the master table (with a link to the
  new page) AND list it under the correct category below. Do not orphan
  pages — every page must be reachable from both the master table and this
  index.

## Categories

### Core Loop
| Flag | Default | Page |
|---|---|---|
| `stream` | `False` | [stream.md](stream.md) |
| `list_available_models` | `False` | [list_available_models.md](list_available_models.md) |
| `enforce_end_tag` | `False` | [enforce_end_tag.md](enforce_end_tag.md) |
| `dynamic_effort` | `False` | [dynamic_effort.md](dynamic_effort.md) |
| `clean_reasoning_at_save` | `True` | [clean_reasoning_at_save.md](clean_reasoning_at_save.md) |

### Artifacts (Workspace Files)
| Flag | Default | Parent | Page |
|---|---|---|---|
| `enable_artefacts` | `True` | — | [enable_artefacts.md](enable_artefacts.md) |
| `auto_activate_artefacts` | `True` | `enable_artefacts` | [auto_activate_artefacts.md](auto_activate_artefacts.md) |
| `enable_notes` | `True` | `enable_artefacts` | [enable_notes.md](enable_notes.md) |
| `enable_skills` | `True` | `enable_artefacts` | [enable_skills.md](enable_skills.md) |
| `enable_inline_widgets` | `False` | `enable_artefacts` | [enable_inline_widgets.md](enable_inline_widgets.md) |
| `enable_forms` | `False` | `enable_artefacts` | [enable_forms.md](enable_forms.md) |
| `enable_books` | `False` | `enable_artefacts` | [enable_books.md](enable_books.md) |
| `enable_image_generation` | `True` | `enable_artefacts`, **TTI gate** | [enable_image_generation.md](enable_image_generation.md) |
| `enable_image_editing` | `True` | `enable_artefacts`, **TTI gate** | [enable_image_editing.md](enable_image_editing.md) |
| `forward_artefact_chunks` | `False` | — | [forward_artefact_chunks.md](forward_artefact_chunks.md) |
| `enable_in_message_status` | `False` | — | [enable_in_message_status.md](enable_in_message_status.md) |

### Memory
| Flag | Default | Parent | Page |
|---|---|---|---|
| `enable_memory` | `True` | — | [enable_memory.md](enable_memory.md) |
| `enable_episodic_memory` | `True` | `enable_memory` | [enable_episodic_memory.md](enable_episodic_memory.md) |
| `enable_auto_dream` | `True` | `enable_memory` | [enable_auto_dream.md](enable_auto_dream.md) |
| `enable_deep_memory_pulling` | `True` | `enable_memory` | [enable_deep_memory_pulling.md](enable_deep_memory_pulling.md) |

### Knowledge & Search
| Flag | Default | Page |
|---|---|---|
| `prehydrate_rag` | `True` | [prehydrate_rag.md](prehydrate_rag.md) |
| `web_search` / `internet_search` | `False` | [web_search.md](web_search.md) *(covers the `internet_search` alias)* |

### Tools & Security
| Flag | Default | Security | Page |
|---|---|---|---|
| `enable_data_tools` | `True` | contextual scan | [enable_data_tools.md](enable_data_tools.md) |
| `enable_code_execution` | `False` | **code execution** | [enable_code_execution.md](enable_code_execution.md) |
| `auto_approve_python` | `False` | **code execution** | [auto_approve_python.md](auto_approve_python.md) |
| `allow_dynamic_tools` | `False` | **code execution** | [allow_dynamic_tools.md](allow_dynamic_tools.md) |
| `enable_vlm_query` | `False` | vision fallback only | [enable_vlm_query.md](enable_vlm_query.md) |
| `enable_computer_use` | `False` | **desktop control**, vision-gated | [enable_computer_use.md](enable_computer_use.md) |
| `enable_powerpoint` | `False` | per-format toolset | [enable_powerpoint.md](enable_powerpoint.md) |
| `enable_word` | `False` | per-format toolset | [enable_word.md](enable_word.md) |
| `enable_excel` | `False` | per-format toolset | [enable_excel.md](enable_excel.md) |
| `suppress_images` | `False` | negative polarity | [suppress_images.md](suppress_images.md) |

Tool-calling wire format reference: [`../TOOLCALLING_FORMAT.md`](../TOOLCALLING_FORMAT.md) (`<tool>`, `<tool_result>`, `<done/>`, registry rules, anti-loop guards).

Legacy alias (**DEPRECATED**): [`enable_office_files.md`](enable_office_files.md) — enables all three per-software office flags. Prefer `enable_powerpoint` / `enable_word` / `enable_excel`.

### RSI (Toolsmith & Skillmaker)

Act-phase flags for the `tool_maker` / `skill_maker` toolsets. Basic
manipulation tools remain available without these flags — see the master table
for the split and the mission briefing in the project memory for constraints.

| Flag | Default | Enables (LLM-visible) | Page |
|---|---|---|---|
| `enable_toolsmith` | `False` | `tool_run_test_agent`, `tool_commit_toolset` | — (in master table) |
| `enable_skillmaker` | `False` | `tool_run_skill_test_agent`, `tool_commit_skill` | — (in master table) |

### Personas & Diagnostics
| Flag | Default | Page |
|---|---|---|
| `orchestrator_mode` | `False` | [orchestrator_mode.md](orchestrator_mode.md) |
| `orchestrator_persona` | `False` | [orchestrator_persona.md](orchestrator_persona.md) |
| `debug` | `False` | [debug.md](debug.md) |
| `debug_export` | `False` | [debug_export.md](debug_export.md) |

---

See [`../CHAT_BOOLEAN_FLAGS.md`](../CHAT_BOOLEAN_FLAGS.md) for the consolidated
reference (mental model, full per-flag semantics, cross-flag interactions,
quick recipes). This file is the routing index — the master is the reference.