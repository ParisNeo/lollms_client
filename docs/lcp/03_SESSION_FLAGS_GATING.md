# 03 — Session Flags & Library Admission Gating

> **Purpose**: How the LCP decides which libraries are allowed to load — the
> office admission sets, the per-software gates, the host-config load-flag
> gate, the fail-closed law, and the session-flag read chain. Cross-links the
> per-flag documentation corpus under `docs/flags/`.

---

## 1. Why gating exists

`default_tools/` contains powerful libraries: code execution, shell access,
computer use, office file manipulation. Loading all of them unconditionally
would (a) surprise users who disabled a capability, (b) waste context on tool
specs the LLM should never see, and (c) widen the attack/accident surface.
The LCP therefore admits each library **only when the corresponding capability
flag is truthy** — and when in doubt, it says no.

**The fail-closed law**: a library with no matching truthy flag is NOT
admitted. Missing flags, null flags, and false flags all mean "no".

---

## 2. The gate machinery

### 2.1 `_OFFICE_LIBRARY_STEMS`

The set of library-name stems belonging to the office family (the exact
membership is defined in the binding source; office-family libraries are
handled by `_office_library_admitted` rather than by generic rules). As of
this writing the office family includes the Word/Excel/PowerPoint/office-file
libraries (`word_tools`, `excel_tools`, `powerpoint_tools`,
`office_files_builder`, `document_editor`).

### 2.2 `_PER_SOFTWARE_GATES`

Maps office-family library stems to the flag(s) that admit them:

| Library stem | Admitting flag(s) |
|---|---|
| `word_tools` | `enable_word` |
| `excel_tools` | `enable_excel` |
| `powerpoint_tools` | `enable_powerpoint` |
| *(office family, no per-software gate — e.g. `document_editor`)* | admitted when **ANY** of `enable_office_files` / `enable_powerpoint` / `enable_word` / `enable_excel` is truthy |

### 2.3 `_office_library_admitted(...)`

The office-family admission function: given a library stem and the flag
values, it applies `_PER_SOFTWARE_GATES` (specific gate first) and falls back
to the office-family any-flag rule for stems without a specific gate.

### 2.4 `TOOL_LIBRARY_LOAD_FLAG` — the host-config gate

Each library's `description.yaml` may declare a **load flag**. The
`TOOL_LIBRARY_LOAD_FLAG` mechanism checks that flag against the host config /
session flags:

- Library declares **no load flag** → the gate **passes** (the library is not
  flag-gated; it may still be gated by other rules).
- Library declares a load flag → admitted only when that flag is truthy in the
  host config / session flags.

This is the generic gate that non-office libraries use (e.g. code-execution
libraries gate on `enable_code_execution`, computer use on
`enable_computer_use`, data tools on `enable_data_tools`).

---

## 3. The session-flag read chain

Flag values are resolved through a chain, from most-specific to
most-general:

```
discussion (per-discussion overrides)
   │
   ▼
session_flags (per-chat-session flags, settable via set_session_flags)
   │
   ▼
host config (LollmsClient / personality configuration)
```

A truthy value at any level admits; the chain exists so a single discussion
can opt into (or out of) a capability without rewriting global config. The
chat-side office-mount seam (`_office_mount_requests`, admission sets,
`set_session_flags`) writes into this chain — documented in
[06_CHAT_MIXIN_INTEGRATION.md](./06_CHAT_MIXIN_INTEGRATION.md) (Step 2).

---

## 4. Decision flowchart — "is this library admitted?"

```
                ┌─────────────────────────────┐
                │ library candidate from       │
                │ _iter_library_files()        │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │ does description.yaml        │
                │ declare a load flag?         │
                └──────┬───────────────┬──────┘
                       │ no            │ yes
                       ▼               ▼
              ┌───────────────┐  ┌─────────────────────────────┐
              │ gate passes   │  │ is the flag truthy in the    │
              │ (unconditional│  │ session-flag chain?          │
              │  for THIS gate)│  └──────┬───────────────┬──────┘
              └───────┬───────┘         │ no            │ yes
                      │                 ▼               │
                      │        ┌─────────────────┐      │
                      │        │ NOT ADMITTED    │      │
                      │        │ (fail-closed)   │      │
                      │        └─────────────────┘      │
                      │                                  │
                      ▼                                  ▼
           ┌──────────────────────────────────────────────────┐
           │ is the library in the office family               │
           │ (_OFFICE_LIBRARY_STEMS)?                          │
           └──────┬───────────────────────────────┬───────────┘
                  │ no                            │ yes
                  ▼                               ▼
        ┌──────────────────┐        ┌────────────────────────────────┐
        │ admitted         │        │ per-software gate in           │
        │ (subject to      │        │ _PER_SOFTWARE_GATES?           │
        │ office rules     │        └──────┬───────────────┬─────────┘
        │ not applying)    │               │ yes           │ no
        └──────────────────┘               ▼               ▼
                             ┌──────────────────┐  ┌──────────────────────────┐
                             │ is that flag     │  │ ANY of enable_office_    │
                             │ truthy?          │  │ files / enable_powerpoint│
                             └────┬───────┬─────┘  │ / enable_word /          │
                                  │ no    │ yes    │ enable_excel truthy?     │
                                  ▼       ▼        └────┬──────────────┬──────┘
                          ┌──────────┐ ┌──────────┐     │ no           │ yes
                          │ NOT      │ │ ADMITTED │     ▼              ▼
                          │ ADMITTED │ └──────────┘ ┌──────────┐ ┌──────────┐
                          └──────────┘              │ NOT      │ │ ADMITTED │
                                                    │ ADMITTED │ └──────────┘
                                                    └──────────┘
```

---

## 5. LCP-relevant flags and their docs

Each flag has a dedicated document under `docs/flags/`. The LCP-relevant set:

| Flag | Gates (among others) | Doc |
|---|---|---|
| `enable_word` | `word_tools`; office-family any-flag rule | [`../flags/enable_word.md`](../flags/enable_word.md) |
| `enable_excel` | `excel_tools`; office-family any-flag rule | [`../flags/enable_excel.md`](../flags/enable_excel.md) |
| `enable_powerpoint` | `powerpoint_tools`; office-family any-flag rule | [`../flags/enable_powerpoint.md`](../flags/enable_powerpoint.md) |
| `enable_office_files` | office-family any-flag rule (`document_editor`, `office_files_builder`, ...) | [`../flags/enable_office_files.md`](../flags/enable_office_files.md) |
| `enable_code_execution` | code-execution libraries (`execute_python`, ...) | [`../flags/enable_code_execution.md`](../flags/enable_code_execution.md) |
| `enable_computer_use` | `computer_use` | [`../flags/enable_computer_use.md`](../flags/enable_computer_use.md) |
| `enable_data_tools` | data-tool libraries | [`../flags/enable_data_tools.md`](../flags/enable_data_tools.md) |
| `allow_dynamic_tools` | dynamic/LLM-authored tool registration | [`../flags/allow_dynamic_tools.md`](../flags/allow_dynamic_tools.md) |
| `list_available_models` | model-listing tools (`models_tools`) | [`../flags/list_available_models.md`](../flags/list_available_models.md) |

The full flags corpus (including chat-behavior flags outside LCP scope) is
indexed at [`../flags/README.md`](../flags/README.md).

---

## 6. Troubleshooting admission

**"My library doesn't appear in discovery."**

1. Is its flag truthy anywhere in the chain (discussion → session_flags →
   config)? Remember: fail-closed.
2. Does its `description.yaml` declare the load flag you think it does?
3. Is it an office-family library? Then check the per-software gate first,
   then the any-flag rule.
4. Only after admission checks, look at load/parse diagnostics
   ([02_TOOL_DISCOVERY_PIPELINE.md](./02_TOOL_DISCOVERY_PIPELINE.md) §4) —
   a library can be admitted and still fail to load.

---

## 7. Key invariants

- Admission is **fail-closed**: no truthy flag ⇒ no admission. Missing, null,
  and false are all "no".
- `_PER_SOFTWARE_GATES` maps `word_tools→("enable_word",)`,
  `excel_tools→("enable_excel",)`, `powerpoint_tools→("enable_powerpoint",)`;
  office-family libraries without a per-software gate (e.g. `document_editor`)
  are admitted when ANY of `enable_office_files` / `enable_powerpoint` /
  `enable_word` / `enable_excel` is truthy.
- `TOOL_LIBRARY_LOAD_FLAG`: a library declaring **no** load flag passes that
  gate; a library declaring one needs the flag truthy.
- Flag values resolve through the chain discussion → session_flags → host
  config.
- Gating happens **before** schema extraction — an unadmitted library costs
  no parse work and no context budget.
- Every LCP-relevant flag has a doc under `docs/flags/`; when adding a new
  library, add its flag doc there too.