# 🏛️ Lollms Agentic: Two-Tier Orchestrator/Worker Doctrine

The `lollms_agentic` package implements the **Two-Tier Execution Doctrine** for the LoLLMS framework. It structurally separates high-level cognitive coordination from atomic tool execution into two isolated agent tiers.

---

## 🎯 1. Core Architectural Separation

```
                       User Request
                            │
                            ▼
              ┌───────────────────────────┐
              │    ORCHESTRATOR TIER      │
              │  (Persistent Context)     │
              │  • PLAN                   │
              │  • DELEGATE               │
              │  • VERIFY (Inspector)     │
              └─────────────┬─────────────┘
                            │
             Delegates task │ Returns plain-data
             + context files│ report envelope
                            ▼
              ┌───────────────────────────┐
              │       WORKER TIER         │
              │   (Disposable Context)    │
              │  • Tool Execution         │
              │  • Artifact Writes        │
              │  • File Edits / Patches   │
              └───────────────────────────┘
```

### A. The Orchestrator Tier
- **Coordination Only**: The Orchestrator plans, delegates, and verifies. It has **no tool-call grammar and no direct file-writing authority** in its context window.
- **Structural Immunity**: Because the Orchestrator's prompt contains zero tool syntax, it is structurally impossible for it to hallucinate or misfire tool executions.
- **Dual-View Context**:
  - **Model View**: The Orchestrator sees the verbatim coordination history: its own `<plan>` steps, `<delegate>` blocks, `<verify>` judgements, and incoming `[WORKER REPORT]` envelopes.
  - **User/UI View**: The user sees natural conversational summaries along with structured execution events (`MSG_TYPE_WORKER_SPAWN_START/END`) or processing blocks.

### B. The Worker Tier
- **Disposable Execution Sandbox**: Each worker is a short-lived sub-agent instantiated to perform a single atomic unit of work.
- **Minimal Context**: The worker receives only the specific task description and the contents of explicitly assigned files (`WORKER_TASK_TEMPLATE`). It has no prior conversation history.
- **Full Tool Doctrine**: The worker has access to the full execution toolset (Python execution, shell commands, file writing, database queries).
- **Clean Termination**: The worker wraps its findings in a `<report>...</report>` envelope and terminates with `<done/>`. Its transient execution context is immediately discarded upon completion.

---

## 📋 2. Orchestration Grammar & Verbs

The Orchestrator communicates through four specific verbs:

### 1. PLAN (`<plan>`)
Emits a numbered checklist of atomic steps required to accomplish the user's goal. This roadmap is automatically tracked in `progress.md`:
```xml
<plan>
1. Inspect dataset structure and identify missing fields
2. Write data processing script to clean records
3. Execute script and verify output CSV
</plan>
```

### 2. DELEGATE (`<delegate>`)
Hands a single atomic step to a specialist worker, specifying the task and required files:
```xml
<delegate>
<task>
Write a Python script 'clean_data.py' that loads 'raw_data.csv', removes null values in the 'age' column, and saves to 'cleaned_data.csv'.
</task>
<context_files>
raw_data.csv
</context_files>
</delegate>
```

### 3. VERIFY (`<verify>`)
Dispatches an independent **Inspector Worker** to verify ground truth in the workspace (checking file existence, running tests, or inspecting outputs) rather than trusting self-reported claims:
```xml
<verify step="2">
Inspector confirmed 'clean_data.py' exists, executes without errors, and produced 'cleaned_data.csv' with 0 nulls.
</verify>
```

### 4. FINISH (`<done/>`)
When all plan steps are marked complete in `progress.md`, the Orchestrator delivers its final conversational response to the user and terminates:
```xml
All data cleaning tasks have been completed and verified. The cleaned dataset is ready in 'cleaned_data.csv'.

<done/>
```

---

## 📦 3. Storage & Versioning Integration

The two-tier engine operates seamlessly across both LoLLMS discussion sessions and standalone personality agents:

| Environment | Versioning Strategy | File Handling |
| :--- | :--- | :--- |
| **`LollmsDiscussion`** | `disable_artefact_versioning=False` | Database-backed history with `.versions/` snapshots and `.lam` logical twins. |
| **`LollmsPersonality`** | `disable_artefact_versioning=True` | Direct live workspace filesystem operations with Git integration. |

---

## 🚀 4. Python API Example

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agentic import AgenticRunner
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "qwen2.5-coder:7b"})
personality = LollmsPersonality(
    name="LeadArchitect",
    workspace_path="./workspace",
    lollms_client=client
)

# Run via two-tier agentic mode
response = personality.chat(
    prompt="Analyze sales.csv and produce a quarterly report summary in report.md",
    orchestrator_mode=True
)

print(response["response"])
```