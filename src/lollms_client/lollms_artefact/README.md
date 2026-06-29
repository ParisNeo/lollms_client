# 📦 lollms_artefact: Dynamic Dual-Stream Artefact Subsystem

The `lollms_artefact` package implements the core versioning, lifecycle management, and file-tracking layers for Lollms. It introduces a **Dual-Stream Storage** architecture designed to bridge the gap between large physical data files on disk and the context window limitations of Large Language Models.

---

## 🏛️ 1. The Dual-Stream Philosophy (.lam Protocol)

Conversational AI agents often experience a conflict between model attention and tool execution:
- **Language Models** require high-level summaries, database schemas, column definitions, and file layouts to plan queries and reason logically without wasting thousands of tokens on raw data.
- **Local Tools** (Python scripts, SQL queries, executors) require the exact, unmodified physical bytes to perform computations and write outputs.

To solve this, the subsystem splits every created or imported data artifact into two distinct streams:

### A. The Physical Twin
* **Location**: Saved on the local file system at `workspace_data/{title}.{ext}` (and versioned inside `artefacts_metadata/{id}/{name}_v{version}{ext}`).
* **Content**: The raw, unmodified bytes of the file (e.g., raw CSV rows, SQLite database binary, or Python source code).
* **Consumer**: Programmatic local tools and sandbox scripts executing in the workspace.

### B. The Logical Twin (`.lam` Metadata File)
* **Location**: Saved inside `artefacts_metadata/{id}/{name}.lam`.
* **Content**: A high-density, text-based abstraction of the file's structure. For a spreadsheet, this includes column names, inferred data types, non-null counts, and a few sample row values.
* **Consumer**: The Large Language Model. This metadata is what is actually injected into the active prompt context zone and evaluated by the token budget calculator.

---

## 👁️ 2. Multi-Tier Visibility Control

To maintain clean and token-efficient context budgets, every registered artifact is assigned a visibility tier that determines how it is represented in the prompt:

| Visibility Tier | Symbol | Prompt Context Behavior |
| :--- | :--- | :--- |
| **`FULL`** | `[C]` | The logical `.lam` content of the file is fully injected verbatim into the active context zone. |
| **`METADATA`** | `[M]` | Only the basic metadata (such as filename, size, and type) is injected, withholding the full schema description. |
| **`TREE_UNLOCKABLE`**| `[U]` | The file is listed only in the directory index. It is excluded from the active context but can be loaded dynamically. |
| **`LOCKED`** | `[L]` | The file is completely excluded from the conversation context and cannot be loaded. |
| **`HIDDEN`** | — | The artifact is completely excluded from both the directory index and the context. |

The LLM can dynamically promote any `[U]` file into its working memory by outputting the file-loading tag:
```xml
<add_files_to_context>
filename.ext
</add_files_to_context>
```

---

## 🧬 3. Integration with LollmsDiscussion

The `ArtefactManager` interacts directly with `LollmsDiscussion` to orchestrate state updates:

```
        LollmsDiscussion (Session State)
              │
              ├──> ArtefactManager
              │         │
              │         ├──> [SQLite Metadata Record] (Maintains version logs)
              │         │
              │         └──> [Physical Workspace] (Writes and version-controls files)
              │
              └──> ChatMixin (Orchestrates tool execution & scans CWD)
```

### A. Automated File-Tracking and Ingestion
During local tool execution, the active directory is snapshotted immediately before and after the run. If a tool writes a new file (such as a Matplotlib chart PNG) or modifies an existing dataset:
1. The new file is automatically detected on disk.
2. Its file type is classified, and the raw bytes are saved as a physical twin.
3. A logical twin (`.lam`) or image reference is compiled.
4. The artifact is committed to the database, incrementing its version.
5. The corresponding reference tags are appended to the conversational message stream.

### B. Self-Healing and Recovery
If a tool or script requests a physical file that is missing from the active `workspace_data/` folder, the manager intercepts the failure, queries the database version log, and restores the exact versioned physical bytes back to the disk folder automatically before the execution begins.

### C. Live Rendering Tags
The chat interface interprets custom tags inserted into the message history:
* `<lollms_artifact id="title" type="atype" version="N" />`: Renders an interactive file card in the chat bubble allowing the user to view or download the file.
* `<artefact_image id="title::N" />`: Directs the chat bubble to render the decoded base64 image pixels inline (e.g., showing a generated plot directly in the conversation).

---

## 🛠️ 4. Class Reference

* **`ArtefactType`**: Registry defining the supported categories (`DATA`, `CODE`, `DOCUMENT`, `IMAGE`, `PRESENTATION`, `NOTE`, `SKILL`).
* **`ArtefactManager`**: Orchestrates database CRUD operations, applies search-and-replace patches, and manages version history squashing.
* **`FileImportMixin`**: Contains multi-modal parser subroutines for importing PDFs, Word documents, PowerPoint presentations, and audio files.