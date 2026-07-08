# 📦 lollms_artefact: Dynamic Artefact & Context Subsystem

The `lollms_artefact` package implements the core versioning, lifecycle management, and file-tracking layers for Lollms. It introduces a **Hybrid Storage Architecture** designed to bridge the gap between large physical data files on disk and the context window limitations of Large Language Models.

---

## 🏛️ 1. The Hybrid Storage Philosophy

Conversational AI agents often experience a conflict between model attention and tool execution:
- **Language Models** require high-level summaries, database schemas, and text content to plan queries and reason logically without wasting thousands of tokens on raw binary data.
- **Local Tools** (Python scripts, SQL queries, executors) require the exact, unmodified physical bytes to perform computations and write outputs.

To solve this, the subsystem uses two distinct storage strategies depending on the file type:

### A. Single-Stream (Text, Code, Documents)
For text-based files (`.py`, `.md`, `.pdf`, `.docx`, `.txt`), the system uses a **Single-Stream** approach:
* **Storage**: The extracted text content is stored directly in the artefact's `content` field and written to `workspace_data/{title}.{ext}`.
* **Context Injection**: The LLM context zone reads directly from the `content` field. The LLM sees the full, verbatim text of the file.
* **No `.lam` files**: Text files do NOT generate Logical Artefact Metadata (`.lam`) files. We do not separate the content from the metadata.

### B. Dual-Stream (.lam Protocol for Binary & Structured Data)
For structured data files (`.csv`, `.db`, `.sqlite`, `.xlsx`), the system uses a **Dual-Stream** approach:
* **Physical Twin**: Saved on the local file system at `workspace_data/{title}.{ext}`. Contains the raw bytes (e.g., raw CSV rows, SQLite binary). Consumed by local tools.
* **Logical Twin (`.lam`)**: Saved inside `artefacts_metadata/{id}/{name}.lam`. Contains a high-density, text-based abstraction of the file's structure (column names, inferred data types, sample values). Consumed by the LLM context zone.

---

## 👁️ 2. Multi-Tier Visibility Control

To maintain clean and token-efficient context budgets, every registered artifact is assigned a visibility tier that determines how it is represented in the prompt:

| Visibility Tier | Symbol | Prompt Context Behavior |
| :--- | :--- | :--- |
| **`FULL`** | `[C]` | The content (for text) or `.lam` schema (for data) is fully injected verbatim into the active context zone. |
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

**Default Visibility Doctrine**: To prevent context window bloat, all newly discovered or tool-generated files are registered with `TREE_UNLOCKABLE` visibility by default. The LLM must explicitly unlock a file to load its content into the active context.

During local tool execution, the active directory is snapshotted immediately before and after the run. If a tool writes a new file (such as a Matplotlib chart PNG) or modifies an existing dataset:
1. The new file is automatically detected on disk.
2. Its file type is classified, and the raw bytes are saved as a physical twin.
3. A logical twin (`.lam`) or image reference is compiled (if applicable).
4. The artifact is committed to the database, incrementing its version.
5. The corresponding reference tags are appended to the conversational message stream.

### B. Self-Healing and Recovery
If a tool or script requests a physical file that is missing from the active `workspace_data/` folder, the manager intercepts the failure, queries the database version log, and restores the exact versioned physical bytes back to the disk folder automatically before the execution begins.

### C. Live Rendering Tags
The chat interface interprets custom tags inserted into the message history:
* `<lollms_artifact id="title" type="atype" version="N" />`: Renders an interactive file card in the chat bubble allowing the user to view or download the file.
* `<artefact_image id="title::N" />`: Directs the chat bubble to render the decoded base64 image pixels inline (e.g., showing a generated plot directly in the conversation).

---

## 🛠️ 4. Dynamic Tool Artefacts (`type="tool"`)

The artefact system natively supports the LLM generating its own executable tools. When the LLM creates an artefact with `type="tool"`, the `ArtefactManager` attempts to register it dynamically.

### Security Gate
To prevent untrusted LLMs from executing arbitrary code, dynamic tool registration is gated by the `allow_dynamic_tools` flag on the active `LollmsDiscussion` instance.
*   If `allow_dynamic_tools` is `False` (the default), the tool artefact is saved as a standard code file but is **NOT** parsed or executed.
*   If `allow_dynamic_tools` is `True`, the manager extracts the Python code and passes it to the active `LCPBinding` for immediate AST parsing and module execution.

### Lifecycle
1.  **Create**: LLM outputs `<artifact type="tool" name="my_tool">def tool_run(): ...</artifact>`.
2.  **Gate Check**: `ArtefactManager._register_tool_artefact()` checks `discussion.allow_dynamic_tools`.
3.  **Register**: If allowed, `LCPBinding.register_tool_from_code("my_tool", code)` is called.
4.  **Execute**: The tool `tool_my_tool` is now available in the active session registry.

---

## 🛠️ 5. Class Reference

*   **`ArtefactType`**: Registry defining the supported categories (`DATA`, `CODE`, `DOCUMENT`, `IMAGE`, `PRESENTATION`, `NOTE`, `SKILL`, `TOOL`, `SCRATCHPAD`).
* **`ArtefactManager`**: Orchestrates database CRUD operations, applies search-and-replace patches, manages version history squashing, and gates dynamic tool registration.
* **`FileImportMixin`**: Contains multi-modal parser subroutines for importing PDFs, Word documents, PowerPoint presentations, and audio files.
