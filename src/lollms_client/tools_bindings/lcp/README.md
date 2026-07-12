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

## ✏️ 6. Updating Artefact Content

You can modify the content of an existing artefact using the `update()` method. By default, this creates a new version in the database, preserving the history of previous states.

```python
art = discussion.artefacts.update(
    title="analysis_script.py",
    new_content="import pandas as pd\nprint('new code')",
    commit_message="Refactored import logic"
)
```

### Overwriting the Current Version

If you want to update the content without creating a new version history entry, you can set `create_new_version=False`. This will overwrite the content of the current active version.

```python
art = discussion.artefacts.update(
    title="temp_notes.md",
    new_content="Updated temporary notes without version bump.",
    create_new_version=False
)
```

---

## ⚠️ 7. Import Conflict Resolution

When importing files into the artefact system, there is a possibility of title collisions (e.g., importing `README.md` from two different sources). The `import_file` method provides an `on_conflict` parameter to define the resolution strategy.

### Strategies

1. **`suffix` (Default)**
   - **Behavior**: If an artifact with the target title already exists, the new file is renamed with an incrementing suffix (e.g., `README_1.md`, `README_2.md`).
   - **Use Case**: Preserving all imported files without losing any data or altering the original artifact.
   - **Result**: Creates a new artifact with the suffixed title. The original artifact remains untouched.

2. **`version`**
   - **Behavior**: The existing artifact is updated, and its version number is incremented. The physical file is overwritten with the new content, but the previous version is preserved in the database history.
   - **Use Case**: Importing an updated version of a file where you want to maintain a clear audit trail of changes.
   - **Result**: Updates the existing artifact and bumps the version (e.g., v1 → v2).

3. **`overwrite`**
   - **Behavior**: The existing artifact's content is replaced with the new content, but the version number is **not** incremented. Previous version history is preserved, but the active version is silently replaced.
   - **Use Case**: Correcting or silently updating a file without polluting the version history.
   - **Result**: Updates the existing artifact. The version number remains the same.

4. **`replace`**
   - **Behavior**: Completely purges all existing versions and history of the artifact, then creates a fresh `v1` baseline with the new content.
   - **Use Case**: Starting over cleanly when the previous iterations are no longer relevant or were imported in error.
   - **Result**: Deletes all previous database records and physical metadata, then creates a new `v1` artifact.

### Example

```python
# Import a file, replacing any existing artifact with the same name completely
discussion.import_file(
    path="path/to/new/README.md",
    mode="text",
    title="README.md",
    on_conflict="replace"
)
```

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

## 🧩 5. Artefact Properties Reference & Handling Guide

Every artifact in the system is represented as a dictionary (record) with a specific set of keys. Understanding the distinction between these properties is critical for correctly creating, updating, and referencing artifacts, especially when dealing with the Dual-Stream storage architecture.

### Core Properties

| Property | Type | Description & Handling Rules |
| :--- | :--- | :--- |
| `title` | `str` | **The primary key.** This is the high-level metadata name used by the LLM and the database to reference the artifact. It may contain subfolder paths (e.g., `My_subfolder/SKILL.md`). It is sanitized via `_sanitize_path_segments` to ensure cross-platform safety. When updating or retrieving an artifact, you query by this title. |
| `physical_path` | `str` | **The disk location.** This stores the exact relative path (including subfolders and extension) where the physical twin resides in `workspace_data/`. If not explicitly provided during `add()` or `update()`, it defaults to the `title`. **CRITICAL**: File-reading tools should be passed the `physical_path` (or `display_path` from the context zone) to ensure they open the correct file on disk. |
| `type` | `str` | The category of the artifact (e.g., `ArtefactType.CODE`, `ArtefactType.DATA`). Determines how the artifact is rendered in the context zone and which tools can operate on it. |
| `content` | `str` | The logical text content. For text/code files, this is the verbatim source code. For `DATA` artifacts, this holds the `.lam` schema description, **NOT** the raw binary bytes. |
| `version` | `int` | The version number. Incremented automatically on `update()` if `bump_version=True`. The `get()` method returns the highest version by default. |
| `visibility` | `str` | The context tier (`FULL`, `TREE_UNLOCKABLE`, `METADATA`, `TREE_LOCKED`, `HIDDEN`). Controls how the artifact appears in the LLM's prompt. See [Section 2: Multi-Tier Visibility Control](#-2-multi-tier-visibility-control). |
| `active` | `bool` | A legacy boolean flag that mirrors `visibility == FULL`. It is `True` if the artifact is fully loaded in context, `False` otherwise. |
| `language` | `str` | The programming or markup language (e.g., `python`, `html`). Used for syntax highlighting in the context zone and to infer file extensions. |
| `file_ext` | `str` | The explicit file extension (e.g., `.csv`, `.db`). **CRITICAL for DATA artifacts**: This determines how the physical file is written to disk and prevents binary corruption. |
| `logical_content` | `str` | Explicit storage for the `.lam` schema text of `DATA` artifacts. While usually mirrored in `content`, this field is the authoritative source for the logical twin during Dual-Stream sync operations. |
| `physical_data` | `bytes` | The raw binary bytes of a `DATA` or `IMAGE` artifact. **CRITICAL**: This field is stripped from the database record by `_get_all_raw()` to prevent JSON serialization crashes. It is only present in the dictionary returned directly by `add()` or `update()`. Never assume `art.get("physical_data")` will return bytes from a database query; rehydrate from disk if needed. |
| `token_count` | `int` | The estimated token count of the `content`. Used by the Context Budget Guard to prevent context overflow. |

### Handling Guidelines

#### 1. Title vs. Physical Path Decoupling
The architecture decouples the database key (`title`) from the disk location (`physical_path`).
*   **Creation**: If you create an artifact with `title="My_subfolder/script.py"`, the `physical_path` automatically mirrors this. The physical file is written to `workspace_data/My_subfolder/script.py`.
*   **Context Injection**: `build_artefacts_context_zone()` displays the `physical_path` to the LLM. When the LLM decides to read a file, it should use this exact string.
*   **Updating**: If you change the `title` during an update (`new_title`), the `physical_path` is updated to match, and the old physical file is deleted from disk.

#### 2. Data Artifact Safety (Binary Corruption Prevention)
`DATA` artifacts (like SQLite databases or CSVs) use the Dual-Stream protocol.
*   **Never write string `content` to a binary file**: The `_sync_to_disk_workspace` method explicitly refuses to write string `content` to `.db`/`.sqlite` files if `physical_data` is missing. This prevents the database header from being overwritten with `.lam` schema text.
*   **Rehydration**: When updating a `DATA` artifact's schema, the `update()` method automatically rehydrates `physical_data` by reading the existing bytes from disk *before* calling `add()`. This ensures the raw binary data is preserved across schema updates.

#### 3. Visibility and Context Budget
*   **Tool-Generated Files**: By default, tool-generated files >100KB are registered with `visibility=TREE_UNLOCKABLE` and `active=False` to prevent context bloat.
*   **Unlocking**: The LLM can use `<unlock_file>` to promote a file to `FULL` visibility. However, the Context Budget Guard blocks unlocking files >50,000 tokens, instructing the LLM to use tools (SQL, grep) instead.

#### 4. Image Artifacts
*   Image artifacts store base64 encoded strings in the `images` list and their MIME types in `image_media_types`.
*   They are an exception to the visibility doctrine: they are always registered with `visibility=FULL` and `active=True` when generated by tools, so they can be hydrated into the LLM's vision context immediately.

---

## 🛠️ 6. Class Reference

*   **`ArtefactType`**: Registry defining the supported categories (`DATA`, `CODE`, `DOCUMENT`, `IMAGE`, `PRESENTATION`, `NOTE`, `SKILL`, `TOOL`, `SCRATCHPAD`).
* **`ArtefactManager`**: Orchestrates database CRUD operations, applies search-and-replace patches, manages version history squashing, and gates dynamic tool registration.
* **`FileImportMixin`**: Contains multi-modal parser subroutines for importing PDFs, Word documents, PowerPoint presentations, and audio files.
