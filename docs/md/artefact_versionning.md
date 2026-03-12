# Artefact Versioning — Documentation

The artefact system uses a **Linear Versioning Model**. Every time an artefact is modified (either by the LLM via XML tags or programmatically via the API), a new immutable record is created in the database with an incremented version number.

---

## 1. How the LLM Sees Versions

The system automatically manages the LLM's context to prevent token waste and confusion:
*   **Latest Version Priority**: Only the version marked as `active` is injected into the context. By default, this is always the most recently created version.
*   **Version Headers**: The LLM receives a header identifying the version it is reading:
    `###[Code] main.py (v3 | 5 total versions exist)`
*   **Reference Awareness**: Because the header includes the total count, the LLM knows it can "look back" by requesting a revert if the current version is broken.

---

## 2. Recovering History (Application Side)

The application can retrieve the full evolution of an artefact using the `get_history` method. This is useful for building "Timeline" or "Undo" UI components.

```python
# Get all versions of a file
history = discussion.artefacts.get_history("app.py")

for version in history:
    print(f"Version: {version['version']}")
    print(f"Created at: {version['created_at']}")
    print(f"Content Length: {len(version['content'])}")
    print(f"Active in Context: {version['active']}")
```

---

## 3. Reverting to a Previous Version

There are two ways to revert an artefact. Both actions create a **new version** that copies the content of the target version, preserving the audit trail.

### A. Programmatic Revert
```python
# Revert 'main.py' to version 1. This creates version 4 with version 1's content.
discussion.artefacts.revert("main.py", target_version=1)
```

### B. LLM-Triggered Revert
The LLM can trigger a revert if it realizes a previous implementation was better:
```xml
<revert_artefact name="main.py" version="1" />
```

---

## 4. Versioning Workflow

1.  **Creation**: `discussion.chat()` results in `<artefact name="test.py">...</artefact>`. **(v1, Active)**
2.  **Update**: The LLM patches the file using SEARCH/REPLACE. **(v2, Active)**. *Note: v1 is automatically marked Inactive.*
3.  **Validation**: The App checks `get_history("test.py")` and sees two entries.
4.  **Token Count**: `get_context_status()` counts only the tokens for **v2**.
5.  **Recovery**: If the user wants to go back, the App calls `revert(title, 1)`. **(v3, Active)**. *v3 content matches v1.*