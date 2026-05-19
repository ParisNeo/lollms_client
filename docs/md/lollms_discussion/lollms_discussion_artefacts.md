# 📌 LollmsDiscussion – Artefacts Management Reference

### Introduction

Artefacts are documents stored in the discussion’s metadata (e.g., code, text, configurations).
The **data zone** is a contextual space visible to the AI, where one or more artefacts can be loaded to enrich responses.

Each artefact contains:

* `title` *(str)* – unique identifier.
* `version` *(int)* – artefact version number.
* `content` *(str)* – artefact body (text/code).
* `created_at`, `updated_at` *(ISO datetime)* – timestamps.
* `is_loaded` *(bool)* – whether the artefact is currently loaded in the data zone.

> ⚠️ Unlike before, the **data zone can now hold multiple artefacts at the same time**.
> Each artefact is wrapped in its own block, separated by blank lines for readability.

---

## 🔹 List artefacts

```python
artefacts = discussion.list_artefacts()
```

* Returns all artefacts.
* Automatically upgrades old schema entries (missing fields).
* Ensures `is_loaded=False` by default.

---

## 🔹 Add an artefact

```python
discussion.add_artefact("spec_doc", "Some technical specification text")
```

* Creates a new artefact (default version = `1`).
* If an artefact with the same `(title, version)` exists, it is replaced.

---

## 🔹 Update an artefact

```python
discussion.update_artefact("spec_doc", "Updated specification text")
```

* Creates a **new version** by incrementing `version`.
* Keeps previous versions intact.

---

## 🔹 Retrieve an artefact

```python
artefact = discussion.get_artefact("spec_doc")       # latest version
artefact_v2 = discussion.get_artefact("spec_doc", 2) # specific version
```

---

## 🔹 Load an artefact into the data zone

```python
discussion.load_artefact_into_data_zone("spec_doc")
discussion.load_artefact_into_data_zone("design_notes", version=2)
```

* Appends the artefact’s content into `discussion_data_zone`.
* Multiple artefacts can be loaded at the same time.
* Each artefact block is separated by a blank line:

```
--- Document: spec_doc v1 ---
Some technical specification text
--- End Document: spec_doc ---

--- Document: design_notes v2 ---
Updated design notes content
--- End Document: design_notes ---
```

* Marks the artefact as `is_loaded=True`.
* Makes the artefact content available to the AI context.

---

## 🔹 Unload an artefact

```python
discussion.unload_artefact_from_data_zone("spec_doc")
```

* Removes only the matching artefact block from `discussion_data_zone`.
* Leaves other artefacts intact.
* Marks the artefact as `is_loaded=False`.

---

## 🔹 Check if an artefact is loaded

```python
discussion.is_artefact_in_data_zone("spec_doc")  # → True / False
```

---

## 🔹 Export the current context as an artefact

```python
discussion.export_as_artefact("saved_context", version=1)
```

* Captures the **current `discussion_data_zone` content** (with all loaded artefacts).
* Saves it as a new artefact in the metadata.
* Useful for archiving snapshots of the AI’s working context.

---

## 🔹 Typical usage flow

1. `add_artefact()` → create new artefacts.
2. `load_artefact_into_data_zone()` → load one or multiple artefacts into the context.
3. `is_artefact_in_data_zone()` → check load state.
4. `unload_artefact_from_data_zone()` → remove only the selected artefact.
5. `export_as_artefact()` → archive the current AI context (all loaded artefacts).

---

## 🔹 Multi-Artefact Workflow Example

```python
from lollms_client import LollmsDiscussion
# --- Create a discussion instance ---
discussion = LollmsDiscussion()

# --- Add artefacts ---
discussion.add_artefact("spec_doc", "Technical specification text v1")
discussion.add_artefact("design_notes", "Design notes content v2", 2)

# --- Load multiple artefacts into the data zone ---
discussion.load_artefact_into_data_zone("spec_doc")
discussion.load_artefact_into_data_zone("design_notes", version=2)

# --- Check which artefacts are loaded ---
print(discussion.is_artefact_in_data_zone("spec_doc"))      # True
print(discussion.is_artefact_in_data_zone("design_notes"))  # True

# --- View the current discussion data zone ---
print(discussion.discussion_data_zone)
"""
--- Document: spec_doc v1 ---
Technical specification text v1
--- End Document: spec_doc ---

--- Document: design_notes v2 ---
Design notes content v2
--- End Document: design_notes ---
"""

# --- Export the current context as a new artefact ---
exported = discussion.export_as_artefact("full_context_snapshot", version=1)
print("Exported artefact:", exported)

# --- Unload one artefact while keeping the other ---
discussion.unload_artefact_from_data_zone("spec_doc")
print(discussion.is_artefact_in_data_zone("spec_doc"))      # False
print(discussion.is_artefact_in_data_zone("design_notes"))  # True

# --- View the updated discussion data zone ---
print(discussion.discussion_data_zone)
"""
--- Document: design_notes v2 ---
Design notes content v2
--- End Document: design_notes ---
"""
```

---

This reference now fully reflects:

* **Multiple artefacts loaded at once**
* **Blank lines between artefacts** for readability
* **Individual unloading** and `is_loaded` tracking
* **Exporting the entire context** as a new artefact
