# Custom Artefact Types — Documentation

The `LollmsDiscussion` artefact system allows you to extend the standard document types with custom categories tailored to your application (e.g., `requirement`, `test_case`, `log`, `blueprint`). 

Registering a custom type ensures the LLM is aware of the category in its instructions and that the system correctly validates and groups these artefacts during token contabilization.

---

## 1. Registering Custom Types

Use the `ArtefactType.register_custom_type` method. This should typically be done during your application's initialization phase.

```python
from lollms_client.lollms_discussion import ArtefactType

# Register a simple type (label defaults to Capitalized name)
ArtefactType.register_custom_type("requirement")

# Register with a specific human-readable label for UI/logging
ArtefactType.register_custom_type("test_case", label="QA Test Case")
ArtefactType.register_custom_type("skill", label="AI Capability")
```

---

## 2. LLM Awareness (System Prompt)

Once a type is registered, it is automatically injected into the **System Prompt** during `chat()` or `simplified_chat()`. The LLM receives updated instructions listing all valid types:

**Auto-generated prompt segment:**
```text
=== ARTEFACT SYSTEM ===
...
Supported types: code, document, file, image, note, requirement, skill, test_case
=== END ARTEFACT INSTRUCTIONS ===
```

This allows the LLM to emit correct XML tags for your custom types:
```xml
<artefact name="PRD_001" type="requirement">
  The system shall support multi-factor authentication.
</artefact>
```

---

## 3. Integration with `get_context_status`

Custom types are first-class citizens in the context status breakdown. When you call `get_context_status()`, the `artefacts` breakdown will group your custom types automatically:

```python
status = discussion.get_context_status()

# Example output snippet:
# "artefacts": {
#     "tokens": 1240,
#     "types": {
#         "code": {"tokens": 800, "count": 2},
#         "requirement": {"tokens": 440, "count": 1}  <-- Custom type identified
#     }
# }
```

---

## 4. Manual Creation

You can also use the `ArtefactManager` API to add custom-typed artefacts programmatically:

```python
discussion.artefacts.add(
    title="Validation Logic",
    artefact_type="test_case",
    content="Check if input is non-null...",
    active=True
)
```

---

## Standard vs. Custom Types

| Type | Persistence | LLM Awareness | Status Grouping |
| :--- | :--- | :--- | :--- |
| **Standard** (`code`, `note`, etc.) | ✅ Yes | ✅ Instructions included | ✅ Individual group |
| **Custom** (`requirement`, etc.) | ✅ Yes | ✅ Instructions included | ✅ Individual group |

**Note:** If the LLM attempts to use a type that has *not* been registered, the system will fallback to the `default_type` (usually `code` or `document`) during XML processing to prevent data loss. Always register your types to ensure proper categorization.