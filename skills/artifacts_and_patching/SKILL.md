---
name: Lollms Artifacts and Surgical Patching
description: Teaches the model how to formulate strategic <coding_plan> blocks, apply indentation-agnostic Aider search/replace patches, manage Git-like version logs, and control in-message status reporting.
author: ParisNeo
version: 1.0.0
category: lollms_client/artifacts
created: 2026-05-24
---

# Lollms Artifacts and Surgical Patching

## 0. UI Header vs. Content Streaming Protocol
When artifacts are built or patched:
- **Header Presentation**: Only file names and structural boundaries (e.g., `Section: 1. Setup`, `def calculate_metrics()`, `class SessionManager`) are displayed in UI titles and subtitles.
- **Content Viewport**: Verbatim content, code lines, table records, and text streams exclusively inside the collapsible artifact container.
- **Immediate Real-time Flush**: Artifacts are synchronized to disk the exact millisecond the closing `</artifact>` tag is received, ensuring persistent safety across interruptions.

This skill explains how to utilize, create, and surgically update persistent artifacts using Lollms.

## 1. Creating and Updating Artifacts
Artifacts are persistent, versioned documents (code, text, notes) stored in discussion metadata. 
- Creating new: Provide 100% of the content inside `<artifact name="filename" type="type">...</artifact>`.
- Updating existing: Use **SEARCH/REPLACE Aider-style patches** to modify targeted regions.

### Visibility Control (`visibility` attribute)

Every artifact accepts an optional `visibility` attribute that controls how much of it is loaded into your context:

| Value | Behavior |
| :--- | :--- |
| `tree_unlockable` (default) | Listed in the workspace tree as `[U]`. Not loaded into context. Use `<unlock_file>` to load it. |
| `full` | Fully loaded into context as `[C]` **permanently**. Use for files you will reference repeatedly. |
| `single_round` | Fully loaded as `[C]` for **exactly one turn**, then auto-demoted to `[U]`. **This is the default for generated images.** |
| `metadata` | Only signatures/schema loaded as `[M]`. |
| `hidden` | Completely excluded from tree and context. |

**🖼️ Image default — `single_round`**: When you generate an image (`<generate_image>`, `<edit_image>`, or a tool produces one), it is registered with `single_round` visibility. You will see the pixels on your **next** round (so you can verify the result and decide whether to edit it), but the payload is freed from context afterwards. If you need to keep an image loaded permanently, emit `<unlock_file>filename.png</unlock_file>` after creation, or pass `visibility="full"` on a subsequent write.

## 2. Strategic Formulation: The Coding Plan
Before outputting any `<artifact>` update, the model must formulate a plan inside `<coding_plan>` to organize its implementation strategy:

```xml
<coding_plan>
  - Goal: Implement safe multiplication
  - Specialist Persona: Python Programmer
  - Target Artifacts: math_ops.py
  - Implementation Details: Add a method compute_multiply(a, b).
</coding_plan>
```

## 3. Surgical Patch Rules
When patching an existing file, the `SEARCH` block must copy lines **exactly as they appear** in the active artifact (matching all leading indentation spaces and comments character-for-character).

```xml
<artifact name="math_ops.py" type="code" language="python">
<<<<<<< SEARCH
        def compute_sum(a, b):
            # Target region
            return a + b
=======
        def compute_sum(a, b):
            # Target region
            return a + b

        def compute_product(a, b):
            # Safe multiplication logic
            return a * b
>>>>>>> REPLACE
</artifact>
```

## 4. Indentation-Agnostic Fuzzy Matching
If the LLM outputs a patch at column-0, or with slightly imperfect indentation, the Lollms Aider patch engine automatically:
1. Calculates the alignment offset of the matched region in the file.
2. Re-indents the `REPLACE` block to align with the rest of the code structure on disk.

## 5. Controlling In-Message Status Reporting
When performing complex tasks, the system can write status updates to the chat bubble. Use `enable_in_message_status` to toggle this behavior:
- `enable_in_message_status=True` (Default): Streaming chunks emit `<processing>` tags showing updates in real-time.
- `enable_in_message_status=False`: The prose stream remains 100% clean, and updates are fired only via background structured callbacks.
```