# `get_context_status` — Documentation

The `get_context_status` method provides a detailed, real-time diagnostic of how the LLM's context window is currently allocated. It precisely accounts for every byte of data that will be sent to the model, including persistent data zones, volatile tool outputs (scratchpad), versioned artefacts, and message history.

---

## Return Structure

The method returns a nested dictionary containing token counts, usage percentages, and a granular breakdown of specific zones.

```python
{
    "max_tokens": int,      # The budget defined by max_context_size (defaults to 8192)
    "current_tokens": int,  # Sum of all tokens in all active zones
    "percent": float,       # (current_tokens / max_tokens) * 100
    "zones": {
        "system_context": {
            "tokens": int,
            "breakdown": {
                "system_prompt": {"tokens": int},
                "memory": {"tokens": int},
                "user_data_zone": {"tokens": int},
                "discussion_data_zone": {"tokens": int},
                "personality_data_zone": {"tokens": int},
                "scratchpad": {"tokens": int},        # Transient tool data
                "pruning_summary": {"tokens": int},   # Summary from context compression
                "artefacts": {                        # Grouped by type
                    "tokens": int,
                    "types": {
                        "code": {"tokens": int, "count": int},
                        "document": {"tokens": int, "count": int},
                        "skill": {"tokens": int, "count": int}
                    }
                }
            }
        },
        "message_history": {
            "tokens": int,
            "breakdown": {
                "text_tokens": int,
                "image_tokens": int,
                "message_count": int
            }
        },
        "discussion_images": {
            "tokens": int,
            "count": int
        }
    }
}
```

---

## Core Components

### 1. Fixed Image Tokenization
To ensure predictable context budgeting across different vision models, **every image consumes exactly 256 tokens**. This includes images attached to specific messages and global discussion-level images.

### 2. The System Context
This represents the "static" instructions and knowledge injected before the conversation starts. It includes:
*   **Persistent Zones**: Memory, user preferences, and task-specific data.
*   **The Scratchpad**: This accounts for the transient tool results and internal technical data generated during agentic turns. It is volatile and never persisted to the database.
*   **Grouped Artefacts**: Active artefacts are contabilized and categorized by type (e.g., `code`, `document`), allowing developers to see which category is consuming the most space.

### 3. Message History
This tracks the linear path of messages from the root to the current `branch_tip_id`.
*   If **Context Compression** has occurred, this only contabilizes messages appearing *after* the pruning point.
*   The tokens include message headers (e.g., `!@>user: `) and the fixed cost for images within those messages.

### 4. Discussion Images
Global images attached to the discussion (not specific to a single message) are contabilized here.

---

## Usage Example

```python
status = discussion.get_context_status()

print(f"Usage: {status['current_tokens']} / {status['max_tokens']} ({status['percent']}%)")

# Check if artefacts are consuming too much context
if "artefacts" in status["zones"]["system_context"]["breakdown"]:
    art_stats = status["zones"]["system_context"]["breakdown"]["artefacts"]
    print(f"Total Artefact Tokens: {art_stats['tokens']}")
    
    for atype, data in art_stats["types"].items():
        print(f" - {atype}: {data['tokens']} tokens ({data['count']} files)")

# Trigger manual pruning if context is over 80%
if status["percent"] > 80:
    discussion.summarize_and_prune(max_tokens=int(status["max_tokens"] * 0.7))
```

## Integration with Logic
The `chat()` loop uses this method at the beginning of every turn to decide if **Context Compression** is required. If the `percent` exceeds the safety threshold (typically 80%), the system will automatically attempt to summarize history or request the deactivation of heavy artefacts to free up space for the model's response.