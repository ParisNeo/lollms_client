# 📜 Lollms History Manager

The `HistoryManager` provides centralized formatting, normalization, and export functions for conversation histories across `LollmsDiscussion` and `LollmsPersonality`.

---

## 🏛️ 1. Two-Tier Context Integration

In the two-tier agentic architecture:
- **Workers** operate in disposable, task-specific contexts that are never appended to the permanent conversation DAG.
- **Orchestrators** maintain a clean, persistent conversation history. The `HistoryManager` exports the exact coordination trace (plans, delegations, and report envelopes) for the orchestrator while providing OpenAI-compliant message alternations.

---

## 🛠️ 2. Core API Methods

### `HistoryManager.export(...)`
Exports the conversation history in the requested format (`openai_chat`, `ollama_chat`, `lollms_text`, or `markdown`).

**Parameters**:
- `context`: The active discussion or personality instance.
- `format_type`: Target output format string.
- `branch`: Chronological list of message nodes.
- `max_allowed_tokens`: Token ceiling for history truncation.
- `suppress_system_prompt`: If `True`, excludes the system context.
- `suppress_images`: If `True`, excludes image payload dictionaries.
- `system_prompt_override`: Explicit system prompt replacement.

### `HistoryManager._normalize_openai_messages(messages)`
Ensures compliance with OpenAI-compatible API schemas:
1. Merges all initial system entries into a single system message at index 0.
2. Combines consecutive messages with the same role into a single message.
3. Guarantees strict `user` / `assistant` role alternation.