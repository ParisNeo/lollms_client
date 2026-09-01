# 📜 Lollms History Manager

The `HistoryManager` is the single source of truth for formatting, sanitizing, and exporting conversation history across both `LollmsDiscussion` and `LollmsPersonality`. 

It ensures that the LLM receives a perfectly structured context window that maximizes cognitive retention while strictly adhering to token limits and API schemas.

## 🧠 Zero-Amnesia & Anti-Mimicry Strategy

A critical issue in autonomous agentic loops is balancing **Cognitive Recall (No Amnesia)** with **Output Purity (No Mimicry)**:

*   **Amnesia Risk**: If the system replaces the assistant's code or skills with opaque placeholders like `[🔒 Action stripped]`, the model forgets what it generated 2 rounds ago and hallucinates that the task is still unstarted or repeats it.
*   **Mimicry Risk**: If the system inserts artificial markers like `[SYSTEM: ...]` or `[🔒 Tool stripped]`, the LLM adopts these templates and begins outputting mock status markers instead of executing real tags (`<skill>`, `<artifact>`, `<tool>`).

### 🛡️ The Two-Zone Solution in `HistoryManager._sanitize_for_context()`

When exporting context to the model, `HistoryManager` splits history into two strictly managed zones based on `distance_from_end`:

1. **Strict Preservation Zone (Last 4 Actions — Zero Amnesia)**:
   - Preserves all functional XML tags (`<skill>`, `<artifact>`, `<note>`, `<tool>`) and their verbatim code/markdown bodies.
   - Strips **only** system runner logs (`<processing>` blocks, `<!-- status:... -->` comments).
   - The model can read its exact prior code and reasoning without cognitive loss.

2. **Clean Compression Zone (Older Actions ≥ 4 — Zero Mimicry)**:
   - For deeply historical messages, bulky tag bodies are compressed into clean self-closing reference tags (e.g. `<artifact name="file.py" type="code" status="saved" />` and `<skill title="name" status="saved" />`).
   - **Eliminates all `[🔒 ...]` placeholder strings**, ensuring the LLM is never exposed to fake markers it could imitate.

## 🔄 Virtual History Integration

During an active agentic loop (`chat()`), the agent generates a `virtual_history` list of actions and system responses that have not yet been committed to the database. 

`HistoryManager.export()` seamlessly appends this `virtual_history` to the sanitized historical branch. Virtual history messages are **always treated as recent** (distance 0) to ensure the agent has full, unstripped visibility of what it is actively doing in the current turn.

## 🛠️ API Reference

### `HistoryManager.export(...)`
Exports the discussion history in the specified format.

**Parameters:**
- `context`: The discussion or personality instance.
- `format_type`: `openai_chat`, `ollama_chat`, `lollms_text`, or `markdown`.
- `branch`: The list of messages (the active branch).
- `virtual_history`: Optional list of active, uncommitted agentic actions.
- `max_allowed_tokens`: If provided, truncates the history from the oldest messages to fit the token budget.
- `suppress_system_prompt` / `suppress_images`: Flags for specific export use cases (e.g., title generation).
- `system_prompt_override`: Forces a specific system prompt instead of the context's default.

### `HistoryManager._normalize_openai_messages(messages)`
Ensures OpenAI API compliance by fusing all system messages into one, merging consecutive same-role messages, and ensuring the first non-system message is a `user` message.