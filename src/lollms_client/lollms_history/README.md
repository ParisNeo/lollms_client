# 📜 Lollms History Manager

The `HistoryManager` is the single source of truth for formatting, sanitizing, and exporting conversation history across both `LollmsDiscussion` and `LollmsPersonality`. 

It ensures that the LLM receives a perfectly structured context window that maximizes cognitive retention while strictly adhering to token limits and API schemas.

## 🧠 The Strict Non-Placeholder Strategy (Anti-Hallucination)

A critical issue in autonomous agentic loops is **Cognitive Thread Loss**. When an agent executes a tool or writes an artifact, the raw XML/JSON can consume massive amounts of the context window. Historically, systems solved this by replacing the assistant's message with an opaque placeholder like `[Assistant executed batched actions]`. 

**The Problem:** After 3-4 rounds, the LLM's context is filled with opaque placeholders. It forgets what it actually said or did, hallucinates that it hasn't done anything, and enters infinite repetition loops trying to execute the same action.

**The Solution:** `HistoryManager` enforces a **Strict Non-Placeholder Strategy** via `_sanitize_for_context()`.

When exporting history, the manager evaluates the `distance_from_end` (how many messages ago the action occurred):

1. **Strict Preservation Zone (Last 4 Actions)**: For the most recent 4 assistant messages, the manager preserves the LLM's raw reasoning and conversational text verbatim. Only bulky structural tags (`<tool>`, `<artifact>`) are stripped to lightweight `[🔒 Tool stripped]` markers. The agent can always see exactly what it said and why it said it for its current active thread.
2. **Aggressive Compression Zone (Older Actions)**: For messages older than 4 rounds, the manager applies aggressive compression, replacing the entire message content with opaque placeholders to maximize context budget for older history.

This guarantees the agent never loses its train of thought on the active task, while still maintaining long-term context efficiency.

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