---
name: Lollms Discussion and Chat
description: Teaches the model how to manage stateful discussions, track conversational trees/branches, navigate alternative sibling replies, and execute context-aware generation.
author: ParisNeo
version: 1.0.0
category: lollms_client/discussion
created: 2026-05-24
---

# Lollms Discussion and Chat

Instead of maintaining a simple list of message dicts, `LollmsDiscussion` organizes conversations as a hierarchical tree, allowing first-class support for branching, alternative model replies, and session persistence.

## 1. Creating a Discussion
A discussion represents a stateful chat session. It can be fully in-memory or persisted to a local SQLite database.

```python
from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "gemma4:e2b"})

# Create an in-memory, stateful discussion
discussion = LollmsDiscussion(lollmsClient=client)
discussion.system_prompt = "You are a helpful software engineering companion."
```

## 2. Managing Chat Messages
Messages are stored in the tree. When adding a message, it automatically attaches to the current active branch tip.

```python
# Add user message
msg_user = discussion.add_message(sender="user", content="What is an RPC?")

# Add assistant response
msg_assistant = discussion.add_message(sender="assistant", content="Remote Procedure Call.")

print(f"Active Branch Tip Message ID: {discussion.active_branch_id}")
```

## 3. Branching and Forking
A branch represents a conversational path from the root message to a leaf. If the user or system wants to explore an alternative path, we "fork" the conversation.

```python
# Fork an alternative branch starting from the first user question
fork_msg = discussion.fork_from(
    message_id=msg_user.id,
    label="Alternative Topic Branch",
    initial_content="Actually, tell me about REST instead."
)

# Switch back and forth between branches
discussion.switch_branch(msg_assistant.id)  # Returns to the original RPC branch
```

## 4. Executing Discussion Chat
The high-level `chat()` method automatically assembles all context zones (system instructions, memory, active artifacts, discussion history) and sends a correctly formatted template to the model.

```python
# High-level chat execution
response = discussion.chat(
    user_message="Explain REST constraints",
    enable_memory=True,
    enable_artefacts=True
)
print(response)
```
