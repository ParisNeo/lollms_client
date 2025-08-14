# LollmsDiscussion: Comprehensive Conversational Context Management

The `LollmsDiscussion` class is a powerful and flexible component within the Lollms ecosystem, designed to manage complex conversational interactions with Large Language Models (LLMs). It provides a robust framework for handling multi-turn dialogues, including conversation history, branching, context pruning, persistent data zones, and multi-modal inputs (like images).

Whether you need a simple in-memory chat session or a persistent, complex conversation with branching capabilities stored in a database, `LollmsDiscussion` offers the tools to manage it effectively.

## Core Concepts

### Discussions vs. Messages

At its heart, `LollmsDiscussion` is a collection of `LollmsMessage` objects.
*   **`LollmsDiscussion`**: Represents an entire conversation, acting as a container for all messages, managing conversation-level metadata, system prompts, data zones, and the active conversational branch.
*   **`LollmsMessage`**: Represents a single turn or entry in the conversation. Each message has a sender, content, and can optionally include images, parent message IDs (for branching), and metadata.

### Persistence: In-Memory vs. Database-Backed

`LollmsDiscussion` supports two modes of operation:
*   **In-Memory**: Discussions created without a `LollmsDataManager` will exist only in the application's memory and will be lost when the application closes. This is suitable for transient, short-lived interactions.
*   **Database-Backed**: By providing a `LollmsDataManager` during creation, discussions and their messages can be persistently stored in a SQLite database (or any other SQLAlchemy-supported database). This enables loading, saving, and managing conversations across application sessions.

### Branching

Conversations are not always linear. `LollmsDiscussion` inherently supports branching, allowing you to explore different responses from a specific point in the conversation. Each message can have a `parent_id`, forming a tree structure. The `active_branch_id` keeps track of the current conversational path.

### Context Pruning (Non-Destructive)

Long conversations can exceed the LLM's context window. `LollmsDiscussion` implements a non-destructive pruning mechanism. Instead of deleting old messages, it summarizes them into a `pruning_summary` and marks a `pruning_point_id`. When exporting the context for the LLM, the summarized text is used for the older parts, preserving the full detail of recent turns.

### Data Zones for Persistent Context

`LollmsDiscussion` offers dedicated "data zones" to inject persistent information into the LLM's context without cluttering the main conversation history:

*   **`system_prompt`**: The primary system-level instruction for the LLM.
*   **`user_data_zone`**: For information specific to the user across multiple discussions (e.g., user preferences, persona details).
*   **`discussion_data_zone`**: For information specific to the current discussion that might not be part of the direct chat (e.g., project details, specific instructions for this session).
*   **`personality_data_zone`**: When using `LollmsPersonality`, this zone is populated with data specific to the loaded personality.
*   **`memory`**: A long-term memory store that can accumulate knowledge extracted from discussions over time using the `memorize` method.

These zones are automatically included in the `lollms_text` export format, providing the LLM with relevant background context.

### Multi-Modality (Image Handling)

`LollmsDiscussion` supports multi-modal interactions, primarily through image handling:

*   **Discussion-Level Images**: Images can be added directly to the discussion, making them part of the overall context, typically included with the system prompt when exporting to multi-modal formats (like OpenAI's chat completion API).
*   **Message-Level Images**: Images can be attached to individual messages, providing visual context for specific turns in the conversation.
*   **Activation State**: Both discussion-level and message-level images have an "active" state, allowing you to dynamically include or exclude them from the LLM's prompt without deleting them.

## Supporting Classes: `LollmsDataManager` and `LollmsMessage`

Before diving into `LollmsDiscussion`, it's helpful to understand its main dependencies:

### `LollmsDataManager`

The `LollmsDataManager` is responsible for:
*   Establishing and managing the database connection.
*   Dynamically creating SQLAlchemy ORM models (`DiscussionModel`, `MessageModel`) based on provided mixins and an optional encryption key.
*   Handling database schema creation and migrations.
*   Providing methods to create, retrieve, list, and delete `Discussion` records from the database.

You typically initialize it once at the start of your application to manage your persistent discussions.

**Example Initialization:**

```python
from lollms_client.lollms_discussion import LollmsDataManager, create_dynamic_models
from pathlib import Path

# Setup database path
db_path = Path("./test_discussion.db")
db_uri = f"sqlite:///{db_path.resolve()}"

# Optional: Add an encryption key for sensitive data in the database
# Make sure to keep this key secure and consistent across sessions if used.
# encryption_key = "my_super_secret_key"
# data_manager = LollmsDataManager(db_uri, encryption_key=encryption_key)

# Or, without encryption:
data_manager = LollmsDataManager(db_uri)

# This will create tables if they don't exist and perform migrations
# data_manager.create_and_migrate_tables()
print(f"Database Manager initialized for {db_uri}")
```

### `LollmsMessage`

`LollmsMessage` is a lightweight proxy class that wraps the underlying SQLAlchemy ORM message object (or a `SimpleNamespace` for in-memory messages). It provides a clean, attribute-based interface to access message data, including `id`, `sender`, `content`, `parent_id`, `tokens`, `images`, `active_images`, `metadata`, etc.

When you interact with messages within a `LollmsDiscussion`, you'll primarily be working with `LollmsMessage` instances.

**Key Features of `LollmsMessage`:**

*   **Proxy Access**: Direct access to underlying ORM object attributes (e.g., `message.content`, `message.sender`).
*   **Image Management**: `get_all_images()`, `get_active_images()`, `toggle_image_activation()`.
*   **Metadata**: `set_metadata_item()`.

You generally don't instantiate `LollmsMessage` directly; instead, you get them from `LollmsDiscussion` methods like `add_message()` or `get_messages()`.

## Class: `LollmsDiscussion`

### Initialization

`LollmsDiscussion` instances are typically created using the `create_new` class method or loaded via `LollmsDataManager`.

```python
from lollms_client.lollms_core import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from pathlib import Path
import os

# Assume lollms_client is already initialized
# For example purposes, we'll create a dummy one.
class MockLollmsClient:
    def count_tokens(self, text): return len(text.split())
    def detokenize(self, tokens): return " ".join(tokens)
    def tokenize(self, text): return text.split()
    def count_image_tokens(self, image_b64): return 100 # Mock value
    def generate_text(self, prompt, **kwargs): return f"AI response to: {prompt}"
    def generate_structured_content(self, prompt, schema, system_prompt, **kwargs):
        if "title" in schema.get("properties", {}): return {"title": "Mocked Discussion Title"}
        return {"result": "mock"}
    def remove_thinking_blocks(self, text):
        return text.replace("<thinking>", "").replace("</thinking>", "").replace("<think>", "").replace("</think>", "")
    # Mock for agentic calls
    def generate_with_mcp_rag(self, prompt, **kwargs):
        print(f"Mock Agentic Call with prompt snippet: {prompt[:100]}...")
        return {"final_answer": "Mock agentic response.", "final_scratchpad": "Agent thought process.", "tool_calls": [], "sources": []}
    # Mock chat function
    def chat(self, discussion_obj, **kwargs):
        # This mocks the internal chat call, which just takes a discussion object
        # and its messages to build context
        branch_tip_id = kwargs.get("branch_tip_id", discussion_obj.active_branch_id)
        # In a real scenario, export() would be called here to build the prompt
        # For this mock, we just return a simple response
        return "Mock response from LollmsClient chat."

lollms_client = MockLollmsClient()

# Initialize LollmsDataManager for persistent discussions
db_path = Path("./test_discussion.db")
if db_path.exists():
    db_path.unlink() # Start fresh for example
data_manager = LollmsDataManager(f"sqlite:///{db_path.resolve()}")

# 1. Create a new in-memory discussion (not saved to DB)
in_memory_discussion = LollmsDiscussion.create_new(
    lollms_client=lollms_client,
    system_prompt="You are a helpful assistant.",
    autosave=False # No autosave for in-memory
)
print(f"New In-Memory Discussion ID: {in_memory_discussion.id}")

# 2. Create a new database-backed discussion
db_discussion_1 = LollmsDiscussion.create_new(
    lollms_client=lollms_client,
    db_manager=data_manager,
    system_prompt="You are a friendly AI companion.",
    discussion_metadata={"topic": "AI capabilities"},
    autosave=True # Automatically commits changes
)
print(f"New DB-Backed Discussion ID: {db_discussion_1.id}")

# 3. Load an existing discussion
# (First, we need to create one if it doesn't exist, as shown above)
# Let's say we want to load db_discussion_1 again in a new session
db_discussion_1.close() # Close the previous session before reloading

loaded_discussion = data_manager.get_discussion(
    lollms_client=lollms_client,
    discussion_id=db_discussion_1.id,
    autosave=True
)
if loaded_discussion:
    print(f"Loaded Discussion ID: {loaded_discussion.id}")
    print(f"Loaded System Prompt: {loaded_discussion.system_prompt}")
else:
    print("Failed to load discussion.")
```

### Attributes

`LollmsDiscussion` exposes several key attributes, many of which are proxied from the underlying database object or in-memory proxy (`_db_discussion`).

*   `id` (str): Unique identifier for the discussion.
*   `system_prompt` (str): The initial system-level instructions for the LLM.
*   `user_data_zone` (str): Persistent data related to the user.
*   `discussion_data_zone` (str): Persistent data specific to this discussion.
*   `personality_data_zone` (str): Data injected by a `LollmsPersonality`.
*   `memory` (str): Long-term knowledge accumulated via `memorize()`.
*   `participants` (dict): Dictionary mapping sender names to roles (e.g., `{"user": "User", "assistant": "AI"}`).
*   `active_branch_id` (str): The ID of the last message in the current active branch.
*   `discussion_metadata` (dict): A flexible dictionary for storing arbitrary discussion-level metadata. Accessed via the `metadata` property.
*   `created_at` (datetime): Timestamp of discussion creation.
*   `updated_at` (datetime): Timestamp of the last modification.
*   `pruning_summary` (str): The summarized content of pruned messages.
*   `pruning_point_id` (str): The ID of the message where pruning started.
*   `images` (List[str]): List of base64-encoded images associated directly with the discussion.
*   `active_images` (List[bool]): Parallel list indicating active status of `images`.
*   `max_context_size` (int): Maximum tokens allowed before pruning.
*   `autosave` (bool): If `True`, changes are automatically committed to the database.
*   `scratchpad` (str): A temporary workspace for agentic operations (not persisted by default).

### Key Methods

#### `add_message(**kwargs) -> LollmsMessage`

Adds a new message to the discussion. This is how you build the conversation history.

**Parameters:**
*   `sender` (str): The name of the sender (e.g., "user", "assistant").
*   `sender_type` (str): The type of sender ("user", "assistant", "system", etc.). Automatically inferred for "user".
*   `content` (str): The text content of the message.
*   `parent_id` (str, optional): The ID of the message this one branches off from. Defaults to `active_branch_id`.
*   `images` (List[str], optional): List of base64-encoded images for this message.
*   `active_images` (List[bool], optional): Activation status for `images`. Defaults to all `True`.
*   `metadata` (dict, optional): Arbitrary message-level metadata.

**Returns:**
*   `LollmsMessage`: The newly created message object.

```python
# Assuming db_discussion_1 is a valid LollmsDiscussion instance
# Add a user message
user_msg_1 = db_discussion_1.add_message(
    sender="user",
    content="Hello, how are you today?",
    metadata={"client_ip": "192.168.1.1"}
)
print(f"Added User Message: {user_msg_1.content} (ID: {user_msg_1.id})")

# Simulate AI response
ai_msg_1 = db_discussion_1.add_message(
    sender="assistant",
    content="I am doing great, thank you! How can I assist you?",
    parent_id=user_msg_1.id # Links to the user's message
)
print(f"Added AI Message: {ai_msg_1.content} (ID: {ai_msg_1.id})")

# Add another user message to the same branch
user_msg_2 = db_discussion_1.add_message(
    sender="user",
    content="Can you tell me a bit about Llama 3?",
    parent_id=ai_msg_1.id
)
print(f"Added User Message: {user_msg_2.content} (ID: {user_msg_2.id})")
```

#### `chat(user_message, personality, branch_tip_id, use_mcps, use_data_store, add_user_message, max_reasoning_steps, images, debug, **kwargs) -> Dict[str, LollmsMessage]`

The primary method for interacting with the LLM. It manages prompt construction, agentic behavior (if enabled), and message addition.

**Parameters:**
*   `user_message` (str): The user's input for the current turn.
*   `personality` (`LollmsPersonality`, optional): An instance of `LollmsPersonality` to influence the system prompt and potentially inject dynamic data.
*   `branch_tip_id` (str, optional): The ID of the message to branch from. If `None`, uses `active_branch_id`.
*   `use_mcps` (bool or List[str], optional): Controls the use of Multi-tool Co-operation Protocol (MCP) tools. `True` for all, `False`/`None` for none, `List[str]` for specific tools.
*   `use_data_store` (Dict[str, Callable], optional): Enables Retrieval Augmented Generation (RAG) by providing query functions for data stores.
*   `add_user_message` (bool): If `True`, a new user message is added. If `False`, it assumes regeneration on the current active user message.
*   `max_reasoning_steps` (int): Max cycles for agentic reasoning (default: 20).
*   `images` (List[str], optional): List of base64-encoded images for this user message.
*   `debug` (bool): If `True`, prints verbose debugging information.
*   `**kwargs`: Additional arguments passed to the underlying LLM generation (e.g., `streaming_callback`, `n_predict`, `temperature`).

**Returns:**
*   `Dict[str, LollmsMessage]`: A dictionary containing the newly added `user_message` and `ai_message` objects.

```python
# Assuming db_discussion_1 and lollms_client are initialized
# And we want to continue the conversation from the last message (user_msg_2)

# Simple chat without agentic features
print("\n--- Simple Chat Example ---")
response_messages = db_discussion_1.chat(
    user_message="Tell me more about the features of Llama 3.",
    streaming_callback=lambda token, msg_type, meta: print(token, end=''), # Simulate streaming
    temperature=0.7
)
print(f"\nAI Response from chat: {response_messages['ai_message'].content}")
print(f"Active branch ID after chat: {db_discussion_1.active_branch_id}")

# Example with dummy images (replace with actual base64 image data)
dummy_image_b64 = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

print("\n--- Multi-modal Chat Example (Simulated) ---")
mm_response = db_discussion_1.chat(
    user_message="What do you see in this image?",
    images=[dummy_image_b64],
    streaming_callback=lambda token, msg_type, meta: print(token, end=''),
    temperature=0.7
)
print(f"\nAI Multi-modal Response: {mm_response['ai_message'].content}")

# Example with simulated agentic call (requires lollms_client.generate_with_mcp_rag to be mocked)
print("\n--- Agentic Chat Example (Simulated) ---")
agent_response = db_discussion_1.chat(
    user_message="Can you search the internet for the latest news on Llama 3?",
    use_mcps=["internet_search"], # Simulate using an internet search tool
    debug=True,
    streaming_callback=lambda token, msg_type, meta: print(token, end='')
)
print(f"\nAgentic AI Response: {agent_response['ai_message'].content}")
print(f"Agentic AI Metadata: {agent_response['ai_message'].metadata}")
```

#### `regenerate_branch(branch_tip_id=None, **kwargs) -> Dict[str, LollmsMessage]`

Deletes the last AI response (if it's an AI message) and re-generates a new one based on the previous user prompt. If the last message is a user message, it simply re-generates the AI response for that user message.

**Parameters:**
*   `branch_tip_id` (str, optional): The ID of the message to regenerate from. If `None`, uses `active_branch_id`.
*   `**kwargs`: Additional arguments passed to the underlying `chat()` method.

**Returns:**
*   `Dict[str, LollmsMessage]`: A dictionary with the original user message and the newly generated AI message.

```python
print("\n--- Regenerate Branch Example ---")
# Assume we just had a chat turn and now we want to try again
# The last message added was an AI message from agent_response
print(f"Before regeneration, active branch tip: {db_discussion_1.active_branch_id}")

regenerated_messages = db_discussion_1.regenerate_branch(
    streaming_callback=lambda token, msg_type, meta: print(token, end='')
)
print(f"\nRegenerated AI Response: {regenerated_messages['ai_message'].content}")
print(f"After regeneration, active branch tip: {db_discussion_1.active_branch_id}")

# If you were to regenerate from a user message directly:
# first_user_msg = db_discussion_1.get_messages(branch_id=user_msg_1.id)[0]
# db_discussion_1.switch_to_branch(first_user_msg.id)
# regenerated_first_response = db_discussion_1.regenerate_branch(...)
```

#### `get_messages(branch_id: Optional[str] = None) -> List[LollmsMessage]`

Retrieves all messages in a specific conversation branch, from root to the specified `branch_id` (leaf message). If `branch_id` is `None`, it returns the messages of the current `active_branch_id`.

**Parameters:**
*   `branch_id` (str, optional): The ID of the leaf message of the desired branch.

**Returns:**
*   `List[LollmsMessage]`: A list of messages in the branch, ordered chronologically.

**Note:** The internal `get_branch(leaf_id)` method is used to trace back from a leaf to the root.

```python
# Get messages for the active branch
print("\n--- Get Messages (Active Branch) ---")
active_branch_messages = db_discussion_1.get_messages()
for msg in active_branch_messages:
    print(f"  {msg.sender.capitalize()} ({msg.id[:8]}): {msg.content[:50]}...")

# Example of branching:
# Suppose we want to explore an alternative response to user_msg_1
print("\n--- Branching Example ---")
ai_msg_branch_2 = db_discussion_1.add_message(
    sender="assistant",
    content="Alternative AI response to: Hello, how are you today? I am a bit sleepy.",
    parent_id=user_msg_1.id # This creates a new branch
)
print(f"Created new branch from {user_msg_1.id} with message {ai_msg_branch_2.id}")

# Now retrieve this new branch
branch_2_messages = db_discussion_1.get_messages(branch_id=ai_msg_branch_2.id)
print(f"Messages in Branch 2 (ending with {ai_msg_branch_2.id[:8]}):")
for msg in branch_2_messages:
    print(f"  {msg.sender.capitalize()} ({msg.id[:8]}): {msg.content[:50]}...")

# Switch active branch to the new one
db_discussion_1.switch_to_branch(ai_msg_branch_2.id)
print(f"Switched active branch to: {db_discussion_1.active_branch_id}")
```

#### `get_message(message_id: str) -> Optional[LollmsMessage]`

Retrieves a single message by its unique ID.

**Parameters:**
*   `message_id` (str): The ID of the message to retrieve.

**Returns:**
*   `LollmsMessage` or `None`: The message object if found, otherwise `None`.

```python
print("\n--- Get Single Message Example ---")
retrieved_msg = db_discussion_1.get_message(user_msg_1.id)
if retrieved_msg:
    print(f"Retrieved message content: {retrieved_msg.content[:50]}...")
```

#### `get_all_messages_flat() -> List[LollmsMessage]`

Retrieves all messages stored for this discussion as a flat list, regardless of their branch. Useful for UI rendering or full data analysis.

**Returns:**
*   `List[LollmsMessage]`: A list of all messages in the discussion.

```python
print("\n--- Get All Messages Flat Example ---")
all_messages = db_discussion_1.get_all_messages_flat()
for msg in all_messages:
    parent_info = f" (Parent: {msg.parent_id[:8]})" if msg.parent_id else ""
    print(f"  - {msg.sender.capitalize()} ({msg.id[:8]}){parent_info}: {msg.content[:50]}...")
```

#### `delete_branch(message_id: str)`

Deletes a message and all its descendant messages, effectively pruning a sub-tree from the conversation. Only supported for database-backed discussions.

**Parameters:**
*   `message_id` (str): The ID of the message at the root of the branch to delete.

```python
print("\n--- Delete Branch Example ---")
# Delete the alternative branch we just created (ai_msg_branch_2)
print(f"Active branch before deletion: {db_discussion_1.active_branch_id}")
db_discussion_1.delete_branch(ai_msg_branch_2.id)
print(f"Active branch after deletion: {db_discussion_1.active_branch_id}") # Should revert to previous branch or a new tip

# Verify deletion (ai_msg_branch_2 should no longer be found)
if db_discussion_1.get_message(ai_msg_branch_2.id) is None:
    print(f"Message {ai_msg_branch_2.id[:8]} and its branch successfully deleted.")
else:
    print(f"Error: Message {ai_msg_branch_2.id[:8]} still found.")
```

#### `export(format_type: str, branch_tip_id: Optional[str] = None, max_allowed_tokens: Optional[int] = None) -> Union[List[Dict], str]`

Exports the discussion history into a specified format for LLM consumption or display. Handles pruning.

**Parameters:**
*   `format_type` (str):
    *   `"lollms_text"`: Native Lollms format (`!@>sender:\ncontent\n`).
    *   `"openai_chat"`: OpenAI Chat Completion API format (list of dicts).
    *   `"ollama_chat"`: Ollama Chat API format (list of dicts).
    *   `"markdown"`: Human-readable Markdown.
*   `branch_tip_id` (str, optional): The ID of the message to use as the end of the context. Defaults to `active_branch_id`.
*   `max_allowed_tokens` (int, optional): Maximum tokens for `"lollms_text"` export, triggering truncation.

**Returns:**
*   `str` for `"lollms_text"` or `"markdown"`.
*   `List[Dict]` for `"openai_chat"` or `"ollama_chat"`.

```python
print("\n--- Export Discussion Example ---")
# Export to Lollms native text format
lollms_text_export = db_discussion_1.export("lollms_text")
print("\nLollms Text Export (first 300 chars):\n")
print(lollms_text_export[:300])

# Export to OpenAI chat format
openai_chat_export = db_discussion_1.export("openai_chat")
print("\nOpenAI Chat Export (first message):\n")
print(openai_chat_export[0])

# Export to Markdown
markdown_export = db_discussion_1.export("markdown")
print("\nMarkdown Export (first 300 chars):\n")
print(markdown_export[:300])

# Example with discussion-level image
print("\n--- Export with Discussion-level Image Example ---")
db_discussion_1.add_discussion_image(dummy_image_b64)
# Note: The mock client doesn't process images visually, but they are added to the prompt structure
openai_chat_with_image = db_discussion_1.export("openai_chat")
print("\nOpenAI Chat Export with Discussion Image (System Message):\n")
print(openai_chat_with_image[0])
```

#### `summarize_and_prune(max_tokens: int, preserve_last_n: int = 4)`

Non-destructively prunes the discussion context by summarizing older messages. This helps keep the conversation within the LLM's context window without losing information.

**Parameters:**
*   `max_tokens` (int): The token limit that triggers the pruning.
*   `preserve_last_n` (int): The number of recent messages to keep in full detail (not summarized).

```python
print("\n--- Context Pruning Example ---")
# Simulate adding a lot of messages to trigger pruning
initial_message_count = len(db_discussion_1.get_messages())
for i in range(10): # Add 10 more message pairs
    db_discussion_1.add_message(sender="user", content=f"User message {i}")
    db_discussion_1.add_message(sender="assistant", content=f"AI response {i}", parent_id=db_discussion_1.active_branch_id)

print(f"Total messages before pruning attempt: {len(db_discussion_1.get_messages())}")

# Try to prune, assuming a small context size (e.g., 200 tokens)
# (Mock client's tokenize method is very simple, so 200 is small)
db_discussion_1.max_context_size = 200 # Set a max context size for the discussion object
db_discussion_1.summarize_and_prune(max_tokens=200, preserve_last_n=2)

print(f"Pruning summary: {db_discussion_1.pruning_summary[:100]}...")
print(f"Pruning point ID: {db_discussion_1.pruning_point_id}")

# Export again to see the effect of pruning
lollms_text_after_pruning = db_discussion_1.export("lollms_text")
print("\nLollms Text Export After Pruning (first 300 chars):\n")
print(lollms_text_after_pruning[:300])
# You should see the summary block in the exported text.
```

#### `memorize(branch_tip_id: Optional[str] = None)`

Analyzes the discussion and extracts key information to append to the discussion's long-term `memory` field. This is useful for storing user preferences, facts, or important conclusions across sessions.

**Parameters:**
*   `branch_tip_id` (str, optional): The ID of the message to use as the end of the context for memory extraction. Defaults to the active branch.

```python
print("\n--- Memorize Example ---")
print(f"Memory before memorizing: {db_discussion_1.memory}")

# Add a specific fact for memorization
db_discussion_1.chat(user_message="My favorite color is blue, and I enjoy hiking.")
db_discussion_1.memorize() # Will analyze the last turn and add to memory

print(f"Memory after memorizing: {db_discussion_1.memory}")

# Add another fact
db_discussion_1.chat(user_message="I'm working on a Python project about AI agents.")
db_discussion_1.memorize()
print(f"Memory after second memorization: {db_discussion_1.memory}")
```

#### `get_full_data_zone() -> str`

Aggregates the content of `user_data_zone`, `discussion_data_zone`, `personality_data_zone`, and `memory` into a single formatted string, ready for inclusion in the LLM prompt.

**Returns:**
*   `str`: The combined content of all active data zones.

```python
print("\n--- Get Full Data Zone Example ---")
db_discussion_1.user_data_zone = "User is a software developer."
db_discussion_1.discussion_data_zone = "This discussion is about project planning."
db_discussion_1.personality_data_zone = "Personality is a cheerful helper."

full_data_zone_content = db_discussion_1.get_full_data_zone()
print(f"Full Data Zone Content:\n{full_data_zone_content}")
```

#### `count_discussion_tokens(format_type: str, branch_tip_id: Optional[str] = None) -> int`

Counts the total number of tokens in the exported discussion content for a given format and branch.

**Parameters:**
*   `format_type` (str): The export format (e.g., `"lollms_text"`, `"openai_chat"`).
*   `branch_tip_id` (str, optional): The ID of the branch tip.

**Returns:**
*   `int`: The total token count.

#### `get_context_status(branch_tip_id: Optional[str] = None) -> Dict[str, Any]`

Provides a detailed breakdown of the context size, including tokens from system prompts, data zones, message history (text and images), and discussion-level images.

**Parameters:**
*   `branch_tip_id` (str, optional): The ID of the message branch to measure. Defaults to the active branch.

**Returns:**
*   `Dict[str, Any]`: A dictionary with detailed token breakdown.

```python
print("\n--- Get Context Status Example ---")
context_status = db_discussion_1.get_context_status()
print(f"Current Tokens: {context_status['current_tokens']}")
print(f"Max Tokens: {context_status['max_tokens']}")
print("Zones Breakdown:")
for zone_name, zone_info in context_status['zones'].items():
    print(f"  - {zone_name}: {zone_info['tokens']} tokens")
    if 'breakdown' in zone_info:
        for sub_name, sub_info in zone_info['breakdown'].items():
            print(f"    -- {sub_name}: {sub_info['tokens']} tokens")
```

#### Image Management (Discussion and Message Level)

`LollmsDiscussion` and `LollmsMessage` objects provide methods to manage images.

**Discussion-Level Image Methods (`LollmsDiscussion`):**

*   `add_discussion_image(image_b64: str)`: Adds a new image to the discussion context.
*   `get_discussion_images() -> List[Dict[str, Union[str, bool]]]` : Returns all discussion images with their activation status.
*   `toggle_discussion_image_activation(index: int, active: Optional[bool] = None)`: Toggles or sets the activation status of a discussion image.
*   `remove_discussion_image(index: int)`: Removes a discussion image.

**Message-Level Image Methods (`LollmsMessage`):**

*   `get_all_images() -> List[Dict[str, Union[str, bool]]]` : Returns all images associated with a message, including activation status.
*   `get_active_images() -> List[str]` : Returns only the base64 strings of active images.
*   `toggle_image_activation(index: int, active: Optional[bool] = None)`: Toggles or sets activation status of a message image.
*   `set_metadata_item(itemname, item_value, discussion)`: Used to update message metadata, including image activation status implicitly when `active_images` is modified via toggle.

```python
print("\n--- Image Management Example ---")
# Add a discussion-level image
db_discussion_1.add_discussion_image(dummy_image_b64)
print(f"Discussion images: {db_discussion_1.get_discussion_images()}")

# Toggle a discussion image
if db_discussion_1.images:
    db_discussion_1.toggle_discussion_image_activation(0, active=False)
    print(f"Discussion images after toggle: {db_discussion_1.get_discussion_images()}")
    db_discussion_1.toggle_discussion_image_activation(0, active=True) # Re-activate for next export
    print(f"Discussion images after re-activation: {db_discussion_1.get_discussion_images()}")

# Add a message with an image
msg_with_image = db_discussion_1.add_message(
    sender="user",
    content="This is a message with an image.",
    images=[dummy_image_b64, dummy_image_b64]
)
print(f"Message {msg_with_image.id[:8]} images: {msg_with_image.get_all_images()}")

# Toggle an image on a message
if msg_with_image.images:
    msg_with_image.toggle_image_activation(0, active=False)
    print(f"Message {msg_with_image.id[:8]} active images after toggle: {msg_with_image.get_active_images()}")
```

#### `switch_to_branch(branch_id: str)`

Changes the `active_branch_id` of the discussion to the specified message ID. All subsequent operations that depend on the "current" conversation path will follow this new branch.

**Parameters:**
*   `branch_id` (str): The ID of the message to set as the tip of the active branch.

```python
print("\n--- Switch to Branch Example ---")
print(f"Current active branch: {db_discussion_1.active_branch_id[:8]}")
# Find an earlier message to switch to
earlier_msg = db_discussion_1.get_messages()[0]
if earlier_msg:
    db_discussion_1.switch_to_branch(earlier_msg.id)
    print(f"Switched active branch to: {db_discussion_1.active_branch_id[:8]}")
    # Now, any chat or export operations will consider this new branch as the active one.
```

#### `auto_title()`

Generates a short, catchy title for the discussion using the LLM and stores it in the `metadata['title']` field.

**Returns:**
*   `str`: The generated title.

```python
print("\n--- Auto Title Example ---")
# Ensure there's some content for auto-titling
db_discussion_1.chat(user_message="Let's talk about the future of AI and space exploration.")
generated_title = db_discussion_1.auto_title()
print(f"Generated Discussion Title: {generated_title}")
print(f"Discussion metadata title: {db_discussion_1.metadata.get('title')}")
```

#### `set_metadata_item(itemname: str, item_value: Any)`

Sets or updates a specific item within the discussion's `discussion_metadata` dictionary.

**Parameters:**
*   `itemname` (str): The key for the metadata item.
*   `item_value` (Any): The value to set for the metadata item.

```python
print("\n--- Set Metadata Item Example ---")
db_discussion_1.set_metadata_item("project_status", "in_progress")
db_discussion_1.set_metadata_item("last_reviewer", "John Doe")
print(f"Updated discussion metadata: {db_discussion_1.metadata}")
```

#### `commit()`, `close()`, `touch()`

These methods manage the persistence state of database-backed discussions.
*   `touch()`: Marks the discussion as updated and triggers an automatic save if `autosave` is `True`.
*   `commit()`: Explicitly commits all pending changes to the database. Useful when `autosave` is `False`.
*   `close()`: Commits any final changes and closes the underlying database session. **Important to call when done with a persistent discussion to release resources.**

```python
print("\n--- Commit & Close Example ---")
# Assuming autosave was True when db_discussion_1 was created,
# changes are already committed. If it was False:
# db_discussion_1.autosave = False
# db_discussion_1.set_metadata_item("manual_save_test", True)
# db_discussion_1.commit() # Manual commit
# print("Manual commit performed.")

db_discussion_1.close()
print("Discussion session closed.")

# Clean up the dummy database file
if db_path.exists():
    db_path.unlink()
    print(f"Cleaned up {db_path}")
```