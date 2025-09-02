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
*   **`memory`**: A long-term memory store. This zone holds the content of "memories" that have been explicitly loaded. The `memorize()` method analyzes a discussion and creates new, structured memories (e.g., about a problem and its solution) that can be managed and loaded into this zone.

These zones are automatically included in the `lollms_text` export format, providing the LLM with relevant background context.

### Artefacts & Memories: Managing Custom Content

Beyond conversation history, `LollmsDiscussion` allows you to manage custom content blobs called **Artefacts** and **Memories**. These are stored within the discussion's metadata and are ideal for tracking different kinds of generated or provided information:
*   **Artefacts**: Best for version-controlled content that evolves with the conversation, such as documents, code snippets, or configuration files. They can be loaded into the `discussion_data_zone`.
*   **Memories**: Designed to capture the essence of a discussion (e.g., a problem and its final solution). They are created by the `memorize()` method and can be loaded into the `memory` data zone to provide long-term context.

This provides a structured way to manage and inject different types of contextual information into the LLM's prompt.

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
from lollms.databases.discussions_database import LollmsDataManager
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
from lollms.lollms_client import LollmsClient
from lollms.databases.discussions_database import LollmsDiscussion, LollmsDataManager
from pathlib import Path
import os

# Assume lollms_client is already initialized
# For example purposes, we'll create a dummy one.
class MockLollmsClient(LollmsClient):
    def __init__(self): self.llm = self # Mock llm attribute
    def count_tokens(self, text): return len(text.split())
    def count_image_tokens(self, image_b64): return 100 # Mock value
    def generate_text(self, prompt, **kwargs): return f"AI response to: {prompt}"
    def generate_structured_content(self, prompt, schema, system_prompt, **kwargs):
        if "title" in schema.get("properties", {}): return {"title": "Mocked Discussion Title", "content":"Mocked content"}
        return {"result": "mock"}
    def remove_thinking_blocks(self, text):
        return text.replace("<thinking>", "").replace("</thinking>", "").replace("<think>", "").replace("</think>", "")
    def generate_with_mcp_rag(self, prompt, **kwargs):
        return {"final_answer": "Mock agentic response.", "final_scratchpad": "Agent thought process.", "tool_calls": [], "sources": []}
    def chat(self, discussion_obj, **kwargs):
        return "Mock response from LollmsClient chat."

lollms_client = MockLollmsClient()
lollms_client.model_name="mock_model"
lollms_client.binding_name="mock_binding"

# Initialize LollmsDataManager for persistent discussions
db_path = Path("./test_discussion.db")
if db_path.exists(): db_path.unlink()
data_manager = LollmsDataManager(f"sqlite:///{db_path.resolve()}")

# 1. Create a new in-memory discussion (not saved to DB)
in_memory_discussion = LollmsDiscussion.create_new(lollms_client=lollms_client)
print(f"New In-Memory Discussion ID: {in_memory_discussion.id}")

# 2. Create a new database-backed discussion
db_discussion_1 = LollmsDiscussion.create_new(
    lollms_client=lollms_client,
    db_manager=data_manager,
    system_prompt="You are a friendly AI companion.",
    discussion_metadata={"topic": "AI capabilities"},
    autosave=True
)
print(f"New DB-Backed Discussion ID: {db_discussion_1.id}")

# 3. Load an existing discussion
db_discussion_1.close()
loaded_discussion = data_manager.get_discussion(
    lollms_client=lollms_client,
    discussion_id=db_discussion_1.id,
    autosave=True
)
if loaded_discussion:
    print(f"Loaded Discussion ID: {loaded_discussion.id}")
else:
    print("Failed to load discussion.")
```

### Attributes

`LollmsDiscussion` exposes several key attributes, many of which are proxied from the underlying database object.

*   `id` (str): Unique identifier for the discussion.
*   `system_prompt` (str): The initial system-level instructions for the LLM.
*   `user_data_zone`, `discussion_data_zone`, `personality_data_zone` (str): Persistent data zones.
*   `memory` (str): Long-term knowledge accumulated via `memorize()` and by loading memories.
*   `participants` (dict): Maps sender names to roles (e.g., `{"user": "User"}`).
*   `active_branch_id` (str): The ID of the last message in the current active branch.
*   `discussion_metadata` (dict): A flexible dictionary for storing arbitrary discussion-level metadata. Accessed via the `metadata` property.
*   `created_at`, `updated_at` (datetime): Timestamps.
*   `pruning_summary` (str), `pruning_point_id` (str): Information for non-destructive pruning.
*   `images` (List[str]), `active_images` (List[bool]): Discussion-level images and their status.
*   `max_context_size` (int), `autosave` (bool), `scratchpad` (str): Configuration and state attributes.

### Key Methods

#### `add_message(**kwargs) -> LollmsMessage`

Adds a new message to the discussion. This is how you build the conversation history.

```python
# Assuming loaded_discussion is a valid LollmsDiscussion instance
user_msg_1 = loaded_discussion.add_message(sender="user", content="Hello!")
ai_msg_1 = loaded_discussion.add_message(sender="assistant", content="Hi there!", parent_id=user_msg_1.id)
print(f"Added User Message: {user_msg_1.content} (ID: {user_msg_1.id})")```

#### `chat(...) -> Dict[str, LollmsMessage]`

The primary method for interacting with the LLM. It manages prompt construction, agentic behavior (if enabled), and message addition.

```python
print("\n--- Simple Chat Example ---")
response_messages = loaded_discussion.chat(user_message="Tell me more about AI.")
print(f"AI Response from chat: {response_messages['ai_message'].content}")
```

#### `regenerate_branch(branch_tip_id=None, **kwargs) -> Dict[str, LollmsMessage]`

Deletes the last AI response and re-generates a new one based on the previous user prompt.

```python
print("\n--- Regenerate Branch Example ---")
regenerated_messages = loaded_discussion.regenerate_branch()
print(f"Regenerated AI Response: {regenerated_messages['ai_message'].content}")```

#### `get_messages(branch_id: Optional[str] = None) -> List[LollmsMessage]`

Retrieves all messages in a specific conversation branch, from root to the specified `branch_id` (leaf message).

```python
# Get messages for the active branch
active_branch_messages = loaded_discussion.get_messages()
for msg in active_branch_messages:
    print(f"  {msg.sender.capitalize()}: {msg.content[:50]}...")
```

#### `get_message(message_id: str) -> Optional[LollmsMessage]`

Retrieves a single message by its unique ID.

```python
retrieved_msg = loaded_discussion.get_message(user_msg_1.id)
```

#### `get_all_messages_flat() -> List[LollmsMessage]`

Retrieves all messages stored for this discussion as a flat list, regardless of their branch.

```python
all_messages = loaded_discussion.get_all_messages_flat()
```

#### `delete_branch(message_id: str)`

Deletes a message and all its descendant messages. Only supported for database-backed discussions.

```python
# Create a temporary branch to delete
temp_msg = loaded_discussion.add_message(sender="assistant", content="Temp branch", parent_id=user_msg_1.id)
loaded_discussion.delete_branch(temp_msg.id)
print(f"Branch starting at {temp_msg.id[:8]} deleted.")
```

#### `export(...) -> Union[List[Dict], str]`

Exports the discussion history into a specified format for LLM consumption or display.

```python
lollms_text_export = loaded_discussion.export("lollms_text")
openai_chat_export = loaded_discussion.export("openai_chat")
```

#### `summarize_and_prune(max_tokens: int, preserve_last_n: int = 4)`

Non-destructively prunes the discussion context by summarizing older messages.

```python
# Simulate pruning for a small context size
loaded_discussion.summarize_and_prune(max_tokens=10, preserve_last_n=2)
if loaded_discussion.pruning_summary:
    print(f"Pruning summary created: {loaded_discussion.pruning_summary[:100]}...")
```

#### `memorize(branch_tip_id: Optional[str] = None)`

Analyzes the discussion and creates a new structured "memory" containing the essence of the conversation (e.g., a problem and its solution). This memory is stored and can be loaded into the `memory` data zone for future reference.

```python
loaded_discussion.chat(user_message="I'm having a bug where my Python script fails to read a file. The error is FileNotFoundError. I fixed it by providing the absolute path instead of the relative one.")
loaded_discussion.memorize()
new_memory = loaded_discussion.list_memories()[-1]
print(f"New memory created: '{new_memory['title']}'")
# To use it, you would load it into the context:
# loaded_discussion.load_memory_into_context(new_memory['title'])
```

#### `get_full_data_zone() -> str`

Aggregates the content of all data zones into a single formatted string.

```python
loaded_discussion.user_data_zone = "User is a dev."
full_data_zone_content = loaded_discussion.get_full_data_zone()
```

#### `count_discussion_tokens(...) -> int` and `get_context_status(...) -> Dict`

Provide detailed information about the token count and context breakdown.

```python
context_status = loaded_discussion.get_context_status()
print(f"Current Tokens: {context_status['current_tokens']}")
```

#### `Image Management`

Methods to manage images at both the discussion and message level.

```python
dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
# Add a discussion-level image
loaded_discussion.add_discussion_image(dummy_image_b64)
# Add a message with an image
msg_with_image = loaded_discussion.add_message(sender="user", content="Image message", images=[dummy_image_b64])
# Toggle activation
msg_with_image.toggle_image_activation(0, active=False)
```

#### `switch_to_branch(branch_id: str)`

Changes the `active_branch_id` of the discussion to the specified message ID.

```python
loaded_discussion.switch_to_branch(user_msg_1.id)
print(f"Switched active branch to: {loaded_discussion.active_branch_id[:8]}")
```

#### `auto_title() -> str` and `set_metadata_item(key, value)`

Manage the discussion's title and other metadata.

```python
generated_title = loaded_discussion.auto_title()
loaded_discussion.set_metadata_item("project_status", "in_progress")
```

#### Artefact & Memory Management

Methods for creating, retrieving, updating, and managing versioned custom content (Artefacts) and discussion summaries (Memories).

*   **`list_artefacts() -> List[Dict]`**: Lists all artefacts.
*   **`add_artefact(title, content, ...)`**: Adds a new artefact.
*   **`load_artefact_into_data_zone(title, ...)`**: Places artefact content into the `discussion_data_zone`.
*   **`list_memories() -> List[Dict]`**: Lists all memories.
*   **`add_memory(title, content, ...)`**: Adds a new memory.
*   **`load_memory_into_context(title)`**: Places memory content into the `memory` data zone.

```python
print("\n--- Artefact Management Example ---")
loaded_discussion.add_artefact(title="My Document", content="Version 1 content.")
artefact_v1 = loaded_discussion.get_artefact("My Document")
print(f"Retrieved '{artefact_v1['title']}' v{artefact_v1['version']}")
loaded_discussion.update_artefact("My Document", "Version 2 is better.")
loaded_discussion.remove_artefact("My Document", version=1)
print(f"Artefacts now: {loaded_discussion.list_artefacts()}")```

#### Cloning, Import/Export, and Serialization

Tools for duplicating discussion context and for saving/loading the entire discussion state.

*   **`clone_without_messages() -> LollmsDiscussion`**: Creates a new discussion with the same context but no messages.
*   **`export_to_json_str() -> str`**: Serializes the entire discussion to a JSON string.
*   **`import_from_json_str(json_str, ...)`**: Creates a new discussion from a JSON string.

```python
print("\n--- Cloning and Serialization Example ---")
# Clone
cloned = loaded_discussion.clone_without_messages()
print(f"Cloned discussion {cloned.id} has {len(cloned.get_all_messages_flat())} messages.")
# Export
json_data = loaded_discussion.export_to_json_str()
# Import
imported = LollmsDiscussion.import_from_json_str(json_data, lollms_client, data_manager)
print(f"Imported discussion {imported.id} with {len(imported.get_all_messages_flat())} messages.")```

#### `commit()`, `close()`, `touch()`

Methods to manage the persistence state of database-backed discussions.

*   `touch()`: Marks the discussion as updated (triggers `autosave`).
*   `commit()`: Explicitly commits all pending changes to the database.
*   `close()`: Commits final changes and closes the database session.

```python
# Important for resource management
loaded_discussion.close()
cloned.close()
imported.close()

# Clean up the dummy database file
if db_path.exists():
    db_path.unlink()
    print(f"\nCleaned up {db_path}")
```