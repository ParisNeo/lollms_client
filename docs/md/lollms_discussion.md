# LollmsDiscussion Class

The `LollmsDiscussion` class is a cornerstone of the `lollms-client` library, designed to represent and manage a single conversation. It provides a robust interface for handling message history, conversation branching, context formatting, and persistence.

## Overview

A `LollmsDiscussion` can be either **in-memory** or **database-backed**, offering flexibility for different use cases.

-   **In-Memory:** Ideal for temporary or transient conversations. The discussion exists only for the duration of the application's runtime.
-   **Database-Backed:** Provides persistence by saving the entire conversation, including all branches and metadata, to a database file (e.g., SQLite). This is perfect for applications that need to retain user chat history.

## Key Features

-   **Message Management:** Add user and AI messages, which are automatically linked to form a conversation tree.
-   **Branching:** The conversation is a tree, not a simple list. This allows for exploring different conversational paths from any point. You can regenerate an AI response, and it will create a new branch.
-   **Context Exporting:** The `export()` method formats the conversation history for various LLM backends (`openai_chat`, `ollama_chat`, `lollms_text`, `markdown`), ensuring compatibility.
-   **Automatic Pruning:** To prevent exceeding the model's context window, it can automatically summarize older parts of the conversation without losing the original data.
-   **Sophisticated Context Layering:** Manage conversation state with multiple, distinct data zones (`user_data_zone`, `discussion_data_zone`, `personality_data_zone`) and a long-term `memory` field, allowing for rich and persistent context.
-   **Stateful Memory:** Empower the AI to build a long-term memory of key facts using the `memorize()` method.
-   **Context Inspection:** Get a detailed breakdown of the prompt context and token usage with `get_context_status()`.
-   **Global Discussion Images:** Attach images that apply to the entire conversation, ideal for setting an overall visual context. These also support selective activation and are persisted to the database.
-   **Selective Image Activation:** Finely control which images, at both the discussion and message level, are sent to a multimodal model.

## Creating a Discussion

The recommended way to create a discussion is using the `LollmsDiscussion.create_new()` class method.

```python
from lollms_client import LollmsClient, LollmsDataManager, LollmsDiscussion

# For an in-memory discussion (lost when the app closes)
lc = LollmsClient(binding_name="ollama", model_name="llama3")
discussion = LollmsDiscussion.create_new(lollms_client=lc, id="my-temp-discussion")

# For a persistent, database-backed discussion
# This will create a 'discussions.db' file if it doesn't exist
db_manager = LollmsDataManager('sqlite:///discussions.db')
discussion_db = LollmsDiscussion.create_new(
    lollms_client=lc, 
    db_manager=db_manager,
    discussion_metadata={"title": "My First DB Chat"}
)
```

## Core Properties

### Data and Memory Zones

`LollmsDiscussion` moves beyond a single `data_zone` to a more structured system of context layers. These string properties allow you to inject specific, persistent information into the AI's system prompt, separate from the main conversational flow. The content of all non-empty zones is automatically formatted and included in the prompt.

#### `system_prompt`
The main instruction set for the AI's persona and core task. It's the foundation of the prompt.
- **Purpose:** Defines who the AI is and what its primary goal is.
- **Example:** `"You are a helpful and friendly assistant."`

#### `memory`
A special zone for storing long-term, cross-discussion information about the user or topics. It is designed to be built up over time using the `memorize()` method.
- **Purpose:** To give the AI a persistent memory that survives across different chat sessions.
- **Example:** `"User's name is Alex.\nUser's favorite programming language is Rust."`

#### `user_data_zone`
Holds information specific to the current user that might be relevant for the session.
- **Purpose:** Storing user preferences, profile details, or session-specific goals.
- **Example:** `"Current project: API development.\nUser is a beginner in Python."`

#### `discussion_data_zone`
Contains context relevant only to the current discussion.
- **Purpose:** Holding summaries, state information, or data relevant to the current conversation topic that needs to be kept in front of the AI.
- **Example:** `"The user has already tried libraries A and B and found them too complex."`

#### `personality_data_zone`
This is where static or dynamic knowledge from a `LollmsPersonality`'s `data_source` is loaded.
- **Purpose:** To provide personalities with their own built-in knowledge bases or rulesets.
- **Example:** `"Rule 1: All code must be documented.\nRule 2: Use type hints."`

#### Example: How Zones are Combined

The `export()` method intelligently combines these zones. If all zones were filled, the effective system prompt would look something like this:

```
!@>system:
You are a helpful and friendly assistant.

-- Memory --
User's name is Alex.
User's favorite programming language is Rust.

-- User Data Zone --
Current project: API development.
User is a beginner in Python.

-- Discussion Data Zone --
The user has already tried libraries A and B and found them too complex.

-- Personality Data Zone --
Rule 1: All code must be documented.
Rule 2: Use type hints.
```
### Other Important Properties

-   `id`: The unique identifier for the discussion.
-   `metadata`: A dictionary for storing any custom metadata, like a title or discussion-level images.
-   `active_branch_id`: The ID of the message at the "tip" of the current conversation branch.
-   `messages`: A list of all `LollmsMessage` objects in the discussion.

## Multimodal Context Management: Discussion and Message Images

Beyond text, `LollmsDiscussion` provides fine-grained control over multimodal inputs at two levels: the entire discussion and individual messages. This allows you to non-destructively activate or deactivate images, which is useful for:

-   **Focusing the AI's attention** on only the most relevant images for a specific query.
-   **Saving costs and time** by not processing unnecessary images with expensive vision models.
-   **Setting a global visual context** for the whole chat while still attaching specific images to user prompts.
-   **Correcting context** if an initial query was ambiguous.
-   **Persisting visual context** alongside text in database-backed discussions.

### Discussion-Level Images
These images are attached directly to the discussion object and are treated as part of the global context, conceptually alongside the system prompt. They are ideal for providing an overarching visual theme or a reference document that should always be available. **Crucially, they are saved within the `discussion_metadata` field and are fully persistent.**

- `discussion.add_discussion_image(image_b64)`: Adds a new image to the discussion and marks it as active.
- `discussion.toggle_discussion_image_activation(index, active=None)`: Toggles or sets the active state of a discussion image.
- `discussion.get_discussion_images()`: Returns a list of all discussion-level images and their active status.

### Message-Level Images
These are images attached to a specific user or assistant message. They are ideal for asking questions about a particular image or providing visual information in a specific turn. They are stored on the message record in the database.

-   `msg.toggle_image_activation(index, active=None)`: The primary method for changing an image's state.
    -   `index`: The zero-based index of the image in the message's `images` list.
    -   `active`: If set to `True` or `False`, it explicitly sets the image's state. If left as `None`, it toggles the current state (active becomes inactive, and vice-versa).
-   `msg.get_active_images()`: Returns a list of base64 strings for *only* the currently active images in that message.

### Example in Action: Separation and Persistence

The test suite now includes `test_image_separation_and_context` and `test_full_multimodal_persistence` to guarantee this behavior. Here's what they confirm:

1.  **Separation:** `discussion.get_discussion_images()` will *only* return images added via `add_discussion_image`. Likewise, a message object's `get_all_images()` will *only* return images passed to `add_message`.
2.  **Aggregation:** `discussion.get_active_images()` correctly gathers all *active* images from both the discussion level and the message history, preparing the complete visual context for the AI.
3.  **Persistence:** When you call `discussion.commit()`, the state of all discussion-level images (including which ones are active/inactive) is saved to the database. Reloading the discussion restores this state perfectly.

## Main Methods

### `chat()`
The `chat()` method is the primary way to interact with the discussion. It handles a full user-to-AI turn, including invoking the advanced agentic capabilities of the `LollmsClient`.

#### Personalities, Tools, and Data Sources

The `chat` method intelligently handles tool activation and data loading when a `LollmsPersonality` is provided. This allows personalities to be configured as self-contained agents with their own default tools and knowledge bases.

**Tool Activation (`use_mcps`):**

1.  **Personality has tools, `use_mcps` is not set:** The agent will use the tools defined in `personality.active_mcps`.
2.  **Personality has tools, `use_mcps` is also set:** The agent will use a *combination* of tools from both the personality and the `use_mcps` parameter for that specific turn. Duplicates are automatically handled. This allows you to augment a personality's default tools on the fly.
3.  **Personality has no tools, `use_mcps` is set:** The agent will use only the tools specified in the `use_mcps` parameter.
4.  **Neither are set:** The agentic turn is not triggered (unless a data store is used), and a simple chat generation occurs.

**Knowledge Loading (`data_source`):**

Before generation, the `chat` method checks for `personality.data_source`:

-   **If it's a `str` (static data):** The string is loaded into the `discussion.personality_data_zone`, making it part of the system context for the current turn.
-   **If it's a `Callable` (dynamic data):**
    1.  The AI first generates a query based on the current conversation.
    2.  The `chat` method calls your function with this query.
    3.  The returned string is loaded into the `discussion.personality_data_zone`.
    4.  The final response generation proceeds with this newly added context.

This makes it easy to create powerful, reusable agents. For a complete, runnable example of building a **Python Coder Agent** that uses both `active_mcps` and a static `data_source`, **please see the "Putting It All Together" section in the main `README.md` file.**

### New Methods for State and Context Management

#### `memorize()`
This method empowers the AI to build its own long-term memory. It analyzes the current conversation, extracts key facts or preferences, and appends them to the `memory` data zone.

- **How it works:** It uses the LLM itself to summarize the most important, long-term takeaways from the recent conversation.
- **Use Case:** Perfect for creating assistants that learn about the user over time, remembering their name, preferences, or past projects without the user needing to repeat themselves.

```python
# User has just said: "My company is called 'Innovatech'."
discussion.chat("My company is called 'Innovatech'.")

# Now, trigger memorization
discussion.memorize() 
discussion.commit() # Save the updated memory to the database

# The discussion.memory field might now contain:
# "... previous memory ...
#
# --- Memory entry from 2024-06-27 10:30:00 UTC ---
# - User's company is named 'Innovatech'."
```

#### `get_context_status()`

Provides a detailed, real-time breakdown of the current prompt context, showing exactly what will be sent to the model and how many tokens each major component occupies. This is crucial for debugging context issues and understanding token usage.

The method accurately reflects the structure of the `lollms_text` format, where all system-level instructions (the main prompt, all data zones, and the pruning summary) are combined into a single system block.

-   **Return Value:** A dictionary containing:
    -   `max_tokens`: The configured maximum token limit for the discussion.
    -   `current_tokens`: The total token count for the entire prompt.
    -   `zones`: A dictionary with keys like:
        -   **`system_context`**: Contains the token count and content of all text-based system information (prompt, data zones, etc.).
        -   **`discussion_images`**: A new zone showing the token count and number of active discussion-level images.
        -   **`message_history`**: Contains token counts for both text and images within the message history, plus a detailed breakdown.

-   **Use Case:** Essential for debugging context issues, visualizing how different data zones contribute to the final prompt, and monitoring token consumption.

### Other Methods
-   `add_message(sender, content, ...)`: Adds a new message.
-   `export(format_type, ...)`: Exports the discussion to a specific format.
-   `commit()`: Saves changes to the database (if DB-backed).
-   `summarize_and_prune()`: Automatically handles context window limits.
-   `count_discussion_tokens()`: Counts the tokens for a given format.
-   `regenerate_branch()`: Deletes the last AI response and generates a new one.
-   `delete_branch(message_id)`: Deletes a message and all its children.