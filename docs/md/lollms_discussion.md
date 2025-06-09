## DOCS FOR: `lollms_client/lollms_discussion.py` and `lollms_client/lollms_types.py`

---
### `lollms_client.lollms_discussion`

**Purpose:**
This module provides classes for managing and representing a conversation or discussion, primarily for use with `LollmsClient` and its associated personalities or LLM interactions. It helps structure the exchange of messages between a user and an AI.

---
#### `LollmsMessage` (Data Class)

*   **Purpose**: Represents a single message within a discussion.
*   **Key Attributes**:
    *   `sender` (str): The originator of the message (e.g., "user", "assistant", "system", or a specific personality name).
    *   `content` (str): The textual content of the message.
    *   `id` (str): A unique identifier for the message (defaults to a UUID).
    *   `metadata` (str): A string field, typically storing JSON, for any additional metadata associated with the message (defaults to "{}").
*   **Methods**:
    *   `to_dict() -> dict`:
        *   **Purpose**: Converts the message object into a dictionary.
        *   **Returns**: A dictionary with keys 'sender', 'content', 'metadata', and 'id'.

---
#### `LollmsDiscussion`

*   **Purpose**: Represents a sequence of messages, forming a complete discussion or conversation. It provides methods to add messages and format the discussion for use as LLM prompt context.
*   **Key Attributes**:
    *   `messages` (List[LollmsMessage]): A list holding all `LollmsMessage` objects in chronological order.
    *   `lollmsClient` (LollmsClient): An instance of `LollmsClient`, used for tokenization when formatting the discussion.
*   **Methods**:
    *   **`__init__(lollmsClient: LollmsClient)`**:
        *   **Purpose**: Initializes a new discussion.
        *   **Parameters**: `lollmsClient` (LollmsClient): The client instance needed for tokenization.
    *   **`add_message(sender: str, content: str, metadata: dict = {})`**:
        *   **Purpose**: Creates a new `LollmsMessage` and adds it to the discussion.
        *   **Parameters**:
            *   `sender` (str): The sender of the message.
            *   `content` (str): The message content.
            *   `metadata` (dict, optional): Metadata for the message (will be converted to a JSON string).
    *   **`save_to_disk(file_path: Union[str, Path])`**:
        *   **Purpose**: Saves the entire discussion (all messages) to a YAML file.
        *   **Parameters**: `file_path` (Union[str, Path]): The path to save the YAML file.
    *   **`format_discussion(max_allowed_tokens: int, splitter_text: str = "!@>") -> str`**:
        *   **Purpose**: Formats the discussion into a single string suitable for use as LLM prompt context, ensuring it does not exceed a specified token limit. Messages are added starting from the most recent until the token limit is approached.
        *   **Parameters**:
            *   `max_allowed_tokens` (int): The maximum number of tokens the formatted string should contain.
            *   `splitter_text` (str, optional): The prefix used to identify the sender in the formatted string (default: "!@>").
        *   **Returns**: A string containing the formatted discussion, truncated to fit within `max_allowed_tokens`. Each message is formatted as `"{splitter_text}{sender}:\n{content}\n"`.

**Usage Example:**
```python
from lollms_client import LollmsClient, LollmsDiscussion

# Assume lc is an initialized LollmsClient instance
# lc = LollmsClient(binding_name="ollama", model_name="mistral:latest")
# For this example, let's mock the client for tokenization if not running a real server
class MockLollmsClient:
    def tokenize(self, text): return list(text) # Simple char-based tokenizer for example
    def detokenize(self, tokens): return "".join(tokens)
lc = MockLollmsClient()


discussion = LollmsDiscussion(lollmsClient=lc)
discussion.add_message(sender="user", content="Hello, AI!")
discussion.add_message(sender="assistant", content="Hello, User! How can I help you today?")
discussion.add_message(sender="user", content="Tell me about large language models.", metadata={"topic": "AI"})

# Save to disk
discussion.save_to_disk("my_conversation.yaml")

# Format for LLM prompt (e.g., max 50 tokens for this example)
# Note: Real tokenization will be model-specific via LollmsClient.
formatted_prompt = discussion.format_discussion(max_allowed_tokens=50)
print("Formatted Prompt (max 50 tokens):")
print(formatted_prompt)
# Expected output (will vary based on actual tokenization and message content):
# !@>assistant:
# Hello, User! How can I help you today?
# !@>user:
# Tell me about large language models.
```

---
### `lollms_client.lollms_types.py`

**Purpose:**
This module defines various enumerations used throughout the `lollms_client` library to represent message types, sender types, completion formats, and other categorical data.

---
#### `MSG_TYPE` (Enum)

*   **Purpose**: Enumerates different types of messages or events that can occur during LLM interaction, especially when streaming. This helps the client application understand the nature of each piece of information received.
*   **Members**:
    *   `MSG_TYPE_CHUNK` (0): A segment of a larger message, typical for streamed responses.
    *   `MSG_TYPE_FULL` (1): A complete message, sent in bulk.
    *   `MSG_TYPE_FULL_INVISIBLE_TO_AI` (2): A complete message intended for the user/UI but not to be fed back to the AI in subsequent turns.
    *   `MSG_TYPE_FULL_INVISIBLE_TO_USER` (3): A complete message intended for internal processing or AI context but not directly shown to the user (e.g., a thought process).
    *   `MSG_TYPE_EXCEPTION` (4): An error or exception occurred.
    *   `MSG_TYPE_WARNING` (5): A warning message.
    *   `MSG_TYPE_INFO` (6): An informational message.
    *   `MSG_TYPE_STEP` (7): An instantaneous step or event in a process.
    *   `MSG_TYPE_STEP_START` (8): Signals the beginning of a multi-part step or process.
    *   `MSG_TYPE_STEP_PROGRESS` (9): A progress update for an ongoing step (e.g., percentage).
    *   `MSG_TYPE_STEP_END` (10): Signals the completion of a multi-part step.
    *   `MSG_TYPE_JSON_INFOS` (11): A JSON payload containing structured information, often used for complex thought processes like Chain of Thought.
    *   `MSG_TYPE_REF` (12): References, typically in the format `[text](path)`.
    *   `MSG_TYPE_CODE` (13): A block of code, potentially JavaScript, intended for execution by the client UI or another system.
    *   `MSG_TYPE_UI` (14): Instructions or data to render a specific UI component (e.g., a Vue.js component).
    *   `MSG_TYPE_NEW_MESSAGE` (15): Signals the start of a new, distinct message from the AI.
    *   `MSG_TYPE_FINISHED_MESSAGE` (17): Signals the end of the current AI message generation.

---
#### `SENDER_TYPES` (Enum)

*   **Purpose**: Enumerates the possible senders of a message in a discussion.
*   **Members**:
    *   `SENDER_TYPES_USER` (0): Message originated from the human user.
    *   `SENDER_TYPES_AI` (1): Message originated from the AI.
    *   `SENDER_TYPES_SYSTEM` (2): Message originated from the system or as a system instruction.

---
#### `SUMMARY_MODE` (Enum)

*   **Purpose**: (Currently seems to have duplicate values) Intended to define different modes for text summarization.
*   **Members**:
    *   `SUMMARY_MODE_SEQUENCIAL` (0)
    *   `SUMMARY_MODE_HIERARCHICAL` (0) (Note: Likely a typo, should probably have a distinct value)

---
#### `ELF_COMPLETION_FORMAT` (Enum)

*   **Purpose**: Specifies the expected prompt and completion format for an LLM, particularly for non-LoLLMs backends that might distinguish between instruction-following and chat-based interaction styles.
*   **Members**:
    *   `Instruct` (0): Indicates an instruction-following format (e.g., a single prompt expecting a direct completion).
    *   `Chat` (1): Indicates a chat-based format, typically involving a sequence of messages with roles (user, assistant, system).
*   **Class Methods**:
    *   `from_string(format_string: str) -> ELF_COMPLETION_FORMAT`: Converts a string (case-insensitive "Instruct" or "Chat") to the corresponding enum member.
*   **Instance Methods**:
    *   `__str__() -> str`: Returns the string name of the enum member (e.g., "Instruct").
