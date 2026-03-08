
## 1. Core Concept: What is LollmsDiscussion?
A `LollmsDiscussion` represents a single conversation thread. Unlike a simple list of messages, it is:
*   **Database-Backed**: Automatically persists to SQLite via `LollmsDataManager`.
*   **Branch-Aware**: Supports "what-if" scenarios by allowing the AI or user to navigate different versions of a conversation history.
*   **Context-Aware**: It dynamically assembles a system prompt by combining persistent facts, session data, and active documents.

### Basic Creation and Persistence
You can create a discussion either as a temporary in-memory object or a persistent database entry.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager

# 1. Setup the Client
lc = LollmsClient(llm_binding_name="lollms", llm_binding_config={"host_address": "http://localhost:9642"})

# 2. Setup Persistence (Optional but recommended)
db = LollmsDataManager("sqlite:///my_vault.db")

# 3. Create a new persistent discussion
discussion = LollmsDiscussion.create_new(
    lollms_client=lc,
    db_manager=db,
    autosave=True,
    system_prompt="You are a collaborative writing assistant."
)
```

## 2. Basic Usage: The Chat Loop
The primary way to interact is the `chat()` method. It handles context assembly, tool calls, and post-processing (like parsing artefacts) automatically.

```python
def on_chunk(text, msg_type, meta):
    if msg_type.name == "MSG_TYPE_CHUNK":
        print(text, end="", flush=True)

# Simple turn
response = discussion.chat(
    user_message="Start a new project called 'Project Phoenix'.",
    streaming_callback=on_chunk
)
```

## 3. Data Zones: Layered Context Management
Data Zones allow you to inject specific types of data into the prompt without polluting the message history. `LollmsDiscussion` assembles these layers every time a message is generated:

| Layer | Attribute | Purpose |
| :--- | :--- | :--- |
| **Memory** | `discussion.memory` | Persistent, long-term facts (User name, career, recurring goals). |
| **User Data** | `discussion.user_data_zone` | Global preferences (Coding style, OS info, languages). |
| **Discussion Data**| `discussion.discussion_data_zone`| Metadata for this specific task (Current chapter, project status). |
| **Personality Data**| `discussion.personality_data_zone`| Temporary tool results (RAG chunks, web search snippets). |
| **Artefacts** | `discussion.artefacts` | Versioned code or documents (Book content, Python scripts). |

```python
# Updating zones to guide the AI
discussion.user_data_zone = "Prefer Python 3.12 and Google Doc style formatting."
discussion.discussion_data_zone = "Project Goal: Write a 10-chapter sci-fi novel."
```

## 4. The Memory System
Memory is used for facts that should survive across many different discussions.
*   **Manual Setting**: `discussion.memory = "User: Alice. Role: Lead Dev."`
*   **Automated Extraction**: Use `memorize()`. This asks the LLM to analyze the current discussion, identify technical solutions or important facts, and return them as a structured object.

```python
# Extract and save key findings from the current session
facts = discussion.memorize()
if facts:
    print(f"Learned: {facts['title']}")
```

## 5. The Versioned Artefact System
Artefacts are persistent documents. They are superior to re-writing whole files because they support **Aider-style incremental patching**, saving massive amounts of context tokens.

### A. Initialization with Metadata
The AI can create an empty structure with metadata like author and description.

```xml
<artefact name="book.md" type="document" author="ParisNeo" description="Master Manuscript">
# Book Title
## Chapter 1
...
</artefact>
```

### B. Iterative Filling (Aider Format)
To add content without re-sending the entire book, the LLM emits a `SEARCH/REPLACE` block. The system finds the exact lines in `SEARCH` and swaps them for `REPLACE`.

```xml
<artefact name="book.md">
<<<<<<< SEARCH
## Chapter 1
...
=======
## Chapter 1: The First Contact
Deep in the silence of the void, a signal flickered.
>>>>>>> REPLACE
</artefact>
```

### C. Versioning and Reverting
Every time an artefact is modified, the version increments. `LollmsDiscussion` allows you to see the version count and revert to a specific point in time.

```python
# Check history
artefacts = discussion.artefacts.list()
for a in artefacts:
    print(f"{a['title']} - Version {a['version']}")

# Revert via code (or the AI can do it via XML tag)
discussion.artefacts.revert("book.md", target_version=1)
```

## 6. Integration and UI Notifications
When using the `chat()` method with a callback, the system emits a `MSG_TYPE_ARTEFACTS_STATE_CHANGED` whenever an artefact is created or modified.

```python
from lollms_client import MSG_TYPE

def callback(text, msg_type, meta):
    if msg_type == MSG_TYPE.MSG_TYPE_ARTEFACTS_STATE_CHANGED:
        # meta["artefacts"] contains the updated state for the UI to render
        print(f"UI NOTIFICATION: Artefacts updated: {text}")

discussion.chat("Update the book summary.", streaming_callback=callback)
```

## 7. Workflow Checklist
1.  **Creation**: Use `create_new` with a `db_manager` for persistence.
2.  **Context**: Populate `user_data_zone` and `discussion_data_zone`.
3.  **Iterate**: Use `chat()` and let the LLM manage documents via `<artefact>` tags.
4.  **Consolidate**: Periodically call `memorize()` to move session insights into long-term `memory`.
5.  **Prune**: Call `summarize_and_prune()` to keep the message history within token limits without losing the "essence" of the conversation.
