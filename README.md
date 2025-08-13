# LoLLMs Client Library

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/lollms_client.svg)](https://badge.fury.io/py/lollms_client)
[![Python Versions](https://img.shields.io/pypi/pyversions/lollms_client.svg)](https://pypi.org/project/lollms-client/)
[![Downloads](https://static.pepy.tech/personalized-badge/lollms-client?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/lollms-client)
[![Documentation - Usage](https://img.shields.io/badge/docs-Usage%20Guide-brightgreen)](DOC_USE.md)
[![Documentation - Developer](https://img.shields.io/badge/docs-Developer%20Guide-blue)](DOC_DEV.md)
[![GitHub stars](https://img.shields.io/github/stars/ParisNeo/lollms_client.svg?style=social&label=Star&maxAge=2592000)](https://github.com/ParisNeo/lollms_client/stargazers/)
[![GitHub issues](https://img.shields.io/github/issues/ParisNeo/lollms_client.svg)](https://github.com/ParisNeo/lollms_client/issues)

**`lollms_client`** is a powerful and flexible Python library designed to simplify interactions with the **LoLLMs (Lord of Large Language Models)** ecosystem and various other Large Language Model (LLM) backends. It provides a unified API for text generation, multimodal operations (text-to-image, text-to-speech, etc.), and robust function calling through the Model Context Protocol (MCP).

Whether you're connecting to a remote LoLLMs server, an Ollama instance, the OpenAI API, or running models locally using GGUF (via `llama-cpp-python` or a managed `llama.cpp` server), Hugging Face Transformers, or vLLM, `lollms-client` offers a consistent and developer-friendly experience.

## Key Features

*   🔌 **Versatile Binding System:** Seamlessly switch between different LLM backends (LoLLMs, Ollama, OpenAI, Llama.cpp, Transformers, vLLM, OpenLLM) without major code changes.
*   🗣️ **Multimodal Support:** Interact with models capable of processing images and generate various outputs like speech (TTS) and images (TTI).
*   🖼️ **Selective Image Activation:** Control which images in a message are active and sent to the model, allowing for fine-grained multimodal context management without deleting the original data.
*   🤖 **Function Calling with MCP:** Empowers LLMs to use external tools and functions through the Model Context Protocol (MCP), with built-in support for local Python tool execution via `local_mcp` binding and its default tools (file I/O, internet search, Python interpreter, image generation).
*   🎭 **Personalities as Agents:** Personalities can now define their own set of required tools (MCPs) and have access to static or dynamic knowledge bases (`data_source`), turning them into self-contained, ready-to-use agents.
*   🚀 **Streaming & Callbacks:** Efficiently handle real-time text generation with customizable callback functions, including during MCP interactions.
*   📑 **Sequential Summarization:** A `summarize` method to process and summarize texts that exceed the model's context window.
*   📝 **Advanced Structured Content Generation:** Reliably generate structured JSON output from natural language prompts using the `generate_structured_content` helper method.
*   💬 **Advanced Discussion Management:** Robustly manage conversation histories with `LollmsDiscussion`, featuring branching, context exporting, and automatic pruning.
*   🧠 **Persistent Memory & Data Zones:** `LollmsDiscussion` now supports multiple, distinct data zones (`user_data_zone`, `discussion_data_zone`, `personality_data_zone`) and a long-term `memory` field. This allows for sophisticated context layering and state management.
*   ✍️ **Automatic Memorization:** A new `memorize()` method allows the AI to analyze a conversation and extract key facts, appending them to the long-term `memory` for recall in future sessions.
*   📊 **Detailed Context Analysis:** The `get_context_status()` method now provides a rich, detailed breakdown of the prompt context, showing the content and token count for each individual component (system prompt, data zones, message history).
*   ⚙️ **Configuration Management:** Flexible ways to configure bindings and generation parameters.
*   🧩 **Extensible:** Designed to easily incorporate new LLM backends and modality services, including custom MCP toolsets.
*   📝 **High-Level Operations:** Includes convenience methods for complex tasks like sequential summarization and deep text analysis directly within `LollmsClient`.

## Installation

You can install `lollms_client` directly from PyPI:

```bash
pip install lollms-client
```

This will install the core library. Some bindings may require additional dependencies (e.g., `llama-cpp-python`, `torch`, `transformers`, `ollama`, `vllm`). The library attempts to manage these using `pipmaster`, but for complex dependencies (especially those requiring compilation like `llama-cpp-python` with GPU support), manual installation might be preferred.

## Core Generation Methods

The `LollmsClient` provides several methods for generating text, catering to different use cases.

### Basic Text Generation (`generate_text`)

This is the most straightforward method for generating a response based on a simple prompt.

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors

# Callback for streaming output
def simple_streaming_callback(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nStreaming Error: {chunk}")
    return True # True to continue streaming

try:
    # Initialize client to connect to a LoLLMs server
    # For other backends, change 'binding_name' and provide necessary parameters.
    # See below for detailed initialization examples for various bindings.
    lc = LollmsClient(
        binding_name="lollms",
        host_address="http://localhost:9600"
    )

    prompt = "Tell me a fun fact about space."
    ASCIIColors.yellow(f"Prompt: {prompt}")

    # Generate text with streaming
    ASCIIColors.green("Streaming Response:")
    response_text = lc.generate_text(
        prompt,
        n_predict=100,
        stream=True,
        streaming_callback=simple_streaming_callback
    )
    print("\n--- End of Stream ---")

    # The 'response_text' variable will contain the full concatenated text
    # if streaming_callback returns True throughout.
    if isinstance(response_text, str):
        ASCIIColors.cyan(f"\nFull streamed text collected: {response_text[:100]}...")
    elif isinstance(response_text, dict) and "error" in response_text:
        ASCIIColors.error(f"Error during generation: {response_text['error']}")

except ValueError as ve:
    ASCIIColors.error(f"Initialization Error: {ve}")
    ASCIIColors.info("Ensure a LoLLMs server is running or configure another binding.")
except ConnectionRefusedError:
    ASCIIColors.error("Connection refused. Is the LoLLMs server running at http://localhost:9600?")
except Exception as e:
    ASCIIColors.error(f"An unexpected error occurred: {e}")

```

### Generating from Message Lists (`generate_from_messages`)

For more complex conversational interactions, you can provide the LLM with a list of messages, similar to the OpenAI Chat Completion API. This allows you to define roles (system, user, assistant) and build multi-turn conversations programmatically.

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors

def streaming_callback_for_messages(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True

try:
    lc = LollmsClient(
        binding_name="ollama", # Or "openai", "claude", "gemini", etc.
        model_name="llama3",
        host_address="http://localhost:11434" # Adjust for your binding
    )

    # Define the conversation history as a list of messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that specializes in programming."},
        {"role": "user", "content": "Hello, what's your name?"},
        {"role": "assistant", "content": "I am an AI assistant created by Google."},
        {"role": "user", "content": "Can you explain recursion in Python?"}
    ]

    ASCIIColors.yellow("\nGenerating response from messages:")
    response_text = lc.generate_from_messages(
        messages=messages,
        n_predict=200,
        stream=True,
        streaming_callback=streaming_callback_for_messages
    )
    print("\n--- End of Message Stream ---")
    ASCIIColors.cyan(f"\nFull collected response: {response_text[:150]}...")

except Exception as e:
    ASCIIColors.error(f"Error during message generation: {e}")

```

### Advanced Structured Content Generation (`generate_structured_content`)

The `generate_structured_content` method is a powerful utility for forcing an LLM's output into a specific JSON format. It's ideal for extracting information, getting consistent tool parameters, or any task requiring reliable, machine-readable output.

```python
from lollms_client import LollmsClient
import json

lc = LollmsClient(binding_name="ollama", model_name="llama3")

text_block = "John Doe is a 34-year-old software engineer from New York. He loves hiking and Python programming."

# Define the exact JSON structure you want
output_template = {
    "full_name": "string",
    "age": "integer",
    "profession": "string",
    "city": "string",
    "hobbies": ["list", "of", "strings"]
}

# Generate the structured data
extracted_data = lc.generate_structured_content(
    prompt=f"Extract the relevant information from the following text:\n\n{text_block}",
    schema=output_template # Note: parameter is 'schema', not 'output_format'
)

if extracted_data:
    print(json.dumps(extracted_data, indent=2))
# Expected output (approximate):
# {
#   "full_name": "John Doe",
#   "age": 34,
#   "profession": "software engineer",
#   "city": "New York",
#   "hobbies": ["hiking", "Python programming"]
# }
```

## Advanced Discussion Management

The `LollmsDiscussion` class is a core component for managing conversational state, including message history, long-term memory, and various context zones.

### Basic Chat with `LollmsDiscussion`

For general conversational agents that need to maintain context across turns, `LollmsDiscussion` simplifies the process. It automatically handles message formatting, history management, and context window limitations.

```python
from lollms_client import LollmsClient, LollmsDiscussion, MSG_TYPE
from ascii_colors import ASCIIColors

# Initialize LollmsClient
lc = LollmsClient(
    binding_name="ollama", 
    model_name="llama3",
    host_address="http://localhost:11434"
)

# Create a new discussion. For persistent discussions, pass a db_manager.
# discussion = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db_manager, autosave=True)
discussion = LollmsDiscussion.create_new(lollms_client=lc, autosave=False) # In-memory for simplicity here

# Define a simple callback for streaming
def chat_callback(chunk: str, msg_type: MSG_TYPE, **kwargs) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True

try:
    ASCIIColors.cyan("> User: Hello, how are you today?")
    response = discussion.chat(
        user_message="Hello, how are you today?",
        streaming_callback=chat_callback
    )
    print("\n") # Newline after stream finishes

    ai_message = response['ai_message']
    user_message = response['user_message']

    ASCIIColors.green(f"< Assistant (Full): {ai_message.content[:100]}...")

    # Now, continue the conversation
    ASCIIColors.cyan("\n> User: Can you recommend a good book?")
    response = discussion.chat(
        user_message="Can you recommend a good book?",
        streaming_callback=chat_callback
    )
    print("\n")

    # You can inspect the full message history
    ASCIIColors.magenta("\n--- Discussion History ---")
    for msg in discussion.get_messages():
        print(f"[{msg.sender.capitalize()}]: {msg.content[:50]}...")

except Exception as e:
    ASCIIColors.error(f"An error occurred during discussion chat: {e}")
```

### Building Stateful Agents with Memory and Data Zones

The `LollmsDiscussion` class provides a sophisticated system for creating stateful agents that can remember information across conversations. This is achieved through a layered system of "context zones" that are automatically combined into the AI's system prompt.

#### Understanding the Context Zones

The AI's context is more than just chat history. It's built from several distinct components, each with a specific purpose:

*   **`system_prompt`**: The foundational layer defining the AI's core identity, persona, and primary instructions.
*   **`memory`**: The AI's long-term, persistent memory. It stores key facts about the user or topics, built up over time using the `memorize()` method.
*   **`user_data_zone`**: Holds session-specific information about the user's current state or goals (e.g., "User is currently working on 'file.py'").
*   **`discussion_data_zone`**: Contains state or meta-information about the current conversational task (e.g., "Step 1 of the plan is complete").
*   **`personality_data_zone`**: A knowledge base or set of rules automatically injected from a `LollmsPersonality`'s `data_source`.
*   **`pruning_summary`**: An automatic, AI-generated summary of the oldest messages in a very long chat, used to conserve tokens without losing the gist of the early conversation.

The `get_context_status()` method is your window into this system, showing you exactly how these zones are combined and how many tokens they consume.

Let's see this in action with a "Personal Assistant" agent that learns about the user over time.

```python
from lollms_client import LollmsClient, LollmsDataManager, LollmsDiscussion, MSG_TYPE
from ascii_colors import ASCIIColors
import json

# --- 1. Setup a persistent database for our discussion ---
db_manager = LollmsDataManager('sqlite:///my_assistant.db')
lc = LollmsClient(binding_name="ollama", model_name="llama3")

# Try to load an existing discussion or create a new one
discussion_id = "user_assistant_chat_1"
discussion = db_manager.get_discussion(lc, discussion_id)
if not discussion:
    ASCIIColors.yellow("Creating a new discussion...")
    discussion = LollmsDiscussion.create_new(
        lollms_client=lc,
        db_manager=db_manager,
        id=discussion_id,
        autosave=True # Important for persistence
    )
    # Let's preset some data in different zones
    discussion.system_prompt = "You are a helpful Personal Assistant."
    discussion.user_data_zone = "User's Name: Alex\nUser's Goal: Learn about AI development."
    discussion.commit()
else:
    ASCIIColors.green("Loaded existing discussion.")


def run_chat_turn(prompt: str):
    """Helper function to run a single chat turn and print details."""
    ASCIIColors.cyan(f"\n> User: {prompt}")

    # --- A. Check context status BEFORE the turn using get_context_status() ---
    ASCIIColors.magenta("\n--- Context Status (Before Generation) ---")
    status = discussion.get_context_status()
    print(f"Max Tokens: {status.get('max_tokens')}, Current Tokens: {status.get('current_tokens')}")
    
    # Print the system context details
    if 'system_context' in status['zones']:
        sys_ctx = status['zones']['system_context']
        print(f"  - System Context Tokens: {sys_ctx['tokens']}")
        # The 'breakdown' shows the individual zones that were combined
        for name, content in sys_ctx.get('breakdown', {}).items():
            print(f"    -> Contains '{name}': {content.split(chr(10))[0]}...")

    # Print the message history details
    if 'message_history' in status['zones']:
        msg_hist = status['zones']['message_history']
        print(f"  - Message History Tokens: {msg_hist['tokens']} ({msg_hist['message_count']} messages)")

    print("------------------------------------------")

    # --- B. Run the chat ---
    ASCIIColors.green("\n< Assistant:")
    response = discussion.chat(
        user_message=prompt,
        streaming_callback=lambda chunk, type, **k: print(chunk, end="", flush=True) if type==MSG_TYPE.MSG_TYPE_CHUNK else None
    )
    print() # Newline after stream

    # --- C. Trigger memorization to update the 'memory' zone ---
    ASCIIColors.yellow("\nTriggering memorization process...")
    discussion.memorize()
    discussion.commit() # Save the new memory to the DB
    ASCIIColors.yellow("Memorization complete.")

# --- Run a few turns ---
run_chat_turn("Hi there! Can you recommend a good Python library for building web APIs?")
run_chat_turn("That sounds great. By the way, my favorite programming language is Rust, I find its safety features amazing.")
run_chat_turn("What was my favorite programming language again?")

# --- Final Inspection of Memory ---
ASCIIColors.magenta("\n--- Final Context Status ---")
status = discussion.get_context_status()
print(f"Max Tokens: {status.get('max_tokens')}, Current Tokens: {status.get('current_tokens')}")
if 'system_context' in status['zones']:
    sys_ctx = status['zones']['system_context']
    print(f"  - System Context Tokens: {sys_ctx['tokens']}")
    for name, content in sys_ctx.get('breakdown', {}).items():
        # Print the full content of the memory zone to verify it was updated
        if name == 'memory':
            ASCIIColors.yellow(f"    -> Full '{name}' content:\n{content}")
        else:
            print(f"    -> Contains '{name}': {content.split(chr(10))[0]}...")
print("------------------------------------------")

```

#### How it Works:

1.  **Persistence & Initialization:** The `LollmsDataManager` saves and loads the discussion. We initialize the `system_prompt` and `user_data_zone` to provide initial context.
2.  **`get_context_status()`:** Before each generation, we call this method. The output shows a `system_context` block with a token count for all combined zones and a `breakdown` field that lets us see the content of each individual zone that contributed to it.
3.  **`memorize()`:** After the user mentions their favorite language, `memorize()` is called. The LLM analyzes the last turn, identifies this new, important fact, and appends it to the `discussion.memory` zone.
4.  **Recall:** In the final turn, when asked to recall the favorite language, the AI has access to the updated `memory` content within its system context and can correctly answer "Rust". This demonstrates true long-term, stateful memory.

### Managing Multimodal Context: Activating and Deactivating Images

When working with multimodal models, you can now control which images in a message are active and sent to the model. This is useful for focusing the AI's attention, saving tokens on expensive vision models, or allowing a user to correct which images are relevant.

This is managed at the `LollmsMessage` level using the `toggle_image_activation()` method.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager
from ascii_colors import ASCIIColors
import base64
from pathlib import Path

# Helper to create a dummy image b64 string
def create_dummy_image(text):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), text, fill=(255,255,0))
    buffer = Path("temp_img.png")
    img.save(buffer, "PNG")
    b64 = base64.b64encode(buffer.read_bytes()).decode('utf-8')
    buffer.unlink()
    return b64

# --- 1. Setup ---
lc = LollmsClient(binding_name="ollama", model_name="llava")
discussion = LollmsDiscussion.create_new(lollms_client=lc)

# --- 2. Add a message with multiple images ---
img1_b64 = create_dummy_image("Image 1")
img2_b64 = create_dummy_image("Image 2: Cat")
img3_b64 = create_dummy_image("Image 3")

discussion.add_message(
    sender="user", 
    content="What is in the second image?", 
    images=[img1_b64, img2_b64, img3_b64]
)
user_message = discussion.get_messages()[-1]

# --- 3. Check the initial state ---
ASCIIColors.magenta("--- Initial State (All 3 Images Active) ---")
status_before = discussion.get_context_status()
print(f"Message History Text:\n{status_before['zones']['message_history']['content']}")

# --- 4. Deactivate irrelevant images ---
ASCIIColors.magenta("\n--- Deactivating images 1 and 3 ---")
user_message.toggle_image_activation(index=0, active=False) # Deactivate first image
user_message.toggle_image_activation(index=2, active=False) # Deactivate third image

# --- 5. Check the new state ---
ASCIIColors.magenta("\n--- New State (Only Image 2 is Active) ---")
status_after = discussion.get_context_status()
print(f"Message History Text:\n{status_after['zones']['message_history']['content']}")

ASCIIColors.green("\nNotice the message now says '(1 image(s) attached)' instead of 3.")
ASCIIColors.green("Only the active image will be sent to the multimodal LLM.")
```

### Putting It All Together: An Advanced Agentic Example

Let's create a **Python Coder Agent**. This agent will use a set of coding rules from a local file as its knowledge base and will be equipped with a tool to execute the code it writes. This demonstrates the synergy between `LollmsPersonality` (with `data_source` and `active_mcps`), `LollmsDiscussion`, and the MCP system.

#### Step 1: Create the Knowledge Base (`coding_rules.txt`)

Create a simple text file with the rules our agent must follow.

```text
# File: coding_rules.txt

1.  All Python functions must include a Google-style docstring.
2.  Use type hints for all function parameters and return values.
3.  The main execution block should be protected by `if __name__ == "__main__":`.
4.  After defining a function, add a simple example of its usage inside the main block.
5.  Print the output of the example usage to the console.
```

#### Step 2: The Main Script (`agent_example.py`)

This script will define the personality, initialize the client, and run the agent.

```python
from pathlib import Path
from lollms_client import LollmsClient, LollmsPersonality, LollmsDiscussion, MSG_TYPE, trace_exception
from ascii_colors import ASCIIColors
import json

# A detailed callback to visualize the agent's process
def agent_callback(chunk: str, msg_type: MSG_TYPE, params: dict = None, **kwargs) -> bool:
    if not params: params = {}
    
    if msg_type == MSG_TYPE.MSG_TYPE_STEP:
        ASCIIColors.yellow(f"\n>> Agent Step: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        ASCIIColors.yellow(f"\n>> Agent Step Start: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
        result = params.get('result', '')
        ASCIIColors.green(f"<< Agent Step End: {chunk} -> Result: {json.dumps(result)[:150]}...")
    elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CONTENT:
        ASCIIColors.magenta(f"🤔 Agent Thought: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
        ASCIIColors.blue(f"🛠️  Agent Action: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_OBSERVATION:
        ASCIIColors.cyan(f"👀 Agent Observation: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True) # Final answer stream
    return True

try:
    # --- 1. Load the knowledge base from the file ---
    rules_path = Path("coding_rules.txt")
    if not rules_path.exists():
        raise FileNotFoundError("Please create the 'coding_rules.txt' file.")
    coding_rules = rules_path.read_text()

    # --- 2. Define the Coder Agent Personality ---
    coder_personality = LollmsPersonality(
        name="Python Coder Agent",
        author="lollms-client",
        category="Coding",
        description="An agent that writes and executes Python code according to specific rules.",
        system_prompt=(
            "You are an expert Python programmer. Your task is to write clean, executable Python code based on the user's request. "
            "You MUST strictly follow all rules provided in the 'Personality Static Data' section. "
            "First, think about the plan. Then, use the `python_code_interpreter` tool to write and execute the code. "
            "Finally, present the code and its output to the user."
        ),
        # A) Attach the static knowledge base
        data_source=coding_rules,
        # B) Equip the agent with a code execution tool
        active_mcps=["python_code_interpreter"]
    )

    # --- 3. Initialize the Client and Discussion ---
    lc = LollmsClient(
        binding_name="ollama",          # Or any capable model binding
        model_name="codellama",         # A code-specialized model is recommended
        mcp_binding_name="local_mcp"    # Enable the local tool execution engine
    )
    discussion = LollmsDiscussion.create_new(lollms_client=lc)
    
    # --- 4. The User's Request ---
    user_prompt = "Write a Python function that takes two numbers and returns their sum."

    ASCIIColors.yellow(f"User Prompt: {user_prompt}")
    print("\n" + "="*50 + "\nAgent is now running...\n" + "="*50)

    # --- 5. Run the Agentic Chat Turn ---
    response = discussion.chat(
        user_message=user_prompt,
        personality=coder_personality,
        streaming_callback=agent_callback
    )

    print("\n\n" + "="*50 + "\nAgent finished.\n" + "="*50)
    
    # --- 6. Inspect the results ---
    ai_message = response['ai_message']
    ASCIIColors.green("\n--- Final Answer from Agent ---")
    print(ai_message.content)
    
    ASCIIColors.magenta("\n--- Tool Calls Made ---")
    print(json.dumps(ai_message.metadata.get("tool_calls", []), indent=2))

except Exception as e:
    trace_exception(e)

```

#### Step 3: What Happens Under the Hood

When you run `agent_example.py`, a sophisticated process unfolds:

1.  **Initialization:** The `LollmsDiscussion.chat()` method is called with the `coder_personality`.
2.  **Knowledge Injection:** The `chat` method sees that `personality.data_source` is a string. It automatically takes the content of `coding_rules.txt` and injects it into the discussion's data zones.
3.  **Tool Activation:** The method also sees `personality.active_mcps`. It enables the `python_code_interpreter` tool for this turn.
4.  **Context Assembly:** The `LollmsClient` assembles a rich prompt for the LLM that includes:
    *   The personality's `system_prompt`.
    *   The content of `coding_rules.txt` (from the data zones).
    *   The list of available tools (including `python_code_interpreter`).
    *   The user's request ("Write a function...").
5.  **Reason and Act:** The LLM, now fully briefed, reasons that it needs to use the `python_code_interpreter` tool. It formulates the Python code *according to the rules it was given*.
6.  **Tool Execution:** The `local_mcp` binding receives the code and executes it in a secure local environment. It captures any output (`stdout`, `stderr`) and results.
7.  **Observation:** The execution results are sent back to the LLM as an "observation."
8.  **Final Synthesis:** The LLM now has the user's request, the rules, the code it wrote, and the code's output. It synthesizes all of this into a final, comprehensive answer for the user.

This example showcases how `lollms-client` allows you to build powerful, knowledgeable, and capable agents by simply composing personalities with data and tools.

## Using LoLLMs Client with Different Bindings

`lollms-client` supports a wide range of LLM backends through its binding system. This section provides practical examples of how to initialize `LollmsClient` for each of the major supported bindings.

### A Note on Configuration

The recommended way to provide credentials and other binding-specific settings is through the `llm_binding_config` dictionary during `LollmsClient` initialization. While many bindings can fall back to reading environment variables (e.g., `OPENAI_API_KEY`), passing them explicitly in the config is clearer and less error-prone.

```python
# General configuration pattern
lc = LollmsClient(
    binding_name="your_binding_name",
    model_name="a_model_name",
    llm_binding_config={
        "specific_api_key_param": "your_api_key_here",
        "another_specific_param": "some_value"
    }
)
```

---

### 1. Local Bindings

These bindings run models directly on your local machine, giving you full control and privacy.

#### **Ollama**

The `ollama` binding connects to a running Ollama server instance on your machine or network.

**Prerequisites:**
*   [Ollama installed and running](https://ollama.com/).
*   Models pulled, e.g., `ollama pull llama3`.

**Usage:**

```python
from lollms_client import LollmsClient

# Configuration for a local Ollama server
lc = LollmsClient(
    binding_name="ollama",
    model_name="llama3",  # Or any other model you have pulled
    host_address="http://localhost:11434" # Default Ollama address
)

# Now you can use lc.generate_text(), lc.chat(), etc.
response = lc.generate_text("Why is the sky blue?")
print(response)
```

#### **PythonLlamaCpp (Local GGUF Models)**

The `pythonllamacpp` binding loads and runs GGUF model files directly using the powerful `llama-cpp-python` library. This is ideal for high-performance, local inference on CPU or GPU.

**Prerequisites:**
*   A GGUF model file downloaded to your machine.
*   `llama-cpp-python` installed. For GPU support, it must be compiled with the correct flags (e.g., `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`).

**Usage:**

```python
from lollms_client import LollmsClient

# --- Configuration for Llama.cpp ---
# Path to your GGUF model file
MODEL_PATH = "/path/to/your/model.gguf" 

# Binding-specific configuration
LLAMACPP_CONFIG = {
    "n_gpu_layers": -1,  # -1 for all layers to GPU, 0 for CPU
    "n_ctx": 4096,       # Context size
    "seed": -1,          # -1 for random seed
    "chat_format": "chatml" # Or another format like 'llama-2'
}

try:
    lc = LollmsClient(
        binding_name="pythonllamacpp",
        model_name=MODEL_PATH, # For this binding, model_name is the file path
        llm_binding_config=LLAMACPP_CONFIG
    )

    response = lc.generate_text("Write a recipe for a great day.")
    print(response)

except Exception as e:
    print(f"Error initializing Llama.cpp binding: {e}")
    print("Please ensure llama-cpp-python is installed and the model path is correct.")

```

---

### 2. Cloud Service Bindings

These bindings connect to hosted LLM APIs from major providers.

#### **OpenAI**

Connects to the official OpenAI API to use models like GPT-4o, GPT-4, and GPT-3.5.

**Prerequisites:**
*   An OpenAI API key.

**Usage:**

```python
from lollms_client import LollmsClient

OPENAI_CONFIG = {
    "service_key": "your_openai_api_key_here" # sk-...
}

lc = LollmsClient(
    binding_name="openai",
    model_name="gpt-4o",
    llm_binding_config=OPENAI_CONFIG
)

response = lc.generate_text("What is the difference between AI and machine learning?")
print(response)
```

#### **Google Gemini**

Connects to Google's Gemini family of models via the Google AI Studio API.

**Prerequisites:**
*   A Google AI Studio API key.

**Usage:**

```python
from lollms_client import LollmsClient

GEMINI_CONFIG = {
    "service_key": "your_google_api_key_here"
}

lc = LollmsClient(
    binding_name="gemini",
    model_name="gemini-1.5-pro-latest",
    llm_binding_config=GEMINI_CONFIG
)

response = lc.generate_text("Summarize the plot of 'Dune' in three sentences.")
print(response)
```

#### **Anthropic Claude**

Connects to Anthropic's API to use the Claude family of models, including Claude 3.5 Sonnet, Opus, and Haiku.

**Prerequisites:**
*   An Anthropic API key.

**Usage:**

```python
from lollms_client import LollmsClient

CLAUDE_CONFIG = {
    "service_key": "your_anthropic_api_key_here"
}

lc = LollmsClient(
    binding_name="claude",
    model_name="claude-3-5-sonnet-20240620",
    llm_binding_config=CLAUDE_CONFIG
)

response = lc.generate_text("What are the core principles of constitutional AI?")
print(response)
```

---

### 3. API Aggregator Bindings

These bindings connect to services that provide access to many different models through a single API.

#### **OpenRouter**

OpenRouter provides a unified, OpenAI-compatible interface to access models from dozens of providers (Google, Anthropic, Mistral, Groq, etc.) with one API key.

**Prerequisites:**
*   An OpenRouter API key (starts with `sk-or-...`).

**Usage:**
Model names must be specified in the format `provider/model-name`.

```python
from lollms_client import LollmsClient

OPENROUTER_CONFIG = {
    "open_router_api_key": "your_openrouter_api_key_here"
}

# Example using a Claude model through OpenRouter
lc = LollmsClient(
    binding_name="open_router",
    model_name="anthropic/claude-3-haiku-20240307",
    llm_binding_config=OPENROUTER_CONFIG
)

response = lc.generate_text("Explain what an API aggregator is, as if to a beginner.")
print(response)
```

#### **Groq**

While Groq is a direct provider, it's famous as an aggregator of speed. It runs open-source models on custom LPU hardware for exceptionally fast inference.

**Prerequisites:**
*   A Groq API key.

**Usage:**

```python
from lollms_client import LollmsClient

GROQ_CONFIG = {
    "groq_api_key": "your_groq_api_key_here"
}

lc = LollmsClient(
    binding_name="groq",
    model_name="llama3-8b-8192",
    llm_binding_config=GROQ_CONFIG
)

response = lc.generate_text("Write a 3-line poem about incredible speed.")
print(response)
```

#### **Hugging Face Inference API**

This connects to the serverless Hugging Face Inference API, allowing experimentation with thousands of open-source models without local hardware.

**Note:** This API can have "cold starts," so the first request might be slow.

**Prerequisites:**
*   A Hugging Face User Access Token (starts with `hf_...`).

**Usage:**

```python
from lollms_client import LollmsClient

HF_CONFIG = {
    "hf_api_key": "your_hugging_face_token_here"
}

lc = LollmsClient(
    binding_name="hugging_face_inference_api",
    model_name="google/gemma-1.1-7b-it",
    llm_binding_config=HF_CONFIG
)

response = lc.generate_text("Write a short story about a robot who discovers music.")
print(response)
```

### Listing Available Models

You can query the active LLM binding to get a list of models it supports or has available. The exact information returned depends on the binding (e.g., Ollama lists local models, OpenAI lists all its API models).

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors

try:
    # Initialize client for Ollama (or any other binding)
    lc = LollmsClient(
        binding_name="ollama",
        model_name="llama3", # A default model is still required for initialization
        host_address="http://localhost:11434"
    )

    ASCIIColors.yellow("\nListing available models for the current binding:")
    available_models = lc.listModels()

    if isinstance(available_models, list):
        for model in available_models:
            # Model structure varies by binding, common fields are 'name'
            model_name = model.get('name', 'N/A')
            model_size = model.get('size', 'N/A') # Common for Ollama
            print(f"- {model_name} (Size: {model_size})")
    elif isinstance(available_models, dict) and "error" in available_models:
        ASCIIColors.error(f"Error listing models: {available_models['error']}")
    else:
        print("Could not retrieve model list or unexpected format.")

except Exception as e:
    ASCIIColors.error(f"An error occurred: {e}")

```

### Sequential Summarization for Long Texts

When dealing with a document, article, or transcript that is too large to fit into a model's context window, the `summarize` method is the solution. It intelligently chunks the text, summarizes each piece, and then synthesizes those summaries into a final, coherent output.

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors

# --- A very long text (imagine this is 10,000+ tokens) ---
long_text = """
The history of computing is a fascinating journey from mechanical contraptions to the powerful devices we use today. 
It began with devices like the abacus, used for arithmetic tasks. In the 19th century, Charles Babbage conceived 
the Analytical Engine, a mechanical computer that was never fully built but laid the groundwork for modern computing. 
...
(many, many paragraphs later)
...
Today, quantum computing promises to revolutionize the field once again, tackling problems currently intractable 
for even the most powerful supercomputers. Researchers are exploring qubits and quantum entanglement to create 
machines that will redefine what is computationally possible, impacting fields from medicine to materials science.
""" * 50 # Simulate a very long text

# --- Callback to see the process in action ---
def summary_callback(chunk: str, msg_type: MSG_TYPE, params: dict = None, **kwargs):
    if msg_type in [MSG_TYPE.MSG_TYPE_STEP_START, MSG_TYPE.MSG_TYPE_STEP_END]:
        ASCIIColors.yellow(f">> {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP:
        ASCIIColors.cyan(f"   {chunk}")
    return True

try:
    lc = LollmsClient(binding_name="ollama", model_name="llama3")

    # The contextual prompt guides the focus of the summary
    context_prompt = "Summarize the text, focusing on the key technological milestones and their inventors."

    ASCIIColors.blue("--- Starting Sequential Summarization ---")
    
    final_summary = lc.sequential_summarize( # Note: changed from summarize to sequential_summarize
        text_to_process=long_text, # Note: changed from text_to_summarize to text_to_process
        contextual_prompt=context_prompt,
        chunk_size_tokens=1000, # Adjust based on your model's context size
        overlap_tokens=200,
        streaming_callback=summary_callback,
        temperature=0.1 # Good for factual summarization
    )
    
    ASCIIColors.blue("\n--- Final Comprehensive Summary ---")
    ASCIIColors.green(final_summary)

except Exception as e:
    print(f"An error occurred: {e}")
```
## Contributing

Contributions are welcome! Whether it's bug reports, feature suggestions, documentation improvements, or new bindings, please feel free to open an issue or submit a pull request on our [GitHub repository](https://github.com/ParisNeo/lollms_client).

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

## Changelog

For a list of changes and updates, please refer to the [CHANGELOG.md](CHANGELOG.md) file.