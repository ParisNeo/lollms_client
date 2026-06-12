## 1. Architectural Overview & Context Integration

`LollmsDiscussion` inherits from multiple specialized mixin classes to isolate concerns while exposing a unified API:

```
                  ┌──────────────────────────────────────────┐
                  │             LollmsDiscussion             │
                  └────┬────────────────────────────────┬────┘
                       │                                │
        ┌──────────────┴──────────────┐  ┌──────────────┴──────────────┐
        │          CoreMixin          │  │          ChatMixin          │
        │   (Database, Message CRUD)  │  │ (chat(), simplified_chat()) │
        └─────────────────────────────┘  └─────────────────────────────┘
                       │                                │
        ┌──────────────┴──────────────┐  ┌──────────────┴──────────────┐
        │         BranchMixin         │  │         MemoryMixin         │
        │    (Tree Forking, Merging)  │  │  (Working vs. Deep Memory)  │
        └─────────────────────────────┘  └─────────────────────────────┘
                       │                                │
        ┌──────────────┴──────────────┐  ┌──────────────┴──────────────┐
        │       FileImportMixin       │  │     InternetImportMixin     │
        │  (Ingest CSV, PDF, SQLite)  │  │   (Google, Web, Wikipedia)  │
        └─────────────────────────────┘  └─────────────────────────────┘
```

The `chat` method sits inside `ChatMixin` but relies heavily on other layers to build the context window before making the LLM call:
* **The Static System Prompt**: Derived from the selected `LollmsPersonality`'s `SOUL.md`.
* **The Global Data Zones**:
  * **User Data Zone**: User preferences and persistent profile rules.
  * **Discussion Data Zone**: Historical background, logs, and static files loaded into the active session.
  * **Personality Data Zone**: Domain-specific static resources linked directly to the selected assistant.
* **The Active Artifacts Zone**: Text, code, presentation slides, notes, or data structures marked as `active` are compiled dynamically. Large files can be summarized sequentially or deactivated automatically under context pressure.
* **The Tiered Memory Zone**: 
  * **Working Memory (Level 1)**: Highly relevant, recently retrieved facts injected directly as context.
  * **Deep Memory (Level 2)**: Only short *handles* (IDs and brief summaries) are injected. If the model determines it needs a deep memory to answer, it emits `<mem_load id="ID"/>` to bring it into the working zone.
  * **Archived Memory (Level 3)**: Dormant memories processed during background consolidation passes (`dream()`).
* **The Scratchpad**: A transient, high-capacity string buffer used to hold massive tool outputs, RAG search results, or intermediate reasoning traces during a single turn. It is cleared automatically once the final response is generated.

---

## 2. Complete Method Signature & Parameters

The method signature of `chat()` is defined as follows:

```python
def chat(
    self,
    user_message: str,
    personality: Optional[LollmsPersonality] = None,
    branch_tip_id: Optional[str] = None,
    tools: Optional[Dict[str, Dict[str, Any]]] = None,
    swarm: Optional[List[LollmsAgent]] = None,
    swarm_config: Optional[SwarmConfig] = None,
    add_user_message: bool = True,
    max_reasoning_steps: int = 20,
    images: Optional[List[str]] = None,
    debug: bool = False,
    remove_thinking_blocks: bool = True,
    enable_image_generation: bool = True,
    enable_image_editing: bool = True,
    auto_activate_artefacts: bool = True,
    enable_show_tools: bool = True,
    enable_extract_artefact: bool = True,
    enable_final_answer: bool = True,
    enable_repl_tools: bool = True,
    enable_inline_widgets: bool = True,
    enable_notes: bool = True,
    enable_skills: bool = False,
    enable_forms: bool = True,
    enable_books: bool = False,
    enable_presentations: bool = False,
    enable_silent_artefact_explanation: bool = True,
    memory_manager: Optional[LollmsMemoryManager] = None,
    enable_artefacts: bool = True,
    enable_memory: bool = True,
    enable_auto_dream: bool = True,
    enable_deep_memory_pulling: bool = True,
    enable_in_message_status: bool = True,
    enable_specialized_events_stream: bool = False,
    **kwargs
) -> Dict[str, Any]
```

### Parameter Explanations

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `user_message` | `str` | *Required* | Raw natural language input from the user. Ignored if `add_user_message=False` is set during branch regenerations. |
| `personality` | `Optional[LollmsPersonality]` | `None` | The active personality object. If `None`, defaults to a `NullPersonality` instance to ensure null-safety. |
| `branch_tip_id` | `Optional[str]` | `None` | The specific message ID in the tree branch to generate from. If `None`, uses `active_branch_id` to extend the current active path. |
| `tools` | `Optional[Dict]` | `None` | Dict of callables wrapped from external MCP bindings, keyed by tool name. |
| `swarm` | `Optional[List[LollmsAgent]]` | `None` | Optional list of agents. If provided, bypasses single-agent execution and delegates the turn to a `SwarmOrchestrator` session. |
| `swarm_config` | `Optional[SwarmConfig]` | `None` | Configuration object for multi-agent swarm operations. |
| `add_user_message` | `bool` | `True` | If `True`, adds a new user-type message to the database before generating. Set to `False` when regenerating or rewinding. |
| `max_reasoning_steps` | `int` | `20` | Hard cap on the number of tool-execution loops allowed in a single agentic turn to prevent runaway loops. |
| `images` | `Optional[List[str]]` | `None` | List of base64-encoded image strings supplied by the user (vision-input). |
| `debug` | `bool` | `False` | Enables verbose terminal output logging intermediate states, prompts, and schema structures. |
| `remove_thinking_blocks` | `bool` | `True` | If `True`, strips any content enclosed in `<think>...</think>` tags from the final user-facing message. |
| `enable_image_generation` | `bool` | `True` | Enables the parsing and execution of `<generate_image>` tags via the active TTI binding. |
| `enable_image_editing` | `bool` | `True` | Enables the parsing and execution of `<edit_image>` tags. |
| `auto_activate_artefacts` | `bool` | `True` | Automatically activates newly created or updated artifacts in the active session context. |
| `enable_show_tools` | `bool` | `True` | Registers the `show_tools()` tool, allowing the LLM to inspect available capabilities. |
| `enable_extract_artefact` | `bool` | `True` | Registers `extract_artifact_text()`, allowing the model to split large artifacts. |
| `enable_final_answer` | `bool` | `True` | Registers `final_answer()`, allowing the model to signal completion in RLM mode. |
| `enable_repl_tools` | `bool` | `True` | Registers in-session text REPL tools (`text_store`, `text_search`, etc.) for managing massive payloads. |
| `enable_inline_widgets` | `bool` | `True` | Enables the parsing of `<lollms_inline>` tags into interactive HTML5 widget containers. |
| `enable_notes` | `bool` | `True` | Enables user-facing `<note>` artifacts. |
| `enable_skills` | `bool` | `False` | Enables LLM-facing `<skill>` artifacts. |
| `enable_forms` | `bool` | `True` | Enables `<lollms_form>` interactive questionnaire cards. |
| `enable_books` | `bool` | `False` | Enables `<book>` artifact generation instructions. |
| `enable_presentations` | `bool` | `False` | Enables `<presentation>` slide-deck HTML templates. |
| `enable_silent_artefact_explanation` | `bool` | `True` | Generates a human-readable summary if the response was entirely consumed by XML tags. |
| `memory_manager` | `Optional[LollmsMemoryManager]` | `None` | Persistent SQLite memory manager. If `None`, falls back to the attached session manager. |
| `enable_artefacts` | `bool` | `True` | Controls the injection of active artifacts into the context window. |
| `enable_memory` | `bool` | `True` | Controls the injection of Working/Deep memory handles. |
| `enable_auto_dream` | `bool` | `True` | Triggers background memory consolidation (`dream()`) on turn completion. |
| `enable_deep_memory_pulling` | `bool` | `True` | Performs proactive semantic retrieval on the deep memory database using keywords in the user query. |
| `enable_in_message_status` | `bool` | `True` | Streams `<processing>` diagnostic log tags to the frontend. |
| `enable_specialized_events_stream` | `bool` | `False` | Streams raw specialist outputs directly to the main chat bubble. |

---

## 3. End-to-End Chat Execution Lifecycle

The following diagram outlines the logical flow of a single `chat()` call:

```
       [User sends Message]
                │
                ▼
     ┌──────────────────────┐
     │ Preflight Retrieval  │ ──► Proactive semantic search on Deep Memory
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │   Context Assembly   │ ──► Gathers system prompts, data zones,
     └──────────┬───────────┘     active artifacts, working memory
                │
                ▼
     ┌──────────────────────┐
     │   Context Check      │ ──► If >95% full: runs sequential reading
     └──────────┬───────────┘     on large artifacts & deactivates them
                │
                ▼
     ┌──────────────────────┐
     │   Fast Path Trial    │ ──► Streams complete prompt to LLM
     └──────────┬───────────┘
                │
                ├──────────────────────────┐
                │                          │
                ▼ (No tags emitted)        ▼ (<agent_mode/> tag emitted)
        [Clean Response]             [Agentic Loop Activated]
                │                          │
                │                          ▼
                │                ┌───────────────────┐
                │                │  Goal Extraction  │ ──► Extracts 1-3 objectives
                │                └─────────┬─────────┘
                │                          │
                │                          ▼
                │                ┌───────────────────┐
                │                │    THINK Phase    │ ──► Generates structured
                │                └─────────┬─────────┘     action plan (JSON schema)
                │                          │
                │                          ▼
                │                ┌───────────────────┐
                │                │     ACT Phase     │ ──► Runs chosen action
                │                └─────────┬─────────┘
                │                          │
                │     ┌────────────────────┴────────────────────┐
                │     ▼ (Tool Call)                             ▼ (Artifact Edit)
                │ [Execute Tool]                       [Surgical Patch-Retry Loop]
                │     │                                         │
                │     └────────────────────┬────────────────────┘
                │                          │
                │                          ▼
                │                ┌───────────────────┐
                │                │   VERIFY Phase    │ ──► Logs results & updates
                │                └─────────┬─────────┘     completed objectives list
                │                          │
                │                          ▼
                │                ┌───────────────────┐
                │                │  Done? (Round N)  │ ──► Loops back to THINK if
                │                └─────────┬─────────┘     objectives remain unfulfilled
                │                          │
                │                          ▼ (All objectives completed)
                │                ┌───────────────────┐
                │                │   Final Answer    │ ──► Model summarizes changes
                │                └─────────┬─────────┘     and answers user
                │                          │
                └──────────────────────────┼──────────────────────────┘
                                           │
                                           ▼
                                 ┌───────────────────┐
                                 │  Post-Processing  │ ──► Cleans code blocks, saves
                                 └─────────┬_________┘     episodes, runs dream() pass
                                           │
                                           ▼
                                 [Render to User]
```

### Stage 1: Preflight Retrieval & Context Assembly
1. **Pre-turn Memory Decay**: The active `LollmsMemoryManager` calculates age-based decay on persistent memories, demoting files falling below the threshold.
2. **Proactive Deep Memory Retrieval**: If `enable_deep_memory_pulling=True`, a TF-IDF semantic query is run over the Level 2 (Deep) and Level 3 (Archived) databases using keywords from `user_message`. Matching entries are promoted to Level 1 (Working Memory).
3. **Budget Verification**: The system counts the total tokens currently in the prompt. If the total context exceeds **95% of the active model limit**, the sequential cognitive reading subroutine is triggered:
   * Large active artifacts (>2,000 tokens) are split into 1,000-token chunks.
   * Each chunk is processed individually to extract key facts relevant to `user_message`.
   * The compiled summaries are saved to the scratchpad, and the original large files are temporarily deactivated to protect the context window.
4. **Context Packaging**: The system prompt is constructed by appending active directives (artifacts, images, notes, forms) to the selected personality's system prompt. This is combined with the active data zones, working memory blocks, and scratchpad content to create a unified system message.

### Stage 2: The Fast Path Trial
1. **First-Pass Generation**: To optimize response latency, the client initiates a direct conversational completion turn first (`stream=True`).
2. **On-Demand Routing Detection**: The streaming relay (`_fast_relay`) parses the incoming chunk stream in real-time:
   * If the model emits the `<agent_mode/>` tag at the start of its output, the fast-path stream is aborted immediately. The partial text is cleared, the system prompt is updated with full tool-calling instructions, and the execution transitions to **Stage 3 (The Agentic Loop)**.
   * If the model answers directly without emitting agent tags, the response is finalized as a normal conversational turn (skipping the multi-round loop).

### Stage 3: The Agentic Loop & State Machine
If agent mode is triggered, the orchestrator begins a multi-round loop (capped at `max_reasoning_steps`):
1. **Goal Extraction (Round 1)**: The model is called with a strict JSON schema to analyze the user's intent and extract 1–3 concrete objectives (e.g., `["search_topic", "create_artifact"]`).
2. **THINK Phase**: The model generates a structured action plan conforming to a JSON schema, choosing exactly one action:
   * `tool_call`: Execute a registered tool.
   * `artifact_action`: Create, modify, or patch an artifact.
   * `complete`: All objectives are met, transition to final answer.
3. **ACT Phase**:
   * For **Tool Calls**: The chosen callable is executed, and the results are stored in the scratchpad.
   * For **Artifact Actions**: The model emits `<artifact>` tags containing code or content updates.
4. **VERIFY Phase**: The orchestrator evaluates the action's results against the declared objectives. Completed objectives are moved to the `completed_objectives` list, and the loop transitions back to the **THINK Phase** for the next step.

---

## 4. Multi-Round Tool Execution & Error Recovery

### Tool Call Parsing & Execution
The system parses `<tool_call>` XML tags containing structured JSON in this format:

```xml
<tool_call>{"name": "search_web_duckduckgo", "parameters": {"query": "LoLLMS architecture"}}</tool_call>
```

To prevent repetitive execution, a semantic and parameter-based duplicate check is applied:
1. The tool parameters are serialized to a stable, sorted JSON string and hashed.
2. If the hash matches a previously executed tool call in the same turn, or if the search query is semantically identical, the execution is blocked.
3. A warning and a correction prompt are injected into the context, forcing the model to either change parameters or proceed to the final answer.

Tool outputs are returned using `LCPResult` structures. These are formatted into clean Markdown blocks for the LLM context, and large outputs (>2,000 characters) are automatically condensed into key summaries in the scratchpad to avoid bloating the context window.

### The Surgical Patch-Retry Loop (Aider SEARCH/REPLACE)
When modifying existing artifacts, the system uses a highly resilient diff engine (`apply_aider_patch`) to apply surgical patches:

```markdown
<<<<<<< SEARCH
[Exact lines currently in the file]
=======
[New lines to replace them with]
>>>>>>> REPLACE
```

If a patch fails to match the file's content character-for-character, the system initiates a self-healing loop:

```
[Patch Match Fails]
        │
        ▼
[Extract Diagnostic Hints] ──► Locates closest matching line and expected line bytes
        │
        ▼
[Build Correction Prompt] ──► Attaches exact file content + search mismatch details
        │
        ▼
[Low-Temp LLM Call] ──► Model attempts to correct the SEARCH block (up to 3 retries)
        │
        ├──────────────────────────┐
        │                          │
        ▼ (Correction succeeds)    ▼ (All retries fail)
  [Apply Patch]            [Full Overwrite Fallback]
```

1. **Diagnostic Extraction**: The system analyzes the mismatch to identify differences in spacing, casing, or line endings.
2. **Correction Prompting**: A prompt containing the exact current file content and the specific mismatch details is sent to the LLM at a low temperature (`temperature=0.1`) to request a corrected patch.
3. **Continuation & Fallback**:
   * On the last retry, the correction prompt explicitly allows a full file rewrite if the surgical patch continues to fail.
   * If all retries are exhausted, the system falls back to a full overwrite if the final response contains no SEARCH markers, preserving the original file otherwise.

---

## 5. Context Budgeting and Synopsis Pruning

To prevent context window exhaustion, the system uses the `summarize_and_prune` mechanism:

```
[Context Window Check]
        │
        ▼
 [Exceeds Limit?]
        │
        ├──────────────────────────┐
        ▼ (Yes)                    ▼ (No)
[History-Heavy?]             [Keep Generating]
        │
        ▼
[Select Past Turns] ──► Keeps last N exchanges as an active conversational anchor
        │
        ▼
[LLM Synopsis Pass] ──► Condenses older turns into a "Project State Synopsis"
        │
        ▼
[Inject Synopsis] ──► Replaces old messages with the compact synopsis in the virtual history
```

1. **State-Based Synopsis**: When the context size approaches the model's limit, the oldest messages in the branch (excluding the last $N$ exchanges, which serve as an active anchor) are selected.
2. **LLM Synthesis**: The LLM compiles these older turns into a dense, technical "Project State Synopsis," capturing all key decisions, file structures, and completed objectives.
3. **Virtual History Injection**: The original verbose messages are removed from the active context window and replaced by this compact synopsis, freeing up critical token space for the remaining generation.

---

## 6. Real-Time Status Streaming Protocol

When `enable_in_message_status=True`, diagnostic logs are streamed inside `<processing>` tags to provide a transparent, real-time view of the agent's actions:

```xml
<processing type="artefact_building" title="🏗️ BUILDING ARTEFACT: main.py">
* Creating new artefact 'main.py'
* ⚙️ Implementing: calculate_metrics()
* ⚙️ Implementing: plot_results()
</processing>
```

To prevent UI rendering issues, the system enforces the following constraints:
1. **Beginning-of-Line Formatting**: Opening `<processing>` tags conditionally prepend a newline (`\n`) if the preceding text does not end with one, ensuring the tag always starts cleanly at the beginning of a line in the markdown parser.
2. **Duplicate/Nest Prevention**: The system maintains an in-memory boolean `proc_has_opened` alongside a real-time string-based scan (`_is_processing_tag_currently_open()`) on `ai_message.content`. If a processing block is already open, any redundant open calls are ignored, guaranteeing that tags are never nested.
3. **Silent-Artifact Summaries**: If the model's response is entirely consumed by XML tags (leaving the user-facing message empty), the system automatically appends a clean, human-readable summary of the completed work (e.g., `📄 Created main.py (version 1) · 120 lines`) at the end of the turn.