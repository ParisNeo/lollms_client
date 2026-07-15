# 🤖 Lollms Agent — High-Grade Agentic System

A fully autonomous, tool-using, self-enhancing agent built on `lollms_client`. Unlike `LollmsDiscussion` (which is optimized for conversational turns with ~20 reasoning rounds), the **Agent** is designed for **long-horizon autonomous tasks** that may require **50–200 reasoning rounds** — such as build→test→fix loops, multi-file refactoring, research synthesis, and iterative problem solving.

---

## 🧠 Architecture Overview

The Agent is composed of five orthogonal subsystems, each fully isolated and independently configurable:

| Subsystem | Purpose |
|---|---|
| **SkillsManager** | Loads `SKILL.md` files from external directories (NOT the workspace). Supports "always_visible", "loadable", and "mixed" modes. The agent can create/update/delete skills via tools. |
| **CapabilityFlags** | Boolean gates: `enable_code_execution`, `enable_sub_agents`, `enable_model_switching`, `enable_image_generation`, `enable_tts`, etc. All dangerous defaults are `False`. |
| **SubAgentSpawner** | Spawns focused child agents with depth/count limits. Children share the workspace but CANNOT spawn further sub-agents (prevents infinite recursion). |
| **ModelSwitcher** | On-the-fly model switching via `load_model`/`unload_model` or `model_name` attribute. |
| **BindingToolsBuilder** | Exposes TTI/TTS/STT/TTM/TTV bindings as callable tools when the corresponding capability flag is enabled. |

### Agentic Loop Lifecycle

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────┐
│  1. Pre-Turn Hydration                    │
│     • Memory decay + associative pull     │
│     • RAG context injection               │
│     • Skills context injection            │
│     • Tool discovery (all 8 sources)      │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  2. Reasoning Loop (max_reasoning_steps)  │
│  ┌─────────────────────────────────────┐ │
│  │ LLM generates text via _AgentStream │ │
│  │ State (intercepts <tool>, <done/>)  │ │
│  └──────────────┬──────────────────────┘ │
│                 │                         │
│        ┌────────┴────────┐               │
│        ▼                 ▼               │
│   <tool> detected    <done/> detected     │
│        │                 │               │
│   Execute tool      Break loop ✓          │
│   Inject result                         │
│   Continue loop                         │
│                 │                         │
│        ┌────────┴────────┐               │
│        ▼                 ▼               │
│   Success           Failure              │
│   Record sig        FailureMemory        │
│   Continue          Block if repeat      │
│                 │                         │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  3. Post-Turn Processing                  │
│     • Strip thinking blocks               │
│     • Process memory tags                 │
│     • Save episodic memory                │
│     • Update internal conversation        │
│     • Return structured result dict       │
└──────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags
from lollms_client.lollms_personality import LollmsPersonality

# 1. Create a client
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "qwen3:32b", "host_address": "http://localhost:11434"},
)

# 2. Define a personality
personality = LollmsPersonality(
    name="Builder",
    author="you",
    category="engineering",
    description="An autonomous code builder and fixer.",
    system_prompt="You are an expert software engineer. You build, test, and fix code autonomously.",
)

# 3. Configure capabilities
caps = CapabilityFlags(
    enable_code_execution=True,      # Allow running Python code
    enable_sub_agents=True,          # Allow spawning child agents
    enable_model_switching=False,    # Keep on one model
    enable_skill_creation=True,      # Allow the agent to learn
    enable_skill_loading=True,
    skills_mode="loadable",          # Skills listed but loaded on demand
    max_sub_agent_depth=2,
    max_sub_agents_per_turn=3,
)

# 4. Create the agent
agent = Agent(
    lc=client,
    personality=personality,
    name="BuilderBot",
    role=AgentRole.IMPLEMENTER,
    workspace_path="./my_workspace",
    capabilities=caps,
    max_tokens_per_turn=8192,        # Large context for code generation
)

# 5. Run a single turn
result = agent.chat(
    prompt="Write a Python script that sorts a CSV file by the second column and saves the result.",
    max_reasoning_steps=30,          # More rounds than discussion's default 20
    temperature=0.3,                 # Low temperature for deterministic code
)

print(result["response"])
print(f"Rounds: {result['rounds']}, Tools called: {len(result['tool_calls'])}")
```

---

## 👜 The Handbag — Unified Agent Resource Folder

The **Handbag** is a self-contained folder that carries **ALL** of an agent's resources in one place. Instead of passing `personality`, `tool_files`, `skills_dirs`, `memory_manager`, and `workspace_path` as separate constructor parameters, you point the agent to a single handbag folder and everything is auto-configured.

### Why Use a Handbag?

| Without Handbag | With Handbag |
|---|---|
| 5+ separate constructor parameters | Single `handbag_path` parameter |
| Resources scattered across the filesystem | Everything in one portable folder |
| Hard to share or version-control an agent | Copy the folder → share the entire agent |
| Manual configuration of each subsystem | Auto-configuration with override semantics |

### Folder Structure

```
my_handbag/
├── handbag.yaml              # Optional manifest (name, default_personality, skills_mode)
├── personalities/            # Personality bundles (subdirs with SOUL.md)
│   ├── researcher/
│   │   └── SOUL.md
│   └── coder/
│       └── SOUL.md
├── tools/                    # Extra LCP tools (.py files or subdirs)
│   ├── my_custom_tool.py
│   └── another_tool/
│       └── another_tool.py
├── skills/                   # SKILL.md files (agent creates/updates these over time)
│   ├── python_patterns/
│   │   └── SKILL.md
│   └── ...
├── rag/                      # RAG documents (text files for retrieval)
│   ├── doc1.txt
│   └── doc2.md
├── memory/                   # Memory database
│   └── memory.db
└── workspace/                # Optional isolated workspace
```

### The `handbag.yaml` Manifest (Optional)

```yaml
name: "My Agent Handbag"
version: "1.0"
description: "A handbag containing all agent resources."
default_personality: "coder"    # Name of the default personality folder
skills_mode: "mixed"             # "always_visible", "loadable", or "mixed"
```

If no manifest is present, the handbag uses sensible defaults (first personality found, `mixed` skills mode).

### What Each Subdirectory Provides

| Subdirectory | Purpose | What It Configures |
|---|---|---|
| `personalities/` | Personality bundles (SOUL.md format) | `personality` parameter |
| `tools/` | Extra LCP tool files (.py) | `tool_files` parameter (appended) |
| `skills/` | SKILL.md knowledge capsules | `skills_dirs` parameter (appended) |
| `rag/` | Text documents for retrieval | RAG data source attached to personality |
| `memory/` | SQLite database for persistent memory | `memory_manager` parameter |
| `workspace/` | Isolated working directory | `workspace_path` parameter |

Missing subdirectories are silently skipped — you only need the folders you actually use.

### Override Semantics (Critical)

The handbag provides **defaults**. Explicit constructor parameters **always override** handbag-provided values:

```python
# Handbag provides a personality, but we override it explicitly
agent = Agent(
    lc=client,
    handbag_path="./my_handbag",       # Provides defaults for everything
    personality=my_custom_personality,  # OVERRIDES handbag's default personality
)

# Handbag provides tool files, and we ADD more on top
agent = Agent(
    lc=client,
    handbag_path="./my_handbag",       # Provides 3 tool files from tools/
    tool_files=["extra_tool.py"],      # ADDED to handbag's 3 tools → 4 total
)

# Handbag provides skills dirs, and we ADD more on top
agent = Agent(
    lc=client,
    handbag_path="./my_handbag",       # Provides skills/ dir
    skills_dirs=["./extra_skills"],    # ADDED to handbag's skills dir
)
```

### RAG Integration

The handbag's `rag/` folder is automatically indexed and attached to the default personality as a RAG data source (if the personality doesn't already have one). The Handbag tries **safestore** for semantic search if available, and falls back to **keyword-based scoring** if not.

```python
# The RAG is automatically attached — no manual configuration needed
agent = Agent(lc=client, handbag_path="./my_handbag")
result = agent.chat(prompt="Search for information about...")
# The agent can query the rag/ documents via its personality's query_data()
```

### Quick Start: Creating and Using a Handbag

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags, Handbag

# 1. Create a handbag structure on disk (one-time setup)
Handbag.create_structure("./my_agent_handbag", name="My Research Agent")

# 2. Add a personality (create personalities/researcher/SOUL.md)
#    Add tools to tools/
#    Add RAG documents to rag/
#    Add SKILL.md files to skills/

# 3. Create the client
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "qwen3:32b", "host_address": "http://localhost:11434"},
)

# 4. Create the agent with JUST the handbag path
agent = Agent(
    lc=client,
    handbag_path="./my_agent_handbag",
    name="ResearchBot",
    role=AgentRole.DOMAIN_EXPERT,
    capabilities=CapabilityFlags(enable_code_execution=True),
)

# 5. Chat — the agent has its personality, tools, skills, RAG, and memory
result = agent.chat(prompt="Analyze the documents in your RAG and summarize key findings.")
```

### Runtime Personality Switching

When a handbag contains multiple personalities, you can switch between them at runtime:

```python
# List all personalities in the handbag
personalities = agent.list_handbag_personalities()
print(personalities)
# {'researcher': LollmsPersonality(...), 'coder': LollmsPersonality(...)}

# Switch to a different personality
agent.switch_handbag_personality("coder")
# The agent now uses the 'coder' personality with the handbag's RAG attached
```

### Backward Compatibility

The `handbag_path` parameter is **purely optional**. All existing Agent constructor parameters continue to work exactly as before:

```python
# This still works perfectly — no handbag needed
agent = Agent(
    lc=client,
    personality=my_personality,
    tool_files=["tool1.py"],
    skills_dirs=["./skills"],
    workspace_path="./workspace",
)

# This also works — handbag provides defaults, explicit params override
agent = Agent(
    lc=client,
    handbag_path="./my_handbag",       # Provides personality, tools, skills, memory
    workspace_path="./custom_workspace",  # Overrides handbag's workspace
)
```

If neither `personality` nor `handbag_path` provides a personality, the Agent raises a clear `ValueError` with instructions on how to fix it.

### Handbag API Reference

| Method | Description |
|---|---|
| `agent.handbag` | Returns the loaded `Handbag` instance, or `None` if no handbag was provided. |
| `agent.list_handbag_personalities()` | Lists all personalities available in the handbag as `{name: LollmsPersonality}`. |
| `agent.switch_handbag_personality(name)` | Switches to a different personality from the handbag. Returns `True` on success. |
| `Handbag.create_structure(path, name)` | Static method that creates a new handbag folder structure on disk. |
| `Handbag.get_default_personality()` | Returns the manifest-specified default personality (or first found). |
| `Handbag.get_personalities()` | Returns all loaded personalities. |
| `Handbag.get_tool_files()` | Returns the list of discovered tool file paths. |
| `Handbag.get_skills_dirs()` | Returns the list of skills directory paths. |
| `Handbag.get_rag_data_source()` | Returns the RAG callable (safestore or keyword-based). |
| `Handbag.create_memory_manager()` | Creates a `LollmsMemoryManager` from the handbag's `memory/` directory. |
| `Handbag.attach_rag_to_personality(p)` | Attaches the handbag's RAG to a personality if it doesn't have its own. |

---

## 🔄 Tutorial: Building a Build → Test → Fix Loop

This is the flagship pattern. The agent writes code, executes it, reads the error output, fixes the code, and repeats until the objective is achieved or `max_reasoning_steps` is exhausted.

### Key Insight: Why `max_reasoning_steps` Should Be High

A `LollmsDiscussion` defaults to `max_reasoning_steps=20` because conversational turns are short. But an **autonomous build→test→fix loop** involves:

| Phase | Typical Rounds |
|---|---|
| Write initial code | 1–2 |
| Execute code | 1 |
| Read error | 1 |
| Fix code (SEARCH/REPLACE) | 1–2 |
| Re-execute | 1 |
| Repeat for multiple bugs | ×3–10 cycles |
| Write tests | 2–3 |
| Run tests | 1 |
| Fix test failures | 2–5 |
| Final verification | 1–2 |

**Total: 20–80 rounds.** We recommend `max_reasoning_steps=100` for complex loops.

### Full Example

```python
#!/usr/bin/env python3
"""
build_test_fix_loop.py
======================
A fully autonomous build→test→fix loop using the Lollms Agent.

The agent is tasked with:
1. Writing a Python module (e.g., a REST API client)
2. Writing unit tests for it
3. Running the tests
4. Fixing any failures
5. Repeating until all tests pass or max rounds is reached

No human intervention required.
"""

import sys
from pathlib import Path
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags
from lollms_client.lollms_personality import LollmsPersonality

def main():
    # ── 1. Client Setup ──
    client = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={
            "model_name": "qwen3:32b",
            "host_address": "http://localhost:11434",
        },
    )

    # ── 2. Personality: Autonomous Engineer ──
    personality = LollmsPersonality(
        name="AutonomousEngineer",
        author="tutorial",
        category="engineering",
        description="An autonomous software engineer that builds, tests, and fixes code iteratively.",
        system_prompt=(
            "You are an Autonomous Software Engineer.\n\n"
            "## Your Mission\n"
            "You operate in a fully autonomous loop. You receive an objective, and you must:\n"
            "1. WRITE the code (using tool_write_file)\n"
            "2. WRITE unit tests (using tool_write_file)\n"
            "3. EXECUTE the tests (using tool_execute_python_code)\n"
            "4. READ the test output\n"
            "5. If tests FAIL, FIX the code (using tool_write_file to overwrite)\n"
            "6. RE-EXECUTE the tests\n"
            "7. Repeat steps 4–6 until ALL tests pass\n"
            "8. When all tests pass, emit <done/>\n\n"
            "## Rules\n"
            "- NEVER ask the user for help. You are autonomous.\n"
            "- If a test fails, read the traceback carefully and fix the ROOT CAUSE.\n"
            "- Do NOT rewrite the entire file for small fixes. Use tool_write_file to overwrite.\n"
            "- After each execution, state what you observed and what you will do next.\n"
            "- When ALL tests pass, write a brief summary and emit <done/>.\n"
            "- If you are stuck after 10 attempts on the same bug, emit <done/> with an explanation.\n"
        ),
    )

    # ── 3. Capabilities ──
    caps = CapabilityFlags(
        enable_code_execution=True,       # CRITICAL: Must be True for build→test→fix
        enable_sub_agents=False,          # Keep it simple for this example
        enable_model_switching=False,
        enable_skill_creation=True,       # Let the agent learn from its mistakes
        enable_skill_loading=True,
        enable_workspace_tools=True,      # tool_write_file, tool_read_file, tool_list_files
        skills_mode="loadable",
    )

    # ── 4. Create Agent ──
    workspace = Path("./build_test_fix_workspace")
    workspace.mkdir(exist_ok=True)

    agent = Agent(
        lc=client,
        personality=personality,
        name="BuildTestFixBot",
        role=AgentRole.IMPLEMENTER,
        workspace_path=str(workspace),
        capabilities=caps,
        max_tokens_per_turn=8192,
        model_params={"temperature": 0.2},  # Low temperature for deterministic code
    )

    # ── 5. The Objective ──
    objective = (
        "## OBJECTIVE\n\n"
        "Build a Python module called `string_utils.py` that contains the following functions:\n\n"
        "1. `reverse_string(s: str) -> str` — Reverses a string.\n"
        "2. `count_vowels(s: str) -> int` — Counts vowels (a, e, i, o, u), case-insensitive.\n"
        "3. `is_palindrome(s: str) -> bool` — Returns True if the string is a palindrome (ignoring case and spaces).\n"
        "4. `title_case(s: str) -> str` — Converts to Title Case, handling multi-word strings.\n\n"
        "## REQUIREMENTS\n\n"
        "1. Write the module to `string_utils.py`.\n"
        "2. Write a comprehensive test file `test_string_utils.py` using the `unittest` framework.\n"
        "   - Include at least 3 test cases per function.\n"
        "   - Include edge cases (empty string, None handling, unicode).\n"
        "3. Run the tests by executing: `exec(open('test_string_utils.py').read())` via tool_execute_python_code.\n"
        "   Or write a runner script and execute that.\n"
        "4. If any test fails, fix the code in `string_utils.py` and re-run.\n"
        "5. Repeat until ALL tests pass.\n"
        "6. When done, emit <done/>.\n"
    )

    # ── 6. Run the Autonomous Loop ──
    print("=" * 70)
    print("🤖 STARTING AUTONOMOUS BUILD → TEST → FIX LOOP")
    print("=" * 70)
    print(f"Workspace: {workspace.resolve()}")
    print(f"Max reasoning steps: 100")
    print()

    result = agent.chat(
        prompt=objective,
        max_reasoning_steps=100,        # HIGH: complex loops need many rounds
        temperature=0.2,
        use_internal_history=True,      # Maintain context across rounds
    )

    # ── 7. Report ──
    print("\n" + "=" * 70)
    print("📊 LOOP COMPLETED")
    print("=" * 70)
    print(f"Total rounds:              {result['rounds']}")
    print(f"Tool calls:                {len(result['tool_calls'])}")
    print(f"Workspace files created:   {len(result['workspace_changes'])}")
    print(f"Skills created:            {result['skills_created']}")
    print(f"Skills updated:            {result['skills_updated']}")
    print(f"Was cancelled:             {result['was_cancelled']}")
    print()

    # List workspace files
    print("📁 Final workspace contents:")
    for f in sorted(workspace.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(workspace)} ({f.stat().st_size:,} bytes)")

    # Show the final response
    print("\n📝 Final Agent Response:")
    print("-" * 70)
    print(result["response"][:2000] + ("..." if len(result["response"]) > 2000 else ""))

if __name__ == "__main__":
    main()
```

---

## 🧪 Tutorial: Sub-Agent Delegation for Complex Tasks

When a task is too complex for a single agent, it can delegate sub-tasks to focused child agents. This is useful for:

- **Research + Synthesis**: One agent researches, another synthesizes
- **Multi-file projects**: Each sub-agent handles one file
- **Parallel exploration**: Try multiple approaches simultaneously

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "qwen3:32b", "host_address": "http://localhost:11434"},
)

# Master orchestrator personality
master_personality = LollmsPersonality(
    name="Orchestrator",
    author="tutorial",
    category="management",
    description="An orchestrator that delegates complex sub-tasks to sub-agents.",
    system_prompt=(
        "You are a Master Orchestrator. You break down complex tasks into sub-tasks "
        "and delegate each to a focused sub-agent using tool_spawn_sub_agent.\n\n"
        "Rules:\n"
        "- Break the user's request into 2–5 independent sub-tasks.\n"
        "- Use tool_spawn_sub_agent for each sub-task with clear instructions.\n"
        "- Collect all sub-agent reports.\n"
        "- Synthesize a final comprehensive answer.\n"
        "- Emit <done/> when the synthesis is complete.\n"
    ),
)

caps = CapabilityFlags(
    enable_code_execution=True,
    enable_sub_agents=True,           # CRITICAL: Enable sub-agent spawning
    max_sub_agent_depth=2,            # Master can spawn children, children cannot spawn
    max_sub_agents_per_turn=5,        # Up to 5 sub-agents per turn
    enable_skill_loading=True,
)

agent = Agent(
    lc=client,
    personality=master_personality,
    name="Orchestrator",
    role=AgentRole.MODERATOR,
    workspace_path="./orchestrator_workspace",
    capabilities=caps,
)

result = agent.chat(
    prompt=(
        "I need a comprehensive report on the following topics:\n"
        "1. The history of Python programming language\n"
        "2. The key differences between REST and GraphQL APIs\n"
        "3. Best practices for async programming in Python\n\n"
        "Delegate each topic to a sub-agent, then synthesize a unified report."
    ),
    max_reasoning_steps=80,           # High: sub-agents + synthesis needs room
    temperature=0.4,
)

print(f"Sub-agents spawned: {result['sub_agents_spawned']}")
print(f"Total rounds: {result['rounds']}")
print(result["response"])
```

---

## 🎓 Tutorial: Self-Enhancing Agent with Skills

The agent can create, update, and load **persistent skills** (SKILL.md files) that survive across sessions. This enables genuine self-improvement.

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags, SkillsManager
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "qwen3:32b", "host_address": "http://localhost:11434"},
)

# ── Custom skills directory ──
# You can pre-populate this with SKILL.md files, or let the agent create them.
skills_dir = "./my_agent_skills"

personality = LollmsPersonality(
    name="SelfImprovingAgent",
    author="tutorial",
    category="general",
    description="An agent that learns from experience and saves reusable skills.",
    system_prompt=(
        "You are a self-improving agent. After solving a problem, you MUST:\n"
        "1. Reflect on what you learned.\n"
        "2. If the lesson is reusable, use tool_create_skill to save it.\n"
        "3. If you discover a better approach to an existing skill, use tool_update_skill.\n"
        "4. Before starting a new task, use tool_list_skills and tool_load_skill "
        "to check if a relevant skill exists.\n\n"
        "Skills are your long-term memory. They make you better over time.\n"
    ),
)

caps = CapabilityFlags(
    enable_code_execution=True,
    enable_skill_creation=True,       # Can create new skills
    enable_skill_loading=True,        # Can load existing skills
    skills_mode="mixed",              # Always-visible skills in prompt + loadable index
)

agent = Agent(
    lc=client,
    personality=personality,
    name="Learner",
    role=AgentRole.FREEFORM,
    workspace_path="./learning_workspace",
    capabilities=caps,
    skills_dirs=[skills_dir],         # Point to our custom skills directory
)

# ── Session 1: Agent learns a lesson ──
result = agent.chat(
    prompt=(
        "I have a CSV file `data.csv` with columns: name, age, salary. "
        "Calculate the average salary for people aged 30-40."
    ),
    max_reasoning_steps=50,
)

# The agent will:
# 1. Load any relevant skills (e.g., "data_analysis_with_pandas")
# 2. Execute the task using tool_execute_python_code
# 3. Create a skill like "CSV Aggregation Pattern" for future use

print("Skills after session 1:")
for skill in agent.list_skills():
    print(f"  - {skill['title']}: {skill['description']}")

# ── Session 2: Agent uses the learned skill ──
# In a new session, the agent will automatically discover and load the skill
# it created in session 1, making it faster and more reliable.
result2 = agent.chat(
    prompt="Calculate the median age for people earning more than 50000 in data.csv.",
    max_reasoning_steps=50,
)
```

---

## 🔧 Tutorial: Model Switching Mid-Task

The agent can switch between models on the fly — useful when a task requires different model strengths (e.g., a reasoning model for planning, a coding model for implementation).

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "qwen3:32b", "host_address": "http://localhost:11434"},
)

personality = LollmsPersonality(
    name="AdaptiveAgent",
    author="tutorial",
    category="general",
    description="An agent that switches models based on task requirements.",
    system_prompt=(
        "You are an adaptive agent with access to multiple models.\n\n"
        "## Model Selection Guidelines\n"
        "- For PLANNING and REASONING: Use a larger model (e.g., qwen3:32b)\n"
        "- For CODE GENERATION: Use a coding-specialized model (e.g., qwen3-coder:14b)\n"
        "- For SIMPLE TASKS: Use a smaller, faster model (e.g., qwen3:4b)\n\n"
        "Use tool_list_models to see available models, then tool_switch_model to switch.\n"
        "Switch BEFORE starting the phase that needs the different model.\n"
    ),
)

caps = CapabilityFlags(
    enable_code_execution=True,
    enable_model_switching=True,      # CRITICAL: Enable mid-task model switching
    enable_sub_agents=True,
)

agent = Agent(
    lc=client,
    personality=personality,
    name="AdaptiveBot",
    role=AgentRole.FREEFORM,
    workspace_path="./adaptive_workspace",
    capabilities=caps,
)

result = agent.chat(
    prompt=(
        "Plan and implement a binary search tree in Python with insertion, deletion, "
        "and traversal methods. Include comprehensive tests.\n\n"
        "Use a reasoning model for the planning phase, then switch to a coding model "
        "for implementation."
    ),
    max_reasoning_steps=80,
)

print(f"Model switches: {result['model_switches']}")
print(result["response"])
```

---

## 📡 Tutorial: Multimodal Agent (Image Generation + TTS)

When the client has TTI/TTS/STT bindings configured, the agent can use them as tools.

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "llama3.2-vision:11b", "host_address": "http://localhost:11434"},
    # TTI binding (e.g., diffusers, openai, etc.)
    tti_binding_name="openai",
    tti_binding_config={"api_key": "sk-...", "model_name": "dall-e-3"},
    # TTS binding (e.g., xtts, piper, etc.)
    tts_binding_name="xtts",
    tts_binding_config={"host": "localhost", "port": 9633, "auto_start_server": True},
)

personality = LollmsPersonality(
    name="MultimodalAgent",
    author="tutorial",
    category="creative",
    description="An agent that can generate images and speech.",
    system_prompt=(
        "You are a creative multimodal agent. You can:\n"
        "- Generate images using tool_generate_image\n"
        "- Edit images using tool_edit_image\n"
        "- Convert text to speech using tool_text_to_speech\n\n"
        "When the user asks for visual or audio content, use the appropriate tool.\n"
    ),
)

caps = CapabilityFlags(
    enable_image_generation=True,
    enable_image_editing=True,
    enable_tts=True,
    enable_code_execution=False,
)

agent = Agent(
    lc=client,
    personality=personality,
    name="CreativeBot",
    role=AgentRole.NARRATOR,
    workspace_path="./creative_workspace",
    capabilities=caps,
)

result = agent.chat(
    prompt=(
        "Create a children's story about a brave little robot. "
        "Generate an illustration for the story and narrate it with text-to-speech."
    ),
    max_reasoning_steps=40,
)

print(f"Images generated: {sum(1 for tc in result['tool_calls'] if tc['name'] == 'tool_generate_image')}")
print(f"Audio generated: {sum(1 for tc in result['tool_calls'] if tc['name'] == 'tool_text_to_speech')}")
```

---

## 📊 Return Value Reference

The `chat()` method returns a structured dictionary:

```python
{
    "response": str,              # Final text response (thinking blocks stripped)
    "tool_calls": [               # List of all tool calls made
        {
            "round": int,         # Which reasoning round
            "name": str,          # Tool name (e.g., "tool_execute_python_code")
            "parameters": dict,   # Parameters passed to the tool
        },
        ...
    ],
    "tool_results": [             # Raw results of each tool call
        {
            "round": int,
            "name": str,
            "result": dict,       # The raw tool return value
            "success": bool,
        },
        ...
    ],
    "rounds": int,                # Total reasoning rounds used
    "workspace_changes": [        # Files created/modified in workspace
        {"action": "created", "path": "script.py", "size": 1234},
        {"action": "modified", "path": "data.csv", "size": 5678},
    ],
    "was_cancelled": bool,        # True if generation was cancelled
    "skills_created": [str],      # Titles of skills created this turn
    "skills_updated": [str],      # Titles of skills updated this turn
    "sub_agents_spawned": int,    # Number of sub-agents spawned
    "model_switches": [str],      # Model names switched to
}
```

---

## 🎯 Best Practices for Autonomous Loops

### 1. Set `max_reasoning_steps` Appropriately

| Task Type | Recommended `max_reasoning_steps` |
|---|---|
| Simple question (no tools) | 5 |
| Single tool call + answer | 10 |
| Write + run a script | 20–30 |
| Build → Test → Fix loop | **80–150** |
| Multi-file project with tests | **100–200** |
| Research + synthesis with sub-agents | 60–100 |

### 2. Use Low Temperature for Code

```python
# For deterministic code generation:
result = agent.chat(prompt="...", temperature=0.1, max_reasoning_steps=100)

# For creative tasks:
result = agent.chat(prompt="...", temperature=0.7, max_reasoning_steps=50)
```

### 3. Design System Prompts for Autonomy

Include explicit instructions like:
- "NEVER ask the user for help. You are autonomous."
- "If a test fails, read the traceback and fix the ROOT CAUSE."
- "When ALL tests pass, emit `<done/>`."
- "If stuck after N attempts, emit `<done/>` with an explanation."

### 4. Use Skills for Cross-Session Learning

```python
# The agent creates skills automatically. To inspect them:
for skill in agent.list_skills():
    print(f"  📚 {skill['title']} [{skill['category']}]: {skill['description']}")

# To manually add a skill:
agent.add_skill(
    title="Always use utf-8 encoding",
    description="When reading/writing files in Python, always specify encoding='utf-8'",
    category="python",
    content="## Rule\nAlways use `open(file, encoding='utf-8')` to avoid UnicodeDecodeError on Windows.\n\n## Example\n```python\nwith open('data.csv', encoding='utf-8') as f:\n    content = f.read()\n```",
    tags=["python", "encoding", "best-practice"],
)
```

### 5. Enable Code Execution Safely

```python
caps = CapabilityFlags(
    enable_code_execution=True,    # The agent CAN run Python code
    # Code runs in the workspace CWD, so it can read/write files there.
    # For production, consider sandboxing the workspace directory.
)
```

### 6. Cancel Long-Running Loops

```python
import threading

# Start a long loop in a background thread
def run_loop():
    result = agent.chat(prompt="...", max_reasoning_steps=200)
    print(result["response"])

thread = threading.Thread(target=run_loop)
thread.start()

# Cancel after 60 seconds if needed
import time
time.sleep(60)
agent.cancel_generation()
thread.join()
```

---

## 📁 SKILL.md File Format

Skills are Markdown files with optional YAML frontmatter:

```markdown
---
title: "CSV Data Analysis Pattern"
description: "Standard workflow for analyzing CSV files with pandas"
category: "data_analysis"
tags: [python, pandas, csv, data]
always_visible: false
---

# CSV Data Analysis Pattern

## Workflow
1. Load with `pd.read_csv(file_name, encoding='utf-8')`
2. Inspect with `df.head()`, `df.info()`, `df.describe()`
3. Filter with boolean indexing
4. Aggregate with `df.groupby()`
5. Save results with `df.to_csv()`

## Common Pitfalls
- Always specify `encoding='utf-8'` to avoid Windows errors
- Use `encoding='utf-8-sig'` if the file has a BOM
- Check for separator: semicolons are common in European CSVs
```

### Directory Structure

```
my_agent_skills/
├── csv_analysis/
│   └── SKILL.md          # Loaded as "CSV Data Analysis Pattern"
├── python_best_practices/
│   └── SKILL.md          # Loaded as "Python Best Practices"
└── debugging_patterns/
    └── SKILL.md          # Loaded as "Debugging Patterns"
```

---

## 🔗 API Reference Summary

### Agent Core

| Method | Description |
|---|---|
| `agent.chat(prompt, ...)` | Main agentic loop. Returns structured dict. |
| `agent.generate(prompt, ...)` | Direct (non-agentic) text generation. |
| `agent.generate_with_tools(prompt, tools, ...)` | Backward-compatible wrapper that delegates to `chat()`. |
| `agent.generate_structured(prompt, schema, ...)` | Structured JSON generation. |
| `agent.cancel_generation()` | Thread-safe cancellation. |
| `agent.clear_conversation()` | Clear internal conversation history. |

### Skills API

| Method | Description |
|---|---|
| `agent.list_skills()` | List all available skills. |
| `agent.add_skill(title, description, category, content, tags)` | Create a new skill. |
| `agent.update_skill(title, new_content)` | Update an existing skill. |
| `agent.delete_skill(title)` | Delete a skill. |
| `agent.reload_skills()` | Reload all skills from disk. |

### Model Switching API

| Method | Description |
|---|---|
| `agent.list_available_models()` | List models available for switching. |
| `agent.get_current_model()` | Get the current model name. |
| `agent.switch_model(model_name)` | Switch to a different model. |
| `agent.restore_original_model()` | Restore the original model. |

### Sub-Agent API

| Method | Description |
|---|---|
| `agent.spawn_sub_agent(instruction, personality_conditioning, model_name)` | Spawn a child agent. |

---

## 🧩 CapabilityFlags Reference

| Flag | Default | Description |
|---|---|---|
| `enable_code_execution` | `False` | Allow running Python code via LCP tools |
| `enable_external_file_access` | `False` | Access files outside workspace |
| `enable_networking` | `False` | Internet/network tools |
| `enable_image_generation` | `True` | TTI binding as tool |
| `enable_image_editing` | `True` | Image editing via TTI |
| `enable_tts` | `False` | TTS binding as tool |
| `enable_stt` | `False` | STT binding as tool |
| `enable_ttm` | `False` | Text-to-music |
| `enable_ttv` | `False` | Text-to-video |
| `enable_sub_agents` | `True` | Spawn child agents |
| `enable_model_switching` | `False` | Switch models mid-task |
| `enable_skill_creation` | `True` | Create new skills |
| `enable_skill_loading` | `True` | Load existing skills |
| `skills_mode` | `"loadable"` | `"always_visible"`, `"loadable"`, or `"mixed"` |
| `max_sub_agent_depth` | `3` | Max recursion depth for sub-agents |
| `max_sub_agents_per_turn` | `5` | Max sub-agents spawned per turn |
| `enable_workspace_tools` | `True` | Built-in file tools (always safe) |

---

## 🏁 Conclusion

The Lollms Agent is designed for **long-horizon autonomous work**. By combining:

- **High `max_reasoning_steps`** (80–200 for complex loops)
- **Code execution** for build→test→fix cycles
- **Skills** for cross-session learning
- **Sub-agents** for task delegation
- **Model switching** for adaptive performance

...you can build fully autonomous systems that iteratively solve complex problems without human intervention.