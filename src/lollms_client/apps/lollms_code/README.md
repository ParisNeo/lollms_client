# lollms_code — Autonomous CLI Coding Agent

A production-grade CLI tool that turns a single prompt into a full autonomous
coding session. Built on `lollms_client`'s high-grade Agent system.

## Features

- **Autonomous Loops**: Write → Test → Fix → Repeat, all from one prompt
- **Persistent Skills**: Creates SKILL.md files that survive across sessions
- **Episodic Memory**: Remembers what worked and what didn't
- **Sub-Agent Delegation**: Spawns focused child agents for complex tasks
- **Model Switching**: Adapts model mid-task for optimal performance
- **Workspace Isolation**: Each project gets its own sandbox
- **Intelligent Context**: Auto-injects workspace files, skills, and memories
- **Interactive REPL**: Multi-turn conversational mode for iterative work

## Installation

```bash
pip install lollms_client[app]
```

## Quick Start

```bash
# Single-prompt autonomous mode
lollms-code "Implement a REST API client with retry logic"

# Interactive REPL mode
lollms-code -i

# Target a specific project
lollms-code --workspace ./myproject "add unit tests for all modules"

# Use a specific model
lollms-code --model qwen3:32b "refactor the database layer"
```

## Configuration

Configuration is loaded from (in priority order):
1. CLI arguments
2. Environment variables (`LOLLMS_CODE_*`)
3. `.env` file in the current directory
4. `~/.lollms_hub/lollms_code/config.json`
5. Built-in defaults

### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOLLMS_CODE_LLM_BINDING` | `ollama` | LLM binding name |
| `LOLLMS_CODE_MODEL` | `qwen3:32b` | Model name |
| `LOLLMS_CODE_HOST` | `http://localhost:11434` | Host address |
| `LOLLMS_CODE_API_KEY` | None | API key for gated services |
| `LOLLMS_CODE_MAX_STEPS` | `100` | Max reasoning steps |
| `LOLLMS_CODE_TEMPERATURE` | `0.3` | Sampling temperature |
| `LOLLMS_CODE_MAX_TOKENS` | `8192` | Max tokens per turn |

## How It Works

```
User Prompt
    │
    ▼
┌──────────────────────────────┐
│  1. Pre-Turn Hydration        │
│     • Memory decay + pull    │
│     • Skills context inject  │
│     • Workspace files scan   │
│     • Tool discovery         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. Autonomous Reasoning Loop  │
│  ┌────────────────────────┐  │
│  │ LLM generates via       │  │
│  │ _AgentStreamState       │  │
│  │ (intercepts <tool>,     │  │
│  │  <done/>)               │  │
│  └──────────┬─────────────┘  │
│             │                │
│    ┌────────┴───────┐       │
│    ▼                ▼       │
│  <tool>          <done/>    │
│    │                │       │
│  Execute         Break ✓    │
│  Inject result             │
│  Continue loop             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. Post-Turn Processing      │
│     • Strip thinking blocks   │
│     • Process memory tags     │
│     • Save episodic memory    │
│     • Create/update skills    │
│     • Return result dict      │
└──────────────────────────────┘
```

## Skills (Cross-Session Learning)

The agent automatically creates SKILL.md files after completing non-trivial
tasks. These persist in `~/.lollms_hub/lollms_code/skills/` and are loaded
in future sessions.

```bash
# List learned skills
lollms-code --list-skills
```

## Interactive Mode Commands

| Command | Description |
|---|---|
| `exit` / `quit` | Exit the REPL |
| `skills` | List all learned skills |
| `clear` | Clear conversation history |
| `models` | List available models for switching |

## Return Value

The `chat()` method returns a structured dictionary:

```python
{
    "response": str,              # Final text response
    "tool_calls": [...],          # All tool calls made
    "tool_results": [...],        # Raw tool results
    "rounds": int,                # Total reasoning rounds
    "workspace_changes": [...],   # Files created/modified
    "was_cancelled": bool,        # Cancellation status
    "skills_created": [...],      # New skills this turn
    "skills_updated": [...],      # Updated skills this turn
    "sub_agents_spawned": int,    # Child agent count
    "model_switches": [...],      # Model switches made
}
```