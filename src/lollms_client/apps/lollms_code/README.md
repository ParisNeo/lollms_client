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

## Configuration: Universal Two-Tier Profiles

`lollms_code` uses the unified **Two-Tier Profile Architecture**:

1. **Connection Layer (`*_BINDINGS_*`)**: Defines server engines (Ollama, OpenAI, vLLM, etc.)
2. **Execution Layer (`*_PROFILES_*`)**: Defines models, vision flags, and routing profiles referencing a binding

Configuration is automatically resolved from:
1. CLI arguments (`--profile`, `--model`, `--binding`, `--host`, `--api-key`)
2. `~/.lollms_client/config.yaml` or `~/.lollms-client/.env` (Config Wizard)
3. Local `.env` files in the workspace
4. `~/.lollms_client/lollms_code/config.json`

### Profile Example (`~/.lollms_client/config.yaml`)

```yaml
llm:
  bindings:
    local_ollama:
      binding_name: ollama
      host_address: http://localhost:11434
    cloud_openai:
      binding_name: openai
      service_key: sk-...
  profiles:
    coder:
      binding_alias: local_ollama
      model_name: qwen2.5-coder:7b
      forced_context_size: 32768
      is_default: true
    gpt4o:
      binding_alias: cloud_openai
      model_name: gpt-4o
      vision_enabled: true
```

### CLI Profile Selection

```bash
# Use a specific profile declared in your configuration
lollms-code --profile gpt4o "Refactor the authentication module"

# Quick override of model on the active profile
lollms-code --model llama3.2:3b "Write a quick test"
```

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