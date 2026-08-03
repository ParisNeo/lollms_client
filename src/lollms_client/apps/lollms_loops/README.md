# 🔄 Lollms Loops: Universal Task-Adaptive Agentic Loop

Lollms Loops is a CLI application that transforms a declarative JSON task profile into a fully autonomous, bounded agentic loop. It implements the canonical four-stage agentic cycle (Perceive → Reason → Act → Observe → Check) with rigorous exit conditions, ensuring the agent adapts to any task without relying on "hope-driven" execution.

## Core Philosophy

In the agentic AI sense, a loop is "LLMs autonomously using tools in a loop." However, a task without a check is just hope. Lollms Loops enforces strict boundaries:
1. **Declarative Task Profiles**: The agent's goal, tools, and success criteria are defined in JSON, not hardcoded.
2. **Bounded Execution**: Hard limits on `max_reasoning_steps` and `timeout_seconds` prevent runaway loops.
3. **Explicit Verification**: The system prompt mandates the agent verify its progress against the success criteria after every action.
4. **FailureMemory Loop Interception**: Inherits the `Agent` class's signature-based loop interceptor. If the agent makes the exact same failing tool call twice, the loop forcefully terminates.

## Installation & Setup

The application is part of `lollms_client`. Ensure you have the base library installed:

```bash
pip install lollms_client
```

## Environment Configuration (`.env`)

Lollms Loops relies on a centralized, 4-tier configuration resolution protocol to locate LLM connection settings. It searches for configurations in the following strict order:

1. **CLI Argument**: `--env path/to/.env`
2. **Current Directory**: A `.env` file in your active terminal folder.
3. **Global Home Directory**: `~/.lollms-client/.env`
4. **OS Environment Variables**: Already exported in your shell.
5. **Interactive Wizard**: If none of the above are found, the CLI automatically launches a wizard to configure your binding, probe for models, and save to `~/.lollms-client/.env`.

### Generating your `.env` file

If you want to configure your environment manually or ahead of time, run the shared wizard:

```bash
python -m lollms_client.lollms_config_cli_env
```

This will guide you through selecting your binding (e.g., Ollama, OpenAI, LlamaCpp), entering your host address and API keys, and automatically fetching available models to select from. It validates the connection before saving.

## Defining a Task Profile

Create a JSON file describing the task. This is the "brain" of the loop.

```json
{
  "goal": "Write a Python script that fetches the current weather for Paris and saves it to weather.txt.",
  "success_criteria": "The file weather.txt exists in the workspace and contains a valid temperature reading.",
  "allowed_tools": ["tool_internet_search", "tool_write_file"],
  "max_reasoning_steps": 15,
  "timeout_seconds": 120,
  "temperature": 0.2,
  "enable_code_execution": false,
  "enable_file_ops": true
}
```

## Running the Loop

Execute the CLI, pointing it to your task profile. It will automatically use your resolved `.env` configuration.

```bash
python -m lollms_client.apps.lollms_loops.cli task.json
```

If you want to force the CLI to use a specific `.env` file for a particular run, use the `--env` flag:

```bash
python -m lollms_client.apps.lollms_loops.cli task.json --env /path/to/custom/.env
```

## How It Works

Lollms Loops wraps the high-grade `lollms_client.Agent` system. When you run a task:

1. **Profile Parsing**: The JSON is loaded into a `TaskProfile` object.
2. **Dynamic System Prompt**: The `TaskProfile.to_system_prompt()` method injects the goal and success criteria into the LLM's system prompt, explicitly instructing it to emit `<done/>` when criteria are met.
3. **Capability Gating**: `TaskProfile.to_capabilities()` translates boolean flags into `CapabilityFlags`, ensuring dangerous capabilities (like `enable_code_execution`) are strictly opt-in.
4. **Bounded Execution**: The `LollmsLoop.run()` method invokes `agent.chat()`. The `Agent` class handles the internal Reason → Act → Observe cycle.
5. **Termination Check**: The loop breaks if the LLM emits `<done/>`, if `max_reasoning_steps` is hit, if the `timeout_seconds` elapse, or if `FailureMemory` detects a duplicate failing tool call.