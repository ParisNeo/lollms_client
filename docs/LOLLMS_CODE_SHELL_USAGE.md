# lollms-code Shell Usage & Multi-Model Orchestration

This guide explains how to drive **lollms-code** non-interactively from a shell, persist a
configuration to reuse across instances, select a specific model profile, and orchestrate
multiple lollms-code instances (one per model) for a power-vs-speed optimization.

> Applies to `lollms-code` v2.0.0 (console script → `lollms_client.apps.lollms_code.cli:main`).

---

## 0. The Default Agent: Optimus Codex

`lollms-code` ships with a high-quality, generic coding agent called **Optimus Codex**. This agent is used by default if you do not specify a `--handbag` path.

- **Soul**: `src/lollms_client/apps/lollms_code/default_handbag/SOUL.md`
- **Persona**: Senior Software Architect & Engineer.
- **Capabilities**:
  - Handles any code type (Python, JS, TS, etc.).
  - Follows SOLID, DRY, and KISS principles.
  - Actively prevents security vulnerabilities.
  - **Skills**: It leverages the `skills/` directory (e.g., `fullstack_development`, `git_workflow_mastery`) to apply specialized methodologies when relevant tasks are detected.

This allows you to run complex engineering tasks immediately without setting up a custom handbag:

```bash
lollms-code "Refactor the login module to use dependency injection"
```

---

## 1. Global CLI Options

These flags can be used with any subcommand (`gui`, `run`, `instances`) or as global arguments.

| Flag | Alias | Meaning |
|------|-------|---------|
| `-i`, `--interactive` | | Start in interactive REPL mode. |
| `--workspace` | | Path to the workspace directory. Defaults to current working directory. |
| `-hb`, `--handbag`, `--handbag-path` | | Path to a custom Handbag folder containing agent resources (SOUL.md, tools, skills, memory). |
| `--model` | | Model name to use. |
| `--llm-binding` | | LLM binding name. |
| `--host` | | Host address for remote bindings. |
| `--api-key` | | API key for gated services. |
| `--context-size` | | Context window size for local models. |
| `--max-steps` | | Maximum reasoning steps (0 or -1 for infinite rounds with warning). |
| `--temperature` | | Sampling temperature. |
| `--max-tokens` | | Maximum tokens per generation turn. |
| `--context-compaction-threshold` | | Context fill threshold (default 0.85 for 85%) to run autonomous file locking and history compaction. |
| `--effort`, `--reasoning-effort` | | Reasoning effort level for thinking models (`none`, `low`, `medium`, `high`, `max`). |
| `--dynamic-effort` | | Enable dynamic reasoning effort adjustment across rounds via `<effort level='...'/>`. |
| `--skills-dir` | | Directory for SKILL.md files. |
| `--enable-model-switching` | | Allow the agent to switch models. |
| `--no-shell-execution` | | Disable autonomous shell command execution. |
| `--allow-computer-use`, `--enable-computer-use` | | Allow desktop automation and computer use tools if model supports vision. |
| `--shell-autonomy` | | Autonomy level for shell and Python execution (`strict`, `safe`, `full_access`). |
| `--auto-approve-python` | | Auto-approve Python execution in safe mode without confirmation prompts. |
| `--no-sub-agents` | | Disable sub-agent delegation. |
| `--no-memory` | | Disable persistent memory. |
| `--list-skills` | | List all learned skills and exit. |
| `--clear-history` | | Clear conversation history and exit. |
| `--config` | | Run configuration wizard and exit. |
| `--zoo` | | Launch interactive Zoo Package Hub Navigator (tools, skills, personas). |
| `--config-path` | | Path to a specific configuration file (`.env`, `.json` or `.yaml`) used by both the client and the wizard. |
| `--llm-profile` | | LLM model profile alias to activate for this session (selects a specific model from the available profiles). |
| `--export-config` | | Save the current configuration (Two-Tier profiles) to a YAML file and exit. |
| `--debug` | | Enable debug logging. |
| `--version` | | Show program's version number and exit. |

---

## 2. Subcommands

### 2.1 `gui`
Launch the lollms_code GUI (NiceGUI native window).

```bash
lollms-code gui
```

### 2.2 `run`
Run a single non-interactive task and return the final output (ideal for shell / multi-agent use).

```bash
lollms-code run "Implement a REST API client with retry logic"
```

Key flags for `run` (in addition to global flags):

| Flag | Meaning |
|------|---------|
| `task` (positional) | The instruction for the agent. |
| `--config <path>` | Use this `.env`/`.json`/`.yaml` **instead of** the global config. |
| `--llm-profile <alias>` | Activate a specific model profile (see §3). |
| `--output <file>` | Write the **final AI text** to this file (separate from artifacts). |
| `--print` | Also print the final AI text to stdout. |
| `--workspace <dir>` | Workspace directory (default: CWD). |
| `--max-steps`, `--temperature`, `--max-tokens`, `--effort`, `--shell-autonomy` | Tuning (mirror the interactive flags). |

Example with an explicit config file and a chosen model:

```bash
lollms-code run "summarize this repo and save the summary to summary.md" \
    --config ./my_config.yaml \
    --llm-profile fast_model \
    --workspace ./ws \
    --output summary.txt \
    --print
```

Exit codes: `0` on success, `130` if the run was cancelled (Ctrl-C), `1` on client error.

### 2.3 `instances`
Spawn and manage multiple lollms-code instances.

```bash
# spawn (non-blocking; prints the instance record incl. its id + pid)
lollms-code instances spawn "write a poem" --config ./shared_config.yaml --llm-profile fast --output out1.txt
lollms-code instances spawn "write a sonnet" --config ./shared_config.yaml --llm-profile big --output out2.txt --wait

# list tracked instances
lollms-code instances list

# wait for a specific instance (or all) to finish
lollms-code instances wait <instance_id>

# kill one, or all
lollms-code instances kill <instance_id>
lollms-code instances kill all
```

> Note: the console-script form is stateless across separate invocations (each process keeps its
> own in-memory registry). For true multi-instance orchestration within a single process, use the
> `InstanceManager` class (below) — that is what an *active agent* uses.

---

## 3. Selecting a model profile with `--llm-profile`

lollms-code's client is built on a **Two-Tier Profile Architecture**:
- **Connection layer** (`llm_binding_profiles`): how to reach an LLM server (host, api key).
- **Execution layer** (`llm_model_profiles`): specific models, each referencing a binding profile.

Every configured model has an **alias**. `--llm-profile <alias>` activates that profile for the
session (internally it sets the profile as the active default and calls `client.switch_model(alias)`).

List your available model profiles from any running client:

```python
from lollms_client import LollmsClient
client = LollmsClient(llm_binding_profiles={...}, llm_model_profiles={...})
for alias, p in client.llm_model_profiles_registry.items():
    print(alias, "->", p.model_name, p.binding_profile_name, "default" if p.is_default else "")
```

Or from a shell (one-shot), simply try the alias:

```bash
lollms-code run "say hello" --llm-profile my_alias
```

This is the key lever for **spawning N agents on N different models** — e.g. a fast/small model
for drafts and a large/powerful model for the final pass.

---

## 4. Persisting a configuration with `save_config`

`LollmsClient.save_config(path, format="yaml")` serializes the **live** two-tier registries
(every modality: `llm`, `tti`, `tts`, `stt`, `ttv`, `ttm`, `connection`, `rag`) into a portable
YAML file. That file can then be handed to another lollms-code instance via `--config`.

Programmatic:

```python
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_profiles={
        "ollama": {"binding_name": "ollama",
                   "binding_config": {"host_address": "http://localhost:11434"},
                   "is_default": True},
    },
    llm_model_profiles={
        "big":  {"binding_profile_name": "ollama", "model_name": "qwen3:32b", "is_default": True},
        "fast": {"binding_profile_name": "ollama", "model_name": "qwen3:4b"},
    },
)
client.save_config("./shared_config.yaml")
```

Then reuse it:

```bash
lollms-code run "write a poem" --config ./shared_config.yaml --llm-profile fast
lollms-code run "grade the poem" --config ./shared_config.yaml --llm-profile big
```

From the CLI you can also export the current client's config and exit:

```bash
lollms-code --export-config ./current.yaml
```

The YAML shape (canonical Two-Tier form):

```yaml
llm:
  bindings:
    ollama:
      binding_name: ollama
      binding_config:
        host_address: http://localhost:11434
      is_default: true
  profiles:
    big:
      binding_profile_name: ollama
      model_name: qwen3:32b
      is_default: true
      vision_enabled: false
    fast:
      binding_profile_name: ollama
      model_name: qwen3:4b
      is_default: false
      vision_enabled: false
```

---

## 5. Spawning & tracking multiple lollms-code instances

lollms-code can **spawn its own child instances** (subprocesses) and track them by PID.
This is powered by the `InstanceManager` module
(`lollms_client.apps.lollms_code.instance_manager`) and the `instances` subcommand.

### 5a. As a console script

```bash
# spawn (non-blocking; prints the instance record incl. its id + pid)
lollms-code instances spawn "write a poem" --config ./shared_config.yaml --llm-profile fast --output out1.txt
lollms-code instances spawn "write a sonnet" --config ./shared_config.yaml --llm-profile big --output out2.txt --wait

# list tracked instances
lollms-code instances list

# wait for a specific instance (or all) to finish
lollms-code instances wait <instance_id>

# kill one, or all
lollms-code instances kill <instance_id>
lollms-code instances kill all
```

> Note: the console-script form is stateless across separate invocations (each process keeps its
> own in-memory registry). For true multi-instance orchestration within a single process, use the
> `InstanceManager` class (below) — that is what an *active agent* uses.

### 5b. As a library (in-process orchestration)

```python
from lollms_client.apps.lollms_code.instance_manager import InstanceManager

mgr = InstanceManager()

a = mgr.spawn("write a haiku", config="cfg.yaml", llm_profile="fast", output="a.txt")
b = mgr.spawn("write a sonnet", config="cfg.yaml", llm_profile="big", output="b.txt")

print(a.id, a.pid, a.alive)

# wait for both
ra, rb = mgr.wait(a.id), mgr.wait(b.id)

# or kill them all if they misbehave
mgr.kill_all()

for s in mgr.status():
    print(s)
```

Each `LollmsCodeInstance` exposes: `id`, `task`, `pid`, `proc` (the `Popen`), `workspace`,
`output`, `config`, `llm_profile`, `started_at`, `returncode`, and the `.alive` property.

### 5c. From inside an active agent (LCP tools)

An agent can orchestrate sibling lollms-code instances through the `instances_tools` LCP tool
library (auto-discovered). It exposes `tool_instances_spawn`, `tool_instances_list`,
`tool_instances_kill`, and `tool_instances_wait` — each operating against the
unified runtime object `env` (see §9) rather than chat-system-specific instances.

---

## 6. Worked example: master model grades N sub-model poems

See `examples/multi_model_poem_orchestrator.py` for a runnable script that:
1. Builds a client from a two-tier config (or `save_config` output).
2. Enumerates **all** available LLM model profiles (N of them).
3. Spawns N `lollms-code run ... --llm-profile <alias>` instances, each writing a poem.
4. Waits for all to finish.
5. Runs a **master** model that reads every poem, grades each `0–10`, and writes `report.md`.

```bash
python examples/multi_model_poem_orchestrator.py --config shared_config.yaml
```

## 7. Interactive chat + debating agents

See `examples/chat_debate_agents_example.py` for a NiceGUI chat interface with a handbag agent
(soul + tools) that can spawn a few agents, let them name themselves, and debate a topic typed
into the chat.

---

## Quick reference

| Goal | Command / API |
|------|---------------|
| One-shot task (Default Agent) | `lollms-code "task"` |
| One-shot, scripted | `lollms-code run "task" --output out.txt` |
| Use a specific Handbag/Agent | `lollms-code run "task" --handbag /path/to/handbag` |
| Use a specific config | `--config path.yaml` |
| Choose a model | `--llm-profile <alias>` |
| Save config to YAML | `client.save_config("cfg.yaml")` / `lollms-code --export-config cfg.yaml` |
| Spawn N instances | `InstanceManager().spawn(...)` / `lollms-code instances spawn ...` |
| Track / wait / kill | `mgr.wait(id)`, `mgr.status()`, `mgr.kill(id)` / `mgr.kill_all()` |

---

## 8. Cross-References

- **`BRANCHING.md`** — conversations held inside lollms-code are message DAGs.
  Every prompt/response pair appends nodes to the active branch
  (`BRANCHING.md` §4); regenerating a reply from the CLI or GUI uses
  `regenerate_branch` (`BRANCHING.md` §7); history persistence across
  `--workspace` sessions follows the reload rules in `BRANCHING.md` §8.
- **`CHAT_BOOLEAN_FLAGS.md`** — most `--shell-autonomy` / `--auto-approve-python`
  / `--no-memory`-style CLI flags map to per-turn `chat()` booleans documented
  there (`enable_code_execution`, `auto_approve_python`, `enable_memory`, ...).
  See its Quick Reference Table for the exact flag names and defaults.
- **`docs/lcp_tool_resource_access.md`** — how tools (including `instances_tools`
  and shell execution) receive the live client via the injected `lc` parameter
  instead of constructing their own. **`env` (see §9)** is the unified runtime
  object that now supersedes those per-mode injections for tool code.
- **`docs/lcp/00_OVERVIEW.md`** — canonical definition of the `env` contract
  (§6) and the chat-system portability rule (§7).

---

## 9. Tool runtime context (`env`) & migration

LCP tools and code-shell snippets no longer receive two mutually exclusive
injections (`lollms_client` in personality/raw mode, `lollms_discussion` in
discussion mode). Both are replaced by **one always-present object, `env`**,
identical in both chat systems. The full contract — `env.llm`,
`env.llm_model_profiles`, `env.llm_switch(name)`, the `tti` /
`tti_model_profiles` / `tti_switch` triple, and the optional
`env.discussion` — is defined once in
[`docs/lcp/00_OVERVIEW.md`](lcp/00_OVERVIEW.md) (§6–§7).

**Migration table:**

| Legacy | Replacement | Notes |
|---|---|---|
| `lollms_client.generate(...)` | `env.llm.generate(...)` | identical object in both modes |
| `lollms_client.tti` | `env.tti` | may be `None`; guard |
| `lollms_discussion` | `env.discussion` | `None` under personality chat |

`lollms_client` and `lollms_discussion` are **still injected but deprecated**
for one release (a `DeprecationWarning` is emitted on first attribute
access) and are then **scheduled for removal**. New tools must use `env`
only.