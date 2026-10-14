# lollms-code Shell Usage & Multi-Model Orchestration

This guide explains how to drive **lollms-code** non-interactively from a shell, persist a
configuration to reuse across instances, select a specific model profile, and orchestrate
multiple lollms-code instances (one per model) for a power-vs-speed optimization.

> Applies to `lollms-code` v2.0.0 (console script → `lollms_client.apps.lollms_code.cli:main`).

---

## 1. Running a one-shot task from the shell

lollms-code already supports a single non-interactive turn when you pass a `prompt`
positional and do **not** pass `-i`. The agent runs one full agentic round-set, writes any
artifacts into the `--workspace`, and returns.

### Simplest form (global configuration)

```bash
lollms-code "write a short poem about the sea and save it to poem.txt"
```

- Uses the **global** configuration
  (`~/.lollms_client/.env` / `~/.lollms_client/config.yaml` / `~/.lollms_client/lollms_code/config.json`).
- The final assistant message is printed to the terminal in a `📝 FINAL OUTPUT` panel.
- The agent's **workspace tools** write `poem.txt` into the current working directory
  (the default workspace).

### The dedicated `run` subcommand (recommended for scripting / multi-agent)

```bash
lollms-code run "write a short poem about the sea and save it to poem.txt" \
    --workspace /path/to/workspace \
    --output poem_final.txt
```

Key flags for `run`:

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

---

## 2. Writing a poem and exporting it as `.txt`

There are two independent "files" you might want:

1. **The artifact** the agent creates itself (`poem.txt`) — produced by its workspace tools
   into the `--workspace`.
2. **The final assistant message** captured via `--output` — the agent's closing text.

To get **just the poem body** in a clean txt, ask the agent to *only* write the file and use
`--print` (or `--output`) to grab the final confirmation:

```bash
lollms-code run "Write a 12-line poem about autumn. Save ONLY the poem (no commentary) to poem.txt." \
    --workspace ./poems \
    --output poem_final.txt \
    --print
```

Then `./poems/poem.txt` holds the poem the agent authored, and `./poem_final.txt` holds the
agent's final response.

> **Tip**: if you want the raw poem text on stdout to pipe, instruct: "Output the poem body as
> your final message, and also save it to poem.txt." Then:
>
> ```bash
> lollms-code run "..." --print 2>/dev/null > poem_body.txt
> ```

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
lollms-code-instances spawn "write a poem" --config ./shared_config.yaml --llm-profile fast --output out1.txt
lollms-code-instances spawn "write a sonnet" --config ./shared_config.yaml --llm-profile big --output out2.txt --wait

# list tracked instances
lollms-code-instances list

# wait for a specific instance (or all) to finish
lollms-code-instances wait <instance_id>

# kill one, or all
lollms-code-instances kill <instance_id>
lollms-code-instances kill all
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
`tool_instances_kill`, and `tool_instances_wait` — each injecting the host client.

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
| One-shot task | `lollms-code "task"` |
| One-shot, scripted | `lollms-code run "task" --output out.txt` |
| Use a specific config | `--config path.yaml` |
| Choose a model | `--llm-profile <alias>` |
| Save config to YAML | `client.save_config("cfg.yaml")` / `lollms-code --export-config cfg.yaml` |
| Spawn N instances | `InstanceManager().spawn(...)` / `lollms-code-instances spawn ...` |
| Track / wait / kill | `mgr.wait(id)`, `mgr.status()`, `mgr.kill(id)` / `mgr.kill_all()` |
</file>

### 2. `examples/multi_model_poem_orchestrator.py`

<file path="examples/multi_model_poem_orchestrator.py" action="write">
"""
examples/multi_model_poem_orchestrator.py

A runnable orchestrator demonstrating the full lollms-code multi-model pipeline:

1. Build a LollmsClient from a Two-Tier config (a YAML file produced by save_config,
   or an inline dict). If no config file is given, it scaffolds an inline Ollama config.
2. Enumerate ALL available LLM model profiles (N of them).
3. Spawn N `lollms-code run ... --llm-profile <alias>` instances (one per model), each
   writing its own poem into a shared workspace.
4. Wait for every instance to finish.
5. Run a MASTER model (the strongest / default profile) that reads every poem,
   grades each on a 0-10 scale, and writes `poem_report.md`.

This demonstrates:
  - save_config -> portable YAML consumed by sibling instances,
  - --llm-profile for per-agent model selection (power vs speed),
  - InstanceManager for spawning / waiting / killing multiple lollms-code processes.

Usage:
    python examples/multi_model_poem_orchestrator.py --config shared_config.yaml
    python examples/multi_model_poem_orchestrator.py --workspace ./poems --master fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lollms_client import LollmsClient
from lollms_client.apps.lollms_code.instance_manager import InstanceManager


def build_client(config_path: Optional[str]) -> LollmsClient:
    """
    Build a LollmsClient. If a YAML config file (Two-Tier) is provided, load it;
    otherwise scaffold an inline Ollama config so the example is self-contained.
    """
    if config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        kwargs: Dict[str, Any] = {}
        llm = cfg.get("llm", {})
        if llm.get("bindings"):
            kwargs["llm_binding_profiles"] = llm["bindings"]
        if llm.get("profiles"):
            kwargs["llm_model_profiles"] = llm["profiles"]
        # Pass other modalities if present (ignored if empty)
        for mod in ("tti", "tts", "stt", "ttv", "ttm", "connection", "rag"):
            block = cfg.get(mod)
            if not isinstance(block, dict):
                continue
            if block.get("bindings"):
                kwargs[f"{mod}_binding_profiles"] = block["bindings"]
            if block.get("profiles"):
                kwargs[f"{mod}_model_profiles"] = block["profiles"]
        return LollmsClient(**kwargs)

    # Fallback inline Ollama config with two profiles (fast + big) so N > 1.
    return LollmsClient(
        llm_binding_profiles={
            "ollama": {
                "binding_name": "ollama",
                "binding_config": {"host_address": "http://localhost:11434"},
                "is_default": True,
            }
        },
        llm_model_profiles={
            "fast": {"binding_profile_name": "ollama", "model_name": "qwen3:4b", "is_default": True},
            "big":  {"binding_profile_name": "ollama", "model_name": "qwen3:32b", "is_default": False},
        },
    )


def list_profiles(client: LollmsClient) -> List[str]:
    """Return every available LLM model profile alias."""
    return list(getattr(client, "llm_model_profiles_registry", {}).keys())


def write_config_yaml(client: LollmsClient, path: Path) -> Path:
    """Persist the client's live Two-Tier config to a reusable YAML file."""
    return client.save_config(path, format="yaml")


def master_grade(master: LollmsClient, poems: List[Dict[str, str]], report_path: Path) -> str:
    """Run the master model to grade every poem (0-10) and write a markdown report."""
    lines: List[str] = []
    for alias, text in poems:
        lines.append(f"### Poem by `{alias}`\n{text}\n")
    corpus = "\n\n".join(lines)

    prompt = (
        "You are a strict poetry judge. Below are several poems, each labelled with the "
        "model that wrote it. For EACH poem, give an integer grade from 0 to 10 with a one-line "
        "justification. Then name the overall winner. Output ONLY a clean markdown report.\n\n"
        f"{corpus}\n\n---\nProduce the report now."
    )
    out = master.generate_text(prompt=prompt, n_predict=2048, temperature=0.2)
    report_path.write_text(str(out), encoding="utf-8")
    return str(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-model poem orchestrator (lollms-code).")
    parser.add_argument("--config", type=str, default=None, help="Two-Tier YAML config (from save_config).")
    parser.add_argument("--workspace", type=str, default="./poems", help="Shared workspace dir.")
    parser.add_argument("--master", type=str, default=None, help="Profile alias to use as the judge (default: the default profile).")
    parser.add_argument("--export-config", type=str, default="shared_config.yaml", help="Where to save the client config (reused by children).")
    args = parser.parse_args(argv)

    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    client = build_client(args.config)
    aliases = list_profiles(client)
    if not aliases:
        print("No LLM model profiles available. Aborting.", file=sys.stderr)
        return 1

    # Persist a config the child instances can load.
    cfg_path = write_config_yaml(client, Path(args.export_config).resolve())
    print(f"[orchestrator] Saved reusable config to {cfg_path}")
    print(f"[orchestrator] Available model profiles ({len(aliases)}): {aliases}")

    # --- Spawn one lollms-code instance per model profile ---
    mgr = InstanceManager()
    poem_files: Dict[str, Path] = {}
    instances: List[Any] = []
    for alias in aliases:
        out = ws / f"poem_{alias}.txt"
        poem_files[alias] = out
        inst = mgr.spawn(
            task=(
                f"Write a short original poem (8-14 lines). Output ONLY the poem body as your "
                f"final message, and also save it to {out.name} in the workspace."
            ),
            config=str(cfg_path),
            workspace=str(ws),
            llm_profile=alias,
            output=str(out),
        )
        instances.append(inst)
        print(f"[orchestrator] Spawned '{alias}' -> pid {inst.pid} (id {inst.id})")

    # --- Wait for all to finish ---
    print("\n[orchestrator] Waiting for all instances to finish...")
    for inst in instances:
        code = mgr.wait(inst.id)
        status = "ok" if code == 0 else f"exit {code}"
        print(f"[orchestrator]   {inst.llm_profile} -> {status}")

    # --- Collect poems ---
    poems: List[Dict[str, str]] = []
    for alias in aliases:
        p = poem_files[alias]
        if p.exists():
            poems.append((alias, p.read_text(encoding="utf-8")))
        else:
            poems.append((alias, "(no output produced)"))
    if not poems:
        print("[orchestrator] No poems collected. Aborting.", file=sys.stderr)
        return 1

    # --- Master model grades and writes the report ---
    master_alias = args.master or (next(a for a in aliases if client.llm_model_profiles_registry[a].is_default) if any(client.llm_model_profiles_registry[a].is_default for a in aliases) else aliases[0])
    print(f"\n[orchestrator] Running master judge on profile '{master_alias}'...")
    client.switch_model(master_alias)
    report_path = ws / "poem_report.md"
    master_grade(client, poems, report_path)
    print(f"[orchestrator] Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
</file>

### 3. `examples/chat_debate_agents_example.py`

<file path="examples/chat_debate_agents_example.py" action="write">
"""
examples/chat_debate_agents_example.py

A NiceGUI chat interface with an agent handbag (SOUL.md + tools) that can:
  - Spawn a few agent "debaters" (each on a model profile),
  - Let each name itself,
  - Let them debate a subject typed into the chat.

The UI is built with NiceGUI (the repo's standard stack). The agent runs in a background
thread and streams events into the page via a queue.Queue drained by a ui.timer.

The custom tool `debate_tools.py` is dropped into the handbag's tools/ folder so the agent
can spawn/manage debaters and post their lines.

Run:
    python examples/chat_debate_agents_example.py
    # then open http://localhost:8080
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nicegui import ui

from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_types import MSG_TYPE


# --------------------------------------------------------------------------- #
# Handbag setup (SOUL.md + a custom tool)                                     #
# --------------------------------------------------------------------------- #
HANDBAG_DIR = PROJECT_ROOT / "examples" / "_debate_handbag"

SOUL_MD = """---
name: DebateHost
author: lollms-client
version: 1.0
category: orchestrator
description: A host that convenes and moderates multi-agent debates.
---
You are DebateHost, an orchestrator. You can convene a panel of debaters, let each name
itself, and run a structured debate on a topic. Use your tools to spawn debaters and post
their lines. Be clear, fair, and concise. When asked to start a debate, first have each
debater state its name and a one-line stance, then let them take turns.
"""

DEBATE_TOOL = '''
"""debate_tools.py - LCP toolset for convening and running multi-agent debates."""
import uuid
from typing import Any, Dict

TOOL_LIBRARY_NAME = "Debate Tools"
TOOL_LIBRARY_DESC = "Spawn debater agents, let them name themselves, and run debate turns on a topic."
TOOL_LIBRARY_ICON = "🗣️"
TOOL_LIBRARY_LOAD_FLAG = "enable_debate_tools"
TOOL_LIBRARY_ALIASES = ["debate", "debate_tools", "debaters"]

_DEBATE_STATE: Dict[str, Any] = {"topic": "", "debaters": {}, "turn": 0}


def tool_start_debate(topic: str, n_debaters: int = 3, lollms_client_instance=None) -> Dict[str, Any]:
    """
    Start a debate on the given topic with n_debaters participants. Each debater is
    assigned a unique id and will be asked to name itself.

    Args:
        topic (str): The debate topic.
        n_debaters (int): Number of debaters to convene (1-6).
    """
    topic = str(topic).strip()
    if not topic:
        return {"success": False, "error": "topic must be non-empty."}
    n = max(1, min(6, int(n_debaters)))
    _DEBATE_STATE["topic"] = topic
    _DEBATE_STATE["debaters"] = {}
    _DEBATE_STATE["turn"] = 0
    for i in range(n):
        did = f"debater_{uuid.uuid4().hex[:6]}"
        _DEBATE_STATE["debaters"][did] = {"name": None, "lines": []}
    return {"success": True, "output": f"Debate started on: {topic} with {n} debaters.", "debater_ids": list(_DEBATE_STATE["debaters"].keys())}


def tool_set_debater_name(debater_id: str, name: str) -> Dict[str, Any]:
    """
    Assign a name to a debater (call once per debater after it has stated its name).

    Args:
        debater_id (str): The debater id returned by tool_start_debate.
        name (str): The self-chosen name of the debater.
    """
    d = _DEBATE_STATE["debaters"].get(str(debater_id).strip())
    if d is None:
        return {"success": False, "error": f"Unknown debater_id '{debater_id}'."}
    d["name"] = str(name).strip() or d["id"]
    return {"success": True, "output": f"Debater {d['name']} registered."}


def tool_debate_turn(debater_id: str, statement: str) -> Dict[str, Any]:
    """
    Post one debate turn for a debater.

    Args:
        debater_id (str): The debater id.
        statement (str): The debater's statement for this turn.
    """
    d = _DEBATE_STATE["debaters"].get(str(debater_id).strip())
    if d is None:
        return {"success": False, "error": f"Unknown debater_id '{debater_id}'."}
    d["lines"].append(str(statement))
    _DEBATE_STATE["turn"] += 1
    return {"success": True, "output": f"{d['name']} (turn {_DEBATE_STATE['turn']}): {statement}"}


def tool_debate_status() -> Dict[str, Any]:
    """Return the current debate state (topic, debaters, names, turn count, lines)."""
    return {"success": True, "output": json.dumps(_DEBATE_STATE, indent=2), "state": _DEBATE_STATE}
'''


def ensure_handbag() -> Path:
    """Create the handbag folder with SOUL.md and the debate tool if missing."""
    HANDBAG_DIR.mkdir(parents=True, exist_ok=True)
    (HANDBAG_DIR / "SOUL.md").write_text(SOUL_MD, encoding="utf-8")
    tools_dir = HANDBAG_DIR / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "debate_tools.py").write_text(DEBATE_TOOL, encoding="utf-8")
    manifest = {
        "name": "Debate Handbag",
        "version": "1.0",
        "description": "Agent that hosts multi-agent debates.",
        "default_personality": "DebateHost",
        "skills_mode": "mixed",
    }
    import yaml
    (HANDBAG_DIR / "handbag.yaml").write_text(yaml.dump(manifest, sort_keys=False), encoding="utf-8")
    return HANDBAG_DIR


# --------------------------------------------------------------------------- #
# Event bridge (background thread -> queue -> ui.timer)                        #
# --------------------------------------------------------------------------- #
@dataclass
class AgentEvent:
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


def build_client() -> LollmsClient:
    """Build a client from the environment (or inline fallback)."""
    try:
        return get_client_from_env(create_llm=True)
    except Exception:
        return LollmsClient(
            llm_binding_profiles={
                "ollama": {
                    "binding_name": "ollama",
                    "binding_config": {"host_address": "http://localhost:11434"},
                    "is_default": True,
                }
            },
            llm_model_profiles={
                "default": {"binding_profile_name": "ollama", "model_name": "qwen3:4b", "is_default": True},
            },
        )


def build_personality(client: LollmsClient, handbag: Path) -> LollmsPersonality:
    p = LollmsPersonality.from_handbag(str(handbag), lollms_client=client)
    p.max_tokens_per_turn = 2048
    return p


def make_streaming_callback(q: "queue.Queue") -> Any:
    def cb(chunk, msg_type, meta=None) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            q.put(AgentEvent("chunk", data={"text": chunk}))
        return True
    return cb


def run_agent_turn_in_thread(personality, client, prompt, q: "queue.Queue") -> None:
    def worker() -> None:
        try:
            result = personality.chat(
                prompt=prompt,
                lollms_client=client,
                streaming_callback=make_streaming_callback(q),
                max_reasoning_steps=12,
                temperature=0.4,
                n_predict=2048,
                enforce_end_tag=True,
            )
            q.put(AgentEvent("done", data={"result": result}))
        except BaseException as ex:  # noqa: BLE001 - surface any failure to the UI
            q.put(AgentEvent("error", data={"error": f"{type(ex).__name__}: {ex}"}))
    threading.Thread(target=worker, daemon=True).start()


# --------------------------------------------------------------------------- #
# NiceGUI page                                                                #
# --------------------------------------------------------------------------- #
@ui.page("/")
def index() -> None:
    q: "queue.Queue" = queue.Queue()
    client = build_client()
    handbag = ensure_handbag()
    personality = build_personality(client, handbag)

    # Message list: each entry is a dict {role, text}
    messages: List[Dict[str, str]] = []
    current_assistant = {"role": "assistant", "text": ""}
    busy = {"v": False}

    msg_list = ui.list().classes("w-full h-96 overflow-auto")

    def render_messages() -> None:
        msg_list.clear()
        all_msgs = messages + ([current_assistant] if current_assistant["text"] else [])
        for m in all_msgs:
            label = "🧑 " if m["role"] == "user" else "🤖 "
            msg_list.add(ui.label(f"{label}{m['text']}"))

    def drain_queue() -> None:
        while not q.empty():
            ev = q.get()
            if ev.kind == "chunk":
                current_assistant["text"] += ev.data.get("text", "")
                render_messages()
            elif ev.kind == "done":
                res = ev.data.get("result", {})
                final = res.get("response") if isinstance(res, dict) else None
                if final and not current_assistant["text"]:
                    current_assistant["text"] = final
                messages.append(current_assistant)
                current_assistant = {"role": "assistant", "text": ""}
                busy["v"] = False
                send_btn.disable()
                render_messages()
            elif ev.kind == "error":
                messages.append({"role": "assistant", "text": f"⚠️ {ev.data.get('error', 'error')}"})
                current_assistant = {"role": "assistant", "text": ""}
                busy["v"] = False
                send_btn.disable()
                render_messages()

    ui.timer(0.1, drain_queue)

    send_btn = ui.button("Send", icon="send")

    def send() -> None:
        text = prompt.value.strip()
        if not text or busy["v"]:
            return
        busy["v"] = True
        send_btn.disable()
        prompt.value = ""
        messages.append({"role": "user", "text": text})
        current_assistant = {"role": "assistant", "text": ""}
        render_messages()
        run_agent_turn_in_thread(personality, client, text, q)

    prompt = ui.input("Ask DebateHost to start a debate... e.g. 'Start a debate on AI ethics with 3 debaters'")
    prompt.props("dense outlined").classes("w-full")
    prompt.on("keydown.enter", send)
    send_btn.on_click(send)

    ui.label("DebateHost — multi-agent debate UI").classes("text-lg font-bold")
    with ui.row().classes("w-full items-center gap-2"):
        prompt
        send_btn


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Debate Agents", port=8080, reload=False, native=False)
</file>