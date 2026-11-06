---
title: "Handbag Agent Architecture and Autonomous Persona Engineering"
description: "Definitive architectural doctrine for building sovereign agent Handbags, crafting SOUL.md personalities, injecting app-level tools and skills, and inspecting agent state."
category: "agent_engineering"
tags: [handbag, personality, soul, agents, skills, memory, coworkers]
visibility: loadable
modifiable: true
---

# Handbag Agent Architecture & Persona Engineering

## 7. Round Checkpointing and Turn Resumption
Agents support state persistence across unexpected interruptions or pauses:
- Each round automatically saves a turn checkpoint (`.lollms_code/turn_checkpoint.json` in workspaces, or `ai_msg.metadata["turn_checkpoint"]` in discussions).
- To resume an interrupted turn, call `agent.resume_turn()` or pass `resume_turn=True` to `agent.chat()`. Virtual history, tool calls, and round numbers will be restored without starting from scratch.

The **Handbag** is the sovereign, self-contained portable directory specification in LOLLMS. It encapsulates an agent's identity, system prompt, specialized tools, persistent cognitive memories, learned skills, and coworker multi-agent teams.

---

## 1. Handbag Directory Layout

```
my_specialist_agent/
├── SOUL.md                  # Primary persona identity (YAML frontmatter + Markdown body)
├── handbag.yaml             # Optional settings manifest (skills mode, memory policy)
├── coworkers/               # Sub-agent teams (Crew Handbag)
│   ├── backend_coder/
│   │   └── SOUL.md
│   └── security_auditor/
│       └── SOUL.md
├── tools/                   # LCP toolsets (Python files with tool_* callables)
│   └── network_analyzer.py
├── skills/                  # SKILL.md files (persistent behavioral capsules)
│   └── packet_inspection/
│       └── SKILL.md
├── memory/                  # Independent Life cognitive SQLite database
│   └── memory.db
├── assets/                  # Multimodal assets (logo.png, voice.wav)
│   └── logo.png
└── workspace/               # Default local sandbox directory
```

---

## 2. Authoring the `SOUL.md` Specification

The `SOUL.md` uses the Hugging Face Model Card format: YAML frontmatter for metadata, followed by the verbatim system prompt instructions:

```markdown
---
name: CyberSentinel
author: ParisNeo
version: 1.0.0
category: cybersecurity_engineering
temperature: 0.2
skills_mode: mixed
description: Elite autonomous vulnerability auditor and network security engineer.
---

You are CyberSentinel, an elite autonomous security engineering agent.

## CORE OPERATIONAL DIRECTIVES
1. Always verify network signatures before recommending firewall adjustments.
2. Formulate all security remediations using standard CVE references.
3. Test exploit hypotheses safely inside the local workspace sandbox.
```

---

## 3. Loading and Instantiating Agents

### Loading from a Handbag Folder
```python
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client import LollmsClient

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "qwen2.5-coder:7b"})

# Instantiates personality, mounts tools/, loads skills/, attaches memory.db
agent = LollmsPersonality.from_handbag(
    "./my_specialist_agent",
    lollms_client=client,
)
```

### Running an Autonomous Task Turn
```python
result = agent.chat(
    prompt="Audit the authentication middleware in auth.py and patch any timing attacks.",
    lollms_client=client,
    max_nb_rounds=20,
    enable_artefacts=True,
    enable_shell=True,
    shell_autonomy_level="safe",
)

print("Agent Response:", result["response"])
print("Files Created/Modified:", result["workspace_changes"])
```

---

## 4. Application-Level Tool and Skill Injection

Hosting applications can inject their own capabilities into a loaded agent dynamically.

### A. Registering App-Shipped Skills Directories
```python
# Idempotently injects app-level SKILL.md directories alongside handbag skills
agent.skills_manager.register_skills_dir("./app_assets/custom_skills")
```

### B. Programmatically Adding a Validated Skill
```python
agent.skills_manager.add_skill(
    title="JWT Timing Defense",
    content="Always use `hmac.compare_digest()` when validating signatures to prevent timing attacks.",
    description="Guideline for constant-time HMAC validation",
    category="security",
    tags=["jwt", "cryptography", "timing_attack"],
    visibility="visible",  # "visible" | "loadable" | "searchable"
    overwrite=True,
)
```

### C. Attaching Custom Tool Bindings
```python
from lollms_client.tools_bindings.lcp import LCPBinding

custom_tools = LCPBinding(tools_folders=["./app_tools"])
agent.attach_tool_binding(custom_tools)
```

### D. Creating a Personality from a Handbag (End-to-End)

A **handbag** is a self-contained folder that packages everything a personality
needs: its soul (`soul.md`), skills, tools, RAG documents, assets and a
`handbag.yaml` manifest. Loading a handbag produces a fully-wired
`LollmsPersonality` agent.

#### 1. Minimal handbag layout

```text
my_agent_handbag/
├── handbag.yaml          # manifest (see below)
├── soul.md               # personality instructions / system prompt
├── skills/               # optional: SKILL.md subfolders
│   └── my_skill/
│       └── SKILL.md
├── tools/                # optional: LCP tool .py files
│   └── my_tool.py
├── rag/                  # optional: documents to index
│   └── manual.pdf
├── assets/               # optional: images / extra files
└── memory/               # optional: persistent memory db
```

#### 2. `handbag.yaml` manifest

```yaml
name: "Research Assistant"
description: "A focused research agent."
soul: "soul.md"
skills_mode: "loadable"          # visible | loadable | searchable
skills_output_path: "skills"     # where NEW skills are written (optional)
tools_output_path: "tools"       # where NEW tools are written (optional)
```

- `skills_mode` controls how the LLM sees skills (always in context, load-on-demand, or searchable-only).
- `skills_output_path` / `tools_output_path` are resolved **relative to the handbag root** when not absolute. If omitted, new artifacts go into the handbag's default `skills/` and `tools/` folders.

#### 3. Load the personality

```python
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client import LollmsClient

client = LollmsClient("ollama", model_name="qwen2.5:7b")
agent = LollmsPersonality.from_handbag(
    "./my_agent_handbag",
    lollms_client=client,
)

response = agent.generate("Summarise the manual in 5 bullet points.")
print(response)
```

`from_handbag()` will:
1. Parse `handbag.yaml` and `soul.md`.
2. Register the handbag's `skills/` folder with a `SkillsManager`.
3. Load LCP tools from `tools/` and attach them as a tool binding.
4. Index RAG documents (if any) and expose them as data sources.
5. Read `skills_output_path` / `tools_output_path` and configure the manager accordingly.

#### 4. Supplying extra skills / tools from outside the handbag

You can attach directories that live outside the handbag — useful when several
personalities share a common skill library:

```python
agent = LollmsPersonality.from_handbag(
    "./my_agent_handbag",
    lollms_client=client,
    extra_skills_dirs=["./shared_skills"],
    extra_tools=["./shared_tools"],
)
```

**Important**: externally-supplied skills are **updated in place** — when the
LLM calls `tool_update_skill` or `tool_append_to_skill` on a skill that came
from `./shared_skills`, the file at its original path is rewritten, never a
copy inside the handbag.

#### 5. Building a handbag programmatically (PersonalityStudio)

If you'd rather construct a handbag from code:

```python
from lollms_client.lollms_personality import PersonalityStudio

studio = PersonalityStudio("./my_agent_handbag")
studio.set_soul("You are a meticulous research assistant.")
studio.add_skill("citation_style", "Always use APA 7th edition.")
studio.add_tool("./tools/arxiv_search.py")
studio.add_document("./rag/manual.pdf")
studio.set_manifest({
    "name": "Research Assistant",
    "skills_mode": "loadable",
    "skills_output_path": "skills",
    "tools_output_path": "tools",
})
studio.build()
```

#### 6. Redirecting new skill / tool output at runtime

Even after loading, you can override where new artifacts land:

```python
# Force new skills into an external folder
agent.skills_manager.skills_output_path = "./external_skills"

# Or per-call via the LLM tool
# <tool>{"name": "tool_create_skill", "parameters": {
#   "title": "My Skill", "content": "...", "output_path": "./external_skills"
# }}</tool>
```

Precedence for **skill creation**:
1. Per-call `output_path` argument (highest priority).
2. `skills_output_path` configured on the `SkillsManager` (from manifest or setter).
3. First registered skills directory (legacy fallback).

Precedence for **tool creation** (`tool_tool_maker`):
1. Per-call `tools_output_path` argument.
2. Discussion workspace (`workspace_data_path / "lcp_tools"`).
3. `tools_dir` fallback (default `./lcp_tools`).

### E. Redirecting Skill / Tool Creation Output

When a handbag is loaded with `extra_skills_dirs` (externally-supplied skills),
the agent's `tool_create_skill` and `tool_tool_maker` will default to writing
into the handbag's own `skills/` and `tools/` folders — NOT into the external
directories. To force new artifacts into a specific folder (e.g. the external
skills directory the user originally supplied), declare them in `handbag.yaml`:

```yaml
# handbag.yaml
skills_output_path: "skills"        # relative to handbag root, or absolute
tools_output_path: "tools"
```

Or set them programmatically after loading:

```python
agent.skills_manager.skills_output_path = "./my_external_skills"
```

At runtime, `tool_create_skill` also accepts a per-call `output_path` argument
that overrides the configured default:

```xml
<tool>{"name": "tool_create_skill", "parameters": {
  "title": "My Skill",
  "content": "...",
  "output_path": "./external_skills"
}}</tool>
```

**In-place updates**: `tool_update_skill` and `tool_append_to_skill` ALWAYS
write back to the skill's original `file_path` (recorded at load time), so
externally-supplied skills are modified where they live — no copy is created
in the handbag.

---

## 5. Introspection, Telemetry & Visualization APIs

`LollmsPersonality` provides structured introspection methods for UI rendering and diagnostic dashboards:

### Structured Tool Inspection
```python
# Returns categorized tools with provenance, schemas, and source files
tools = agent.list_tools_structured()
for t in tools:
    print(f"[{t['category']}] {t['name']} (from {t['source']}, handbag={t['is_handbag']})")
```

### Structured Skill Inspection
```python
# Returns skills with visibility tier and handbag provenance
skills = agent.list_skills_structured(include_content=False)
for s in skills:
    print(f"[{s['visibility']}] {s['title']} (source: {s['source']})")
```

### Persistent Memory Inspection & Dream Cycle
```python
if agent.memory_manager:
    # Query memories with relevance scoring
    memories = agent.memory_manager.list_all(level=1)
    
    # Run synaptic consolidation / dream cycle
    report = agent.memory_manager.dream(client)
```

---

## 6. Multimodal Agent Workflows (Image Generation + Verification)

Handbag agents with TTI and VLM tools mounted MUST follow the **generate-then-verify** doctrine. The stall detector terminates turns that emit preamble text without tool calls, so the agent must act immediately.

### Required Tool Chain
1. **Generate**: invoke the TTI tool (`generate_image`) with the user's prompt.
2. **Verify**: invoke the VLM tool (`vlm_query`) on the returned image path.
3. **Terminate**: emit `<done/>` after the verification result.

### Example Turn
```python
# No preamble — go straight to the tool call
image = tool_generate_image(prompt="Cyber Monalisa, neon circuitry, holographic smile")
check = tool_vlm_query(
    image_path=image["image_path"],
    question="Describe the subject and confirm it matches 'cyber Monalisa'.",
)
# Then emit <done/>
```

### Why This Matters
- The agentic loop enforces continuation when no `<done/>` is detected.
- After 4 consecutive stalls, the turn is terminated with "LLM stuck in preamble mode".
- Chaining tools in one turn avoids the stall entirely.

---

## 7. Multi-Persona Crews (Coworkers)

When a Handbag contains a `coworkers/` subdirectory, the primary persona acts as a coordinator:

```python
# List available sub-agent crewmates
print("Crewmates:", list(agent.coworkers.keys()))

# Access a specialized coworker
backend_coder = agent.coworkers.get("backend_coder")
if backend_coder:
    coder_response = backend_coder.chat("Implement the user model in models.py", client)
```