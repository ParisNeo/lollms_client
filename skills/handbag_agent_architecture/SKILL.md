---
title: "Handbag Agent Architecture and Autonomous Persona Engineering"
description: "Definitive architectural doctrine for building sovereign agent Handbags, crafting SOUL.md personalities, injecting app-level tools and skills, and inspecting agent state."
category: "agent_engineering"
tags: [handbag, personality, soul, agents, skills, memory, coworkers]
visibility: visible
modifiable: true
---

# Handbag Agent Architecture & Persona Engineering

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

## 6. Multi-Persona Crews (Coworkers)

When a Handbag contains a `coworkers/` subdirectory, the primary persona acts as a coordinator:

```python
# List available sub-agent crewmates
print("Crewmates:", list(agent.coworkers.keys()))

# Access a specialized coworker
backend_coder = agent.coworkers.get("backend_coder")
if backend_coder:
    coder_response = backend_coder.chat("Implement the user model in models.py", client)
```