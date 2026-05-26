---
name: Lollms Swarms and Multi-Agent Orchestration
description: Teaches how to coordinate multiple specialized agents, distribute tasks, establish consensus, and manage role-based multi-agent swarms using LollmsSwarm.
author: ParisNeo
version: 1.0.0
category: lollms_client/swarm
created: 2026-05-24
---

# Lollms Swarms and Multi-Agent Orchestration

This skill explains how to build, configure, and orchestrate role-based multi-agent swarms using the `lollms_client` library.

## 1. Core Concepts: Agents & Roles
An `Agent` is a stateful entity assigned a specific `AgentRole` (e.g. `DOMAIN_EXPERT`, `AUDITOR`, `ORCHESTRATOR`) and a custom personality/system prompt.
- **Specialized Persona**: Each agent is configured with specialized system prompts so they operate with high domain accuracy.
- **Stateful Memory**: Agents track their own thought blocks and conversation histories within their assigned discussion branches.

## 2. Setting Up an Agent
Instantiate an agent using a loaded `LollmsClient` and custom metadata.

```python
from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, AgentRole
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "gemma4:e2b"})

# Define expert personality
python_expert = LollmsPersonality(
    name="PythonExpert",
    system_prompt="You are a Senior Python Architect. Focus on clean code, SOLID principles, and type-safety."
)

# Instantiate the Agent
agent = Agent(
    lc=client,
    personality=python_expert,
    name="PythonExpert",
    role=AgentRole.DOMAIN_EXPERT,
    model_params={"temperature": 0.2}
)
```

## 3. Orchestrating Swarms
The `SwarmOrchestrator` manages collaborative task distribution across several specialized agents. It handles the round-robin discussions, collects intermediate answers, and facilitates the consensus pass.

```python
from lollms_client.lollms_swarm import SwarmOrchestrator, SwarmConfig

# Configure the swarm
swarm_config = SwarmConfig(
    max_rounds=3,
    consensus_threshold=0.85,
    enable_peer_review=True
)

# Initialize the orchestrator inside a discussion
orchestrator = SwarmOrchestrator(
    discussion=discussion,
    agents=[agent_developer, agent_auditor],
    config=swarm_config
)

# Dispatch a query to the swarm
result = orchestrator.run("Design a thread-safe connection pool for SQLite.")
print(f"Final Answer: {result['response']}")
```

## 4. Multi-Agent Event Flow
During execution, the swarm communicates progress through specific message types:
- `MSG_TYPE_SWARM_AGENT_START`: An agent begins generating its turn.
- `MSG_TYPE_SWARM_AGENT_END`: An agent finishes its generation.
- `MSG_TYPE_SWARM_CONSENSUS`: The orchestrator evaluates the similarity of responses and reports on consensus strength.