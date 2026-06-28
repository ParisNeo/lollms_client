# 🧠 Lollms Memory: Multi-Level Cognitive Memory & Semantic Graph System

The `lollms_memory` module provides a stateful, human-brain-inspired cognitive memory system for LLM agents. By combining **Petroff's Power-Law Decay**, **Spreading Activation**, and a **Semantic Graph Ontology**, this system allows agents to persistently organize, recall, decay, and synchronize knowledge across long conversation turns.

---

## 🏛️ 1. Multi-Level Memory Architecture

The memory system is divided into five progressive tiers, mimicking short-term working attention and long-term consolidation:

| Tier Level | Memory Type | Storage Location | Lifetime / Scope | Context Ingestion |
| :--- | :--- | :--- | :--- | :--- |
| **Level 0** | **Volatile Scratchpad** | In-Process Memory | Single Turn (Cleared) | Appended before the last user prompt |
| **Level 1** | **Working Memory** | SQLite (`main.db` / `shared.db`) | Active Session | Rendered verbatim in the prompt context |
| **Level 2** | **Deep Memory** | SQLite (`main.db` / `shared.db`) | Inactive / Latent | Injected as lightweight ID handles only |
| **Level 3** | **Archived Memory** | SQLite (`main.db` / `shared.db`) | Highly decayed | Completely excluded; evaluated during Dream Cycle |
| **Level 4** | **Episodic Memory** | SQLite (`main.db` / `shared.db`) | Permanent History | Interaction logs used for retrospective queries |

---

## 📈 2. Petroff's Power-Law Decay & Spreading Activation

Rather than simple linear timeouts, the memory manager implements cognitive mathematical decay:

### Petroff's Power-Law Decay
The activation energy ($B_i$) of a memory node is calculated from its complete retrieval log history:

$$B_i = \ln \left( \sum_{j} (t - t_j)^{-d} \right)$$

*   Where $t_j$ is the timestamp of the $j$-th retrieval/tag event.
*   Where $d$ is the decay rate parameter (`decay_rate_per_day` in `MemoryConfig`, default `0.5`).
*   **Synaptic Demotion**: If a node's activation energy drops below the `demotion_threshold`, it is moved from **Level 1 (Working)** to **Level 2 (Deep)**. If it drops below `archive_threshold`, it is moved to **Level 3 (Archived)**.

### Spreading Activation
When an active memory node is retrieved or updated, energy is spread **multiplicatively** to its semantically linked neighbors:

$$A_{\text{neighbor}} = A_{\text{source}} \times P_{\text{spread}}$$

*   Where $P_{\text{spread}}$ is the attenuation multiplier (`spread_probability` in `MemoryConfig`, default `0.9`).
*   **Pre-Warming**: This raises the activation of linked concepts in Deep Memory, bringing their handles to the attention of the LLM without bloating the immediate context.

---

## 🔗 3. Ontological Schema & Semantic Graph (TBox / ABox)

The memory system models data using standard semantic web and knowledge graph paradigms:

### A. The TBox (Terminological Schema)
Defines the valid classes of concepts and their allowed relationship verbs:

*   **Node Classes**:
    *   `CONCEPT`: Abstract ideas, subjects, tools, or entities.
    *   `PREFERENCE`: User guidelines, constraints, and custom personality rules.
    *   `EVENT`: Milestone occurrences, episodes, or tool outputs.
    *   `DECISION`: Architectural choices, code designs, or lessons learned.
*   **Relationship Verbs**:
    *   `RELATED_TO` (Default/Associative)
    *   `PREFERS` (Preference mapping)
    *   `IMPLEMENTS` (Code realization)
    *   `CONTRADICTS` (Logic conflicts)
    *   `SUPPORTS` (Logical validations)
    *   `TEMPORAL_AFTER` (Chronological ordering)
    *   `PART_OF` (Decomposition/Composition)

### B. The ABox (Assertional Instances)
The actual facts saved by the LLM are stored as Semantic Triples:
```text
(user --[PREFERS]--> rust_and_go)
(complex_plot.py --[IMPLEMENTS]--> data_aggregation)
```

---

## 🚀 4. Interaction XML Tags

The LLM interacts with the memory system using custom XML tags inside its response stream. The system parser intercepts these tags, executes the operations on the database, and strips them before displaying the text to the user:

### Create a New Memory
```xml
<mem_new tags="user_preference,coding" subject="user" predicate="PREFERS" object="rust" importance="0.9">
  The user prefers Rust for all systems programming tasks.
</mem_new>
```

### Update an Existing Memory
```xml
<mem_update id="a1b2c3d4">
  The user prefers Rust and Go for all systems programming tasks.
</mem_update>
```

### Tag/Acknowledge Retrieval
When referencing information from Working Memory, the LLM must tag the node to reinforce its importance:
```xml
<mem_tag id="a1b2c3d4" />
```

### Promote Deep Memory to Working
```xml
<mem_load id="e5f6g7h8" />
```

### Create a Graph Relationship
```xml
<mem_rel source="a1b2c3d4" target="e5f6g7h8" type="SUPPORTS" weight="1.0" />
```

### Soft-Delete a Memory
```xml
<mem_delete id="a1b2c3d4" />
```

---

## 🗃️ 5. Dual-Database Architecture

To provide both private session tracking and shared project learning, the memory system utilizes a **Dual-Database Attachment** paradigm over SQLite:

1.  **Private Local Database (`main.memories`)**
    *   **Scope**: Bound strictly to the current discussion session.
    *   **Stores**: Local episodic interaction logs, temporary task variables, and session-specific events.
2.  **Shared Semantic Database (`shared_mem_db.memories`)**
    *   **Scope**: Shared across all discussions inside a given project workspace.
    *   **Stores**: Persistent user preferences, validated technical lessons, code style standards, and shared design constraints.
    *   **Trigger**: Semantic engrams carrying `#preference`, `#standard`, or `#technical_lesson` tags are automatically routed and committed to the attached shared database.

---

## 💤 6. The Dream Cycle (Synaptic Consolidation)

The `dream()` pass is an asynchronous consolidation routine designed to run periodically (or on-demand):

1.  **Soft-Delete Purge**: Permanently deletes any memory nodes whose importance has decayed to `0.0`.
2.  **Centrality Auditing**: Computes PageRank-like weighted degree centrality across all nodes to identify highly connected "keystone" memories.
3.  **Synaptic Fusion**: Merges redundant or highly overlapping memories (sharing identical tags or categories) into a single, high-density note to optimize storage.
4.  **Synaptic Auditing**: Uses an LLM to automatically categorize and index un-tagged or "orphaned" memory nodes.
5.  **Forgetting Pass**: Faded memories that fall below `forget_threshold` are subjected to a final "forgetting evaluation" by the LLM. If evaluated as obsolete, they are permanently purged.