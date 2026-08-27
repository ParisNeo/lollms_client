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
Defines the valid classes of concepts and their allowed relationship verbs (implemented in the `MemoryOntology` class):

*   **Node Classes**:
    *   `CONCEPT`: Abstract ideas, subjects, tools, or entities.
    *   `PREFERENCE`: User guidelines, constraints, and custom personality rules.
    *   `EVENT`: Milestone occurrences, episodes, or tool outputs.
    *   `DECISION`: Architectural choices, code designs, or lessons learned.
*   **Relationship Verbs (Predicates)**:
    *   `RELATED_TO` (Default/Associative)
    *   `PREFERS` (Preference mapping)
    *   `IMPLEMENTS` (Code realization)
    *   `CONTRADICTS` (Logic conflicts)
    *   `SUPPORTS` (Logical validations)
    *   `TEMPORAL_AFTER` (Chronological ordering)
    *   `PART_OF` (Decomposition/Composition)

### B. The ABox (Assertional Instances)
The actual facts saved by the LLM are stored as Semantic Triples. If the LLM omits the ontology attributes, the `auto_extract_ontology_from_content` function attempts to surgically infer them from the text content:
```text
(user --[PREFERS]--> rust_and_go)
(complex_plot.py --[IMPLEMENTS]--> data_aggregation)
```

---

## 🚀 4. Interaction XML Tags

The LLM interacts with the memory system using custom XML tags inside its response stream. The system parser intercepts these tags, executes the operations on the database, and strips them before displaying the text to the user.

### Controlling Episodic Memory

By default, the system automatically saves substantial conversations as episodic memories (Level 4). You can control this behavior using the `enable_episodic_memory` parameter in the `chat()` method:

```python
# Disable episodic memory saving (privacy mode)
response = discussion.chat(
    user_message="Tell me a joke",
    enable_episodic_memory=False  # Conversation won't be saved to episodic memory
)

# Enable episodic memory saving (default behavior)
response = discussion.chat(
    user_message="Explain quantum computing",
    enable_episodic_memory=True  # Conversation will be saved if substantial
)
```

**When to disable episodic memory:**
- Privacy-sensitive applications where conversation history shouldn't persist
- Temporary/scratch conversations that don't represent meaningful events
- When you want manual control over what gets saved (use `<mem_new>` tags explicitly)
- Testing/development scenarios where you don't want to pollute the memory database

**When episodic memory is saved (when enabled):**
- Conversations longer than 200 characters
- Turns that used tools or created artifacts
- Non-trivial exchanges (not just greetings like "hi" or "thanks")

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

## 🛠️ 5. Application APIs & Direct Management

Beyond LLM-initiated XML tags, `LollmsMemoryManager` exposes direct Python APIs for building user interfaces and performing explicit queries:

*   **`list_all(level, search_query, page, page_size)`**: Paginated, searchable retrieval of all memories. Ideal for rendering administrative control panels.
*   **`edit_memory(memory_id, content, importance, level, tags, subject_group)`**: Manually overwrite any field of a memory record.
*   **`query(text, top_k, level)`**: Fast, on-the-fly TF-IDF keyword matching over the database. It filters stop words and weights results by Term Frequency-Inverse Document Frequency alongside the memory's base importance.
*   **`auto_pull_deep_memories(user_message, top_k)`**: Proactively scans user input for keywords matching Level 2 (Deep) memories and promotes them to Level 1 (Working) before the LLM even responds.

---

## 🗃️ 6. Dual-Database Architecture

To provide both private session tracking and shared project learning, the memory system utilizes a **Dual-Database Attachment** paradigm over SQLite:

1.  **Private Local Database (`main.memories`)**: Bound strictly to the current discussion session.
2.  **Shared Semantic Database (`shared_mem_db.memories`)**: Shared across all discussions inside a given project workspace.

When a `shared_db_path` is provided during initialization, the manager dynamically executes `ATTACH DATABASE` and constructs a cross-schema `UNION ALL` query layer (`_q()`). This allows the application to query and interact with memories from both the local and shared schemas transparently as a single unified graph.

---

## 💤 7. The Dream Cycle (Synaptic Consolidation)

The `dream()` pass is an asynchronous consolidation routine designed to run periodically (or on-demand):

1.  **Soft-Delete Purge**: Permanently deletes any memory nodes whose importance has decayed to `0.0`.
2.  **Centrality Auditing**: Computes PageRank-like weighted degree centrality across all nodes to identify highly connected "keystone" memories.
3.  **Synaptic Fusion**: Merges redundant or highly overlapping memories (sharing identical tags or categories) into a single, high-density note to optimize storage.
4.  **Synaptic Auditing**: Uses an LLM to automatically categorize and index un-tagged or "orphaned" memory nodes.
5.  **Forgetting Pass**: Faded memories that fall below `forget_threshold` are subjected to a final "forgetting evaluation" by the LLM. If evaluated as obsolete, they are permanently purged.

---

## 🧩 8. SSAM Engine (Sovereign Semantic Autarkic Memory)

Exported alongside the main manager is the `SSAMEngine`, a standalone, highly isolated semantic memory engine that strictly enforces LTI/STI (Long-Term Identifier / Short-Term Identifier) separation.

### Architecture
*   **LTI (Long-Term Identifier)**: The immutable baseline fact securely anchored in the SQLite database.
*   **STI (Short-Term Identifier)**: A volatile, active projection of an LTI inside the working memory sandbox. Changes made to an STI do not affect the LTI until explicitly committed.

### Key Operations
*   **`add_lti()`**: Saves a permanent fact to the LTI table.
*   **`load_to_working_memory()`**: Projects an LTI into the active STI sandbox, computing its history and activation.
*   **`spread_activation()`**: Spreads energy multiplicatively from a source STI to linked semantic neighbors in the graph, pre-warming them.
*   **`commit_sandbox_changes(sti_id)`**: Executes a transaction commit protocol that validates and saves STI modifications back to the permanent LTI database.
*   **`purge_sandbox()`**: Discards the transient working sandbox, reverting any uncommitted cognitive drift.