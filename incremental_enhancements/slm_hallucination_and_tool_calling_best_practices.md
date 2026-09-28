# 🧠 Research & Best Practices: Mitigating Tool-Bypass Hallucinations in Small Language Models (SLMs)

## 1. Executive Summary
This document analyzes the cognitive mechanism behind **Tool-Bypass Hallucinations** in Small Language Models (SLMs, 7B–8B parameter range, such as Qwen 2.5/3 VL 8B, Llama 3 8B, Mistral 7B) during agentic workflows.

It explains why runtime regex heuristics like "Phantom `<done/>` Interception" are dangerous, and details the research-grounded architectural remedies: **Few-Shot In-Context Exemplars**, **Positive Syntax Steering**, and **De-noised Skill Schemas**.

---

## 2. Theoretical Background: The "Tool-Bypass" Error in SLMs

In agentic systems, hallucinations are not limited to factual errors; they manifest as functional errors:
1. **Tool-Selection Hallucination**: Calling a tool that does not exist.
2. **Parameter Hallucination**: Passing invalid types or hallucinated arguments.
3. **Tool-Bypass Hallucination (Simulation over Execution)**: The model describes performing the action in prose (*"Step 1: I will load... Step 2: I will execute..."*), fails to emit the tool or artifact syntax, and prematurely considers the turn complete with `<done/>`.

### Why Small Models Suffer from Tool-Bypass
- **Autoregressive Text Inertia**: 8B models are pre-trained predominantly on prose. Once an SLM starts generating a markdown list or paragraph, the probability of continuing that narrative stream is significantly higher than switching syntactical registers to emit an XML block (`<tool>` or `<artifact>`).
- **Prompt Header Parrot Effect**: When complex skills (like `file_organization`) contain procedural headers (`### Step 1: Scan...`, `### Step 2: Generate...`), small models mistakenly replicate those markdown headers as conversational text rather than following the underlying action logic.
- **Negative Constraint Decay**: Instructions like *"DO NOT write plans without tools"* suffer from semantic inversion in SLMs: the model attends strongly to the concepts *"write plans"* and produces them.

---

## 3. Why "Phantom `<done/>` Interception" Was Dangerous

The previous heuristic attempted to catch tool-bypass by inspecting the model's text:
```python
if ss.was_done_detected():
    if not actions_done and has_action_intent:
        force_continuation()
```
### The Failure Mode:
When an agent legitimately finishes work and provides a retrospective summary to the user (*"Here is what was completed: Step 1: I organized files. Step 2: I verified the output. <done/>"*), the regex falsely matched the words `Step 1` and `organized files` as future intent, rejected `<done/>`, and locked the conversation into an infinite loop.

**Verdict**: The system must treat `<done/>` as sovereign and heal the **causes** of tool-bypass at the prompt and in-context grounding layers.

---

## 4. Best Practices Applied to Heal the Causes

### A. Few-Shot In-Context Exemplars (LangChain & Berkeley Leaderboard Research)
Research demonstrates that small models transition from zero-shot simulation to deterministic execution when provided with compact, few-shot demonstrations:
```text
User: "Execute the migration plan mapping.yaml"
Assistant:
<tool>{"name": "tool_organize_files_from_plan", "parameters": {"plan_file": "mapping.yaml", "move_files": true}}</tool>
```
Demonstrating that the answer begins immediately with the functional tag anchors the autoregressive generation directly onto the action token.

### B. Positive Syntax Steering (Action-First Token Prefix)
Instead of negative warnings, the system prompt specifies:
- *"When performing a task, output the action tag (`<tool>`, `<artifact>`, `<unlock_file>`) as your FIRST tokens. Never write prose checklists explaining what you plan to do."*

### C. Skill Header De-noising
In `skills/file_organization/SKILL.md`, repetitive headers (`### Step 1: Scan...`, `### Step 2: Generate...`) were stripped, preventing the model from parroting those exact headers as conversational text.