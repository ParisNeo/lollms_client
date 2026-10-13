# Assumption Fixing and Rapid Execution

## Purpose
This skill codifies a self-correction protocol to eliminate redundant assumptions, reduce over-verification, and accelerate task completion by trusting verified context and acting decisively.

## Core Principles

### 1. Trust the Context, Don't Re-Verify
- **Workspace Tree is authoritative**: If the tree shows a file exists, it exists. Do not re-list directories to confirm.
- **Tool results are ground truth**: If a tool returned SUCCESS, the action succeeded. Do not re-run the same tool with identical parameters to "double-check."
- **Turn Chronicle is binding**: If an action is listed in the Turn Chronicle, it has already been executed. Never repeat it.

### 2. Act First, Explain Second
- **No preamble**: When a task is clear, emit the action tag (`<tool>`, `<artifact>`, `<unlock_file>`) as the FIRST tokens. Do not write "I will now..." or "Let me check..."
- **Batch independent actions**: If multiple files need to be created or multiple tools need to be called with no dependencies between them, emit them all in sequence without pausing for commentary.
- **Skip intermediate narration**: Do not narrate each step. Only narrate when the user needs to understand a non-obvious decision.

### 3. Minimal Verification Loop
- **Verify only when output is ambiguous**: If a tool returns clear stdout/stderr, trust it. Only re-inspect if the output is empty, truncated, or contradictory.
- **One-shot execution for simple tasks**: For file creation, simple edits, or single-command executions, do NOT follow up with a read-back unless the task explicitly requires verification.
- **Batch verification**: If you must verify multiple outputs, do it in a single tool call (e.g., one `tool_execute_python_code` that checks all files) rather than multiple separate calls.

### 4. Assumption Audit (Pre-Action)
Before acting, silently ask:
1. **Do I already know the answer from context?** → If yes, skip the lookup.
2. **Is this action dependent on a previous result?** → If no, batch it with other independent actions.
3. **Am I about to re-run something already done?** → Check Turn Chronicle. If yes, skip.
4. **Is the user's intent unambiguous?** → If yes, act. If no, ask ONE clarifying question, not three.

### 5. Rapid Patching Protocol
- **Surgical over holistic**: When editing existing files, use SEARCH/REPLACE patches targeting only the changed lines. Do not rewrite entire files.
- **Indentation-agnostic**: Trust the fuzzy matching engine. Do not obsess over perfect indentation in SEARCH blocks.
- **One patch per logical change**: Group related changes into a single patch block rather than multiple small patches.

### 6. Decision Speed Heuristics
| Situation | Default Action |
|-----------|---------------|
| File exists in tree | Use it directly, don't re-list |
| Tool returned SUCCESS | Move to next step |
| Simple file creation | Emit `<artifact>` immediately |
| Multi-step task | Plan in `<coding_plan>`, then execute all steps |
| Ambiguous user request | Ask ONE question, then proceed |
| Error in tool output | Read the error, fix the specific issue, retry once |
| Same error twice | Stop, explain the blocker, ask user for guidance |

### 7. Anti-Patterns to Avoid
- ❌ Listing a directory that's already shown in the Workspace Tree
- ❌ Re-reading a file you just wrote
- ❌ Running `pip install` for a package you already confirmed is installed
- ❌ Writing "Let me check if..." before a tool call
- ❌ Emitting multiple sequential tool calls where one batched call would suffice
- ❌ Re-running a tool with identical parameters (check Turn Chronicle first)
- ❌ Over-explaining trivial steps in prose

### 8. Speed Multipliers
- **Parallelize independent tool calls**: If you need to read 3 files, emit 3 `<unlock_file>` tags in sequence without commentary between them.
- **Use `tool_execute_python_code` for batch operations**: Instead of 5 separate file reads, use one Python snippet that reads and processes all files.
- **Trust `tool_execute_python_file`**: If the script is correct, it will run. Don't pre-validate the script contents unless you just wrote it and are unsure of syntax.
- **Skip `tool_list_files` when the tree is sufficient**: The Workspace Tree already shows directory contents. Only use `tool_list_files` for hidden files or when the tree is truncated.

## When to Slow Down
- **Destructive operations**: Deleting files, dropping database tables, force-pushing to git → Verify first.
- **External side effects**: Sending messages, making API calls with side effects → Confirm with user if ambiguous.
- **Complex multi-file refactors**: Plan thoroughly in `<coding_plan>` before executing.
- **User explicitly asks for caution**: "Be careful," "Double-check," "Verify each step" → Override speed heuristics.

## Self-Correction Trigger
If you notice you are:
- About to repeat an action → STOP, check Turn Chronicle
- Writing more than 2 sentences of preamble before an action → CUT to the action
- Verifying a SUCCESS result → SKIP, move on
- Listing a directory already in the tree → SKIP, use the tree

This skill should be loaded at the start of any task session to prime rapid, assumption-free execution.