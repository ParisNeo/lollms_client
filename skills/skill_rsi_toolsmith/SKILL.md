---
name: skill_rsi_toolsmith
title: "Skills RSI Toolsmith Workflow"
description: "Self-improvement workflow for SKILL.md files: read → modify → test → (test agent) → commit. Use this whenever the operator asks you to add, refine, or repair a skill in the current session."
category: self_improvement
tags:
  - rsi
  - skill_authoring
  - meta
visibility: loadable
tools:
  - tool_read_skill_source
  - tool_write_skill_source
  - tool_test_skill_source
  - tool_list_skill_versions
  - tool_revert_skill
  - tool_discard_skill_changes
  - tool_load_skill
---

# Skills RSI Toolsmith Workflow

Use this skill to evolve the agent's own skills safely, mirroring the RSI
toolsmith loop that already exists for LCP toolsets.

## When to use

Trigger this skill when the operator says things like:

- "Improve the X skill so it also handles Y."
- "The current `desktop_automation` skill keeps failing — figure out why and fix it."
- "Add a new skill for Z."
- "Revert my last skill change / discard everything since the last commit."

Do **not** trigger for one-off answers that do not require editing a `SKILL.md`.

## Mandatory workflow

Follow the exact step order. **Never skip `tool_test_skill_source` before
`tool_commit_skill`.**

### 1. Locate the skill

List candidates by calling `tool_list_skills` (or `tool_load_skill` when you
already know the slug) to confirm the skill is registered and its current
`SKILL.md` is on disk. Do **not** invent a filename — `skill_maker` only
resolves **registered** skills.

### 2. Read the current source

```
tool_read_skill_source(skill_name_or_path="<slug>")
```

Read the full source returned. Identify the exact lines that need to change.
If the skill is not in the list of active skills, stop and report to the
operator — you must not create a skill file outside the registered paths.

### 3. Modify

Prefer **Aider-style SEARCH/REPLACE** blocks over full rewrites:

```
tool_write_skill_source(
    skill_name_or_path="<slug>",
    new_source="""
<<<<<<< SEARCH
old lines verbatim
=======
new lines
>>>>>>> REPLACE
""",
    reason="short rationale"
)
```

The write is applied in memory first, the resulting `SKILL.md` is re-validated
(YAML frontmatter + required fields + non-trivial body), and only then is it
written to disk. A broken patch is refused — you never land a partial edit.

### 4. Validate

```
tool_test_skill_source(skill_name_or_path="<slug>")
```

This checks:

- YAML frontmatter parses
- Required fields (`name`, `description`) are present
- Body is non-trivial (>= 30 chars)
- Declared `tools:` list is extractable

A **failure here means you must go back to step 3**. Never advance to the
test agent or to commit on a failing validation.

### 5. (Optional) Exercise the skill with a constrained child agent

Only available when the operator enabled `enable_skillmaker`. Use it:

- When the skill declares tools whose behaviour you actually need to check
  (e.g. `tool_load_skill`, `tool_list_skills`).
- Never for skills that declare `system_shell`, `execute_python`, or any
  code-execution tool — the child refuses those for safety.

```
tool_run_skill_test_agent(
    skill_name_or_path="<slug>",
    test_task="Load the skill and confirm tool_load_skill returns its body.",
    max_steps=4,
    timeout_sec=60,
)
```

The transcript comes back in the response — paste a one-paragraph summary of
the verdict into your next message to the operator.

### 6. Commit (only when the operator asked you to, or when the task is
complete and the operator's directive implicitly authorises it)

```
tool_commit_skill(
    skill_name_or_path="<slug>",
    commit_message="<short rationale>"
)
```

The commit writes a **file-system release** under `<skill_dir>/.rsi/commits/`
(with the full source embedded) and, if git is available in this repo, also
performs a scoped `git commit` for the skill folder. Both mechanisms are
reported in the result. Then the personality's `skills_manager` is hot-reloaded
so the new content is visible in the next turn.

### 7. Revert / discard

If a committed change turns out to be wrong, use:

- `tool_list_skill_versions` — enumerate snapshots and releases.
- `tool_revert_skill(skill_name_or_path, backup_index=0)` — undo the last
  write; the current content is auto-snapshotted first, so a further revert is
  always possible.
- `tool_discard_skill_changes(skill_name_or_path)` — hard reset to the last
  committed state (release manifest / git HEAD / oldest snapshot).

## Invariants — do not violate

1. **Never write a skill that was not previously registered.** If the operator
   asks for a brand-new skill, ask them to create the initial `SKILL.md`
   (or use the GUI) first, then improve it via this workflow.
2. **Never skip `tool_test_skill_source`.** A commit without validation is
   refused by `tool_commit_skill` anyway, but emitting an unvalidated write is
   still bad form.
3. **Never run the child test agent on a skill that declares code-execution
   tools.** It will refuse, and it will report the offending tool names.
4. **Never invent frontmatter fields.** The structural validator only requires
   `name` + `description`, but downstream tools rely on `title`, `category`,
   `tags`, `visibility`, and `tools:` — keep them consistent with the existing
   shape of the file you are editing.
5. **Never call `tool_discard_skill_changes` without an explicit operator
   request.** It wipes every intermediate edit for that skill.

## Failure recovery checklist

| Symptom | Action |
|---|---|
| `tool_write_skill_source` returns "frontmatter rejected" | Read the exact error; the candidate is missing a `name:` or `description:` line or did not open with `---`. Fix and retry. |
| `tool_test_skill_source` reports `structure_ok: false` | Body is too short (< 30 chars) or a required field is missing. Retry from step 3. |
| `tool_run_skill_test_agent` returns `Refused: the skill declares code-execution tools [...]` | The skill's `tools:` list includes a forbidden name. Remove it or test the skill manually. |
| `tool_commit_skill` returns "Refusing to commit — the skill fails validation" | Never happens after a successful step 4 — means the file was modified between step 4 and step 6 (rare). Re-read, re-validate, retry. |
| You need to undo everything | `tool_discard_skill_changes(skill_name_or_path)` — returns to last release. |