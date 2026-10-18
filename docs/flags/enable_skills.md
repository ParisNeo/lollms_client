# `enable_skills`

```python
enable_skills: bool = True
```

**Category**: Artifacts

## What it does

Enables the `<skill>` tag. Routing is personality-aware:

- **Handbag personality** (has `skills_manager`): skills are written directly
  into the handbag's skills folder (`create_skill`/`update_skill`). Read-only
  skills are refused with a clear failure block.
- **Stateless/manual personality**: skills are stored as discussion artifacts
  (`<title>.md`, skill type).

## What to expect

- Requires `enable_artefacts=True`.
- With a Handbag personality, `<skill>` writes go to the personality's skills
  folder and **persist across discussions**.

## Recovering generated stuff

- **Handbag**: `personality.skills_manager.skills` — skills persist across
  discussions.
- **Stateless**: `discussion.artefacts.get("<title>.md")`.

## Usage

```python
discussion.chat("Create a skill for SQL joins", enable_skills=True)
```

## Related flags

- `enable_artefacts` — master switch.