# `enable_memory`

```python
enable_memory: bool = True
```

**Category**: Memory — **MASTER SWITCH**

## What it does

Enables the memory subsystem for the turn: memory system instructions, decay,
budget enforcement, `<mem_*>` tag processing (`<mem_new>`, `<mem_update>`,
`<mem_tag>`, `<mem_load>`, `<mem_delete>`, `<mem_search>`, `<mem_rel>`), and the
Active Memories context block.

**Manager resolution order**: personality's own `memory_manager` → discussion's
`_get_memory_manager()` → explicit `memory_manager` argument →
`self.memory_manager`.

## What to expect

- `False`: no memory operations occur and memory tags are not processed. A
  `<mem_search>` still intercepts but the manager is absent so no results are
  injected.
- `True`: memories are hydrated into context; the model can create/update/load
  memories via tags.

## Usage

```python
discussion.chat(msg, enable_memory=True)   # default
discussion.chat(msg, enable_memory=False)  # memory-less turn
```

## Related flags

- `enable_episodic_memory`, `enable_auto_dream`, `enable_deep_memory_pulling` —
  sub-flags inert without this master or a resolvable manager.