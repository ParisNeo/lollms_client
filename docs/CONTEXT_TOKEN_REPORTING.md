
# Context Token Reporting — How to Inspect a Discussion's Context Budget

This document explains how to read the **token cost of every part of a discussion's context**: the system prompt, the memory zones, the artefacts, the conversation history, the images, and the memory manager blocks. The authoritative source is `LollmsDiscussion.get_context_status()` in `src/lollms_client/lollms_discussion/_mixin_utils.py`.

---

## 1. The Single Entry Point

```python
status = discussion.get_context_status()          # active branch
status = discussion.get_context_status(branch_tip_id="msg-id")  # a specific branch
```

`get_context_status()` is a synchronous method on the discussion object. It returns a **plain `dict`** that is safe to serialize, log, or display in a UI. It never raises for missing content — empty zones are simply omitted.

### Top-level shape

```python
{
    "max_tokens":    8192,        # self.max_context_size (fallback 8192)
    "current_tokens": 4210,       # sum of every zone's tokens
    "percent":        51.39,      # current_tokens / max_tokens * 100 (2 decimals)
    "zones":          { ... },    # detailed per-zone breakdown (see §2)
    "images":         { ... },    # additive image summary (see §4)
}
```

> **Convention:** `current_tokens` is always the arithmetic sum of `z["tokens"]` across `status["zones"].values()`. If you need a number to compare against the model's context window, use `current_tokens`, never reassemble zones yourself.

---

## 2. The `zones` Dictionary

Each key present in `zones` is a context **zone** that actually contributes text. Zones with zero content are **not** included, so always use `.get(...)` when reading.

```python
sys = status["zones"].get("system_context", {"tokens": 0})
hist = status["zones"].get("message_history", {"tokens": 0})
arts = status["zones"].get("artefacts", {"tokens": 0})
```

### 2.1 `system_context` — the system block

```python
status["zones"]["system_context"] = {
    "tokens": 3187,
    "breakdown": {
        "system_prompt":        {"tokens": 812},
        "memory":               {"tokens": 0},     # discussion.memory blob
        "user_data_zone":       {"tokens": 0},
        "discussion_data_zone": {"tokens": 1203},
        "personality_data_zone":{"tokens": 74},
        "scratchpad":           {"tokens": 0},
        "pruning_summary":      {"tokens": 412},
        "working_memory":       {"tokens": 566},   # memory manager
        "deep_memory":          {"tokens": 120},   # memory manager handles
    }
}
```

| Sub-key | Source attribute | Notes |
|---|---|---|
| `system_prompt` | `discussion.system_prompt` | The base system instructions. |
| `memory` | `discussion.memory` | Legacy free-form memory blob. |
| `user_data_zone` | `discussion.user_data_zone` | Pinned user data. |
| `discussion_data_zone` | `discussion.discussion_data_zone` | Loaded artefacts appear here too. |
| `personality_data_zone` | `discussion.personality_data_zone` | Injected personality context. |
| `scratchpad` | `discussion.scratchpad` | Agent scratchpad state. |
| `pruning_summary` | `discussion.pruning_summary` | The persisted technical synopsis (`--- Conversation Summary ---`). |
| `working_memory` | `memory_manager.build_working_zone()` | Only present when a memory manager is attached. |
| `deep_memory` | `memory_manager.build_handles_zone()` | Only present when a memory manager is attached. |

> **Rule:** `system_context.tokens` is the token count of the *formatted* block (`!@>system:\n<everything>\n`), which is slightly larger than the sum of sub-keys because of the role header. Use sub-keys for **attribution**, the parent `tokens` for **budget**.

### 2.2 `message_history` — the conversation

```python
status["zones"]["message_history"] = {
    "tokens": 1874,
    "breakdown": {
        "text_tokens":  1798,
        "image_tokens": 76,
        "message_count": 12
    }
}
```

- `message_count` is the number of rendered messages — it honours pruning (only messages after `pruning_point_id` are counted).
- `text_tokens` is the sum of each message's cached `msg.tokens` **plus** a fixed `\n(N image(s) attached)` annotation when images are active. Cached `msg.tokens` stores the **text-only** count so it stays stable when images are toggled.
- `image_tokens` is estimated per message image via `_image_tokens_for()` — see §5.

### 2.3 `artefacts` — active artefacts

```python
status["zones"]["artefacts"] = {
    "tokens": 940,
    "types": {
        "document": {"tokens": 720, "count": 1},
        "code":     {"tokens": 220, "count": 1}
    }
}
```

Only **active** artefacts are counted (`self.artefacts.list(active_only=True)`). The per-type breakdown lets a UI show "code is eating 220 tokens, documents 720". The per-artefact `token_count` is cached on the artefact record.

### 2.4 `discussion_images` — pinned discussion images

```python
status["zones"]["discussion_images"] = {"tokens": 210, "count": 3}
```

Present only when the discussion has **active** discussion-level images (added via `add_discussion_image(...)`). Counts images whose `active` flag is `True` and estimates their tokens with `_image_tokens_for()`.

---

## 3. Reading the Numbers — Worked Examples

### "Why is my context nearly full?"

```python
status = discussion.get_context_status()
total = status["current_tokens"]
max_ctx = status["max_tokens"]

for name, zone in sorted(status["zones"].items(),
                         key=lambda kv: kv[1].get("tokens", 0),
                         reverse=True):
    pct = zone.get("tokens", 0) / total * 100 if total else 0
    print(f"{name:<20} {zone.get('tokens',0):>7} tokens  ({pct:5.1f}%)")
print(f"TOTAL {total}/{max_ctx}  ({status['percent']}%)")
```

This prints zones largest-first — the top line is your culprit. Common patterns:

- **`message_history` dominates** → consider `summarize_and_prune(max_tokens=...)`.
- **`artefacts` dominates** → `unload_artefact_from_data_zone(title)`.
- **`system_context.working_memory` dominates** → shrink episodic memory or disable deep memory pulling.

### "Is a specific artefact worth keeping loaded?"

```python
arts = status["zones"].get("artefacts", {})
for atype, info in arts.get("types", {}).items():
    print(f"{atype}: {info['count']} file(s), {info['tokens']} tokens")
```

To attribute an artefact *by name*, iterate the record directly:

```python
for art in discussion.artefacts.list(active_only=True):
    print(f"{art['title']:<40} {art.get('token_count', 0):>7} tokens")
```

### "How much is images costing me?"

```python
status["images"]
# {
#   "discussion_images": {"count": 3,   "tokens": 210},
#   "message_images":    {"count": 12,  "tokens": 76},
#   "total_images":      15,
#   "total_image_tokens":286
# }
```

This top-level summary is *additive* — it duplicates the relevant bits of `zones` in a flat layout so a UI can surface image usage without walking zone sub-dicts.

---

## 4. `images` — Flat Image Summary

| Field | Meaning |
|---|---|
| `discussion_images.count` | Number of active discussion-level images. |
| `discussion_images.tokens` | Their estimated token cost. |
| `message_images.count` | Total active images across rendered messages. |
| `message_images.tokens` | Their estimated token cost. |
| `total_images` | Sum of both counts. |
| `total_image_tokens` | Sum of both token costs. |

Sourced identically to `zones.discussion_images` and `zones.message_history.breakdown.image_tokens` — **do not double count** if you already consume those zones.

---

## 5. The Token Counting Contract

### Text tokens

All cached token counts go through `self.lollmsClient.count_tokens(text)`. Results are memoised per content hash in `discussion.metadata["_token_cache"]`:

```python
{"zone_system_prompt": {"hash": "…", "tokens": 812}, …}
```

When anything changes, the affected cache entry is recomputed on next call and the discussion is `touch()`-ed + `commit()`-ed so the cost survives reloads.

### Image tokens

`_image_tokens_for(img)` uses, in order:

1. `self.lollmsClient.count_image_tokens(img)` if the binding implements it.
2. A conservative heuristic of **1 token per 750 base64 chars** (`max(1, len(raw) // 750)`).

Bindings that need precise accounting (e.g. OpenAI, Claude, Gemini, Qwen-VL) override `count_image_tokens`; bindings without it (e.g. raw Ollama) fall back to the heuristic.

### Caching contract (important for callers)

If you call `get_context_status()` in a tight UI loop, it is **hook-safe**: on an unchanged discussion, it makes **zero** tokenizer calls and **zero** DB writes. Only when a hash changes does it call `count_tokens` and commit. If you suspect a stale number, force a recompute by mutating content (which changes the hash) or clearing `discussion.metadata["_token_cache"]`.

---

## 6. Per-message Inspection

You can inspect a single message without recomputing the whole discussion:

```python
for msg in discussion.get_branch(discussion.active_branch_id):
    print(f"{msg.id}  sender={msg.sender:<9} tokens={msg.tokens or 0:>5} "
          f"imgs={len(msg.get_active_images())} active={msg.tokens is not None}")
```

Two flags matter:

| Field | Meaning |
|---|---|
| `msg.tokens` | `None` / `-1` if never computed; otherwise the **text-only** token count for this message. |
| `msg.get_active_images()` | Active images attached to this message; their tokens are **not** in `msg.tokens`. |

If you need a single-message *full* cost, do not do it manually in production — `get_context_status()` is the sanctioned path because it persists the text-only count and layers image cost on top per the rule in §2.2.

---

## 7. Async / Large-Context Caution

`get_context_status()` is synchronous and, in the worst case, calls `count_tokens` once per uncached message. On a 10k-token discussion over a network binding (Ollama API tokenizer, remote OpenAI), the **first** call may take a noticeable fraction of a second. Subsequent calls are cached and effectively free.

Practical rules:

1. Call it **once per UI refresh**, not per message render.
2. On startup after a reload, expect one full recompute — that's the warm-up.
3. If you only need the aggregate (`current_tokens`, `percent`), still call it once; the per-zone data comes free.

---

## 8. Cheat Sheet

```python
s = discussion.get_context_status()

s["current_tokens"]                    # total tokens in context
s["max_tokens"]                        # budget
s["percent"]                           # % used

s["zones"]["system_context"]["tokens"]           # system block total
s["zones"]["system_context"]["breakdown"]        # per-zone attribution
s["zones"]["message_history"]["tokens"]          # conversation total
s["zones"]["message_history"]["breakdown"]       # text vs images
s["zones"].get("artefacts", {})                  # active artefacts
s["zones"].get("discussion_images", {})          # pinned images
s["images"]["total_image_tokens"]                # flat image summary
```

---

## 9. See Also

- `src/lollms_client/lollms_discussion/_mixin_utils.py` → `UtilsMixin.get_context_status()` — the authoritative implementation.
- `src/lollms_client/lollms_discussion/_mixin_utils.py` → `UtilsMixin._image_tokens_for()` — image token estimation.
- `docs/IMAGE_TOKEN_ACCOUNTING.md` — image-token accounting rules per binding.
- `docs/CHAT_BOOLEAN_FLAGS.md` — flags that gate the zones surfaced here (`enable_memory`, `enable_artefacts`, etc.).
