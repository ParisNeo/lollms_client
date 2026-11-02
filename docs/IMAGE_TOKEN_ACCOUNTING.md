# `get_context_status()` — Complete Reference

`get_context_status()` returns a snapshot of how the token budget is currently
distributed across every context source — system prompt, message history, images,
artefacts, working memory, and deep memory — so a UI can render an accurate
context-composition panel. It is **read-only**: it never prunes, truncates, or
mutates the discussion.

## 1. Quick start

```python
status = disc.get_context_status()
# {
#   "max_tokens": 8192,
#   "current_tokens": 1433,
#   "percent": 17.49,
#   "zones": {
#       "system_context":    {"tokens": 120, ...},
#       "message_history":   {"tokens": 900, "breakdown": {...}},
#       "discussion_images": {"tokens": 256, "count": 1},
#       "artefacts":         {"tokens": 0, ...},
#       "working_memory":    {"tokens": 100, ...},
#       "deep_memory":       {"tokens": 57, ...},
#   },
#   "images": {
#       "discussion_images": {"count": 1, "tokens": 256},
#       "message_images":    {"count": 0, "tokens": 0},
#       "total_images": 1,
#       "total_image_tokens": 256,
#   },
# }
```

## 2. The return contract — full reference

`get_context_status(branch_tip_id=None) -> Dict[str, Any]`

### 2.1 `max_tokens` (int)
The model's context window size, taken from the binding's `ctx_size` (or the
binding default when unset). This is the denominator for `percent`.

### 2.2 `current_tokens` (int)
The sum of all zone token counts. **Invariant:**
`current_tokens == sum(zone["tokens"] for zone in zones.values())`.

### 2.3 `percent` (float)
`current_tokens / max_tokens * 100`. Convenience value for progress bars.

### 2.4 `zones` (dict)
Per-source token accounting. The mixin ships these zone labels; if your build
differs slightly, trust the runtime dict — the shape and meaning are stable.

| Zone | Shape | Meaning |
|------|-------|---------|
| `system_context` | `{"tokens": int, ...}` | System prompt / persona / handbag |
| `message_history` | `{"tokens": int, "breakdown": {"text_tokens": int, "image_tokens": int, "message_count": int}}` | Rendered conversation turns |
| `discussion_images` | `{"tokens": int, "count": int}` | **Conditional** — present only when active discussion-level images exist |
| `artefacts` | `{"tokens": int, ...}` | Loaded artefacts |
| `working_memory` | `{"tokens": int, ...}` | Short-term memory |
| `deep_memory` | `{"tokens": int, ...}` | Long-term / episodic memory |

### 2.5 The additive `images` block
A top-level summary so any UI can surface image usage without parsing zone
sub-dicts:

| Key | Meaning |
|-----|---------|
| `images["discussion_images"]["count"]` | Number of active discussion-level images |
| `images["discussion_images"]["tokens"]` | Token cost of those images |
| `images["message_images"]["count"]` | Number of active per-message images |
| `images["message_images"]["tokens"]` | Token cost of those images |
| `images["total_images"]` | Sum of both counts |
| `images["total_image_tokens"]` | Sum of both token costs |

The `images` block is **purely additive** — existing callers that only read
`current_tokens` / `percent` / `zones` are unaffected.

## 3. How tokens are computed

### 3.1 Text tokenizer chain
Text is tokenized via `self.lollmsClient.count_tokens(text)`.

### 3.2 Per-message `msg.tokens` text-only cache
Each message persists a **text-only** token count in `msg.tokens`. This value
excludes image tokens **and** excludes the `\n(N image(s) attached)` annotation
suffix. At status time, the annotation delta is added on top, so the cache stays
stable across image toggles.

### 3.3 Image tokenizer
`_image_tokens_for(img)` uses the binding's `count_image_tokens(img)` when
available; otherwise it falls back to a conservative heuristic (see §4.4).

## 4. Image token accounting deep-dive

### 4.1 Discussion-level images
Images attached via `add_discussion_image(...)` live in `self.images` and are
counted into `zones["discussion_images"]` (active only).

### 4.2 Per-message images
Images attached to a message (`message.images`) are counted per message via
`msg.get_active_images()` and folded into `zones["message_history"]["breakdown"]["image_tokens"]`.

### 4.3 Step-by-step counting pipeline

```
per message:  msg.get_active_images() → [_image_tokens_for(img) for img] = img_toks
              + tokenizer("!@>sender:\ncontent\n(N image(s) attached)")
                → history_breakdown["image_tokens"] += img_toks
                → history_breakdown["text_tokens"] += msg_toks
                → zones["message_history"]["tokens"] = text_tokens + image_tokens
discussion:   get_discussion_images() (active only) → [_image_tokens_for(i["data"]) ...]
                → zones["discussion_images"] = {"tokens": N, "count": K}
top-level:    images["discussion_images"] = {"count": K, "tokens": N}
              images["message_images"]    = {"count": M, "tokens": T}
              images["total_images"]      = K + M
              images["total_image_tokens"]= N + T
```

### 4.4 Heuristic fallback (exact formula)
When `count_image_tokens` is absent or raises, the estimate is:

```
max(1, len(base64) // 750)
```

for any non-empty image string. This guarantees a real image never shows 0
tokens — coarse, but never silently dropped.

### 4.5 Cache invalidation on image mutators
`_invalidate_discussion_image_token_cache()` fires from `add_discussion_image`,
`toggle_discussion_image_activation`, and `remove_discussion_image`, so the next
`get_context_status()` call reflects the change immediately.

## 5. Worked example (fully runnable)

```python
import base64
from pathlib import Path
from lollms_client import LollmsClient

client = LollmsClient("your_binding_name")
disc = client.create_discussion("image-status-demo")

# 1) Attach a discussion-level image
img_b64 = base64.b64encode(Path("cat.png").read_bytes()).decode()
disc.add_discussion_image(img_b64, source="user", active=True)

# 2) Add a message that also carries an image
disc.add_message(sender="User", content="What is this image?", sender_type="user")
branch = disc.get_branch(disc.active_branch_id)
branch[-1].images = [{"data": img_b64, "active": True}]
disc.commit()

# 3) Inspect the context composition
status = disc.get_context_status()
print("Budget:",    f"{status['current_tokens']}/{status['max_tokens']} ({status['percent']}%)")
print("Images:",    status["images"])
print("History:",   status["zones"]["message_history"]["breakdown"])
print("Disc imgs:", status["zones"].get("discussion_images"))
```

`N` and `M` are **token estimates**; when the binding lacks `count_image_tokens`
they come from `max(1, len(base64)//750)` — coarse, but **never 0 for a real
image** (that is the feature guarantee).

## 6. Interpreting the numbers / UI integration

| Context-composition UI element | Data source |
|---|---|
| System prompt / persona bar | `zones["system_context"]["tokens"]` |
| Chat history bar | `zones["message_history"]["tokens"]` |
| Image chip (discussion + messages combined) | `images["total_images"]` (count) · `images["total_image_tokens"]` (tokens) |
| Artefacts bar | `zones["artefacts"]["tokens"]` |
| Working memory bar | `zones["working_memory"]["tokens"]` |
| Deep memory bar | `zones["deep_memory"]["tokens"]` |
| Total budget ring / bar | `current_tokens` / `max_tokens` / `percent` |

## 7. Interaction with pruning & the branch model

- `get_context_status` is **read-only** — it never prunes or truncates.
- `branch_tip_id` narrows the computation to the ancestor chain of the given
  tip; it defaults to `self.active_branch_id`.
- Pruning is a separate operation (`summarize_and_prune(...)`), invoked by the
  host — never by `get_context_status`.

## 8. Backward compatibility

- The `images` block is **purely additive**.
- `count_discussion_tokens()` reads only `current_tokens` and is unaffected.
- Existing `zones` keys are unchanged.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Image tokens show 0 in tests | `MockClient.count_image_tokens` returns 0 (harness stubs it) | Expected; override in tests or use a real binding |
| `AttributeError: count_image_tokens` | Should not happen post-fix | Verify `_image_tokens_for` is on disk; it guards via `getattr` |
| `images` missing from top-level status | Regression — landing absent | Confirm `result["images"]` block is present in `_mixin_utils.py` |
| Totals don't add up | `current_tokens` invariant violated | A zone wasn't updated; check the mutator that changed state |
| Image annotation added twice | `msg.tokens` reused incorrectly | Ensure `msg.tokens` stays text-only; annotation delta added at status only |

## 10. Related docs & flags

- [`docs/flags/suppress_images.md`](flags/suppress_images.md) — suppressing image
  injection into the prompt.
- [`docs/BRANCHING.md`](BRANCHING.md) — the branch model that `branch_tip_id`
  walks.