# LCP Tool Resource Access

How LCP tools (and the bindings they live in) reach auxiliary resources: the main LLM
for custom generation, the TTI service for image generation, TTS/TTM for audio, and
the active discussion for conversational awareness.

---

## 1. Overview — The Access Chain

`LollmsClient` is the **single point of access** a tool receives. All service bindings
hang off it:

```text
                    ┌──────────────────────┐
        Tool ───►   │    LollmsClient      │
                    └─────────┬────────────┘
              ┌───────────────┼───────────────────┬───────────────┐
              ▼               ▼                   ▼               ▼
      .llm_binding      .tti_binding        .tts_binding      .discussion
   LollmsLLMBinding   LollmsTTIBinding    LollmsTTSBinding  LollmsDiscussion
   ├─ model_name      └─ generate_image()  └─ synthesize()  └─ .messages
   ├─ generate()          (may be None)       (may be None)    (may be None)
   ├─ generate_with_images()
   └─ tokenize() / count_tokens()
```

---

## 2. Accessing the LLM from a Tool

Any tool receiving the `LollmsClient` (as `lc` or `client`) reaches the LLM via
`lc.llm_binding`:

```python
def summarize_document(lc, text: str) -> str:
    prompt = f"Summarize the following in 3 bullet points:\n\n{text}"
    return lc.llm_binding.generate(prompt, n_predict=512)
```

Available on `lc.llm_binding`:

| Member | Purpose |
|---|---|
| `generate(prompt, n_predict=..., streaming_callback=...)` | Standard text generation |
| `generate_with_images(prompt, images=..., ...)` | Multimodal generation |
| `generate_from_messages(messages, ...)` | Chat-style generation from a message list |
| `model_name` | Loaded model identifier |
| `tokenize(text)` / `count_tokens(text)` | Token accounting |
| `embed(text)` | Embeddings (if the binding supports it) |

Some LCP tool signatures pass the binding directly (`llm: LollmsLLMBinding`) — the
same methods are available on that object.

---

## 3. Accessing TTI (Text-to-Image)

`lc.tti_binding` is **optional and may be `None`**. Always guard:

```python
def make_illustration(lc, scene: str):
    if lc.tti_binding is None:
        return {"error": "No TTI binding configured."}
    return lc.tti_binding.generate_image(prompt=scene)
```

---

## 4. Accessing TTS / TTM

Same optional-binding pattern:

```python
if lc.tts_binding is not None:
    audio = lc.tts_binding.synthesize(text="Hello world")

if lc.ttm_binding is not None:
    music = lc.ttm_binding.generate_music(prompt="lo-fi beat")
```

---

## 5. Accessing the Discussion (Conversational Context)

Tools needing conversation history use `lc.discussion`:

```python
def recall_last_user_request(lc) -> str:
    if lc.discussion is None:
        return ""
    for msg in reversed(lc.discussion.messages):
        if msg.get("sender_type") == "user":
            return msg.get("content", "")
    return ""
```

---

## 6. Tool Signature Conventions

Tools can declare any of three resource-injection styles:

| Signature parameter | Receives | Grants access to |
|---|---|---|
| `lc: LollmsClient` | The full client | LLM, TTI, TTS, TTM, discussion — everything |
| `llm: LollmsLLMBinding` | Only the LLM binding | Text / multimodal generation |
| `discussion: LollmsDiscussion` | Only the discussion | Conversation state & history |

Prefer the narrowest signature that satisfies the tool's need, but use `lc` when the
tool crosses modalities (e.g., generates an image *and* narrates it).

---

## 7. Safety Note

> **Never re-instantiate bindings inside a tool.** Bindings are registered with
> `LollmsClient` at construction; creating a second instance duplicates model
> connections, wastes VRAM/API quota, and bypasses the client's lifecycle management.
> Always use the bindings provided via `lc.llm_binding`, `lc.tti_binding`, etc.

---

## 8. Summary Table

| Resource | Access pattern | Guard |
|---|---|---|
| LLM generation | `lc.llm_binding.generate(...)` | Always available if a binding is loaded |
| Multimodal LLM | `lc.llm_binding.generate_with_images(...)` | Check the model supports vision |
| TTI (image gen) | `lc.tti_binding.generate_image(...)` | `if lc.tti_binding is not None` |
| TTS (speech) | `lc.tts_binding.synthesize(...)` | `if lc.tts_binding is not None` |
| TTM (music) | `lc.ttm_binding.generate_music(...)` | `if lc.ttm_binding is not None` |
| Discussion | `lc.discussion.messages` | `if lc.discussion is not None` |

---

## 9. Confirm Handler — Authorization and Long-Running Execution

The same `confirm_handler`-seam documented here for **execution authorization**
(in `execute_python` / `execute_python_data_query`) is also used to govern
**long-running execution**:

> When a sandboxed child exceeds its wall-clock budget, the handler receives a
> structured dict with `action_type="timeout"` and can answer `"continue"` or
> `"stop"`. See
> [`execute_python_timeout_policy.md`](execute_python_timeout_policy.md) for
> the full timeout contract, defaults (1 h), and dispatch rules.

This means a host that registers a single handler gets **both** capabilities
without any code change — legacy handlers that return `allow`/`always`/`reject`
are automatically mapped onto the timeout protocol (`allow`/`always` → continue,
`reject` → stop), and a broken handler safely defaults to *stop* rather than
allowing a hang to extend.

---

## 10. Imports for Examples

```python
from lollms_client import (
    LollmsClient,
    LollmsLLMBinding,
    LollmsDiscussion,
    MSG_TYPE,
)
```