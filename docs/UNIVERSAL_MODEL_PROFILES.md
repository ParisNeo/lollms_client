# Universal Model Profiles — The Two-Tier Architecture

> **Authoritative guide** for creating, configuring, persisting, and switching model profiles across **all eight modalities** of `LollmsClient`: LLM (text generation), TTI (text-to-image), TTS (text-to-speech), STT (speech-to-text), TTV (text-to-video), TTM (text-to-music), Connection, and RAG.
>
> **Applies to**: `lollms_client` (core profiles machinery in `src/lollms_client/lollms_core.py`).

---

## Table of Contents

1. [Overview — Why Two Tiers?](#1-overview--why-two-tiers)
2. [The Two Dataclasses](#2-the-two-dataclasses)
3. [Creating Profiles — Constructor Kwargs & Formats](#3-creating-profiles--constructor-kwargs--formats)
4. [Registry Internals](#4-registry-internals)
5. [Switching Models & Modalities](#5-switching-models--modalities)
6. [The Instantiation Pipeline](#6-the-instantiation-pipeline)
7. [Default Resolution & the Single-Default Invariant](#7-default-resolution--the-single-default-invariant)
8. [Per-Modality Guide (LLM, TTI, TTS, TTM, STT, TTV, Connection, RAG)](#8-per-modality-guide)
9. [Persistence Formats](#9-persistence-formats)
   - 9.4 [Discovering Available Bindings](#94-discovering-available-bindings)
   - 9.5 [Inspecting a Binding's Configuration Parameters](#95-inspecting-a-bindings-configuration-parameters)
10. [Host-App Integration (CLI, GUI, Programmatic)](#10-host-app-integration)
11. [Consumer Tools & Agents](#11-consumer-tools--agents)
12. [Migration & Legacy Compatibility](#12-migration--legacy-compatibility)
13. [API Quick Reference](#13-api-quick-reference)

---

## 1. Overview — Why Two Tiers?

`LollmsClient` manages **eight generation/perception modalities** through a single, uniform profile system. Before the two-tier design, each modality required a binding *name* plus a *config* dict passed directly to the constructor — one model per modality, no way to declare several, no lazy loading.

The **Universal Two-Tier Profile Architecture** solves this with two declarative layers:

| Tier | Profile Class | Describes | Example |
|---|---|---|---|
| **Connection Layer** | `LollmsBindingProfile` | *Where* to connect: the backend engine/server, its host, port, API key, library name | `localhost llama.cpp server`, `OpenAI API endpoint` |
| **Execution Layer** | `LollmsModelProfile` | *What* to run: a specific model on a named server, plus vision flags, context size, routing rules | `Qwen3-VL-8B` on the local server |

```
┌────────────────────────────────────────────────────────────────────┐
│                        LollmsClient                                │
│                                                                    │
│  Connection Layer            Execution Layer                       │
│  ┌───────────────────────┐   ┌──────────────────────────────────┐  │
│  │ LollmsBindingProfile  │◄──┤ LollmsModelProfile               │  │
│  │  name                 │   │  name                            │  │
│  │  binding_name         │   │  binding_profile_name ───────────┘  │
│  │  binding_config       │   │  model_name                        │
│  │  is_default           │   │  is_default / vision_enabled / …   │
│  └───────────────────────┘   └──────────────────────────────────┘  │
│          ▲                              ▲                          │
│   "Declare once                       "Declare many —             │
│    per server"                         one per model"              │
└────────────────────────────────────────────────────────────────────┘
```

**Key properties:**

- **Universal** — the *identical* dataclasses, kwargs, registries, and switch API work for all 8 modalities.
- **Declarative** — profiles are plain configuration; no engines start at declaration time.
- **Lazy** — only the model profile marked `is_default=True` is instantiated at startup. Every other model is instantiated **on first switch**, preserving RAM and VRAM.
- **Backward compatible** — legacy `llm_binding_name` / `llm_binding_config` (and `extra_llms`) are auto-registered as `"master"` profiles.

### The Eight Modalities

| Modality | Direction | Client attribute | Helper methods |
|---|---|---|---|
| `llm` | text generation | `client.llm` | `generate_text`, `generate_from_messages`, `generate_with_tools`, `chat` |
| `tti` | text→image | `client.tti` | `generate_image`, `edit_image`, `generate_omni` |
| `tts` | text→speech | `client.tts` | `generate_audio` |
| `stt` | speech→text | `client.stt` | `transcribe_audio`, `transcribe_audio_with_diarization` |
| `ttv` | text→video | `client.ttv` | `generate_video` |
| `ttm` | text→music | `client.ttm` | `generate_music`, `generate_song`, `generate_song_from_lyrics` |
| `connection` | outbound messaging | `client.connection` | `send_connection_message` (auto-switches by `channel_alias`) |
| `rag` | retrieval / knowledge graph | `client.rag` | `query_rag`, `add_document_to_rag`, `add_text_to_rag`, `query_sparql` (auto-switch by `store_alias`) |

> **Tools (MCP/LCP) are the exception**: `tools_binding_name` / `tools_binding_config` remain direct constructor parameters — the tools binding is intentionally *not* part of the two-tier profile system yet (see `_update_binding` bypass in `update_tools_binding`).

### Common Misconceptions (Verified Facts)

| Claim | Reality |
|---|---|
| "There is a `ModelProfile` dataclass" | ❌ The classes are `LollmsBindingProfile` and `LollmsModelProfile` (both defined in `lollms_core.py`) |
| "Profiles carry a `provider` or `service_key` field" | ❌ Connection credentials live inside `binding_config: Dict[str, Any]` (e.g. `{"host": ..., "service_key": ...}`) |
| "`lollms_bindings_utils.py` contains profile machinery" | ❌ It is a pure binding-discovery utility (`list_bindings`, `get_binding_desc`) |
| "You switch LLMs with `switch_llm`" | ❌ The method is `switch_model` (LLM); other modalities use `switch_tti`, `switch_tts`, … |
| "Bindings validate profiles via `_verify_profile_support`" | ❌ Bindings never see profiles — they receive merged plain constructor kwargs at instantiation time |
| "Profiles exist only for LLM/TTI/TTS/TTM" | ❌ All 8 modalities (incl. Connection & RAG) share the same two-tier system |

---

## 2. The Two Dataclasses

Both live in `src/lollms_client/lollms_core.py` and are importable from there:

```python
from lollms_client.lollms_core import LollmsBindingProfile, LollmsModelProfile
```

### 2.1 `LollmsBindingProfile` — Connection Layer

```python
@dataclass
class LollmsBindingProfile:
    """
    Declarative profile for a modality binding engine/server (Connection Layer).
    Defines the backend connection (e.g., host, api_key, binding library).
    """
    name: str
    binding_name: str
    binding_config: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | `str` | ✅ | Profile alias — the key used in the kwargs dict. Also stored on the object; the dict key wins on registration |
| `binding_name` | `str` | ✅ | Backend engine identifier, resolved by the modality's binding manager (e.g. `"llama_cpp_server"`, `"openai"`, `"diffusers"`, `"xtts"`, `"smart_router"`) |
| `binding_config` | `Dict[str, Any]` | optional | Connection parameters passed verbatim to the binding constructor: `host`, `port`, `service_key`, `verify_ssl_certificate`, `model_name`, … |
| `is_default` | `bool` | default `False` | Applies to **model** profiles primarily; binding profiles are referenced by name from model profiles |

**Design rule:** declare **one binding profile per backend server/endpoint**. If three models live on the same llama.cpp server, they should share one binding profile and differ only in `model_name`.

### 2.2 `LollmsModelProfile` — Execution Layer

```python
@dataclass
class LollmsModelProfile:
    """
    Declarative profile for a specific model (Execution Layer).
    References a binding profile and defines model-specific execution parameters.
    """
    name: str
    binding_profile_name: str
    model_name: Optional[str] = None
    is_default: bool = False
    vision_enabled: bool = False
    forced_context_size: Optional[int] = None
    routing_config: Optional[Dict[str, Any]] = None
    glm_image_embedding: bool = False
    supported_reasoning_efforts: Optional[List[str]] = None
    video_enabled: bool = False
    supported_reasoning_efforts: Optional[List[str]] = None   # (duplicate declaration — see note)
```

> ⚠️ **On-disk artifact**: `supported_reasoning_efforts` is declared **twice** in the source dataclass (a verbatim artifact). In Python the second declaration wins; behavior is identical to a single declaration. Do not "fix" this when writing profile code.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | required | Model alias — the key you pass to `switch_model(alias)` / `switch_tti(alias)` … |
| `binding_profile_name` | `str` | required | **Foreign key** into the same modality's binding-profile registry |
| `model_name` | `Optional[str]` | `None` | Model identifier on the backend. `None` → fall back to the server's default model (a startup warning is emitted for non-`master` LLM profiles) |
| `is_default` | `bool` | `False` | Eagerly instantiated at startup; the active model until you switch |
| `vision_enabled` | `bool` | `False` | Marks the model as vision-capable (VLM). Copied onto the instantiated binding; consulted by `has_vision_capability()` and the VLM fallback chain |
| `forced_context_size` | `Optional[int]` | `None` | Overrides the binding-reported context size |
| `routing_config` | `Optional[Dict[str, Any]]` | `None` | Routing metadata (used e.g. by the Smart Router meta-binding) |
| `glm_image_embedding` | `bool` | `False` | Enables GLM-style image-embedding handling |
| `supported_reasoning_efforts` | `Optional[List[str]]` | `None` | Restricts the reasoning-effort vocabulary for thinking models (e.g. `["low", "medium", "high"]`) |
| `video_enabled` | `bool` | `False` | Marks the model as video-input-capable; consulted by `has_video_capability()` |

**Which fields apply to which modality?** All fields exist on every profile object (uniform dataclass), but the instantiation pipeline only *applies* the LLM-oriented flags (`vision_enabled`, `forced_context_size`, `routing_config`, `glm_image_embedding`, `supported_reasoning_efforts`, `video_enabled`) to bindings that expose the corresponding attributes — for non-LLM modalities they are simply ignored unless the binding has the attribute (see §6).

---

## 3. Creating Profiles — Constructor Kwargs & Formats

### 3.1 The Sixteen Profile Kwargs

The constructor accepts one binding-profiles kwarg and one model-profiles kwarg **per modality**:

| Modality | Connection kwarg | Execution kwarg |
|---|---|---|
| LLM | `llm_binding_profiles` | `llm_model_profiles` |
| TTI | `tti_binding_profiles` | `tti_model_profiles` |
| TTS | `tts_binding_profiles` | `tts_model_profiles` |
| STT | `stt_binding_profiles` | `stt_model_profiles` |
| TTV | `ttv_binding_profiles` | `ttv_model_profiles` |
| TTM | `ttm_binding_profiles` | `ttm_model_profiles` |
| Connection | `connection_binding_profiles` | `connection_model_profiles` |
| RAG | `rag_binding_profiles` | `rag_model_profiles` |

Each kwarg accepts:

```python
Dict[str, Union[Dict[str, Any], LollmsBindingProfile]]   # binding profiles
Dict[str, Union[Dict[str, Any], LollmsModelProfile]]     # model profiles
```

i.e. a **dict keyed by alias**, whose values are either instantiated dataclasses or **plain dicts that are auto-converted** (§3.3).

### 3.2 Quick Start — Copy-Paste Example

```python
from lollms_client import LollmsClient
from lollms_client.lollms_core import LollmsBindingProfile, LollmsModelProfile

client = LollmsClient(
    llm_binding_profiles={
        "local_server": LollmsBindingProfile(
            name="local_server",
            binding_name="llama_cpp_server",
            binding_config={"host_address": "127.0.0.1", "port": 8080, "model_name": "qwen3-4b-q4_k_m.gguf"},
        ),
        "cloud": LollmsBindingProfile(
            name="cloud",
            binding_name="openai",
            binding_config={"host_address": "api.openai.com", "service_key": "<API-KEY>", "verify_ssl_certificate": True},
        ),
    },
    llm_model_profiles={
        "qwen_local": LollmsModelProfile(
            name="qwen_local",
            binding_profile_name="local_server",
            model_name="qwen3-4b-q4_k_m.gguf",
            is_default=True,
        ),
        "gpt_cloud": LollmsModelProfile(
            name="gpt_cloud",
            binding_profile_name="cloud",
            model_name="gpt-4o-mini",
            vision_enabled=True,
        ),
        "vlm_local": LollmsModelProfile(
            name="vlm_local",
            binding_profile_name="local_server",
            model_name="qwen3-vl-8b.gguf",
            vision_enabled=True,
        ),
    },
)

client.switch_model("vlm_local")            # lazily loads the VLM
client.generate_text(prompt="Hello!")       # uses the VLM now
```

### 3.3 Raw-Dict Auto-Conversion & Accepted Aliases

If you pass plain dicts, `_register_model_profiles` converts them using **lenient alias acceptance**:

| Canonical field | Accepted dict keys | Notes |
|---|---|---|
| `binding_profile_name` | `binding_profile_name` **or** `binding_alias` | Falls back to `"master"` when neither is present |
| `model_name` | `model_name` | `None` allowed (server default) |
| `is_default` | `is_default` | First-registered default wins; later duplicates are demoted |
| `vision_enabled` | `vision_enabled` | bool |
| `forced_context_size` | `forced_context_size` | int or None |
| `routing_config` | `routing_config` **or** `routing_profile` | Dict |
| `glm_image_embedding` | `glm_image_embedding` | bool |
| `supported_reasoning_efforts` | `supported_reasoning_efforts` **or** `reasoning_efforts` | **List** or comma-separated **string** (e.g. `"low,medium,high"` → parsed to a list) |
| `video_enabled` | `video_enabled` | bool |

For **binding profiles** passed as dicts, `_register_binding_profiles` accepts `binding_name`, `binding_config` and `is_default` — and additionally **hoists** the SSL keys `verify_ssl_certificate` / `certificate_file_path` from the top level of the dict into `binding_config` if they were not already there (a convenience for flattened configs).

The same profile as raw dicts:

```python
client = LollmsClient(
    llm_binding_profiles={
        "local_server": {
            "binding_name": "llama_cpp_server",
            "binding_config": {"host_address": "127.0.0.1", "port": 8080},
        },
    },
    llm_model_profiles={
        "qwen_local": {
            "binding_profile_name": "local_server",     # or "binding_alias"
            "model_name": "qwen3-4b-q4_k_m.gguf",
            "is_default": True,
            "supported_reasoning_efforts": "low,medium,high",  # comma-string OK
        },
    },
)
```

---

## 4. Registry Internals

After construction, the client holds **two registries per modality** — declarative configs, never engines:

| Modality | Binding-profile registry | Model-profile registry | Instance cache | Active alias |
|---|---|---|---|---|
| llm | `llm_binding_profiles_registry` | `llm_model_profiles_registry` | `llms` | `_active_llm_alias` |
| tti | `tti_binding_profiles_registry` | `tti_model_profiles_registry` | `ttis` | `_active_tti_alias` |
| tts | `tts_binding_profiles_registry` | `tts_model_profiles_registry` | `tts_bindings` | `_active_tts_alias` |
| stt | `stt_binding_profiles_registry` | `stt_model_profiles_registry` | `stts` | `_active_stt_alias` |
| ttv | `ttv_binding_profiles_registry` | `ttv_model_profiles_registry` | `ttvs` | `_active_ttv_alias` |
| ttm | `ttm_binding_profiles_registry` | `ttm_model_profiles_registry` | `ttms` | `_active_ttm_alias` |
| connection | `connection_binding_profiles_registry` | `connection_model_profiles_registry` | `connections` | `_active_connection_alias` |
| rag | `rag_binding_profiles_registry` | `rag_model_profiles_registry` | `rags` | `_active_rag_alias` |

Three kinds of state, cleanly separated:

- **`*_binding_profiles_registry`** — `Dict[str, LollmsBindingProfile]`. Pure configuration; never instantiated on its own.
- **`*_model_profiles_registry`** — `Dict[str, LollmsModelProfile]`. Pure configuration.
- **`{llms, ttis, tts_bindings, stts, ttvs, ttms, connections, rags}`** — **instance caches**: alias → live binding object. Populated lazily by `_switch_modality` / `find_available_vlm`.

The active binding of each modality is additionally surfaced on the convenience attributes `client.llm`, `client.tti`, `client.tts`, `client.stt`, `client.ttv`, `client.ttm`, `client.connection`, `client.rag`.

Registration order: binding profiles are registered first (with legacy fallback), then model profiles (which may reference any binding profile), then the default model is eagerly instantiated per modality.

---

## 5. Switching Models & Modalities

### 5.1 The Switch API

Every modality exposes the same `switch_*(alias, callback=None) -> bool` signature:

```python
client.switch_model(alias, callback=None)       # LLM
client.switch_tti(alias, callback=None)         # text→image
client.switch_tts(alias, callback=None)         # text→speech
client.switch_stt(alias, callback=None)         # speech→text
client.switch_ttv(alias, callback=None)         # text→video
client.switch_ttm(alias, callback=None)         # text→music
client.switch_connection(alias, callback=None)  # outbound channel
client.switch_rag(alias, callback=None)         # RAG store
```

All delegate to the generic engine:

```python
_switch_modality(alias, model_registry, binding_registry, instance_cache,
                 manager, modality, attr_name, active_alias_attr, callback=None) -> bool
```

**What `_switch_modality` does, step by step:**

1. **Validates** the alias against the model-profile registry — unknown alias → error log listing available aliases, `return False`.
2. **Cache lookup**: if the alias already lives in the instance cache, mount it instantly (no re-instantiation, no server round-trips).
3. **Lazy instantiation**: otherwise call `_instantiate_binding_from_profile` (§6); failure → `return False`.
4. **Cache + mount**: store in the instance cache, set the modality attribute (`client.llm` etc.).
5. **LLM housekeeping** (`modality == "llm"` only): reset the remote-tokenizer health flag and clear the context-size cache.
6. **Local resource guard**: if the binding `is_local()`, call `ensure_model_loaded(binding, model_name)` — which may trigger cooperative VRAM reclamation across other local bindings (§8.1.4).
7. **Record the active alias** (`_active_llm_alias` etc.) and log the switch.

### 5.2 Legacy Mount Aliases

For backward compatibility, `mount_*` methods simply delegate:

| Legacy call | Delegates to |
|---|---|
| `mount_llm(alias)` | `switch_model(alias)` |
| `mount_tti(alias)` | `switch_tti(alias)` |
| `mount_tts(alias)` | `switch_tts(alias)` |
| `mount_stt(alias)` | `switch_stt(alias)` |
| `mount_ttv(alias)` | `switch_ttv(alias)` |
| `mount_ttm(alias)` | `switch_ttm(alias)` |
| `mount_connection(alias)` | `switch_connection(alias)` |
| `mount_rag(alias)` | `switch_rag(alias)` |

> The names say "mount" but the semantics are identical to switching — they exist purely so older code keeps working.

### 5.3 Updating a Binding (Legacy Recreate Path)

```python
client.update_llm_binding(binding_name, config=None)
client.update_tts_binding(binding_name, config=None)
client.update_tti_binding(binding_name, config=None)
client.update_stt_binding(binding_name, config=None)
client.update_ttv_binding(binding_name, config=None)
client.update_ttm_binding(binding_name, config=None)
client.update_connection_binding(binding_name, config=None)
client.update_rag_binding(binding_name, config=None)
```

`_update_binding` **recreates the `"master"` binding profile** with the given name/config, ensures a `"master"` model profile points at it, evicts the cached instance, and switches to it. Use this for runtime re-pointing of a single-server setup (e.g. a GUI settings dialog).

Exception: `update_tools_binding(binding_name, config)` bypasses profiles entirely and directly re-creates the tools binding (tools are not yet two-tier).

---

## 6. The Instantiation Pipeline

`_instantiate_binding_from_profile(alias, model_profile, manager, modality, callback=None)` turns a declarative model profile into a live binding:

```
model_profile (LollmsModelProfile)
        │
        ▼ binding_profile_name
binding_registry lookup ──► LollmsBindingProfile
        │                     (missing → error, return None)
        ▼
copy binding_config
        │
        ├─ string→bool coercion of "verify_ssl_certificate"
        ├─ modality != "llm" → b_config.setdefault("wait_for_server", False)
        │    (secondary modalities must not block the main thread on spawn)
        ├─ model_profile.model_name → b_config["model_name"]
        │    └─ modality == "rag" → also db_path / store_name defaults
        ├─ modality == "llm" → inject user_name / ai_name
        ├─ modality == "rag" → inject lollms_client=self
        ├─ debug, system_dir, cwd injection
        ▼
manager.create_binding(binding_name=binding_profile.binding_name, **b_config)
        │
        ▼
post-apply execution flags onto the live binding:
  vision_enabled, video_enabled, forced_context_size, routing_config,
  glm_image_embedding, supported_reasoning_efforts
```

**Key contract — bindings never see profiles.** The binding constructor receives *merged plain kwargs*; there is no `_verify_profile_support`-style API. Matching is purely by the `binding_name` string at `create_binding` time. This keeps bindings decoupled from the profile system entirely.

**Flag application is attribute-gated:** `forced_context_size` and `routing_config` are only applied when the binding already has the attribute (`hasattr` check); `vision_enabled`, `video_enabled`, `glm_image_embedding`, and `supported_reasoning_efforts` use `getattr` fallbacks and will `setattr` when absent. Practically: declare vision/video/reasoning flags on LLM model profiles; for other modalities they are inert unless the binding exposes them.

> ⚠️ **Source artifact note:** the on-disk `_instantiate_binding_from_profile` contains a copy-paste-duplicated block (the SSL-sanitize / `wait_for_server` / model_name-injection segment appears three times). It is harmless — the later copies overwrite the same keys with identical values — but be aware when reading the code.

### What `binding_config` keys mean per binding

Each binding's constructor signature defines its accepted `binding_config` keys (see the binding's `description.yaml` or `binding_config.py` via `get_binding_desc()` in `lollms_bindings_utils.py`). Common patterns:

| Binding family | Typical `binding_config` keys |
|---|---|
| `llama_cpp_server` | `host_address`, `port`, `model_name`, `verify_ssl_certificate`, `server_exe_path` … |
| `openai` / cloud APIs | `host_address`, `service_key`, `model_name`, `verify_ssl_certificate` |
| `diffusers` (TTI/TTV/TTM) | `host`, `port`, `auto_start_server`, `model_name` |
| `xtts` / `piper_tts` (TTS) | `host`, `port`, `auto_start_server`, `voice` … |
| `whisper` (STT) | `host`, `port`, `model_size` … |
| `smart_router` | `model_profiles` (child dicts) + routing strategy, see §8.1.2 |

---

## 7. Default Resolution & the Single-Default Invariant

At startup, `_eagerly_instantiate_default` runs for every modality and enforces **exactly one default**:

1. **Multiple defaults?** The *first* registered default wins; the rest are demoted (`is_default = False`) with a warning listing the choice.
2. **Default selection precedence:**
   - the profile with `is_default=True` → else
   - the alias `"master"` (the legacy compatibility profile) → else
   - the **first alias in the registry** — auto-promoted to default (and the promotion is *persisted to disk* so the warning never repeats).
3. **Missing model names:** LLM model profiles without a `model_name` (other than `"master"`) trigger a startup warning — they will fall back to the server's default model.

**Single-default invariant enforcement points:**

| Stage | Mechanism |
|---|---|
| Registration (`_register_model_profiles`) | First `is_default=True` wins; subsequent defaults are silently demoted |
| Startup (`_eagerly_instantiate_default`) | Demotes multiples with a warning, auto-promotes a default when none exists |
| Persistence (`lollms_config_api.set_default_profile`) | Exclusive setter — sets the chosen default and clears all others |
| Auto-repair (`_save_single_default_profiles_to_disk`) | Writes the resolved single default back to YAML/env files so warnings never repeat |

---

## 8. Per-Modality Guide

### 8.1 LLM — Text Generation

LLM is the richest modality: it is the default generation engine, hosts vision/VLM routing, reasoning-effort control, and the Smart Router meta-binding.

#### 8.1.1 Basic Multi-Model Setup

```python
client = LollmsClient(
    llm_binding_profiles={
        "local": {
            "binding_name": "llama_cpp_server",
            "binding_config": {"host_address": "127.0.0.1", "port": 8080},
        },
    },
    llm_model_profiles={
        "coder": {
            "binding_profile_name": "local",
            "model_name": "qwen3-coder-30b.gguf",
            "is_default": True,
        },
        "fast": {
            "binding_profile_name": "local",
            "model_name": "qwen3-4b-q4_k_m.gguf",
        },
        "vlm": {
            "binding_profile_name": "local",
            "model_name": "qwen3-vl-8b.gguf",
            "vision_enabled": True,
        },
    },
)
```

All three profiles share the `"local"` connection; only `"coder"` loads at startup. Switching is instant after first load (instance cache).

#### 8.1.2 Smart Router (Cost/Quality/Latency Routing)

The Smart Router is a **meta-binding** (`binding_name="smart_router"`): its own binding profile holds child model profiles, and a routing strategy picks the best child per prompt.

```python
client = LollmsClient(
    llm_binding_profiles={
        "router": {
            "binding_name": "smart_router",
            "binding_config": {
                "strategy": "balanced",
                "model_profiles": {
                    "small_fast": {
                        "binding_name": "llama_cpp_server",
                        "binding_config": {"host_address": "127.0.0.1", "port": 8080},
                        "model_name": "qwen3-4b-q4_k_m.gguf",
                        "routing_profile": {
                            "description": "Small generalist, low latency",
                            "complexity_tier": 1,
                            "cost_per_1k_tokens": 0.0,
                            "avg_latency_ms": 120,
                            "priority": 2,
                        },
                    },
                    "big_quality": {
                        "binding_name": "openai",
                        "binding_config": {"host_address": "api.openai.com", "service_key": "<API-KEY>"},
                        "model_name": "gpt-4o-mini",
                        "routing_profile": {
                            "description": "High quality generalist",
                            "complexity_tier": 3,
                            "cost_per_1k_tokens": 0.15,
                            "avg_latency_ms": 800,
                            "priority": 1,
                        },
                    },
                },
            },
        },
    },
    llm_model_profiles={
        "smart": {
            "binding_profile_name": "router",
            "model_name": None,
            "is_default": True,
        },
    },
)
```

Routing strategy weights (subject match → complexity → cost/latency):

| Strategy | Subject match | Complexity | Cost | Latency |
|---|---|---|---|---|
| `balanced` | 0.35 | 0.35 | 0.15 | 0.15 |
| `cost_optimized` | 0.20 | 0.20 | 0.50 | 0.10 |
| `quality_optimized` | 0.40 | 0.40 | 0.00 | 0.20 |

Selection: hard vision filter → TF-IDF subject match vs child descriptions → complexity-tier match → weighted score → priority tie-break → graceful degradation to the first available child.

#### 8.1.3 Vision, VLM Fallback & Reasoning Efforts

- **`vision_enabled: True`** marks a profile as a VLM. `has_vision_capability()` checks (in order): binding flags → `models_dir`/`_find_mmproj` probe (llama.cpp-style local bindings) → router child bindings → the **active model profile's** `vision_enabled` flag.
- **Automatic VLM fallback:** if the active model lacks vision and an image is passed, `_sanitize_images_for_active_llm` strips the image and substitutes a **text description generated by another VLM** found via `find_available_vlm()` — which searches (1) router children, (2) the instantiated `llms` cache, (3) declared-but-not-yet-instantiated vision profiles (lazily instantiating them, caching into `llms[alias]`). Descriptions are cached by image hash in `_image_description_cache`.
- **`supported_reasoning_efforts`** constrains the reasoning-effort vocabulary for thinking models (e.g. `["low", "medium", "high"]`, or `"low,medium,high"` in raw dicts). `generate_text` / `generate_from_messages` normalize `think` / `reasoning_effort` kwargs against it.

#### 8.1.4 Cooperative VRAM Management

Pass `cooperative_vram_management=True` (or rely on the local resource guard) and the client coordinates local bindings: `free_local_binding_resources(except_binding=...)` asks other local bindings with active resources to `unload_model()`, then retries the load (`ensure_model_loaded`). `_cooperative_unload_except(active_modality)` runs before each modality call — so generating an image can temporarily evict the LLM (and vice versa) on VRAM-constrained machines.

### 8.2 TTI — Text→Image

```python
client = LollmsClient(
    llm_binding_name="openai",                              # whatever your main LLM is
    llm_binding_config={"service_key": "<API-KEY>"},
    tti_binding_profiles={
        "local_sd": {
            "binding_name": "diffusers",
            "binding_config": {"host": "127.0.0.1", "port": 14211, "auto_start_server": True},
        },
    },
    tti_model_profiles={
        "sdxl": {
            "binding_profile_name": "local_sd",
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "is_default": True,
        },
        "flux": {
            "binding_profile_name": "local_sd",
            "model_name": "black-forest-labs/FLUX.1-schnell",
        },
    },
)
client.switch_tti("flux")
client.generate_image(prompt="a lighthouse at dawn")
```

Notes:

- The diffusers TTI server may run on its own port (e.g. 14211) — declared per binding profile.
- Personality-based apps (lollms-code, handbag agents) expose TTI tools that lazily switch to the **registry default** TTI profile on first use (see §11).
- `install_model(model_name, modality="tti")` / `pull_model(...)` delegate to the binding's model download support.

### 8.3 TTS — Text→Speech

```python
client = LollmsClient(
    tts_binding_profiles={
        "xtts_local": {
            "binding_name": "xtts",
            "binding_config": {"host": "127.0.0.1", "port": 8020, "auto_start_server": True},
        },
    },
    tts_model_profiles={
        "voice_main": {
            "binding_profile_name": "xtts_local",
            "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
            "is_default": True,
        },
    },
)
client.generate_audio(prompt="Hello from LOLLMS!", voice="main")
```

Multiple voices or engines → multiple model profiles sharing one binding profile (voice choice happens at call time via the binding's own `voice` parameter; the profile pins the engine).

### 8.4 TTM — Text→Music

```python
client = LollmsClient(
    ttm_binding_profiles={
        "local_ttm": {
            "binding_name": "diffusers",
            "binding_config": {"host": "127.0.0.1", "port": 9637, "auto_start_server": True},
        },
    },
    ttm_model_profiles={
        "minimax_music": {
            "binding_profile_name": "local_ttm",
            "model_name": "MiniMaxAI/MiniMax-Music3",
            "is_default": True,
        },
    },
)
client.generate_music(prompt="upbeat lo-fi beat with vinyl crackle", duration=30)
client.generate_song_from_lyrics(prompt="a ballad about the sea", lyrics="...", duration=60)
```

TTM tools in personality apps use the **active** `client.ttm` binding directly (no per-call alias parameter), but `client.switch_ttm(alias)` fully supports hot-swapping music models at runtime.

### 8.5 STT — Speech→Text

```python
client = LollmsClient(
    stt_binding_profiles={
        "whisper_local": {
            "binding_name": "whisper",
            "binding_config": {"host": "127.0.0.1", "port": 8021, "auto_start_server": True},
        },
    },
    stt_model_profiles={
        "whisper_medium": {
            "binding_profile_name": "whisper_local",
            "model_name": "medium",
            "is_default": True,
        },
    },
)
client.transcribe_audio(audio_file_path="meeting.wav")
```

### 8.6 TTV — Text→Video

```python
client = LollmsClient(
    ttv_binding_profiles={
        "local_ttv": {
            "binding_name": "diffusers",
            "binding_config": {"host": "127.0.0.1", "port": 14212, "auto_start_server": True},
        },
    },
    ttv_model_profiles={
        "svd": {
            "binding_profile_name": "local_ttv",
            "model_name": "stabilityai/stable-video-diffusion-img2vid-xt",
            "is_default": True,
        },
    },
)
client.generate_video(prompt="waves on a rocky shore")
```

### 8.7 Connection — Outbound Messaging

Connection profiles represent **channels** (e.g. webhook endpoints). The helper `send_connection_message` **auto-switches** when you pass a `channel_alias`:

```python
client = LollmsClient(
    connection_binding_profiles={
        "webhook_prod": {
            "binding_name": "generic_webhook",
            "binding_config": {"url": "https://example.com/hook"},
        },
    },
    connection_model_profiles={
        "alerts": {
            "binding_profile_name": "webhook_prod",
            "is_default": True,
        },
    },
)
client.send_connection_message("Deploy finished", channel_alias="alerts")
```

If `channel_alias` differs from the active alias, `switch_connection(channel_alias)` runs first — no manual switching needed.

### 8.8 RAG — Retrieval & Knowledge Graphs

RAG model profiles represent **stores**. The `model_name` doubles as the store identity: at instantiation, it is injected both as `model_name` and as default `db_path` / `store_name`. Helper methods auto-switch by `store_alias`:

```python
client = LollmsClient(
    rag_binding_profiles={
        "safe_store_local": {
            "binding_name": "safe_store",
            "binding_config": {},
        },
    },
    rag_model_profiles={
        "docs": {
            "binding_profile_name": "safe_store_local",
            "model_name": "project_docs",
            "is_default": True,
        },
        "archive": {
            "binding_profile_name": "safe_store_local",
            "model_name": "old_archive",
        },
    },
)
client.add_document_to_rag("manual.pdf", store_alias="docs")
results = client.query_rag("installation steps", top_k=5, store_alias="docs")
client.query_sparql("SELECT ?s WHERE { ?s ?p ?o } LIMIT 10", store_alias="archive")
```

> RAG requires the optional `safe_store` dependency; if unavailable, the RAG manager is `None` and RAG profiles are inert (graceful degradation).

---

## 9. Persistence Formats

Host applications persist profiles through `lollms_config_api.py`, which serializes to a **flat env-style map** and a **Two-Tier YAML** structure.

### 9.1 Flat Env-Style Map (Primary Format)

Keys are `UPPERCASE` modality-prefixed:

```
LLM_BINDINGS_<ALIAS>_BINDING_NAME
LLM_BINDINGS_<ALIAS>_<PARAM>                  # flattened binding_config entries
LLM_PROFILES_<ALIAS>_BINDING_ALIAS
LLM_PROFILES_<ALIAS>_MODEL_NAME
LLM_PROFILES_<ALIAS>_INSTANCE_NAME
LLM_PROFILES_<ALIAS>_IS_DEFAULT
LLM_PROFILES_<ALIAS>_VISION_ENABLED
LLM_PROFILES_<ALIAS>_VIDEO_ENABLED
LLM_PROFILES_<ALIAS>_GLM_IMAGE_EMBEDDING
LLM_PROFILES_<ALIAS>_SUPPORTED_REASONING_EFFORTS
LLM_PROFILES_<ALIAS>_FORCED_CONTEXT_SIZE
LLM_PROFILES_<ALIAS>_ROUTING_<KEY>            # flattened routing_config entries
```

The same pattern repeats with `TTS_`, `TTI_`, `STT_`, `TTV_`, `TTM_`, `CONNECTION_`, `RAG_` prefixes.

Example `.env` lines:

```env
LLM_BINDINGS_LOCAL_BINDING_NAME=llama_cpp_server
LLM_BINDINGS_LOCAL_HOST_ADDRESS=127.0.0.1
LLM_BINDINGS_LOCAL_PORT=8080
LLM_PROFILES_QWEN_BINDING_ALIAS=local
LLM_PROFILES_QWEN_MODEL_NAME=qwen3-4b-q4_k_m.gguf
LLM_PROFILES_QWEN_IS_DEFAULT=true
LLM_PROFILES_VLM_VISION_ENABLED=true
LLM_PROFILES_VLM_MODEL_NAME=qwen3-vl-8b.gguf
TTM_BINDINGS_LOCAL_TTM_BINDING_NAME=diffusers
TTM_BINDINGS_LOCAL_TTM_PORT=9637
TTM_PROFILES_MINIMAX_BINDING_ALIAS=local_ttm
TTM_PROFILES_MINIMAX_MODEL_NAME=MiniMaxAI/MiniMax-Music3
TTM_PROFILES_MINIMAX_IS_DEFAULT=true
```

### 9.2 Two-Tier YAML

```yaml
llm:
  bindings:
    local:
      binding_name: llama_cpp_server
      binding_config:
        host_address: 127.0.0.1
        port: 8080
  profiles:
    qwen:
      binding_profile_name: local
      model_name: qwen3-4b-q4_k_m.gguf
      is_default: true
    vlm:
      binding_profile_name: local
      model_name: qwen3-vl-8b.gguf
      vision_enabled: true

ttm:
  bindings:
    local_ttm:
      binding_name: diffusers
      binding_config:
        host: 127.0.0.1
        port: 9637
        auto_start_server: true
  profiles:
    minimax:
      binding_profile_name: local_ttm
      model_name: MiniMaxAI/MiniMax-Music3
      is_default: true
```

### 9.3 `lollms_config_api` Function Reference

| Function | Purpose |
|---|---|
| `load_config_map(conf_file, conf_dict, global_config, local_env)` | Loads the flat map from YAML/JSON/`.env` sources with precedence |
| `get_binding_params(map, modality, alias)` / `get_profile_params(...)` | Reads one binding/model profile's entries |
| `set_binding_params(map, modality, alias, binding_name, params)` | Writes a binding profile |
| `set_profile_params(map, modality, alias, binding_alias, model_name, instance_name, is_default, vision_enabled, video_enabled, glm_image_embedding, supported_reasoning_efforts, forced_context_size, routing_config)` | Writes a model profile |
| `set_default_profile(...)` | **Exclusive** default setter (clears other defaults — the invariant) |
| `delete_binding(...)` / `delete_profile(...)` | Removes entries; deleting the default auto-promotes the next one |
| `save_config_map(map, ...)` | Serializes to yaml / json / env |
| `build_client_config(map, **modalities)` | Map → constructor kwargs (`{modality}_binding_profiles`, `{modality}_model_profiles`, plus legacy `{modality}_binding_name`/`{modality}_binding_config`) |
| `build_client_from_map(...)` | Map → ready `LollmsClient` |

### 9.4 Discovering Available Bindings

Before you can declare a `LollmsBindingProfile`, you need to know **which bindings exist for a given modality** and **what parameters each one accepts**. The client ships with discovery utilities for exactly this.

#### 9.4.1 Low-Level: `list_bindings` (from `lollms_bindings_utils`)

```python
from lollms_client.lollms_bindings_utils import list_bindings, BINDING_TYPES
```

| Function | Signature | Returns |
|---|---|---|
| `list_bindings` | `(binding_type: str = "llm", custom_bindings_dir: Union[Path, str, None] = None) -> List[str]` | **Sorted list of binding folder-name strings** (e.g. `["llama_cpp_server", "openai", "ollama", …]`). Only directories containing an `__init__.py` are included. Returns `[]` if the folder is absent. |

`custom_bindings_dir` overrides the default `src/lollms_client/<folder>/` lookup — useful for out-of-tree binding packs.

**The `BINDING_TYPES` mapping** (modality key → folder name inside `src/lollms_client/`):

| Modality key | Folder scanned |
|---|---|
| `"llm"` | `llm_bindings/` |
| `"tti"` | `tti_bindings/` |
| `"stt"` | `stt_bindings/` |
| `"tts"` | `tts_bindings/` |
| `"ttm"` | `ttm_bindings/` |
| `"ttv"` | `ttv_bindings/` |
| `"mcp"` | `tools_bindings/` |
| `"rag"` | `rag_bindings/` |

**Example:**

```python
# All installed LLM bindings
llm_backends = list_bindings("llm")
# → ["azure_openai", "gemini", "grok", "groq", "hugging_face_inference_api",
#    "litellm", "llama_cpp_server", "lollms", "lollms_webui", "mistral",
#    "novita_ai", "ollama", "open_router", "openai", "openllm", "openwebui",
#    "perplexity", "sglang", "smart_router", "tensor_rt", "transformers", "vllm"]

# All TTM (music) bindings
music_backends = list_bindings("ttm")
# → ["audiocraft", "beatoven_ai", "diffusers", "lollms", "replicate",
#    "stability_ai", "suno_ttm", "topmediai"]

# A custom bindings directory
custom = list_bindings("llm", custom_bindings_dir="/opt/my-custom-bindings/llm")
```

> **Connection** is special-cased: it is not in `BINDING_TYPES`. Use `lollms_client.lollms_connection_binding.get_available_bindings()` to list connection channels.

#### 9.4.2 High-Level: `discover_bindings` (from `lollms_config_api`)

```python
from lollms_client.lollms_config_api import discover_bindings
```

| Function | Signature | Returns |
|---|---|---|
| `discover_bindings` | `(modality: str) -> List[Dict[str, Any]]` | **Deduped list of `{"name": "<binding_name>"}` dicts**. Validates `modality` against `SUPPORTED_MODALITIES = ("llm", "tti", "tts", "stt", "ttm", "ttv", "connection")`. Returns `[]` on any error. |

This is the **UI-ready** wrapper: it calls `list_bindings` under the hood (or `get_available_bindings()` for `connection`), normalizes the result to a list of dicts, and is the function the CLI/GUI use to populate binding dropdowns.

**Example:**

```python
available = discover_bindings("stt")
# → [{"name": "lollms"}, {"name": "whisper"}, {"name": "whispercpp"}]

for entry in available:
    print(f"  {entry['name']}")
```

> **Tip**: use `discover_bindings` when building dynamic UIs or scripts that iterate over all backends; use `list_bindings` when you just need a plain sorted list of strings.

---

### 9.5 Inspecting a Binding's Configuration Parameters

Once you know which binding to use, you need to know **which `binding_config` keys it accepts**. Two functions cover this:

#### 9.5.1 `get_binding_schema` — UI-Ready Parameter List (from `lollms_config_api`)

```python
from lollms_client.lollms_config_api import get_binding_schema
```

| Function | Signature | Returns |
|---|---|---|
| `get_binding_schema` | `(binding_name: str, modality: str) -> List[Dict[str, Any]]` | **List of parameter dicts**, each with exactly: `{"name", "type", "mandatory", "default", "description"}` |

**Behavior:**
- Calls `get_binding_desc(binding_name, modality)` internally (see §9.5.2) and merges `global_input_parameters` + `model_input_parameters`.
- **Excludes `model_name`** (that field belongs to the Execution Layer / `LollmsModelProfile`, not to `binding_config`).
- **De-duplicates** by parameter name.
- **Fallback**: if no description is found, returns a minimal 3-field schema: `host_address` (str, default `http://localhost:9642`), `service_key` (str, default `""`), `verify_ssl_certificate` (bool, default `False`).

**Worked example — building a Whisper STT binding profile:**

```python
from lollms_client.lollms_config_api import discover_bindings, get_binding_schema
from lollms_client.lollms_core import LollmsBindingProfile, LollmsModelProfile

# Step 1: discover available STT bindings
stt_backends = discover_bindings("stt")
print(stt_backends)
# → [{"name": "lollms"}, {"name": "whisper"}, {"name": "whispercpp"}]

# Step 2: inspect whisper's parameters
schema = get_binding_schema("whisper", "stt")
for param in schema:
    required = " *" if param["mandatory"] else ""
    print(f"  {param['name']}{required}  ({param['type']})  "
          f"default={param['default']!r}  — {param['description']}")

# Step 3: build the binding profile from the discovered params
whisper_config = {
    "host": "127.0.0.1",
    "port": 9633,
    "auto_start_server": True,
}
client = LollmsClient(
    stt_binding_profiles={
        "whisper_local": LollmsBindingProfile(
            name="whisper_local",
            binding_name="whisper",
            binding_config=whisper_config,
        ),
    },
    stt_model_profiles={
        "whisper_medium": LollmsModelProfile(
            name="whisper_medium",
            binding_profile_name="whisper_local",
            model_name="medium",
            is_default=True,
        ),
    },
)
```

#### 9.5.2 `get_binding_desc` — Low-Level Description Loader (from `lollms_bindings_utils`)

```python
from lollms_client.lollms_bindings_utils import get_binding_desc
```

| Function | Signature | Returns |
|---|---|---|
| `get_binding_desc` | `(binding_name: str, binding_type: str) -> Dict` | The parsed description: either the return value of `binding_config.py::get_binding_desc()` (preferred), or the YAML content of `description.yaml`. Returns `{"error": "..."}` if neither exists or the binding is not found. |

**Resolution order (verbatim):**

1. If `<binding_dir>/binding_config.py` exists and exposes a callable `get_binding_desc()`, import and call it (cached in `sys.modules`; exceptions fall through).
2. Otherwise, read `<binding_dir>/description.yaml` via `yaml.safe_load`.
3. If neither exists: return `{"error": "No description found …"}`.

**The `description.yaml` format** (illustrated with `stt_bindings/whisper/description.yaml`):

```yaml
title: Whisper STT Binding
author: ParisNeo
description: >
  A resilient Speech-To-Text (STT) binding with a shared server daemon.

global_input_parameters:
  - name: host
    type: str
    description: "The host address for the shared Whisper server."
    mandatory: false
    default: "127.0.0.1"
  - name: port
    type: int
    description: "The port on which the shared Whisper server daemon runs."
    mandatory: false
    default: 9633
  - name: auto_start_server
    type: bool
    description: "Automatically spawn the shared server daemon if not running."
    mandatory: false
    default: true
  - name: service_key
    type: str
    description: "Optional API key for remote Whisper servers."
    mandatory: false
    default: ""

model_input_parameters:
  - name: model_name
    type: str
    description: "The Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large')."
    mandatory: false
    default: "base"

commands:
  - name: status
    title: Daemon Status
    description: "Show whether the shared Whisper daemon is running."
    parameters: []
    output:
      - name: is_running
        type: bool
```

Key points:

- **`global_input_parameters`** = connection/server parameters → go into `LollmsBindingProfile.binding_config`.
- **`model_input_parameters`** = per-model parameters → `model_name` is excluded from `get_binding_schema` output because it belongs to `LollmsModelProfile.model_name`, not `binding_config`.
- **`commands`** = optional daemon commands (used by host apps for status/management UIs); not relevant to profile construction.

> **Cross-reference**: §6 "What `binding_config` keys mean per binding" provides a hand-written summary of common parameter patterns across binding families. The discovery API above is the *authoritative, machine-readable* source — always prefer it over hard-coded key lists.

#### 9.5.3 There Is No `validate` / `build_config` Helper

There is **no** dedicated `validate_user_config(...)` or `build_binding_config(...)` convenience function. The intended workflow is:

1. `discover_bindings(modality)` → pick a binding name.
2. `get_binding_schema(binding_name, modality)` → collect user input per field (UI form or manual dict).
3. Pass the resulting dict as `binding_config` on a `LollmsBindingProfile`.
4. Persist via `set_binding_params` (§9.3).

---

### 9.6 Where Defaults Get Auto-Persisted

`_save_single_default_profiles_to_disk` writes resolved single defaults back to (first existing wins, `system_dir` prioritized):

- `<system_dir>/config.yaml`, `~/.lollms_client/config.yaml`, `<cwd>/.lollms_code/config.yaml`
- `<system_dir>/.env`, `~/.lollms_client/.env`, `<cwd>/.lollms_code/.env`

YAML: phantom `master` profiles without a `model_name` are purged; the chosen default is set `is_default: true`; demoted/others set `false`. `.env`: `<MODALITY>_PROFILES_<ALIAS>_IS_DEFAULT=true` lines for demoted aliases are flipped to `false`.

> 🔐 **Security note**: keep `service_key` values in environment variables or untracked `.env` files — never commit populated YAML with credentials.

---

## 10. Host-App Integration

### 10.1 lollms-code CLI (`apps/lollms_code/cli.py`)

- `CodeAgentConfig` extracts profiles for **all 8 modalities** from env/yaml sources; preferences-only `config.json` explicitly ignores profile keys.
- `create_client(config)` passes `llm_binding_profiles` + `llm_model_profiles` when both exist; otherwise it falls back to legacy `llm_binding_name` + `llm_binding_config`.
- TTI/TTS/STT (and other secondary modalities) are mounted via the legacy `{modality}_binding_name` + `{modality}_binding_config` path resolved from config/env.
- The `--profile` CLI flag marks the chosen LLM profile `is_default` at startup.

### 10.2 GUI Bridge

The GUI configuration bridge converts its settings model into the same flat map (§9.1) and builds the client via `build_client_config` / `build_client_from_map`.

### 10.3 Programmatic Round-Trip

```python
from lollms_client.lollms_config_api import (
    load_config_map, set_binding_params, set_profile_params,
    set_default_profile, build_client_config, build_client_from_map,
)

cfg_map = load_config_map(conf_file="profiles.yaml")
set_binding_params(cfg_map, "llm", "local", "llama_cpp_server",
                   {"host_address": "127.0.0.1", "port": 8080})
set_profile_params(cfg_map, "llm", "qwen", binding_alias="local",
                   model_name="qwen3-4b-q4_k_m.gguf", is_default=True)
client_kwargs = build_client_config(cfg_map, "llm", "tti", "tts", "ttm")
client = LollmsClient(**client_kwargs)
```

---

## 11. Consumer Tools & Agents

### 11.1 `models_tools` LCP Library (Runtime Model Switching for Agents)

Gated by `TOOL_LIBRARY_LOAD_FLAG = "enable_auto_model_selection"`. When enabled, agents get:

| Tool | Signature | Uses |
|---|---|---|
| `tool_list_models` | `(lollms_client_instance=None)` | `llm_model_profiles_registry`, `llms` cache, `_active_llm_alias` |
| `tool_switch_model` | `(model_alias, lollms_client_instance=None)` | `client.switch_model(alias)` |
| `tool_spawn_child_agent` | `(task, model_alias, system_prompt="", agent_name="", n_predict=None, ...)` | `_instantiate_binding_from_profile(alias, profile, llm_binding_manager, "llm")` into the `llms` cache; child runs on a daemon thread |
| `tool_get_child_agent_result` | `(agent_id, remove_if_done=True)` | Child-agent registry (capped, purges finished) |

### 11.2 Personality Tool Closures

- **TTI tools** lazily resolve the **registry default** TTI profile on first use and `switch_tti(default_alias)` — so a personality app works even when only one TTI profile is declared.
- **Connection tools** do the same for the default connection profile (`switch_connection`).
- **RAG tools** (`tool_query_rag`, `tool_sparql_query`, `tool_add_document_to_rag`, `tool_get_rag_info`) accept `store_alias` and rely on the client's auto-switch.
- **TTM tools** (`tool_generate_music`, `tool_generate_song`) use the active `client.ttm` directly.

### 11.3 Sub-Agents

`SubAgentSpawner.spawn` runs a child agent on a target model alias and **restores the original alias** via `client.switch_model(orig_alias)` afterward — the parent's active model is never silently changed.

### 11.4 Legacy `ModelSwitcher` — Don't Confuse

The personality's `ModelSwitcher.switch_model` is the **legacy in-binding switcher**: it calls `llm.unload_model()` / `llm.load_model()` or sets `llm.model_name` on the *current binding*. It does **not** use the profile system. For multi-model setups, prefer `client.switch_model(alias)`.

---

## 12. Migration & Legacy Compatibility

Legacy parameters are auto-registered into the two-tier system — nothing breaks:

| Legacy input | Auto-registered as |
|---|---|
| `llm_binding_name="llama_cpp_server"`, `llm_binding_config={...}` | Binding profile `"master"` + model profile `"master"` (config's `model_name` carried over) |
| `extra_llms={"alias": {"binding_name": ..., "binding_config": {...}}}` | Model profiles referencing an auto-created `"legacy_master_binding"` binding profile |

**Migration recipe:**

1. Replace `llm_binding_name`/`llm_binding_config` with `llm_binding_profiles` + `llm_model_profiles`.
2. Give each server a named binding profile; give each model its own model profile referencing it.
3. Mark exactly one model profile `is_default=True`.
4. Replace `client.switch_active_model(name)` calls with `client.switch_model(alias)`.
5. Optional: persist via `lollms_config_api` (§9) so restarts restore the same layout.

Legacy `mount_llm` / `update_llm_binding` keep working throughout (§5.2, §5.3).

---

## 13. API Quick Reference

### Dataclasses (verbatim from `lollms_core.py`)

```python
@dataclass
class LollmsBindingProfile:
    name: str
    binding_name: str
    binding_config: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False

@dataclass
class LollmsModelProfile:
    name: str
    binding_profile_name: str
    model_name: Optional[str] = None
    is_default: bool = False
    vision_enabled: bool = False
    forced_context_size: Optional[int] = None
    routing_config: Optional[Dict[str, Any]] = None
    glm_image_embedding: bool = False
    supported_reasoning_efforts: Optional[List[str]] = None
    video_enabled: bool = False
    supported_reasoning_efforts: Optional[List[str]] = None
```

### Constructor kwargs

| Modality | Binding profiles kwarg | Model profiles kwarg |
|---|---|---|
| llm | `llm_binding_profiles` | `llm_model_profiles` |
| tti | `tti_binding_profiles` | `tti_model_profiles` |
| tts | `tts_binding_profiles` | `tts_model_profiles` |
| stt | `stt_binding_profiles` | `stt_model_profiles` |
| ttv | `ttv_binding_profiles` | `ttv_model_profiles` |
| ttm | `ttm_binding_profiles` | `ttm_model_profiles` |
| connection | `connection_binding_profiles` | `connection_model_profiles` |
| rag | `rag_binding_profiles` | `rag_model_profiles` |

### Registries, caches, aliases

| Modality | Binding registry | Model registry | Instance cache | Active alias attr | Switch method |
|---|---|---|---|---|---|
| llm | `llm_binding_profiles_registry` | `llm_model_profiles_registry` | `llms` | `_active_llm_alias` | `switch_model` |
| tti | `tti_binding_profiles_registry` | `tti_model_profiles_registry` | `ttis` | `_active_tti_alias` | `switch_tti` |
| tts | `tts_binding_profiles_registry` | `tts_model_profiles_registry` | `tts_bindings` | `_active_tts_alias` | `switch_tts` |
| stt | `stt_binding_profiles_registry` | `stt_model_profiles_registry` | `stts` | `_active_stt_alias` | `switch_stt` |
| ttv | `ttv_binding_profiles_registry` | `ttv_model_profiles_registry` | `ttvs` | `_active_ttv_alias` | `switch_ttv` |
| ttm | `ttm_binding_profiles_registry` | `ttm_model_profiles_registry` | `ttms` | `_active_ttm_alias` | `switch_ttm` |
| connection | `connection_binding_profiles_registry` | `connection_model_profiles_registry` | `connections` | `_active_connection_alias` | `switch_connection` |
| rag | `rag_binding_profiles_registry` | `rag_model_profiles_registry` | `rags` | `_active_rag_alias` | `switch_rag` |

### Key internal methods

| Method | Signature | Role |
|---|---|---|
| `_register_binding_profiles` | `(profiles_dict, registry, modality_name, legacy_binding_name=None, legacy_binding_config=None)` | Populates a binding-profile registry (with legacy auto-master) |
| `_register_model_profiles` | `(profiles_dict, registry, modality_name, binding_registry)` | Populates a model-profile registry (dict auto-conversion, first-default-wins) |
| `_instantiate_binding_from_profile` | `(alias, model_profile, manager, modality, callback=None) -> Optional[Any]` | Merges configs → `create_binding` → applies execution flags |
| `_switch_modality` | `(alias, model_registry, binding_registry, instance_cache, manager, modality, attr_name, active_alias_attr, callback=None) -> bool` | Cache-aware lazy switch engine |
| `_update_binding` | `(binding_name, config, binding_registry, model_registry, instance_cache, switch_method, modality)` | Recreates the `"master"` binding profile and switches to it |
| `_eagerly_instantiate_default` | `(model_registry, switch_method, modality_name)` | Startup default selection + invariant enforcement |
| `_save_single_default_profiles_to_disk` | `(resolved_defaults)` | Auto-persists resolved defaults (YAML + env) |

---

*Cross-references: [README §1–2 Universal Two-Tier Profile Architecture & Smart Router](../src/lollms_client/README.md) · [client_creation_and_profiles skill](../skills/client_creation_and_profiles/SKILL.md) · [Connection bindings skill](../skills/connection_bindings/SKILL.md)*