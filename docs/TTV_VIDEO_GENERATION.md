# 🎬 Text-to-Video (TTV) — Complete Guide

> **Authoritative guide** for the `ttv` modality of `LollmsClient`: configuring TTV profiles, generating video from text, using optional lip-sync and first/last-frame interpolation, discovering and installing models, switching bindings at runtime, and writing your own TTV binding.
>
> **Applies to**: `src/lollms_client/lollms_ttv_binding.py` (base class + manager) and every binding under `src/lollms_client/ttv_bindings/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The `LollmsTTVBinding` Contract](#2-the-lollmsttvbinding-contract)
3. [Configuring TTV Profiles](#3-configuring-ttv-profiles)
4. [Generating Video](#4-generating-video)
5. [Optional Capabilities — Lip-Sync & First/Last Frame](#5-optional-capabilities--lip-sync--firstlast-frame)
6. [Switching Bindings at Runtime](#6-switching-bindings-at-runtime)
7. [Model Discovery, Zoos & Installation](#7-model-discovery-zoos--installation)
8. [Available Bindings](#8-available-bindings)
9. [Writing a Custom TTV Binding](#9-writing-a-custom-ttv-binding)
10. [VRAM, Multi-Process & Shared Daemons](#10-vram-multiprocess--shared-daemons)
11. [API Quick Reference](#11-api-quick-reference)

---

## 1. Overview

The **Text-to-Video (TTV)** modality turns a text prompt into an encoded video (`mp4`, returned as `bytes`). It sits alongside the other generation modalities (`ti`, `tts`, `ttm`) inside `LollmsClient` and shares the **Universal Two-Tier Profile Architecture**:

- **Connection Layer** — `LollmsBindingProfile`: *where* to run (the backend engine/server, its host, port, model directory).
- **Execution Layer** — `LollmsModelProfile`: *what* to run (a specific model on a named binding).

Only the model profile marked `is_default=True` is instantiated at startup; every other TTV profile is **lazily loaded** on first `switch_ttv()` / first use.

Beyond plain text-to-video, bindings may opt into two advanced abilities:

| Ability | Method | Default |
| :--- | :--- | :--- |
| Text → Video | `generate_video(prompt, **kwargs)` | **Required** (abstract) |
| Image + Audio → talking video (lip-sync) | `generate_lipsync_video(image, audio, prompt="", **kwargs)` | Opt-in; raises `NotImplementedError` |
| First frame + Last frame → interpolated video | `generate_video_first_and_last_frame(prompt, start_image, end_image, **kwargs)` | Opt-in; raises `NotImplementedError` |

Always check `supports_lipsync()` / `supports_first_last_frame()` (or `get_capabilities()`) **before** calling an optional method.

---

## 2. The `LollmsTTVBinding` Contract

All TTV bindings inherit from `LollmsTTVBinding` (in `src/lollms_client/lollms_ttv_binding.py`). The base class defines the full public contract.

### Abstract (must be implemented)

| Method | Signature | Returns |
| :--- | :--- | :--- |
| `generate_video` | `(prompt: str, **kwargs) -> bytes` | Encoded video bytes |
| `list_models` | `(**kwargs) -> List[str]` | Available model names |

### Optional (opt-in — override + gate)

| Method | Signature | Default behavior |
| :--- | :--- | :--- |
| `supports_lipsync` | `() -> bool` | `False` |
| `generate_lipsync_video` | `(image: MediaInput, audio: MediaInput, prompt: str = "", **kwargs) -> bytes` | raises `NotImplementedError` |
| `supports_first_last_frame` | `() -> bool` | `False` |
| `generate_video_first_and_last_frame` | `(prompt: str, start_image: MediaInput, end_image: MediaInput, **kwargs) -> bytes` | raises `NotImplementedError` |
| `get_capabilities` | `() -> Dict[str, bool]` | `{text_to_video, image_to_video, lipsync, first_last_frame}` — computed from the `supports_*` methods |

### Model Zoo & Installation (optional — generic defaults)

| Method | Signature | Default behavior |
| :--- | :--- | :--- |
| `get_zoo` | `() -> List[Dict[str, Any]]` | `[]` (each entry: `name`, `description`, `size`, `type`, `link`) |
| `download_from_zoo` | `(index: int, progress_callback=None) -> dict` | `{"status": False, "message": "Not implemented"}` |
| `install_model` | `(model_name: str, **kwargs) -> dict` | delegates to `pull_model` if present, else `{"status": False, ...}` |
| `pull_model` | `(model_name: str, **kwargs) -> dict` | delegates to `install_model` if present, else `{"status": False, ...}` |

### The `MediaInput` Type

```python
MediaInput = Union[str, Path, bytes]
```

Every image/audio argument (`image`, `audio`, `start_image`, `end_image`) accepts **either** a filesystem path **or** raw bytes. When passing raw bytes, supply the corresponding extension hint kwarg (`image_ext`, `audio_ext`, e.g. `"png"`, `"wav"`) so the binding can decode it.

### Optional kwargs a binding may honor

Bindings **ignore unknown kwargs rather than rejecting them**, so it is always safe to pass the full documented set:

`negative_prompt`, `seed`, `fps`, `width`, `height`, `num_frames`, `num_inference_steps`, `guidance_scale`, plus `image_ext` / `audio_ext` when using raw bytes.

---

## 3. Configuring TTV Profiles

### Constructor kwargs

| Modality | Binding kwarg | Model kwarg |
| :--- | :--- | :--- |
| TTV | `ttv_binding_profiles` | `ttv_model_profiles` |

Each accepts a dict keyed by alias, whose values are `LollmsBindingProfile` / `LollmsModelProfile` instances (or plain dicts, auto-converted).

### Example — local diffusers server

```python
from lollms_client import LollmsClient

client = LollmsClient(
    ttv_binding_profiles={
        "local_ttv": {
            "binding_name": "diffusers",
            "binding_config": {
                "host": "127.0.0.1",
                "port": 14212,
                "auto_start_server": True,
            },
        },
    },
    ttv_model_profiles={
        "svd": {
            "binding_profile_name": "local_ttv",
            "model_name": "stabilityai/stable-video-diffusion-img2vid-xt",
            "is_default": True,
        },
        "svd_alt": {
            "binding_profile_name": "local_ttv",
            "model_name": "another-video-model",
        },
    },
)
```

### Example — dataclass form

```python
from lollms_client import LollmsClient
from lollms_client.lollms_core import LollmsBindingProfile, LollmsModelProfile

client = LollmsClient(
    ttv_binding_profiles={
        "local_ttv": LollmsBindingProfile(
            name="local_ttv",
            binding_name="diffusers",
            binding_config={"host": "127.0.0.1", "port": 14212, "auto_start_server": True},
        ),
    },
    ttv_model_profiles={
        "svd": LollmsModelProfile(
            name="svd",
            binding_profile_name="local_ttv",
            model_name="stabilityai/stable-video-diffusion-img2vid-xt",
            is_default=True,
        ),
    },
)
```

> **Note**: the TTV server commonly runs on its own port. Multiple models on the same server should share one binding profile and differ only in `model_name` (they can be lazily swapped at runtime, subject to the server's memory).

---

## 4. Generating Video

### Text-to-Video

```python
video_bytes = client.generate_video(
    prompt="waves crashing on a rocky shore at sunset, cinematic",
    negative_prompt="blurry, static",
    seed=42,
    fps=24,
    width=1024,
    height=576,
    num_frames=64,
    num_inference_steps=25,
    guidance_scale=7.5,
)

with open("output/shore.mp4", "wb") as f:
    f.write(video_bytes)
```

The returned value is always the encoded video (`mp4`) as `bytes` — write it to disk, stream it to a client, or hand it to any downstream processor.

### Image-to-Video via First/Last Frame (when supported)

Bindings that advertise `image_to_video` / `first_last_frame` can interpolate between two reference images:

```python
if client.ttv.supports_first_last_frame():
    flf_bytes = client.ttv.generate_video_first_and_last_frame(
        prompt="a flower blooming in the sun",
        start_image="bud.png",
        end_image="bloom.png",
        fps=24,
        num_frames=48,
    )
    with open("output/bloom.mp4", "wb") as f:
        f.write(flf_bytes)
```

---

## 5. Optional Capabilities — Lip-Sync & First/Last Frame

### Checking Capabilities

```python
caps = client.ttv.get_capabilities()
# {"text_to_video": True, "image_to_video": True, "lipsync": False, "first_last_frame": True}
```

`get_capabilities()` is the single authoritative probe for UIs. Bindings may return `False` **at runtime** (e.g., a loaded model cannot accept keyframes), so always re-check rather than caching `True` forever.

### Lip-Sync (Portrait + Audio → Talking Video)

Requires `supports_lipsync() == True`.

```python
if client.ttv.supports_lipsync():
    dictation = client.ttv.generate_lipsync_video(
        image="portrait.jpg",     # path or bytes
        audio="speech.wav",       # path or bytes
        prompt="the subject speaks with a warm smile",
        seed=7,
        fps=25,
    )
    with open("output/dictation.mp4", "wb") as f:
        f.write(dictation)
```

When passing raw bytes:

```python
with open("portrait.jpg", "rb") as f:
    img = f.read()
with open("speech.wav", "rb") as f:
    aud = f.read()

client.ttv.generate_lipsync_video(img, aud, image_ext="jpg", audio_ext="wav")
```

---

## 6. Switching Bindings at Runtime

Every modality shares the same switch engine. For TTV:

```python
client.switch_ttv("svd_alt")   # lazy-loads if not already cached, then mounts
client.mount_ttv("svd_alt")    # legacy alias — identical semantics
client.update_ttv_binding("diffusers", {"host": "127.0.0.1", "port": 14212})  # re-point the "master" profile
```

`switch_ttv(alias, callback=None)` returns `bool` (`True` on success). Failure (unknown alias, backend error) logs the precise error and returns `False`; the previously active binding stays mounted.

---

## 7. Model Discovery, Zoos & Installation

### Enumerate installed TTV bindings

```python
from lollms_client.lollms_bindings_utils import list_bindings

print(list_bindings("ttv"))
# ['diffusers', 'lollms', 'lollms_minimax_h3', 'minimaxh3', 'novita_ai_ttv', 'sd.cpp', 'vllm_omni']
```

### List models for a binding without a full client

```python
from lollms_client.lollms_ttv_binding import list_binding_models

models = list_binding_models(
    ttv_binding_name="diffusers",
    ttv_binding_config={"host": "127.0.0.1", "port": 14212},
)
```

### Browse and download from a binding's zoo

```python
for entry in client.ttv.get_zoo():
    print(entry["name"], entry["size"], entry["link"])

# Download the entry at a given index (progress_callback is optional)
result = client.ttv.download_from_zoo(0, progress_callback=lambda p: print(p))

# Or install/pull a known model by name
client.ttv.install_model("stabilityai/stable-video-diffusion-img2vid-xt")
```

Because `install_model` and `pull_model` delegate to each other, calling either works on any binding that implements one of them; a binding that implements neither returns `{"status": False, "message": "..."}`.

### Inspecting a binding's config schema

```python
from lollms_client.lollms_config_api import get_binding_schema

for p in get_binding_schema("diffusers", "ttv"):
    print(p["name"], p["type"], p["mandatory"], p["default"])
```

---

## 8. Available Bindings

TTV bindings live under `src/lollms_client/ttv_bindings/`:

| Binding | Backend / Characteristic |
| :--- | :--- |
| `diffusers` | Local Diffusers-based video generation server (shared daemon on a dedicated port). |
| `lollms` | Delegates to a LoLLMS WebUI/server video endpoint. |
| `lollms_minimax_h3` | LoLLMS-hosted wrapper around the MiniMax H3 video model family. |
| `minimaxh3` | Native MiniMax H3 binding (text-to-video, with lip-sync/FL2V descriptors). |
| `novita_ai_ttv` | Cloud TTV via the Novita AI API. |
| `sd.cpp` | Local `sd.cpp` runtime with model hub, presets, and media helpers. |
| `vllm_omni` | vLLM-Omni multimodal server (text + vision + video). |

Each binding ships a `description.yaml` (or `binding_config.py`) declaring its `global_input_parameters` and `model_input_parameters`; use `get_binding_schema` / `get_binding_desc` to read them programmatically.

---

## 9. Writing a Custom TTV Binding

Create a new folder under `ttv_bindings/` with an `__init__.py` that exposes `BindingName` and a class subclassing `LollmsTTVBinding`.

```python
# src/lollms_client/ttv_bindings/my_ttv/__init__.py
from typing import List
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, MediaInput

BindingName = "MyTTVBinding"

class MyTTVBinding(LollmsTTVBinding):
    def __init__(self, host="127.0.0.1", port=9999, model_name=None, **kwargs):
        super().__init__(binding_name="my_ttv", **kwargs)
        self.host = host
        self.port = port
        self.model_name = model_name

    # ---- Required ----
    def generate_video(self, prompt: str, **kwargs) -> bytes:
        # ... call your backend, return mp4 bytes ...
        return b"..."

    def list_models(self, **kwargs) -> List[str]:
        return ["my-video-model"]

    # ---- Optional: opt into lip-sync ----
    def supports_lipsync(self) -> bool:
        return True

    def generate_lipsync_video(self, image: MediaInput, audio: MediaInput, prompt: str = "", **kwargs) -> bytes:
        # image/audio are path or bytes; honor image_ext/audio_ext when bytes
        return b"..."
```

Rules:

- `BindingName` **must** be defined at module level and name the class in the module.
- The class **must** subclass `LollmsTTVBinding`.
- Override optional methods **only** if you also advertise them via the matching `supports_*` method.
- Unknown kwargs must be ignored, not rejected.

The `LollmsTTVBindingManager` discovers folders that contain an `__init__.py` (directory names with dots like `sd.cpp` are supported via file-based import) and instantiates `binding_class(**kwargs)`.

### Add a `description.yaml`

```yaml
title: My TTV Binding
author: You
description: A minimal Text-to-Video binding.

global_input_parameters:
  - name: host
    type: str
    mandatory: false
    default: "127.0.0.1"
    description: "Backend host address."
  - name: port
    type: int
    mandatory: false
    default: 9999
    description: "Backend port."

model_input_parameters:
  - name: model_name
    type: str
    mandatory: false
    default: "my-video-model"
    description: "Model to load."
```

`get_binding_schema("my_ttv", "ttv")` will then surface `host` and `port` (excluding `model_name`, which belongs to the model profile).

---

## 10. VRAM, Multi-Process & Shared Daemons

- **Cooperative VRAM management**: enable `cooperative_vram_management=True` on the client so that loading a TTV model can reclaim VRAM from the LLM/TTI bindings first (and vice-versa). TTV bindings report `is_local()` when they run a local server, which triggers the resource guard.
- **Shared daemons**: several processes can share one TTV server by pointing multiple binding profiles at the same `host`/`port`. The first process starts the server; others attach (see the Phenix shared-daemon architecture in `phenix_docs/`).
- **`auto_start_server`**: when `True`, the binding spawns the local server on first use and (by design) does **not** block the main thread indefinitely for non-LLM modalities (`wait_for_server` defaults to `False` for them).

---

## 11. API Quick Reference

### Client-level

| Example | Purpose |
| :--- | :--- |
| `client.ttv` | Active TTV binding instance |
| `client.generate_video(prompt, **kwargs)` | Generate video bytes |
| `client.switch_ttv(alias, callback=None) -> bool` | Switch/lazy-load a model profile |
| `client.mount_ttv(alias) -> bool` | Legacy alias for `switch_ttv` |
| `client.update_ttv_binding(binding_name, config=None)` | Re-create the `"master"` profile & switch |

### Binding-level (`client.ttv.*`)

| Method | Returns | Notes |
| :--- | :--- | :--- |
| `generate_video(prompt, **kwargs)` | `bytes` | Required |
| `list_models(**kwargs)` | `List[str]` | Required |
| `generate_lipsync_video(image, audio, prompt="", **kwargs)` | `bytes` | Opt-in |
| `generate_video_first_and_last_frame(prompt, start_image, end_image, **kwargs)` | `bytes` | Opt-in |
| `supports_lipsync()` | `bool` | Capability probe |
| `supports_first_last_frame()` | `bool` | Capability probe |
| `get_capabilities()` | `Dict[str, bool]` | UI-friendly capability map |
| `get_zoo()` | `List[Dict]` | Zoo entries |
| `download_from_zoo(index, progress_callback=None)` | `dict` | `{"status", "message", ...}` |
| `install_model(model_name, **kwargs)` | `dict` | Delegates to `pull_model` |
| `pull_model(model_name, **kwargs)` | `dict` | Delegates to `install_model` |

### Utilities

| Function | Module | Purpose |
| :--- | :--- | :--- |
| `list_binding_models(ttv_binding_name, ttv_binding_config=None, ttv_bindings_dir=...)` | `lollms_ttv_binding` | List models without a full client |
| `list_bindings("ttv")` | `lollms_bindings_utils` | Enumerate installed TTV bindings |
| `get_binding_schema(binding_name, "ttv")` | `lollms_config_api` | List binding_config parameters |
| `get_binding_desc(binding_name, "ttv")` | `lollms_bindings_utils` | Raw description (yaml/config loader) |

---

*See also: [`UNIVERSAL_MODEL_PROFILES.md`](UNIVERSAL_MODEL_PROFILES.md) §8.6 for the TTV profile entry in the cross-modality guide.*