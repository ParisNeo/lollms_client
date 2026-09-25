# 🦅 Project Phenix: The Sovereign Multi-User AI Architecture

**Author:** ParisNeo & Lollms Team  
**Scope:** Multi-Process Architecture, Shared Model Server Singletons, IPC, Dynamic Micro-Batching & Generative Audio

---

## 1. Executive Vision

**Project Phenix** redefines how local generative AI services are hosted and consumed across modern distributed and multi-process applications (FastAPI/Uvicorn multi-worker services, Celery task queues, asynchronous desktop GUIs, and command-line interfaces).

Historically, local AI execution required each process to load model weights directly into VRAM, leading to catastrophic **Out-of-Memory (OOM) crashes**, **zero tensor batching**, and **severe port drifting**.

Phenix eliminates these bottlenecks through a **Zero-Port-Drift Shared Model Server Daemon Architecture** combined with **Event-Driven Dynamic Continuous Micro-Batching**.

---

## 2. Core Pillars of Phenix

```
[ FastAPI Worker 1 ] ──┐
                       ├─► [ Canonical Loopback TCP / HTTP ] ──► [ Phenix Shared Daemon ]
[ CLI Agent (lollms) ] ──┤   [Constant-Time HMAC Authenticated]     │  - Single VRAM Weight Footprint
                       │                                            │  - Dynamic Micro-Batch Queue
[ Desktop GUI App ]  ──┘                                            │  - GPU OOM CPU Escalation
                                                                    ▼
                                                            [ Output Streams ]
```

### 1. The Self-Spawning Shared Daemon ("First Instance Wins")
- **Zero Configuration**: Any worker process needing a modality (STT Whisper, TTI Diffusers, TTS XTTS, TTM Diffusers) attempts to reach the shared daemon on its canonical loopback port.
- **Fast Loopback Probing**: If the server is already active, workers attach in under **1.5ms** without spawning duplicate processes.
- **Atomic Concurrency Protection**: If the server is uninitialized, workers enter a cross-process `FileLock` (`timeout=120s`) and execute a **double-checked probe**. Exactly one worker launches the background daemon; all other workers cleanly wait and attach.
- **Immutable Lifetime**: The daemon outlives individual client processes. Worker garbage collection (`__del__`) **never kills the shared daemon**. Clean termination is managed via an authenticated `/shutdown` RPC.

### 2. Event-Driven Continuous Dynamic Micro-Batching
- **0% Idle CPU Usage**: Processing threads sleep on OS synchronization events (`threading.Event.wait()`), awakening only when client jobs arrive.
- **Adaptive Collection Windows**: Gathers concurrent requests across an adaptive window (e.g. 20ms) and processes them in a single batched tensor pass.
- **Fair Future Slicing**: Batch results are distributed back to client processes via thread-safe `Future` objects.

### 3. Canonical Port Registry (Zero Port Drift)
| Modality | Binding | Canonical Port | Default Model / Target |
| :--- | :--- | :--- | :--- |
| **TTI** (Image) | `diffusers` | **`9632`** | FLUX.1 / SDXL / Qwen-Image-Edit |
| **STT** (Speech-to-Text) | `whisper` | **`9633`** | OpenAI Whisper (tiny to large-v3) |
| **TTS** (Speech) | `xtts` | **`9634`** | Coqui XTTS v2 (voice cloning) |
| **TTS** (Speech) | `piper_tts` | **`9635`** | Piper ONNX Neural TTS |
| **TTS** (Speech) | `bark` | **`9636`** | Suno Bark Generative Audio |
| **TTM** (Music/Song) | `diffusers` | **`9637`** | **`MiniMaxAI/MiniMax-Music3`** |

### 4. Studio-Grade Full Song Generation (MiniMax Music 3)
Project Phenix introduces native support for full-song synthesis. Unlike older instrumental-only engines, `MiniMaxAI/MiniMax-Music3` accepts **musical style prompts** alongside **structured lyrics** (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`), producing up to 5-minute audio tracks with expressive vocals and realistic arrangements.

---

## 3. Documentation Map

- [Shared Daemon IPC & Security Architecture](SHARED_DAEMON_IPC.md)
- [Text-to-Music & Full Song Generation Guide](TTM_MUSIC_AND_SONGS.md)
- [Modality Port & Daemon Lifecycle Registry](MODALITY_PORT_REGISTRY.md)
- [Phenix Unified API Reference](API_REFERENCE.md)