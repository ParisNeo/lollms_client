# 📡 Modality Port & Daemon Lifecycle Registry

---

## 1. Canonical Loopback Port Assignments

To eliminate port collisions and ensure predictable process mutualization, every background daemon is allocated a dedicated port:

| Modality | Binding | Canonical Port | Process Type | Log File Path |
| :--- | :--- | :--- | :--- | :--- |
| **TTI** (Image) | `diffusers` | **`9632`** | FastAPI / Uvicorn | `data/tti_models/diffusers/diffusers_server.log` |
| **STT** (Audio Transcribe) | `whisper` | **`9633`** | FastAPI / Uvicorn | `data/stt_models/whisper/whisper_server.log` |
| **TTS** (Voice Clone) | `xtts` | **`9634`** | FastAPI / Uvicorn | `data/tts_models/xtts/xtts_server.log` |
| **TTS** (Neural Voice) | `piper_tts` | **`9635`** | FastAPI / Uvicorn | `data/tts_models/piper/piper_server.log` |
| **TTS** (Generative Audio) | `bark` | **`9636`** | FastAPI / Uvicorn | `data/tts_models/bark/bark_server.log` |
| **TTM** (Music & Songs) | `diffusers` | **`9637`** | FastAPI / Uvicorn | `data/ttm_models/diffusers/diffusers_ttm_server.log` |
| **LLM** (Default Core) | `lollms` | **`9642`** | LoLLMs Native Server | Managed externally |

---

## 2. Standardized Daemon Endpoints

Every Phenix-compliant daemon implements these core endpoints:

### Health & Telemetry
- `GET /health`: Fast, lightweight probe returning `{"status": "ok"}` (used by `is_server_running()`).
- `GET /status`: Returns running status, active model, device allocation (`cuda`/`cpu`), and queue backlog. Requires HMAC token if configured.
- `GET /ps`: Returns process status, VRAM footprint, active workers, and queue size.

### Generation & Control
- `POST /shutdown`: Authenticated termination RPC. Allows graceful teardown during server maintenance or test cleanup.
- `POST /unload_model`: Frees model weights and releases GPU VRAM immediately without stopping the HTTP daemon.

---

## 3. Daemon Command Line Usage

Daemons can be started independently or inspected via terminal:

```bash
# 1. Start Diffusers TTM Daemon on port 9637
python -m lollms_client.ttm_bindings.diffusers.server.main --host 127.0.0.1 --port 9637 --model-name MiniMaxAI/MiniMax-Music3

# 2. Start Whisper STT Daemon on port 9633 with micro-batching
python -m lollms_client.stt_bindings.whisper.server.main --host 127.0.0.1 --port 9633 --batch-window 0.02 --max-batch-size 8

# 3. Start Piper TTS Daemon on port 9635
python -m lollms_client.tts_bindings.piper_tts.server.main --host 127.0.0.1 --port 9635
```