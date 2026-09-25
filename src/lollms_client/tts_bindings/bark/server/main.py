# File: lollms_client/tts_bindings/bark/server/main.py
import os
import io
import time
import wave
import hmac
import secrets
import threading
import queue
import argparse
import traceback
from pathlib import Path
from typing import Optional, List
from concurrent.futures import Future

import pipmaster as pm
pm.ensure_packages(["fastapi", "uvicorn", "pydantic", "ascii_colors>=0.11.10", "torch", "torchaudio", "numpy"])

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Header
from pydantic import BaseModel
from ascii_colors import ASCIIColors

# --- Bark TTS Implementation ---
try:
    ASCIIColors.info("Server: Loading Bark dependencies...")
    from bark import SAMPLE_RATE, generate_audio as bark_generate_audio, preload_models
    from bark.generation import set_seed
    ASCIIColors.green("Server: Bark dependencies loaded successfully")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bark_available = True
except Exception as e:
    ASCIIColors.error(f"Server: Failed to load Bark dependencies: {e}")
    bark_available = False
    SAMPLE_RATE = 24000


class GenerationRequest(BaseModel):
    text: str
    voice: Optional[str] = "v2/en_speaker_6"
    temperature: Optional[float] = 0.7
    silence_duration: Optional[float] = 0.25
    seed: Optional[int] = None


class VoiceRequest(BaseModel):
    voice: str


class BarkJob:
    __slots__ = ("future", "req")

    def __init__(self, future: Future, req: GenerationRequest):
        self.future = future
        self.req = req


class BarkServer:
    def __init__(self, batch_window: float = 0.02, max_batch_size: int = 4):
        self.model_loaded = False
        self.current_voice = "v2/en_speaker_6"
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.available_voices = self._get_available_voices()
        self.queue: queue.Queue[Optional[BarkJob]] = queue.Queue()
        self._stop_event = threading.Event()
        self.model_lock = threading.Lock()

        if bark_available:
            self._initialize_model()

        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()

    def _initialize_model(self):
        try:
            ASCIIColors.info("Server: Preloading Bark models...")
            preload_models()
            self.model_loaded = True
            ASCIIColors.green("Server: Bark model loaded successfully")
        except Exception as e:
            ASCIIColors.error(f"Server: Error initializing Bark model: {e}")
            self.model_loaded = False

    def _get_available_voices(self) -> List[str]:
        voices = []
        langs = ["en", "zh", "fr", "de", "hi", "it", "ja", "ko", "pl", "pt", "ru", "es", "tr"]
        for lang in langs:
            for i in range(10):
                voices.append(f"v2/{lang}_speaker_{i}")
        return voices

    def _batch_worker(self):
        while not self._stop_event.is_set():
            try:
                first_job = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if first_job is None:
                break

            batch = [first_job]
            if self.batch_window > 0:
                time.sleep(self.batch_window)

            while len(batch) < self.max_batch_size:
                try:
                    job = self.queue.get_nowait()
                    if job is None:
                        self._stop_event.set()
                        break
                    batch.append(job)
                except queue.Empty:
                    break

            for job in batch:
                try:
                    req = job.req
                    speaker_voice = req.voice or self.current_voice
                    if req.seed is not None:
                        set_seed(req.seed)

                    with self.model_lock:
                        audio_array = bark_generate_audio(
                            req.text,
                            history_prompt=speaker_voice,
                            text_temp=req.temperature,
                            waveform_temp=req.temperature
                        )

                    if req.silence_duration and req.silence_duration > 0:
                        silence_samples = int(SAMPLE_RATE * req.silence_duration)
                        silence = np.zeros(silence_samples, dtype=audio_array.dtype)
                        audio_array = np.concatenate([audio_array, silence])

                    audio_array = (audio_array * 32767).astype(np.int16)
                    buf = io.BytesIO()
                    with wave.open(buf, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(SAMPLE_RATE)
                        wav_file.writeframes(audio_array.tobytes())

                    job.future.set_result(buf.getvalue())
                except Exception as e:
                    job.future.set_exception(e)


bark_server: Optional[BarkServer] = None
auth_token: Optional[str] = None

app = FastAPI(title="Bark TTS Shared Server")
router = APIRouter()


def verify_auth_token(
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    if not auth_token:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_server_token:
        token = x_server_token.strip()

    if not token or not hmac.compare_digest(token, auth_token):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    return {
        "status": "running",
        "bark_available": bark_available,
        "model_loaded": bark_server.model_loaded if bark_server else False,
        "current_voice": bark_server.current_voice if bark_server else None,
        "device": device if bark_available else "CPU"
    }


@router.post("/shutdown")
def shutdown(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)

    def _delayed_exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down"}


@router.post("/generate_audio")
def generate_audio(
    request: GenerationRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not bark_available or not bark_server:
        raise HTTPException(status_code=500, detail="Bark is not available.")

    fut: Future = Future()
    job = BarkJob(future=fut, req=request)
    bark_server.queue.put(job)
    try:
        audio_bytes = fut.result(timeout=300)
        from fastapi.responses import Response
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_voices")
def list_voices(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    return {"voices": bark_server.available_voices if bark_server else []}


@router.get("/list_models")
def list_models():
    return {"models": ["bark"]}


@router.post("/set_voice")
def set_voice(
    request: VoiceRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not bark_server:
        raise HTTPException(status_code=500, detail="Server uninitialized")
    bark_server.current_voice = request.voice
    return {"success": True, "message": f"Voice set to {request.voice}"}


app.include_router(router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bark TTS Shared Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9636)
    parser.add_argument("--cache-dir", type=str, default="./data/tts_models/bark")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--batch-window", type=float, default=0.02)
    parser.add_argument("--max-batch-size", type=int, default=4)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    auth_token = args.token
    token_file = cache_path / "bark_server.token"
    if not auth_token:
        if token_file.exists():
            try:
                auth_token = token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        if not auth_token:
            auth_token = secrets.token_hex(16)
            try:
                fd = os.open(str(token_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(auth_token)
            except Exception:
                token_file.write_text(auth_token, encoding="utf-8")

    bark_server = BarkServer(batch_window=args.batch_window, max_batch_size=args.max_batch_size)

    ASCIIColors.cyan("─── Bark TTS Shared Server ───────────────────────────────────")
    ASCIIColors.green(f"Host:Port      : http://{args.host}:{args.port}")
    ASCIIColors.green(f"Cache Path     : {cache_path}")
    ASCIIColors.green(f"Micro-Batching : window={args.batch_window}s, max_batch={args.max_batch_size}")
    ASCIIColors.green(f"Auth Protected : {'Yes' if auth_token else 'No'}")
    ASCIIColors.cyan("────────────────────────────────────────────────────────────────")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="warning")