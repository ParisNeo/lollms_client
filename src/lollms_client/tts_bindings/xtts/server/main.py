import os
import io
import re
import wave
import time
import hmac
import secrets
import threading
import queue
import argparse
import traceback
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List
from concurrent.futures import Future

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import pipmaster as pm
pm.ensure_packages(["fastapi", "uvicorn", "pydantic", "ascii_colors>=0.11.10", "torch", "numpy"])

import numpy as np
import torch
import uvicorn
import fastapi
from fastapi import FastAPI, APIRouter, HTTPException, File, Form, UploadFile, Header
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from ascii_colors import ASCIIColors

try:
    ASCIIColors.info("Server: Loading XTTS dependencies...")
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ASCIIColors.green(f"Server: XTTS dependencies loaded on device: {device}")
    xtts_available = True
except Exception as e:
    ASCIIColors.error(f"Server: Failed to load XTTS dependencies: {e}")
    xtts_available = False


class GenerationRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "en"
    speaker_wav: Optional[str] = None
    split_sentences: Optional[bool] = True


class VoiceUploadResponse(BaseModel):
    success: bool
    voice_name: str
    message: str


class XTTSJob:
    __slots__ = ("future", "req")

    def __init__(self, future: Future, req: GenerationRequest):
        self.future = future
        self.req = req


class XTTSServer:
    def __init__(self, voices_dir: Path):
        self.model = None
        self.model_loaded = False
        self.available_models = ["tts_models/multilingual/multi-dataset/xtts_v2"]
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.available_voices = self._load_available_voices()
        self.queue: queue.Queue[Optional[XTTSJob]] = queue.Queue()
        self._stop_event = threading.Event()
        self.model_lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.worker_thread.start()

    def _ensure_model_loaded(self):
        with self.model_lock:
            if self.model_loaded and self.model is not None:
                return
            if not xtts_available:
                raise RuntimeError("XTTS library not available.")
            ASCIIColors.yellow("Server: Initializing XTTS model...")
            self.model = TTS(self.available_models[0]).to(device)
            self.model_loaded = True
            ASCIIColors.green("Server: XTTS model loaded successfully.")

    def _load_available_voices(self) -> List[str]:
        self.voices_dir.mkdir(exist_ok=True, parents=True)
        found_voices = {p.stem for p in self.voices_dir.glob("*.[wW][aA][vV]")}
        found_voices.update({p.stem for p in self.voices_dir.glob("*.[mM][pP]3")})
        all_voices = {"default_voice"}.union(found_voices)
        return sorted(list(all_voices))

    def _get_speaker_wav_path(self, voice_name: str) -> Optional[str]:
        if not voice_name:
            return None
        if os.path.isabs(voice_name) and os.path.exists(voice_name):
            return voice_name
        mp3_path = self.voices_dir / f"{voice_name}.mp3"
        if mp3_path.exists():
            return str(mp3_path)
        wav_path = self.voices_dir / f"{voice_name}.wav"
        if wav_path.exists():
            return str(wav_path)
        return None

    def _chunk_text(self, text: str, max_chunk_length: int = 200) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                phrases = re.split(r'(?<=[,;])\s+', sentence)
                current_split = ""
                for phrase in phrases:
                    if len(current_split) + len(phrase) + 1 <= max_chunk_length:
                        current_split = (current_split + " " + phrase).strip() if current_split else phrase
                    else:
                        if current_split:
                            chunks.append(current_split)
                        current_split = phrase
                if current_split:
                    chunks.append(current_split.strip())
            else:
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                    current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                break

            try:
                self._ensure_model_loaded()
                req = job.req
                text_to_generate = req.text
                voice_to_find = req.speaker_wav or req.voice or "default_voice"
                speaker_wav_path = self._get_speaker_wav_path(voice_to_find)
                if not speaker_wav_path and voice_to_find != "default_voice":
                    speaker_wav_path = self._get_speaker_wav_path("default_voice")

                if not speaker_wav_path:
                    raise RuntimeError(f"Speaker reference audio file '{voice_to_find}' not found in {self.voices_dir}.")

                if len(text_to_generate) > 250:
                    chunks = self._chunk_text(text_to_generate, 200)
                    all_audio_parts = []
                    sample_rate = None
                    for chunk in chunks:
                        with self.model_lock:
                            wav_chunks = self.model.tts(
                                text=chunk,
                                speaker_wav=speaker_wav_path,
                                language=req.language or "en",
                                split_sentences=False
                            )
                        audio_part = np.array(wav_chunks, dtype=np.float32)
                        all_audio_parts.append(audio_part)
                        if sample_rate is None:
                            sample_rate = self.model.synthesizer.output_sample_rate
                    audio_data = np.concatenate(all_audio_parts)
                else:
                    with self.model_lock:
                        wav_chunks = self.model.tts(
                            text=text_to_generate,
                            speaker_wav=speaker_wav_path,
                            language=req.language or "en",
                            split_sentences=req.split_sentences
                        )
                    audio_data = np.array(wav_chunks, dtype=np.float32)
                    sample_rate = self.model.synthesizer.output_sample_rate

                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                job.future.set_result(buf.getvalue())
            except Exception as e:
                job.future.set_exception(e)


xtts_server: Optional[XTTSServer] = None
auth_token: Optional[str] = None

app = FastAPI(title="XTTS Shared Server")
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
        "xtts_available": xtts_available,
        "model_loaded": xtts_server.model_loaded if xtts_server else False,
        "device": device if xtts_available else "N/A"
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
    if not xtts_available or not xtts_server:
        raise HTTPException(status_code=500, detail="XTTS is not available.")

    fut: Future = Future()
    job = XTTSJob(future=fut, req=request)
    xtts_server.queue.put(job)
    try:
        audio_bytes = fut.result(timeout=300)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_voice")
async def upload_voice(
    voice_file: UploadFile = File(...),
    voice_name: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not xtts_server:
        raise HTTPException(status_code=500, detail="Server uninitialized")
    allowed_extensions = {'.wav', '.mp3'}
    file_ext = Path(voice_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 voice files are supported.")

    safe_name = re.sub(r'[^\w\-_]', '_', voice_name or Path(voice_file.filename).stem)
    target_path = xtts_server.voices_dir / f"{safe_name}{file_ext}"
    content = await voice_file.read()
    target_path.write_bytes(content)
    xtts_server.available_voices = xtts_server._load_available_voices()
    return VoiceUploadResponse(success=True, voice_name=safe_name, message=f"Voice '{safe_name}' uploaded successfully.")


@router.get("/list_voices")
def list_voices(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    return {"voices": xtts_server.available_voices if xtts_server else []}


@router.get("/list_models")
def list_models():
    return {"models": xtts_server.available_models if xtts_server else []}


app.include_router(router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LoLLMs XTTS Shared Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9634)
    parser.add_argument("--voices-dir", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    v_dir = Path(args.voices_dir).resolve() if args.voices_dir else Path(__file__).parent / "voices"
    v_dir.mkdir(parents=True, exist_ok=True)
    xtts_server = XTTSServer(voices_dir=v_dir)

    auth_token = args.token
    token_file = v_dir / "xtts_server.token"
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

    ASCIIColors.cyan("─── LoLLMs XTTS Shared Server ────────────────────────────────")
    ASCIIColors.green(f"Host:Port      : http://{args.host}:{args.port}")
    ASCIIColors.green(f"Voices Dir     : {v_dir}")
    ASCIIColors.green(f"Auth Protected : {'Yes' if auth_token else 'No'}")
    ASCIIColors.cyan("────────────────────────────────────────────────────────────────")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="warning")