# File: lollms_client/tts_bindings/piper/server/main.py
import os
import io
import time
import hmac
import secrets
import threading
import queue
import argparse
import traceback
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import Future

import pipmaster as pm
pm.ensure_packages(["fastapi", "uvicorn", "pydantic", "ascii_colors>=0.11.10", "soundfile", "numpy"])

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from ascii_colors import ASCIIColors

# --- Piper TTS Implementation ---
try:
    ASCIIColors.info("Server: Loading Piper dependencies...")
    import piper
    import numpy as np
    import soundfile as sf
    ASCIIColors.green("Server: Piper dependencies loaded successfully")
    piper_available = True
except Exception as e:
    ASCIIColors.error(f"Server: Failed to load Piper dependencies: {e}")
    piper_available = False


class GenerationRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speaker_id: Optional[int] = None
    length_scale: Optional[float] = 1.0
    noise_scale: Optional[float] = 0.667
    noise_w: Optional[float] = 0.8


class VoiceRequest(BaseModel):
    voice: str


class DownloadRequest(BaseModel):
    voice: str


class PiperJob:
    __slots__ = ("future", "req")

    def __init__(self, future: Future, req: GenerationRequest):
        self.future = future
        self.req = req


class PiperServer:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.current_voice = None
        self.loaded_models: Dict[str, Any] = {}
        self.model_lock = threading.Lock()
        self.queue: queue.Queue[Optional[PiperJob]] = queue.Queue()
        self._stop_event = threading.Event()
        self.available_voices = self._get_available_voice_list()
        self.installed_voices = self._scan_installed_models()

        self.worker_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.worker_thread.start()

    def _get_available_voice_list(self) -> Dict[str, Dict]:
        return {
            "en_US-lessac-medium": {
                "language": "en_US", "quality": "medium", "description": "US English, female, clear",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
            },
            "en_US-lessac-low": {
                "language": "en_US", "quality": "low", "description": "US English, female, fast",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx"
            },
            "en_US-ryan-high": {
                "language": "en_US", "quality": "high", "description": "US English, male, high quality",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
            },
            "en_US-ryan-medium": {
                "language": "en_US", "quality": "medium", "description": "US English, male, balanced",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
            },
            "en_GB-alan-medium": {
                "language": "en_GB", "quality": "medium", "description": "British English, male",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
            },
            "fr_FR-siwis-medium": {
                "language": "fr_FR", "quality": "medium", "description": "French, female",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx"
            },
            "de_DE-thorsten-medium": {
                "language": "de_DE", "quality": "medium", "description": "German, male",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
            },
            "es_ES-mls_9972-low": {
                "language": "es_ES", "quality": "low", "description": "Spanish, female",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx"
            },
            "it_IT-riccardo-x_low": {
                "language": "it_IT", "quality": "x_low", "description": "Italian, male, fast",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx"
            },
        }

    def _scan_installed_models(self) -> List[str]:
        installed = []
        for onnx_file in self.models_dir.glob("*.onnx"):
            voice_name = onnx_file.stem
            json_file = onnx_file.with_suffix('.onnx.json')
            if json_file.exists():
                installed.append(voice_name)
        return installed

    def download_voice(self, voice_name: str) -> bool:
        if voice_name not in self.available_voices:
            raise ValueError(f"Voice '{voice_name}' not available")

        import requests
        voice_info = self.available_voices[voice_name]
        model_url = voice_info["url"]
        config_url = model_url + ".json"

        model_path = self.models_dir / f"{voice_name}.onnx"
        config_path = self.models_dir / f"{voice_name}.onnx.json"

        if model_path.exists() and config_path.exists():
            if voice_name not in self.installed_voices:
                self.installed_voices.append(voice_name)
            return True

        try:
            ASCIIColors.info(f"Downloading voice '{voice_name}'...")
            r_model = requests.get(model_url, timeout=120)
            r_model.raise_for_status()
            model_path.write_bytes(r_model.content)

            r_config = requests.get(config_url, timeout=60)
            r_config.raise_for_status()
            config_path.write_bytes(r_config.content)

            if voice_name not in self.installed_voices:
                self.installed_voices.append(voice_name)
            ASCIIColors.green(f"Downloaded voice '{voice_name}' successfully.")
            return True
        except Exception as e:
            model_path.unlink(missing_ok=True)
            config_path.unlink(missing_ok=True)
            raise e

    def _load_model(self, voice_name: str):
        with self.model_lock:
            if voice_name in self.loaded_models:
                return self.loaded_models[voice_name]

            model_path = self.models_dir / f"{voice_name}.onnx"
            config_path = self.models_dir / f"{voice_name}.onnx.json"

            if not (model_path.exists() and config_path.exists()):
                ASCIIColors.info(f"Voice '{voice_name}' not found locally. Auto-downloading...")
                self.download_voice(voice_name)

            voice = piper.PiperVoice.load(str(model_path), config_path=str(config_path))
            self.loaded_models[voice_name] = voice
            return voice

    def _generation_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                break

            try:
                req = job.req
                target_voice = req.voice or self.current_voice
                if not target_voice and self.installed_voices:
                    target_voice = self.installed_voices[0]
                elif not target_voice:
                    target_voice = "en_US-lessac-medium"

                voice_model = self._load_model(target_voice)
                audio_stream = io.BytesIO()
                voice_model.synthesize(
                    req.text,
                    audio_stream,
                    speaker_id=req.speaker_id,
                    length_scale=req.length_scale,
                    noise_scale=req.noise_scale,
                    noise_w=req.noise_w,
                )
                audio_stream.seek(0)
                job.future.set_result(audio_stream.getvalue())
            except Exception as e:
                job.future.set_exception(e)


piper_server: Optional[PiperServer] = None
auth_token: Optional[str] = None

app = FastAPI(title="Piper TTS Shared Server")
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
        "piper_available": piper_available,
        "current_voice": piper_server.current_voice if piper_server else None,
        "installed_voices": piper_server.installed_voices if piper_server else [],
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
    if not piper_available or not piper_server:
        raise HTTPException(status_code=500, detail="Piper is not available.")

    fut: Future = Future()
    job = PiperJob(future=fut, req=request)
    piper_server.queue.put(job)
    try:
        audio_bytes = fut.result(timeout=120)
        from fastapi.responses import Response
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_voices")
def list_voices(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    return {"voices": piper_server.installed_voices if piper_server else []}


@router.get("/list_models")
def list_models():
    return {"models": ["piper"]}


@router.post("/download_voice")
def download_voice(
    request: DownloadRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not piper_server:
        raise HTTPException(status_code=500, detail="Server uninitialized")
    try:
        success = piper_server.download_voice(request.voice)
        return {"success": success, "message": f"Voice '{request.voice}' downloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_voice")
def set_voice(
    request: VoiceRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not piper_server:
        raise HTTPException(status_code=500, detail="Server uninitialized")
    piper_server.current_voice = request.voice
    return {"success": True, "message": f"Voice set to {request.voice}"}


app.include_router(router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Piper TTS Shared Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9635)
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    models_path = Path(args.models_dir).resolve()
    piper_server = PiperServer(models_dir=models_path)
    auth_token = args.token

    token_file = models_path / "piper_server.token"
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

    ASCIIColors.cyan("─── Piper TTS Shared Server ──────────────────────────────────")
    ASCIIColors.green(f"Host:Port      : http://{args.host}:{args.port}")
    ASCIIColors.green(f"Models Dir     : {models_path}")
    ASCIIColors.green(f"Auth Protected : {'Yes' if auth_token else 'No'}")
    ASCIIColors.cyan("────────────────────────────────────────────────────────────────")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="warning")