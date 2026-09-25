# lollms_client/ttm_bindings/diffusers/server/main.py
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
from typing import Optional, List, Dict, Any
from concurrent.futures import Future

import pipmaster as pm
pm.ensure_packages(["fastapi", "uvicorn", "pydantic", "ascii_colors>=0.11.10", "torch", "soundfile", "numpy", "diffusers"])

import numpy as np
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel, Field
from ascii_colors import ASCIIColors, trace_exception

device = "cuda" if torch.cuda.is_available() else "cpu"


class MusicGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description of the musical style, mood, BPM, instruments")
    duration: float = Field(default=15.0, description="Duration in seconds")
    model_name: Optional[str] = Field(default=None)
    num_inference_steps: int = Field(default=30)
    guidance_scale: float = Field(default=7.0)
    seed: int = Field(default=-1)


class SongGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Musical style and structure prompt (e.g. genre, vocal style, instruments)")
    lyrics: str = Field(default="", description="Song lyrics with optional section markers like [Verse], [Chorus]")
    duration: float = Field(default=60.0, description="Duration in seconds (up to 300s for MiniMax-Music3)")
    model_name: Optional[str] = Field(default=None)
    num_inference_steps: int = Field(default=30)
    seed: int = Field(default=-1)


class PullModelRequest(BaseModel):
    model_name: str


class TTMJob:
    __slots__ = ("future", "task", "payload")

    def __init__(self, future: Future, task: str, payload: dict):
        self.future = future
        self.task = task
        self.payload = payload


class DiffusersTTMServer:
    def __init__(self, models_cache_dir: Path, default_model: str = "MiniMaxAI/MiniMax-Music3"):
        self.models_cache_dir = models_cache_dir
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_model_name: Optional[str] = None
        self.pipeline: Optional[Any] = None
        self.default_model = default_model
        self.lock = threading.Lock()
        self.queue: queue.Queue[Optional[TTMJob]] = queue.Queue()
        self._stop_event = threading.Event()
        self.last_used_time = time.time()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _load_model(self, model_name: str):
        if self.pipeline is not None and self.loaded_model_name == model_name:
            return

        with self.lock:
            if self.pipeline is not None and self.loaded_model_name == model_name:
                return

            if self.pipeline is not None:
                ASCIIColors.info(f"[TTM Server] Unloading previous model '{self.loaded_model_name}'...")
                del self.pipeline
                self.pipeline = None
                self.loaded_model_name = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ASCIIColors.info(f"[TTM Server] Loading music model '{model_name}' on {device}...")
            target_dtype = torch.bfloat16 if device == "cuda" else torch.float32

            # Route by model architecture
            if "minimax" in model_name.lower():
                try:
                    from diffusers import ModularPipeline
                    self.pipeline = ModularPipeline.from_pretrained(
                        model_name,
                        cache_dir=str(self.models_cache_dir),
                        torch_dtype=target_dtype,
                    )
                except Exception:
                    from diffusers import DiffusionPipeline
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        model_name,
                        cache_dir=str(self.models_cache_dir),
                        torch_dtype=target_dtype,
                    )
            elif "stable-audio" in model_name.lower():
                from diffusers import StableAudioPipeline
                self.pipeline = StableAudioPipeline.from_pretrained(
                    model_name,
                    cache_dir=str(self.models_cache_dir),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                )
            elif "audioldm" in model_name.lower():
                from diffusers import AudioLDM2Pipeline
                self.pipeline = AudioLDM2Pipeline.from_pretrained(
                    model_name,
                    cache_dir=str(self.models_cache_dir),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                )
            else:
                from diffusers import DiffusionPipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    cache_dir=str(self.models_cache_dir),
                    torch_dtype=target_dtype,
                )

            if hasattr(self.pipeline, "to") and device == "cuda":
                self.pipeline.to(device)

            self.loaded_model_name = model_name
            self.last_used_time = time.time()
            ASCIIColors.success(f"[TTM Server] Model '{model_name}' loaded successfully on {device}.")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                break

            try:
                payload = job.payload
                req_model = payload.get("model_name") or self.default_model
                self._load_model(req_model)
                self.last_used_time = time.time()

                audio_bytes = self._execute_generation(job.task, payload)
                job.future.set_result(audio_bytes)
            except Exception as e:
                trace_exception(e)
                job.future.set_exception(e)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _execute_generation(self, task: str, payload: dict) -> bytes:
        prompt = payload.get("prompt", "")
        lyrics = payload.get("lyrics", "")
        duration = float(payload.get("duration", 15.0))
        steps = int(payload.get("num_inference_steps", 30))
        seed = int(payload.get("seed", -1))

        generator = None
        if seed != -1:
            generator = torch.Generator(device=device).manual_seed(seed)

        model_name = (self.loaded_model_name or "").lower()

        # 1. MiniMax-Music3 execution (lyrics + structured caption)
        if "minimax" in model_name:
            input_text = lyrics if lyrics else "[Verse]\n" + prompt
            instructions = prompt
            kwargs: Dict[str, Any] = {
                "input": input_text,
                "instructions": instructions,
                "num_inference_steps": steps,
            }
            if generator:
                kwargs["generator"] = generator

            with torch.no_grad():
                output = self.pipeline(**kwargs)

            # Convert output to stereo WAV bytes
            buf = io.BytesIO()
            if hasattr(output, "audios") and len(output.audios) > 0:
                audio_arr = output.audios[0]
                sr = getattr(self.pipeline, "sample_rate", 32000)
                sf.write(buf, audio_arr, sr, format="WAV")
            else:
                raise RuntimeError("MiniMax Music 3 pipeline produced no audio output.")
            return buf.getvalue()

        # 2. Stable Audio Open execution
        elif "stable-audio" in model_name:
            kwargs = {
                "prompt": prompt,
                "audio_end_in_s": duration,
                "num_inference_steps": steps,
            }
            if generator:
                kwargs["generator"] = generator

            with torch.no_grad():
                output = self.pipeline(**kwargs)

            buf = io.BytesIO()
            audio_arr = output.audios[0].T.float().cpu().numpy()
            sr = self.pipeline.vae.sampling_rate
            sf.write(buf, audio_arr, sr, format="WAV")
            return buf.getvalue()

        # 3. AudioLDM2 execution
        elif "audioldm" in model_name:
            kwargs = {
                "prompt": prompt,
                "audio_length_in_s": duration,
                "num_inference_steps": steps,
            }
            if generator:
                kwargs["generator"] = generator

            with torch.no_grad():
                output = self.pipeline(**kwargs)

            buf = io.BytesIO()
            audio_arr = output.audios[0]
            sf.write(buf, audio_arr, 16000, format="WAV")
            return buf.getvalue()

        # 4. Generic fallback pipeline
        else:
            kwargs = {"prompt": prompt}
            if generator:
                kwargs["generator"] = generator
            with torch.no_grad():
                output = self.pipeline(**kwargs)

            buf = io.BytesIO()
            audio_arr = output.audios[0] if hasattr(output, "audios") else output[0]
            if isinstance(audio_arr, torch.Tensor):
                audio_arr = audio_arr.cpu().numpy()
            sf.write(buf, audio_arr, 32000, format="WAV")
            return buf.getvalue()


server_instance: Optional[DiffusersTTMServer] = None
auth_token: Optional[str] = None

app = FastAPI(title="Diffusers TTM Shared Server")
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
        "loaded_model": server_instance.loaded_model_name if server_instance else None,
        "device": device,
        "queue_size": server_instance.queue.qsize() if server_instance else 0,
    }


@router.post("/shutdown")
def shutdown(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)

    def _delayed_exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down"}


@router.post("/generate_music")
def generate_music(
    request: MusicGenerationRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server uninitialized")

    fut: Future = Future()
    job = TTMJob(future=fut, task="generate_music", payload=request.dict())
    server_instance.queue.put(job)
    try:
        audio_bytes = fut.result(timeout=600)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_song")
def generate_song(
    request: SongGenerationRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server uninitialized")

    fut: Future = Future()
    job = TTMJob(future=fut, task="generate_song", payload=request.dict())
    server_instance.queue.put(job)
    try:
        audio_bytes = fut.result(timeout=1200)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_models")
def list_models(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    return {
        "models": [
            "MiniMaxAI/MiniMax-Music3",
            "stabilityai/stable-audio-open-1.0",
            "cvssp/audioldm2-music",
            "cvssp/audioldm2-large",
            "facebook/musicgen-small",
        ]
    }


@router.post("/pull_model")
def pull_model(
    request: PullModelRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server uninitialized")

    try:
        from huggingface_hub import snapshot_download
        ASCIIColors.info(f"Downloading model '{request.model_name}'...")
        snapshot_download(
            repo_id=request.model_name,
            local_dir=server_instance.models_cache_dir / request.model_name.replace("/", "__")
        )
        return {"status": True, "message": f"Model '{request.model_name}' downloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ps")
def ps():
    if not server_instance:
        return []
    return [{
        "model_name": server_instance.loaded_model_name,
        "is_loaded": server_instance.pipeline is not None,
        "device": device,
        "queue_size": server_instance.queue.qsize(),
        "last_used": time.ctime(server_instance.last_used_time),
    }]


app.include_router(router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diffusers TTM Shared Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9637)
    parser.add_argument("--cache-dir", type=str, default="./data/ttm_models/diffusers")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="MiniMaxAI/MiniMax-Music3")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    auth_token = args.token
    token_file = cache_path / "diffusers_ttm.token"
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

    server_instance = DiffusersTTMServer(models_cache_dir=cache_path, default_model=args.model_name)

    ASCIIColors.cyan("--- Diffusers TTM Shared Server ---------------------------------")
    ASCIIColors.green(f"Host:Port      : http://{args.host}:{args.port}")
    ASCIIColors.green(f"Default Model  : {args.model_name}")
    ASCIIColors.green(f"Cache Path     : {cache_path}")
    ASCIIColors.green(f"Auth Protected : {'Yes' if auth_token else 'No'}")
    ASCIIColors.cyan("----------------------------------------------------------------")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="warning")