# lollms_client/ttv_bindings/diffusers/server/main.py
"""
Shared singleton FastAPI daemon for Text-to-Video (TTV) generation.

Loads Hugging Face Diffusers video pipelines (LTX-Video, CogVideoX, AnimateDiff,
or any generic DiffusionPipeline) and serializes generation through a single
worker thread so that concurrent clients share one GPU-resident model.

Endpoints:
    GET  /health          – liveness probe
    GET  /status          – daemon status (loaded model, device, queue size)
    POST /shutdown        – graceful shutdown
    POST /generate_video  – generate a video from a text prompt
    GET  /list_models     – list preset + locally cached models
    POST /pull_model      – download a model from Hugging Face
    GET  /ps              – process status of the loaded model
"""
from __future__ import annotations

import argparse
import hmac
import io
import os
import secrets
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import pipmaster as pm

pm.ensure_packages([
    "fastapi",
    "uvicorn",
    "pydantic",
    "ascii_colors>=0.11.10",
    "torch",
    "diffusers",
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "pillow",
])

import numpy as np
import torch
import uvicorn
from ascii_colors import ASCIIColors, trace_exception
from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Request models ──────────────────────────────────────────────────────────
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the video.")
    negative_prompt: str = Field(default="", description="Negative prompt.")
    width: int = Field(default=768)
    height: int = Field(default=512)
    num_frames: int = Field(default=24)
    num_inference_steps: int = Field(default=30)
    guidance_scale: float = Field(default=7.5)
    seed: int = Field(default=-1)
    fps: int = Field(default=8)
    model_name: Optional[str] = Field(default=None)


class PullModelRequest(BaseModel):
    model_name: str


class TTVJob:
    __slots__ = ("future", "payload")

    def __init__(self, future: Future, payload: Dict[str, Any]):
        self.future = future
        self.payload = payload


# ── Server ──────────────────────────────────────────────────────────────────
class DiffusersTTVServer:
    def __init__(self, models_cache_dir: Path, default_model: str = "Lightricks/LTX-Video"):
        self.models_cache_dir = models_cache_dir
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_model = default_model
        self.loaded_model_name: Optional[str] = None
        self.pipeline: Optional[Any] = None
        self.lock = threading.Lock()
        self.queue: Queue[Optional[TTVJob]] = Queue()
        self._stop_event = threading.Event()
        self.last_used_time = time.time()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    # ── Model loading ───────────────────────────────────────────────────────
    def _resolve_model_source(self, model_name: str) -> str:
        flat = model_name.replace("/", "__")
        local_dir = self.models_cache_dir / flat
        if local_dir.exists() and any(local_dir.iterdir()):
            return str(local_dir)
        return model_name

    def _load_model(self, model_name: str) -> None:
        if self.pipeline is not None and self.loaded_model_name == model_name:
            return

        with self.lock:
            if self.pipeline is not None and self.loaded_model_name == model_name:
                return

            if self.pipeline is not None:
                ASCIIColors.info(f"[TTV Server] Unloading previous model '{self.loaded_model_name}'...")
                del self.pipeline
                self.pipeline = None
                self.loaded_model_name = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            source = self._resolve_model_source(model_name)
            ASCIIColors.info(f"[TTV Server] Loading video model '{model_name}' (source: {source}) on {device}...")
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            lower = model_name.lower()

            try:
                if "ltx" in lower:
                    from diffusers import LTXPipeline
                    self.pipeline = LTXPipeline.from_pretrained(
                        source, torch_dtype=dtype, cache_dir=str(self.models_cache_dir)
                    )
                elif "cogvideox" in lower:
                    from diffusers import CogVideoXPipeline
                    self.pipeline = CogVideoXPipeline.from_pretrained(
                        source, torch_dtype=dtype, cache_dir=str(self.models_cache_dir)
                    )
                elif "animatediff" in lower:
                    from diffusers import AnimateDiffPipeline, MotionAdapter
                    adapter = MotionAdapter.from_pretrained(
                        "guoyww/animatediff-motion-adapter-v1-5-2",
                        torch_dtype=dtype,
                    )
                    self.pipeline = AnimateDiffPipeline.from_pretrained(
                        source, motion_adapter=adapter, torch_dtype=dtype,
                        cache_dir=str(self.models_cache_dir),
                    )
                else:
                    from diffusers import DiffusionPipeline
                    self.pipeline = DiffusionPipeline.from_pretrained(
                        source, torch_dtype=dtype, cache_dir=str(self.models_cache_dir)
                    )

                if hasattr(self.pipeline, "to") and device == "cuda":
                    self.pipeline.to(device)
                if hasattr(self.pipeline, "enable_model_cpu_offload") and device == "cuda":
                    try:
                        self.pipeline.enable_model_cpu_offload()
                    except Exception:
                        pass

                self.loaded_model_name = model_name
                self.last_used_time = time.time()
                ASCIIColors.success(f"[TTV Server] Model '{model_name}' loaded on {device}.")
            except Exception as e:
                trace_exception(e)
                self.pipeline = None
                self.loaded_model_name = None
                raise

    # ── Worker loop ─────────────────────────────────────────────────────────
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1.0)
            except Empty:
                continue
            if job is None:
                break
            try:
                payload = job.payload
                req_model = payload.get("model_name") or self.default_model
                self._load_model(req_model)
                self.last_used_time = time.time()
                video_bytes = self._execute_generation(payload)
                job.future.set_result(video_bytes)
            except Exception as e:
                trace_exception(e)
                job.future.set_exception(e)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── Generation ──────────────────────────────────────────────────────────
    def _execute_generation(self, payload: Dict[str, Any]) -> bytes:
        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "")
        width = int(payload.get("width", 768))
        height = int(payload.get("height", 512))
        num_frames = int(payload.get("num_frames", 24))
        steps = int(payload.get("num_inference_steps", 30))
        guidance = float(payload.get("guidance_scale", 7.5))
        seed = int(payload.get("seed", -1))
        fps = int(payload.get("fps", 8))

        generator = None
        if seed != -1:
            generator = torch.Generator(device=device).manual_seed(seed)

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if generator is not None:
            kwargs["generator"] = generator

        # Model-family specific kwargs
        lower = (self.loaded_model_name or "").lower()
        if "ltx" in lower:
            kwargs.update({
                "width": width,
                "height": height,
                "num_frames": num_frames,
            })
        elif "cogvideox" in lower:
            kwargs.update({
                "num_frames": num_frames,
            })
        elif "animatediff" in lower:
            kwargs.update({
                "num_frames": num_frames,
                "width": width,
                "height": height,
            })
        else:
            # Generic fallback – pass what the pipeline accepts
            kwargs.update({
                "num_frames": num_frames,
                "width": width,
                "height": height,
            })

        with torch.no_grad():
            output = self.pipeline(**kwargs)

        frames = getattr(output, "frames", None)
        if frames is None:
            raise RuntimeError("Video pipeline returned no 'frames' attribute.")

        # Normalize frames to a list of HxWx3 uint8 arrays
        if isinstance(frames, list) and frames and isinstance(frames[0], list):
            frames = frames[0]
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            frames = list(frames)

        normalized: List[np.ndarray] = []
        for f in frames:
            if isinstance(f, torch.Tensor):
                f = f.cpu().numpy()
            if isinstance(f, np.ndarray):
                if f.dtype != np.uint8:
                    f = np.clip(f * 255.0, 0, 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
                normalized.append(f)
            else:
                # PIL Image
                normalized.append(np.array(f.convert("RGB")))

        if not normalized:
            raise RuntimeError("Video pipeline produced zero frames.")

        # Encode to MP4 via imageio
        import imageio
        buf = io.BytesIO()
        writer = imageio.get_writer(
            buf, format="mp4", mode="I", fps=fps,
            codec="libx264", quality=8, macro_block_size=1,
        )
        try:
            for frame in normalized:
                writer.append_data(frame)
        finally:
            writer.close()
        return buf.getvalue()


# ── FastAPI app ─────────────────────────────────────────────────────────────
server_instance: Optional[DiffusersTTVServer] = None
auth_token: Optional[str] = None

app = FastAPI(title="Diffusers TTV Shared Server")
router = APIRouter()


def verify_auth_token(
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None),
) -> None:
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
def status(
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None),
):
    verify_auth_token(authorization, x_server_token)
    return {
        "status": "running",
        "loaded_model": server_instance.loaded_model_name if server_instance else None,
        "device": device,
        "queue_size": server_instance.queue.qsize() if server_instance else 0,
    }


@router.post("/shutdown")
def shutdown(
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None),
):
    verify_auth_token(authorization, x_server_token)

    def _delayed_exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down"}


@router.post("/generate_video")
def generate_video(
    request: VideoGenerationRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None),
):
    verify_auth_token(authorization, x_server_token)
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server uninitialized")

    fut: Future = Future()
    job = TTVJob(future=fut, payload=request.dict())
    server_instance.queue.put(job)
    try:
        video_bytes = fut.result(timeout=1800)
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_models")
def list_models(
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None),
):
    verify_auth_token(authorization, x_server_token)
    presets = [
        "Lightricks/LTX-Video",
        "THUDM/CogVideoX-2b",
        "THUDM/CogVideoX-5b",
        "guoyww/animatediff-motion-adapter-v1-5-2",
        "damo-vilab/text-to-video-ms-1.7b",
        "ali-vilab/i2vgen-xl",
    ]
    discovered: List[str] = []
    if server_instance and server_instance.models_cache_dir.exists():
        for d in server_instance.models_cache_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                discovered.append(d.name.replace("__", "/"))
    return {"models": list(dict.fromkeys(presets + discovered))}


@router.post("/pull_model")
def pull_model(
    request: PullModelRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None),
):
    verify_auth_token(authorization, x_server_token)
    if not server_instance:
        raise HTTPException(status_code=500, detail="Server uninitialized")
    try:
        from huggingface_hub import snapshot_download
        ASCIIColors.info(f"[TTV Server] Downloading model '{request.model_name}'...")
        snapshot_download(
            repo_id=request.model_name,
            local_dir=server_instance.models_cache_dir / request.model_name.replace("/", "__"),
        )
        return {"status": True, "message": f"Model '{request.model_name}' downloaded."}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusers TTV Shared Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9634)
    parser.add_argument("--cache-dir", type=str, default="./data/ttv_models/diffusers")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Lightricks/LTX-Video")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    auth_token = args.token
    token_file = cache_path / "diffusers_ttv.token"
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
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(auth_token)
            except Exception:
                token_file.write_text(auth_token, encoding="utf-8")

    server_instance = DiffusersTTVServer(models_cache_dir=cache_path, default_model=args.model_name)

    ASCIIColors.cyan("─── Diffusers TTV Shared Server ──────────────────────────────")
    ASCIIColors.green(f"Host:Port      : http://{args.host}:{args.port}")
    ASCIIColors.green(f"Default Model  : {args.model_name}")
    ASCIIColors.green(f"Cache Path     : {cache_path}")
    ASCIIColors.green(f"Auth Protected : {'Yes' if auth_token else 'No'}")
    ASCIIColors.cyan("────────────────────────────────────────────────────────────────")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="warning")