import os
import gc
import time
import threading
import queue
import hashlib
import argparse
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from concurrent.futures import Future

import pipmaster as pm
pm.ensure_packages(["fastapi", "uvicorn", "ascii_colors>=0.11.10", "filelock", "pydantic"])

import torch
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ascii_colors import ASCIIColors, trace_exception

# Ensure whisper is installed in the server environment
try:
    import whisper
except ImportError:
    ASCIIColors.error("openai-whisper is not installed in the server environment.")
    import sys
    sys.exit(-1)


class TranscriptionRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64 encoded audio data")
    model_name: Optional[str] = Field(default=None, description="Whisper model size to use")
    language: Optional[str] = Field(default=None, description="Language code (e.g. 'en')")
    task: str = Field(default="transcribe", description="'transcribe' or 'translate'")
    fp16: Optional[bool] = Field(default=None, description="Override fp16 usage")
    device: Optional[str] = Field(default=None, description="Compute device override: 'cuda', 'cpu', or 'auto' (auto = CUDA if available). On GPU OOM the server degrades to CPU automatically.")
    filename: Optional[str] = Field(default=None, description="Original filename; used only to pick a safe temp-file extension for FFmpeg decoding")


class ModelManager:
    def __init__(self, config: Dict[str, Any], models_cache_dir: Path):
        self.config = config
        self.models_cache_dir = models_cache_dir
        self.model = None
        self.loaded_model_name = None
        requested_device = config.get("device", "auto")
        if requested_device in ("auto", "", None):
            self.device = self._auto_detect_device()
        else:
            self.device = requested_device
        self.device = str(self.device).lower()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            ASCIIColors.warning(f"CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        self.last_used_time = time.time()
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.worker_thread.start()

    def _auto_detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_model_path(self, model_name: str) -> Path:
        return self.models_cache_dir / f"{model_name}.pt"

    def is_loaded(self) -> bool:
        return self.model is not None

    def stop(self):
        self._stop_event.set()
        self.queue.put(None)
        self.worker_thread.join(timeout=5)

    def _load_whisper_model(self, model_name: str):
        if self.model is not None and self.loaded_model_name == model_name:
            return

        if self.model is not None:
            self._unload_model()

        import filelock
        cache_dir = self.models_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_file = cache_dir / f"{model_name}.lock"
        lock = filelock.FileLock(lock_file, timeout=300)

        load_error: Optional[Exception] = None
        try:
            with lock:
                ASCIIColors.info(f"Loading Whisper model '{model_name}' on device '{self.device}'...")
                self.model = whisper.load_model(model_name, device=self.device)
                self.loaded_model_name = model_name
                self.last_used_time = time.time()
                ASCIIColors.green(f"Whisper model '{model_name}' loaded successfully on '{self.device}'.")
        except Exception as e:
            load_error = e
            self.model = None
            self.loaded_model_name = None

        if load_error is not None and self.device != "cpu":
            err_lower = str(load_error).lower()
            is_memory_error = (
                "out of memory" in err_lower
                or "not enough memory" in err_lower
                or "cuda" in err_lower
                or "alloc" in err_lower
                or isinstance(load_error, torch.cuda.OutOfMemoryError)
            )
            if is_memory_error:
                ASCIIColors.warning(
                    f"GPU memory exhaustion detected while loading '{model_name}' on '{self.device}'. "
                    f"Falling back to CPU + main memory. Error: {load_error}"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                try:
                    self.model = whisper.load_model(model_name, device="cpu")
                    self.loaded_model_name = model_name
                    self.last_used_time = time.time()
                    ASCIIColors.green(
                        f"Whisper model '{model_name}' loaded successfully on CPU fallback "
                        f"(GPU unavailable/out of VRAM)."
                    )
                    return
                except Exception as cpu_err:
                    self.model = None
                    self.loaded_model_name = None
                    raise RuntimeError(
                        f"Failed to load Whisper model '{model_name}' on both GPU and CPU. "
                        f"GPU error: {load_error} | CPU error: {cpu_err}"
                    ) from cpu_err

        if load_error is not None:
            raise RuntimeError(f"Failed to load Whisper model '{model_name}': {load_error}")
        try:
            if lock_file.exists() and not lock.is_locked:
                lock_file.unlink()
        except Exception:
            pass

    def _unload_model(self):
        if self.model is not None:
            ASCIIColors.info(f"Unloading Whisper model '{self.loaded_model_name}' to free resources...")
            del self.model
            self.model = None
            self.loaded_model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _transcription_worker(self):
        while not self._stop_event.is_set():
            try:
                job = self.queue.get(timeout=1)
                if job is None:
                    break
                future, model_name, audio_path, transcribe_args = job
                
                try:
                    with self.lock:
                        self.last_used_time = time.time()
                        if self.loaded_model_name != model_name:
                            self._load_whisper_model(model_name)
                    
                    if self.model is None:
                        future.set_exception(RuntimeError("Model failed to load"))
                        continue

                    ASCIIColors.info(f"Transcribing {Path(audio_path).name} with {self.loaded_model_name}...")
                    result = self.model.transcribe(str(audio_path), **transcribe_args)
                    future.set_result(result.get("text", "").strip())
                except Exception as e:
                    err_lower = str(e).lower()
                    is_memory_error = (
                        "out of memory" in err_lower
                        or "not enough memory" in err_lower
                        or "cuda" in err_lower
                        or "alloc" in err_lower
                        or isinstance(e, torch.cuda.OutOfMemoryError)
                    )
                    if is_memory_error and self.device != "cpu":
                        ASCIIColors.warning(
                            f"GPU OOM during transcription of '{Path(audio_path).name}'. "
                            f"Unloading model and escalating to CPU + main memory."
                        )
                        with self.lock:
                            self._unload_model()
                            self.device = "cpu"
                        try:
                            with self.lock:
                                self._load_whisper_model(model_name)
                            transcribe_args["fp16"] = False
                            result = self.model.transcribe(str(audio_path), **transcribe_args)
                            future.set_result(result.get("text", "").strip())
                            ASCIIColors.success(
                                f"Transcription of '{Path(audio_path).name}' succeeded on CPU fallback."
                            )
                            continue
                        except Exception as retry_e:
                            future.set_exception(retry_e)
                    else:
                        future.set_exception(e)
            except queue.Empty:
                continue


class WhisperRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._managers = {}
                cls._instance._registry_lock = threading.Lock()
                cls._instance.models_cache_dir = kwargs.get("models_cache_dir")
        return cls._instance

    def get_manager(self, model_name: str, device: str) -> ModelManager:
        key_data = (model_name, device)
        key = hashlib.sha256(str(key_data).encode('utf-8')).hexdigest()
        with self._registry_lock:
            if key not in self._managers:
                config = {"model_name": model_name, "device": device}
                self._managers[key] = ModelManager(config, self.models_cache_dir)
            return self._managers[key]

    def get_active_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return [m for m in self._managers.values() if m.is_loaded()]

    def get_all_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return list(self._managers.values())


class ServerState:
    def __init__(self, models_cache_dir: Path):
        self.models_cache_dir = models_cache_dir
        self.registry = WhisperRegistry(models_cache_dir=models_cache_dir)

state: Optional[ServerState] = None

app = FastAPI(title="Whisper STT Server")
router = APIRouter()

@router.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    import base64
    import tempfile
    
    model_name = request.model_name or "base"
    temp_file = None
    try:
        requested_device = (request.device or "auto").lower().strip()
        if requested_device in ("auto", ""):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device.startswith("cuda") and not torch.cuda.is_available():
            ASCIIColors.warning("CUDA requested for transcription but not available. Using CPU.")
            device = "cpu"
        else:
            device = requested_device
        manager = state.registry.get_manager(model_name=model_name, device=device)

        audio_bytes = base64.b64decode(request.audio_b64, validate=False)
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio payload.")
        max_audio_bytes = 100 * 1024 * 1024
        if len(audio_bytes) > max_audio_bytes:
            raise HTTPException(status_code=413, detail=f"Audio payload too large: {len(audio_bytes):,} bytes (limit: {max_audio_bytes:,}).")

        original_ext = ""
        if request.filename and "." in request.filename:
            original_ext = Path(request.filename).suffix.lower()
        allowed_exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif", ".webm", ".mp4", ".mkv"}
        safe_suffix = original_ext if original_ext in allowed_exts else ".wav"

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix)
        try:
            temp_file.write(audio_bytes)
        finally:
            temp_file.close()

        transcribe_args = {
            "language": request.language,
            "task": request.task,
        }
        if request.fp16 is not None:
            transcribe_args["fp16"] = request.fp16
        else:
            transcribe_args["fp16"] = (device == "cuda")

        future = Future()
        manager.queue.put((future, model_name, temp_file.name, transcribe_args))
        text = future.result()
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        trace_exception(e)
        error_text = str(e)
        friendly = f"Whisper transcription failed on the server: {error_text}"
        lowered = error_text.lower()
        if "not a valid win32 application" in lowered or "ffmpeg" in lowered:
            friendly += (
                " | Likely cause: FFmpeg is missing or incompatible. "
                "Ensure FFmpeg is installed and available in the server venv PATH "
                "(whisper uses it to decode audio)."
            )
        elif "out of memory" in lowered or "cuda" in lowered:
            friendly += (
                " | Likely cause: GPU OOM while loading the Whisper model. "
                "Try a smaller model_name (e.g. 'base' or 'small') or free VRAM."
            )
        elif "download" in lowered or "checksum" in lowered or "connection" in lowered:
            friendly += (
                " | Likely cause: failed to download the Whisper model. "
                "Check network access or pre-download the model into the cache dir."
            )
        raise HTTPException(status_code=500, detail=friendly)
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)

@router.get("/status")
def status():
    return {
        "status": "running",
        "active_models": [m.loaded_model_name for m in state.registry.get_active_managers() if m.is_loaded()]
    }

@router.get("/ps")
def ps():
    return [{
        "model_name": m.loaded_model_name,
        "is_loaded": m.is_loaded(),
        "device": m.device,
        "queue_size": m.queue.qsize(),
        "last_used": time.ctime(m.last_used_time)
    } for m in state.registry.get_all_managers()]

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Whisper STT Server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9633)
    parser.add_argument("--cache-dir", type=str, required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    state = ServerState(models_cache_dir=cache_path)
    
    ASCIIColors.cyan("─── Whisper STT Server ───────────────────────────────────────")
    ASCIIColors.green(f"Starting on http://{args.host}:{args.port}")
    ASCIIColors.green(f"Cache path  : {cache_path.resolve()}")
    ASCIIColors.cyan("────────────────────────────────────────────────────────────────")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
