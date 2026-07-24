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


class ModelManager:
    def __init__(self, config: Dict[str, Any], models_cache_dir: Path):
        self.config = config
        self.models_cache_dir = models_cache_dir
        self.model = None
        self.loaded_model_name = None
        self.device = config.get("device", self._auto_detect_device())
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

        try:
            with lock:
                ASCIIColors.info(f"Loading Whisper model '{model_name}' on device '{self.device}'...")
                self.model = whisper.load_model(model_name, device=self.device)
                self.loaded_model_name = model_name
                self.last_used_time = time.time()
                ASCIIColors.green(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            self.model = None
            self.loaded_model_name = None
            raise RuntimeError(f"Failed to load Whisper model '{model_name}': {e}")
        finally:
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
                    if "out of memory" in str(e).lower():
                        ASCIIColors.warning("OOM detected during transcription. Attempting to free VRAM.")
                        with self.lock:
                            self._unload_model()
                        try:
                            with self.lock:
                                self._load_whisper_model(model_name)
                            result = self.model.transcribe(str(audio_path), **transcribe_args)
                            future.set_result(result.get("text", "").strip())
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
    
    try:
        model_name = request.model_name or "base"
        
        # Auto-detect device if not specified globally, default to cuda if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        manager = state.registry.get_manager(model_name=model_name, device=device)
        
        # Decode base64 to temp file
        audio_bytes = base64.b64decode(request.audio_b64)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_bytes)
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
        
        text = future.result()  # Blocks until transcription is complete
        
        # Cleanup temp file
        Path(temp_file.name).unlink(missing_ok=True)
        
        return {"text": text}
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))

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
