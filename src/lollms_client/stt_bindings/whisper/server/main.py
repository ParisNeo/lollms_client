import os
import gc
import time
import hmac
import secrets
import threading
import queue
import hashlib
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import Future

import pipmaster as pm
pm.ensure_packages(["fastapi", "uvicorn", "ascii_colors>=0.11.10", "filelock", "pydantic", "torch", "numpy"])

import numpy as np
import torch
from fastapi import FastAPI, APIRouter, HTTPException, Header
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
    device: Optional[str] = Field(default=None, description="Compute device override: 'cuda', 'cpu', or 'auto'")
    filename: Optional[str] = Field(default=None, description="Original filename for FFmpeg extension hint")


class TranscriptionJob:
    __slots__ = ("future", "model_name", "audio_path", "transcribe_args")

    def __init__(self, future: Future, model_name: str, audio_path: str, transcribe_args: dict):
        self.future = future
        self.model_name = model_name
        self.audio_path = audio_path
        self.transcribe_args = transcribe_args


class ModelManager:
    def __init__(
        self,
        config: Dict[str, Any],
        models_cache_dir: Path,
        batch_window: float = 0.02,
        max_batch_size: int = 8,
    ):
        self.config = config
        self.models_cache_dir = models_cache_dir
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.model = None
        self.loaded_model_name = None

        requested_device = config.get("device", "auto")
        if requested_device in ("auto", "", None):
            self.device = self._auto_detect_device()
        else:
            self.device = requested_device
        self.device = str(self.device).lower()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            ASCIIColors.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        self.last_used_time = time.time()
        self.lock = threading.Lock()
        self.queue: queue.Queue[Optional[TranscriptionJob]] = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()

    def _auto_detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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
                    f"Falling back to CPU. Error: {load_error}"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                try:
                    self.model = whisper.load_model(model_name, device="cpu")
                    self.loaded_model_name = model_name
                    self.last_used_time = time.time()
                    ASCIIColors.green(f"Whisper model '{model_name}' loaded on CPU fallback.")
                    return
                except Exception as cpu_err:
                    self.model = None
                    self.loaded_model_name = None
                    raise RuntimeError(
                        f"Failed to load Whisper model '{model_name}' on GPU and CPU: {cpu_err}"
                    ) from cpu_err

        if load_error is not None:
            raise RuntimeError(f"Failed to load Whisper model '{model_name}': {load_error}")

        try:
            if lock_file.exists() and not lock.is_locked:
                lock_file.unlink(missing_ok=True)
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

    def _batch_worker(self):
        while not self._stop_event.is_set():
            try:
                first_job = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if first_job is None:
                break

            batch: List[TranscriptionJob] = [first_job]

            if self.batch_window > 0:
                time.sleep(self.batch_window)

            while len(batch) < self.max_batch_size:
                try:
                    next_job = self.queue.get_nowait()
                    if next_job is None:
                        self._stop_event.set()
                        break
                    batch.append(next_job)
                except queue.Empty:
                    break

            try:
                self._process_batch(batch)
            except Exception as e:
                trace_exception(e)
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(e)

    def _process_batch(self, batch: List[TranscriptionJob]):
        if not batch:
            return

        with self.lock:
            self.last_used_time = time.time()
            target_model = batch[0].model_name
            if self.loaded_model_name != target_model:
                self._load_whisper_model(target_model)

        if self.model is None:
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(RuntimeError(f"Model '{target_model}' failed to load."))
            return

        short_jobs: List[Tuple[TranscriptionJob, np.ndarray]] = []
        long_or_fallback_jobs: List[TranscriptionJob] = []

        for job in batch:
            try:
                audio = whisper.load_audio(job.audio_path)
                duration = len(audio) / whisper.audio.SAMPLE_RATE
                if duration <= 30.0:
                    short_jobs.append((job, audio))
                else:
                    long_or_fallback_jobs.append(job)
            except Exception as e:
                job.future.set_exception(e)

        if short_jobs:
            groups: Dict[Tuple[str, Optional[str], Optional[bool]], List[Tuple[TranscriptionJob, np.ndarray]]] = {}
            for job, audio in short_jobs:
                key = (
                    job.transcribe_args.get("task", "transcribe"),
                    job.transcribe_args.get("language"),
                    job.transcribe_args.get("fp16")
                )
                groups.setdefault(key, []).append((job, audio))

            for (task, language, fp16), group in groups.items():
                if len(group) == 1:
                    job, _ = group[0]
                    self._transcribe_single_with_recovery(job)
                else:
                    try:
                        mels = []
                        for _, audio in group:
                            padded = whisper.pad_or_trim(audio)
                            mel = whisper.log_mel_spectrogram(padded, n_mels=self.model.dims.n_mels)
                            mels.append(mel)

                        stacked_mels = torch.stack(mels, dim=0).to(self.device)
                        use_fp16 = fp16 if fp16 is not None else (self.device == "cuda")
                        options = whisper.DecodingOptions(
                            task=task,
                            language=language,
                            fp16=use_fp16,
                            without_timestamps=True
                        )
                        with torch.no_grad():
                            decoded_results = whisper.decode(self.model, stacked_mels, options)

                        for idx, (job, _) in enumerate(group):
                            if not job.future.done():
                                job.future.set_result(decoded_results[idx].text.strip())
                        ASCIIColors.green(f"Batched transcription complete for {len(group)} short audio jobs.")
                    except Exception as batch_err:
                        ASCIIColors.warning(f"Batch decoding failed ({batch_err}), falling back to individual transcription.")
                        for job, _ in group:
                            if not job.future.done():
                                self._transcribe_single_with_recovery(job)

        for job in long_or_fallback_jobs:
            if not job.future.done():
                self._transcribe_single_with_recovery(job)

    def _transcribe_single_with_recovery(self, job: TranscriptionJob):
        try:
            result = self.model.transcribe(str(job.audio_path), **job.transcribe_args)
            job.future.set_result(result.get("text", "").strip())
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
                    f"GPU OOM during transcription of '{Path(job.audio_path).name}'. "
                    f"Escalating model to CPU fallback."
                )
                with self.lock:
                    self._unload_model()
                    self.device = "cpu"
                    try:
                        self._load_whisper_model(job.model_name)
                    except Exception as load_err:
                        job.future.set_exception(load_err)
                        return
                try:
                    job.transcribe_args["fp16"] = False
                    result = self.model.transcribe(str(job.audio_path), **job.transcribe_args)
                    job.future.set_result(result.get("text", "").strip())
                except Exception as retry_err:
                    job.future.set_exception(retry_err)
            else:
                job.future.set_exception(e)


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
                cls._instance.batch_window = kwargs.get("batch_window", 0.02)
                cls._instance.max_batch_size = kwargs.get("max_batch_size", 8)
        return cls._instance

    def get_manager(self, model_name: str, device: str) -> ModelManager:
        key_data = (model_name, device)
        key = hashlib.sha256(str(key_data).encode('utf-8')).hexdigest()
        with self._registry_lock:
            if key not in self._managers:
                config = {"model_name": model_name, "device": device}
                self._managers[key] = ModelManager(
                    config,
                    self.models_cache_dir,
                    batch_window=self.batch_window,
                    max_batch_size=self.max_batch_size,
                )
            return self._managers[key]

    def get_active_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return [m for m in self._managers.values() if m.is_loaded()]

    def get_all_managers(self) -> List[ModelManager]:
        with self._registry_lock:
            return list(self._managers.values())


class ServerState:
    def __init__(self, models_cache_dir: Path, auth_token: Optional[str] = None, batch_window: float = 0.02, max_batch_size: int = 8):
        self.models_cache_dir = models_cache_dir
        self.auth_token = auth_token
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.registry = WhisperRegistry(
            models_cache_dir=models_cache_dir,
            batch_window=batch_window,
            max_batch_size=max_batch_size,
        )


state: Optional[ServerState] = None

app = FastAPI(title="Whisper STT Server", description="Shared model server with dynamic continuous micro-batching.")
router = APIRouter()


def verify_auth_token(
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    """Enforces constant-time authentication token validation if configured on the server."""
    if not state or not state.auth_token:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_server_token:
        token = x_server_token.strip()

    if not token or not hmac.compare_digest(token, state.auth_token):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing authentication token")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    return {
        "status": "running",
        "batch_window": state.batch_window if state else 0.02,
        "max_batch_size": state.max_batch_size if state else 8,
        "active_models": [m.loaded_model_name for m in state.registry.get_active_managers() if m.is_loaded()] if state else []
    }


@router.get("/ps")
def ps(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)
    if not state:
        return []
    return [{
        "model_name": m.loaded_model_name,
        "is_loaded": m.is_loaded(),
        "device": m.device,
        "queue_size": m.queue.qsize(),
        "last_used": time.ctime(m.last_used_time)
    } for m in state.registry.get_all_managers()]


@router.post("/shutdown")
def shutdown(authorization: Optional[str] = Header(None), x_server_token: Optional[str] = Header(None)):
    verify_auth_token(authorization, x_server_token)

    def _delayed_exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down"}


@router.post("/transcribe")
async def transcribe(
    request: TranscriptionRequest,
    authorization: Optional[str] = Header(None),
    x_server_token: Optional[str] = Header(None)
):
    verify_auth_token(authorization, x_server_token)
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
            raise HTTPException(status_code=413, detail=f"Audio payload too large: {len(audio_bytes):,} bytes.")

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

        future: Future = Future()
        job = TranscriptionJob(
            future=future,
            model_name=model_name,
            audio_path=temp_file.name,
            transcribe_args=transcribe_args
        )
        manager.queue.put(job)
        text = future.result(timeout=600)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        trace_exception(e)
        error_text = str(e)
        friendly = f"Whisper transcription failed: {error_text}"
        lowered = error_text.lower()
        if "ffmpeg" in lowered:
            friendly += " | Ensure FFmpeg is installed and accessible."
        elif "out of memory" in lowered or "cuda" in lowered:
            friendly += " | GPU memory limit exceeded. Try a smaller model size or CPU execution."
        raise HTTPException(status_code=500, detail=friendly)
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Whisper STT Shared Daemon Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9633)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--batch-window", type=float, default=0.02)
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    auth_token = args.token
    token_file = cache_path / "whisper_server.token"
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

    state = ServerState(
        models_cache_dir=cache_path,
        auth_token=auth_token,
        batch_window=args.batch_window,
        max_batch_size=args.max_batch_size,
    )

    ASCIIColors.cyan("--- Whisper STT Shared Server ----------------------------------")
    ASCIIColors.green(f"Host:Port      : http://{args.host}:{args.port}")
    ASCIIColors.green(f"Cache Path     : {cache_path}")
    ASCIIColors.green(f"Micro-Batching : window={args.batch_window}s, max_batch={args.max_batch_size}")
    ASCIIColors.green(f"Auth Protected : {'Yes' if auth_token else 'No'}")
    ASCIIColors.cyan("----------------------------------------------------------------")

    uvicorn.run(app, host=args.host, port=args.port, reload=False, log_level="warning")