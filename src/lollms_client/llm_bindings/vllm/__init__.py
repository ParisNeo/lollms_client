# lollms_client/llm_bindings/vllm/__init__.py
from __future__ import annotations

import atexit
import base64
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pipmaster as pm

pm.ensure_packages([
    "openai",
    "filelock",
    "requests",
    "psutil",
    "tqdm",
    "huggingface_hub"
])

import openai
import psutil
import requests
from ascii_colors import ASCIIColors, trace_exception
from filelock import FileLock
from huggingface_hub import hf_hub_download, snapshot_download

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE

BindingName = "VLLMBinding"
DEFAULT_MODELS_FOLDER = Path.home() / ".lollms" / "bindings_models" / "vllm_models"


def get_free_port(start_port: int = 8000, max_port: int = 9000) -> int:
    """Finds an available TCP port on localhost."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port available for vLLM server in range.")


def _read_file_as_base64(path: Union[str, Path]) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_markdown_path(s: str) -> str:
    s = s.strip()
    if s.startswith("[") and s.endswith(")"):
        lb, rb = s.find("["), s.find("]")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb + 1 : rb].strip()
    return s


def _guess_mime_from_name(name: str, default: str = "image/jpeg") -> str:
    mime, _ = mimetypes.guess_type(name)
    return mime or default


def _to_data_url(b64_str: str, mime: str) -> str:
    return f"data:{mime};base64,{b64_str}"


def normalize_image_input(img: Any, default_mime: str = "image/jpeg", glm_format: bool = False) -> Dict[str, Any]:
    if isinstance(img, dict):
        if "url" in img and isinstance(img["url"], str):
            return {"type": "image_url", "image_url": {"url": img["url"]}}
        if "data" in img and isinstance(img["data"], str):
            mime = img.get("mime", default_mime)
            raw = img["data"]
            url = raw if raw.startswith(("http://", "https://", "data:")) else _to_data_url(raw, mime)
            return {"type": "image_url", "image_url": {"url": url}}
        if "path" in img and isinstance(img["path"], str):
            p = _extract_markdown_path(img["path"])
            b64 = _read_file_as_base64(p)
            mime = _guess_mime_from_name(p, default_mime)
            return {"type": "image_url", "image_url": {"url": _to_data_url(b64, mime)}}
        raise ValueError("Unsupported dict format for image input")

    if isinstance(img, str):
        s = _extract_markdown_path(img)
        if s.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": s}}
        if os.path.exists(s) or (":" in s and "\\" in s) or s.startswith(("/", ".")):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            return {"type": "image_url", "image_url": {"url": _to_data_url(b64, mime)}}
        return {"type": "image_url", "image_url": {"url": _to_data_url(s, default_mime)}}

    raise ValueError("Unsupported image input type")


def normalize_video_input(video: Any, default_mime: str = "video/mp4") -> Dict[str, Any]:
    if isinstance(video, dict):
        if "url" in video and isinstance(video["url"], str):
            return {"type": "video_url", "video_url": {"url": video["url"]}}
        if "data" in video and isinstance(video["data"], str):
            mime = video.get("mime", default_mime)
            raw = video["data"]
            url = raw if raw.startswith(("http://", "https://", "data:")) else _to_data_url(raw, mime)
            return {"type": "video_url", "video_url": {"url": url}}
        if "path" in video and isinstance(video["path"], str):
            p = _extract_markdown_path(video["path"])
            b64 = _read_file_as_base64(p)
            mime = _guess_mime_from_name(p, default_mime)
            return {"type": "video_url", "video_url": {"url": _to_data_url(b64, mime)}}
        raise ValueError("Unsupported dict format for video input")

    if isinstance(video, str):
        s = _extract_markdown_path(video)
        if s.startswith(("http://", "https://", "data:")):
            return {"type": "video_url", "video_url": {"url": s}}
        if os.path.exists(s) or (":" in s and "\\" in s) or s.startswith(("/", ".")):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            return {"type": "video_url", "video_url": {"url": _to_data_url(b64, mime)}}
        return {"type": "video_url", "video_url": {"url": _to_data_url(s, default_mime)}}

    raise ValueError("Unsupported video input type")


def extract_reasoning(obj: Any) -> Optional[str]:
    if obj is None:
        return None

    candidate_keys = (
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_text",
        "thought",
        "thoughts",
    )

    if isinstance(obj, dict):
        for k in candidate_keys:
            val = obj.get(k)
            if val is not None and val != "":
                return str(val)
        return None

    for k in candidate_keys:
        try:
            val = getattr(obj, k, None)
            if val is not None and val != "":
                return str(val)
        except Exception:
            pass

    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        for k in candidate_keys:
            val = model_extra.get(k)
            if val is not None and val != "":
                return str(val)

    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        for k in candidate_keys:
            val = obj_dict.get(k)
            if val is not None and val != "":
                return str(val)

    return None


class _StreamThinkingHandler:
    def __init__(self, streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None):
        self.callback = streaming_callback
        self.in_dedicated_reasoning = False
        self.dedicated_reasoning_opened = False
        self.in_content_thinking = False
        self.buffer = ""
        self.output = ""

    def process_reasoning(self, reasoning: str) -> bool:
        if not reasoning:
            return True

        if not self.in_dedicated_reasoning:
            self.in_dedicated_reasoning = True
            self.dedicated_reasoning_opened = True
            open_tag = "<think>\n"
            self.output += open_tag
            if self.callback:
                if self.callback(open_tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                    return False

        self.output += reasoning
        if self.callback:
            if self.callback(reasoning, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                return False

        return True

    def _close_dedicated_reasoning(self) -> bool:
        if self.in_dedicated_reasoning:
            self.in_dedicated_reasoning = False
            close_tag = "\n</think>\n"
            self.output += close_tag
            if self.callback:
                if self.callback(close_tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                    return False
        return True

    def process_content(self, content: str) -> bool:
        if not self._close_dedicated_reasoning():
            return False

        if not content:
            return True

        if self.dedicated_reasoning_opened:
            self.output += content
            if self.callback:
                return self.callback(content, MSG_TYPE.MSG_TYPE_CHUNK) is not False
            return True

        text = self.buffer + content
        self.buffer = ""

        open_tag_re = re.compile(r'<(think|thinking)>', re.IGNORECASE)
        close_tag_re = re.compile(r'</(think|thinking)>', re.IGNORECASE)

        while text:
            if not self.in_content_thinking:
                m = open_tag_re.search(text)
                if m:
                    pre = text[:m.start()]
                    if pre:
                        self.output += pre
                        if self.callback and self.callback(pre, MSG_TYPE.MSG_TYPE_CHUNK) is False:
                            return False

                    tag = m.group(0)
                    self.output += tag
                    self.in_content_thinking = True
                    if self.callback and self.callback(tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                        return False

                    text = text[m.end():]
                else:
                    for i in range(min(len(text), 10), 0, -1):
                        suffix = text[-i:].lower()
                        if "<thinking"[:i] == suffix or "<think"[:i] == suffix:
                            self.buffer = text[-i:]
                            text = text[:-i]
                            break

                    if text:
                        self.output += text
                        if self.callback and self.callback(text, MSG_TYPE.MSG_TYPE_CHUNK) is False:
                            return False
                    break
            else:
                m = close_tag_re.search(text)
                if m:
                    thought_part = text[:m.start()]
                    if thought_part:
                        self.output += thought_part
                        if self.callback and self.callback(thought_part, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                            return False

                    tag = m.group(0)
                    self.output += tag
                    self.in_content_thinking = False
                    if self.callback and self.callback(tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                        return False

                    text = text[m.end():]
                else:
                    for i in range(min(len(text), 11), 0, -1):
                        suffix = text[-i:].lower()
                        if "</thinking"[:i] == suffix or "</think"[:i] == suffix:
                            self.buffer = text[-i:]
                            text = text[:-i]
                            break

                    if text:
                        self.output += text
                        if self.callback and self.callback(text, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                            return False
                    break

        return True

    def flush(self) -> str:
        self._close_dedicated_reasoning()

        if self.buffer:
            msg_type = MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK if self.in_content_thinking else MSG_TYPE.MSG_TYPE_CHUNK
            self.output += self.buffer
            if self.callback:
                self.callback(self.buffer, msg_type)
            self.buffer = ""

        if self.in_content_thinking:
            self.in_content_thinking = False
            close_tag = "\n</think>\n"
            self.output += close_tag
            if self.callback:
                self.callback(close_tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)

        return self.output


class VLLMBinding(LollmsLLMBinding):
    """
    High-performance vLLM binding with multi-process server mutualization and recipe presets.
    """

    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)

        # ── 1. Apply Preset if specified ──────────────────────────────────────
        self.preset_name = kwargs.get("preset")
        effective_kwargs = self._load_preset_defaults(self.preset_name)
        effective_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        self.config = effective_kwargs

        self.model_name = effective_kwargs.get("model_name")
        self.host = effective_kwargs.get("host", "127.0.0.1")
        self.port = effective_kwargs.get("port")
        self.tensor_parallel_size = effective_kwargs.get("tensor_parallel_size")
        self.pipeline_parallel_size = effective_kwargs.get("pipeline_parallel_size")
        self.gpu_memory_utilization = effective_kwargs.get("gpu_memory_utilization", 0.90)
        self.quantization = effective_kwargs.get("quantization")
        self.dtype = effective_kwargs.get("dtype", "auto")
        self.max_model_len = effective_kwargs.get("max_model_len")
        self.kv_cache_dtype = effective_kwargs.get("kv_cache_dtype")
        self.block_size = effective_kwargs.get("block_size")
        self.enable_prefix_caching = effective_kwargs.get("enable_prefix_caching")
        self.enable_chunked_prefill = effective_kwargs.get("enable_chunked_prefill")
        self.max_num_batched_tokens = effective_kwargs.get("max_num_batched_tokens")
        self.max_num_seqs = effective_kwargs.get("max_num_seqs")
        self.speculative_config = effective_kwargs.get("speculative_config")
        self.tool_call_parser = effective_kwargs.get("tool_call_parser")
        self.reasoning_parser = effective_kwargs.get("reasoning_parser")
        self.enable_expert_parallel = effective_kwargs.get("enable_expert_parallel")
        self.cpu_offload_gb = effective_kwargs.get("cpu_offload_gb")
        self.trust_remote_code = effective_kwargs.get("trust_remote_code")
        self.enforce_eager = effective_kwargs.get("enforce_eager")
        self.idle_timeout = float(effective_kwargs.get("idle_timeout", 600.0))
        self.max_active_models = int(effective_kwargs.get("max_active_models", 1))

        self.glm_image_embedding = bool(effective_kwargs.get("glm_image_embedding", False))
        self.video_enabled = bool(effective_kwargs.get("video_enabled", False))

        raw_efforts = effective_kwargs.get("supported_reasoning_efforts")
        if isinstance(raw_efforts, str) and raw_efforts.strip():
            self.supported_reasoning_efforts = [s.strip() for s in raw_efforts.split(",") if s.strip()]
        elif isinstance(raw_efforts, list):
            self.supported_reasoning_efforts = raw_efforts
        else:
            default_efforts = ["low", "high", "max"] if self.glm_image_embedding else ["low", "medium", "high"]
            self.supported_reasoning_efforts = default_efforts

        self.models_folder = Path(effective_kwargs.get("models_folder") or DEFAULT_MODELS_FOLDER).resolve()
        self.models_folder.mkdir(parents=True, exist_ok=True)
        self.servers_dir = self.models_folder / "servers"
        self.servers_dir.mkdir(parents=True, exist_ok=True)

        self.global_lock_path = self.models_folder / "global_vllm_manager.lock"

        self.client: Optional[openai.OpenAI] = None
        self.active_server_info: Optional[Dict[str, Any]] = None

        if self.model_name and kwargs.get("auto_start_server", False):
            self.load_model(self.model_name)

        atexit.register(self.cleanup_client_registration)

    @staticmethod
    def _load_preset_defaults(preset_id: Optional[str]) -> Dict[str, Any]:
        """Loads preset configuration parameters from description.yaml."""
        if not preset_id:
            return {}
        try:
            import yaml
            desc_file = Path(__file__).parent / "description.yaml"
            if desc_file.exists():
                with open(desc_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    presets = data.get("presets", [])
                    for p in presets:
                        if p.get("id") == preset_id:
                            ASCIIColors.info(f"[vLLM Preset] Loaded recipe '{p.get('title')}'")
                            return dict(p.get("parameters", {}))
        except Exception as e:
            ASCIIColors.warning(f"Could not load vLLM preset '{preset_id}': {e}")
        return {}

    def _get_registry_file(self, model_name: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "__", model_name)
        return self.servers_dir / f"{safe_name}.json"

    def _get_server_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        reg_file = self._get_registry_file(model_name)
        if not reg_file.exists():
            return None
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            pid = info.get("pid")
            port = info.get("port")
            if pid and psutil.pid_exists(pid):
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
                    if r.status_code == 200:
                        return info
                except Exception:
                    pass
            reg_file.unlink(missing_ok=True)
        except Exception:
            reg_file.unlink(missing_ok=True)
        return None

    def _build_server_command(self, model_name: str, port: int) -> List[str]:
        """
        Builds the vLLM OpenAI-compatible server command line.
        CRITICAL: All arguments with None, empty string, or False boolean values are filtered out.
        """
        cmd: List[str] = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_name),
            "--host", str(self.host),
            "--port", str(port),
        ]

        def _add_arg(flag: str, val: Any) -> None:
            if val is None:
                return
            if isinstance(val, bool):
                if val is True:
                    cmd.append(flag)
            elif isinstance(val, (int, float)):
                cmd.extend([flag, str(val)])
            elif isinstance(val, (dict, list)):
                cmd.extend([flag, json.dumps(val)])
            else:
                s_val = str(val).strip()
                if s_val and s_val.lower() not in ("null", "none"):
                    cmd.extend([flag, s_val])

        _add_arg("--tensor-parallel-size", self.tensor_parallel_size)
        _add_arg("--pipeline-parallel-size", self.pipeline_parallel_size)
        _add_arg("--gpu-memory-utilization", self.gpu_memory_utilization)
        _add_arg("--dtype", self.dtype if self.dtype and self.dtype != "auto" else None)
        _add_arg("--quantization", self.quantization)
        _add_arg("--max-model-len", self.max_model_len)
        _add_arg("--kv-cache-dtype", self.kv_cache_dtype if self.kv_cache_dtype and self.kv_cache_dtype != "auto" else None)
        _add_arg("--block-size", self.block_size)
        _add_arg("--enable-prefix-caching", self.enable_prefix_caching)
        _add_arg("--enable-chunked-prefill", self.enable_chunked_prefill)
        _add_arg("--max-num-batched-tokens", self.max_num_batched_tokens)
        _add_arg("--max-num-seqs", self.max_num_seqs)
        _add_arg("--speculative-config", self.speculative_config)
        _add_arg("--tool-call-parser", self.tool_call_parser)
        _add_arg("--reasoning-parser", self.reasoning_parser)
        _add_arg("--enable-expert-parallel", self.enable_expert_parallel)
        _add_arg("--cpu-offload-gb", self.cpu_offload_gb)
        _add_arg("--trust-remote-code", self.trust_remote_code)
        _add_arg("--enforce-eager", self.enforce_eager)

        return cmd

    def _spawn_server_daemon(self, model_name: str, port: int) -> Tuple[int, str]:
        cmd = self._build_server_command(model_name, port)

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "__", model_name)
        log_path = self.servers_dir / f"{safe_name}.log"
        log_f = open(log_path, "w", encoding="utf-8")

        popen_kwargs: Dict[str, Any] = {
            "stdout": log_f,
            "stderr": subprocess.STDOUT,
        }
        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True

        ASCIIColors.info(f"[vLLM Mutualization] Spawning daemon for '{model_name}' on port {port}...")
        ASCIIColors.info(f"Command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, **popen_kwargs)

        deadline = time.time() + 300
        while time.time() < deadline:
            if proc.poll() is not None:
                log_f.close()
                err_snippet = ""
                if log_path.exists():
                    try:
                        err_snippet = log_path.read_text(encoding="utf-8", errors="ignore")[-1000:]
                    except Exception:
                        pass
                raise RuntimeError(
                    f"vLLM server exited prematurely with code {proc.returncode}.\nLog snippet:\n{err_snippet}"
                )
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                if r.status_code == 200:
                    log_f.close()
                    url = f"http://{self.host}:{port}/v1"
                    return proc.pid, url
            except Exception:
                pass
            time.sleep(1.0)

        proc.terminate()
        log_f.close()
        raise TimeoutError(f"vLLM server for '{model_name}' did not become ready within 300 seconds.")

    def load_model(self, model_name: str) -> bool:
        self.models_folder.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.global_lock_path), timeout=360)

        with lock:
            info = self._get_server_info(model_name)
            my_pid = os.getpid()

            if info:
                clients = info.get("clients", [])
                if my_pid not in clients:
                    clients.append(my_pid)
                    info["clients"] = clients
                info["last_used"] = time.time()
                reg_file = self._get_registry_file(model_name)
                with open(reg_file, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)

                self.active_server_info = info
                self.model_name = model_name
                self.client = openai.OpenAI(base_url=info["url"], api_key="EMPTY")
                ASCIIColors.success(f"[vLLM Mutualization] Attached to existing shared vLLM server on port {info['port']} (PID {info['pid']}).")
                return True

            port = self.port or get_free_port()
            pid, url = self._spawn_server_daemon(model_name, port)

            info = {
                "model_name": model_name,
                "pid": pid,
                "port": port,
                "url": url,
                "clients": [my_pid],
                "started_at": time.time(),
                "last_used": time.time(),
            }
            reg_file = self._get_registry_file(model_name)
            with open(reg_file, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)

            self.active_server_info = info
            self.model_name = model_name
            self.client = openai.OpenAI(base_url=url, api_key="EMPTY")
            ASCIIColors.success(f"[vLLM Mutualization] New shared vLLM server initialized on port {port} (PID {pid}).")
            return True

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        target = model_name or self.model_name
        if not target:
            return False

        lock = FileLock(str(self.global_lock_path), timeout=60)
        with lock:
            info = self._get_server_info(target)
            if not info:
                return False

            pid = info.get("pid")
            my_pid = os.getpid()
            clients = [c for c in info.get("clients", []) if c != my_pid and psutil.pid_exists(c)]

            if clients:
                info["clients"] = clients
                with open(self._get_registry_file(target), "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)
                ASCIIColors.info(f"[vLLM Mutualization] Disconnected from server '{target}' (other clients still active: {clients}).")
                return True

            ASCIIColors.info(f"[vLLM Mutualization] Terminating shared vLLM daemon '{target}' (PID {pid})...")
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            self._get_registry_file(target).unlink(missing_ok=True)
            self.client = None
            self.active_server_info = None
            return True

    def cleanup_client_registration(self):
        if self.model_name:
            try:
                self.unload_model(self.model_name)
            except Exception:
                pass

    def generate_text(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        system_prompt: str = "",
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        split: Optional[bool] = False,
        user_keyword: Optional[str] = "!@>user:",
        ai_keyword: Optional[str] = "!@>assistant:",
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        if not self.client:
            raise RuntimeError("vLLM model server is not loaded. Call load_model() first.")

        effort = self.get_effective_reasoning_effort(think=think, reasoning_effort=reasoning_effort)
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]

        media_blocks = []
        if images:
            media_blocks.extend([normalize_image_input(img, glm_format=self.glm_image_embedding) for img in images])
        if videos:
            media_blocks.extend([normalize_video_input(vid) for vid in videos])

        if media_blocks:
            if split:
                messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
                last = messages[-1]
                last["content"] = [{"type": "text", "text": last["content"]}] + media_blocks
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": prompt}] + media_blocks})
        else:
            if split:
                messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": n_predict or 2048,
            "temperature": temperature if temperature is not None else 0.7,
            "top_p": top_p if top_p is not None else 0.95,
            "stream": stream if stream is not None else (streaming_callback is not None),
        }
        if seed is not None:
            params["seed"] = seed

        if effort is not None and not self.glm_image_embedding:
            params.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = True

        output = ""
        try:
            if params["stream"]:
                handler = _StreamThinkingHandler(streaming_callback)
                stream_res = self.client.chat.completions.create(**params)
                for chunk in stream_res:
                    if self.is_cancelled():
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    r_text = extract_reasoning(delta)
                    c_text = getattr(delta, "content", None)

                    if r_text:
                        if not handler.process_reasoning(r_text):
                            break
                        continue
                    if c_text:
                        if not handler.process_content(c_text):
                            break
                output = handler.flush()
            else:
                resp = self.client.chat.completions.create(**params)
                msg_obj = resp.choices[0].message
                r_text = extract_reasoning(msg_obj)
                c_text = msg_obj.content or ""
                if r_text and not c_text.strip().startswith(("<think>", "<thinking>")):
                    output = f"<think>\n{r_text}\n</think>\n{c_text}"
                else:
                    output = c_text
        except Exception as e:
            trace_exception(e)
            return {"status": "error", "message": f"vLLM API error: {e}"}

        return output

    def generate_from_messages(
        self,
        messages: List[Dict],
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs,
    ) -> Union[str, dict]:
        if not self.client:
            raise RuntimeError("vLLM model server is not loaded. Call load_model() first.")

        def normalize_msg(msg: Dict) -> Dict:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts = []
            images = []
            videos = []

            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") in ("image_url", "input_image"):
                        val = item.get("image_url")
                        if isinstance(val, dict):
                            val = val.get("url") or val.get("base64")
                        if isinstance(val, str) and val:
                            images.append(val)
                    elif item.get("type") in ("video_url", "video"):
                        val = item.get("video_url")
                        if isinstance(val, dict):
                            val = val.get("url")
                        if isinstance(val, str) and val:
                            videos.append(val)

            if "images" in msg and msg["images"]:
                images.extend(msg["images"])
            if "videos" in msg and msg["videos"]:
                videos.extend(msg["videos"])

            text_content = "\n".join(p for p in text_parts if p.strip())
            if not images and not videos:
                return {"role": role, "content": text_content}

            openai_content = []
            if text_content:
                openai_content.append({"type": "text", "text": text_content})
            for img in images:
                openai_content.append(normalize_image_input(img, glm_format=self.glm_image_embedding))
            for vid in videos:
                openai_content.append(normalize_video_input(vid))
            return {"role": role, "content": openai_content}

        vllm_messages = [normalize_msg(m) for m in messages]
        effort = self.get_effective_reasoning_effort(think=think, reasoning_effort=reasoning_effort)

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": vllm_messages,
            "max_tokens": n_predict or 2048,
            "temperature": temperature if temperature is not None else 0.7,
            "top_p": top_p if top_p is not None else 0.95,
            "stream": stream if stream is not None else (streaming_callback is not None),
        }
        if seed is not None:
            params["seed"] = seed

        if effort is not None and not self.glm_image_embedding:
            params.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = True

        raw_tools = kwargs.get("tools")
        if raw_tools and isinstance(raw_tools, list):
            params["tools"] = raw_tools
            params["tool_choice"] = "auto"

        output = ""
        try:
            if params["stream"]:
                handler = _StreamThinkingHandler(streaming_callback)
                res = self.client.chat.completions.create(**params)
                for chunk in res:
                    if self.is_cancelled():
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    r_text = extract_reasoning(delta)
                    c_text = getattr(delta, "content", None)
                    if r_text:
                        if not handler.process_reasoning(r_text):
                            break
                        continue
                    if c_text:
                        if not handler.process_content(c_text):
                            break
                output = handler.flush()
            else:
                res = self.client.chat.completions.create(**params)
                msg_obj = res.choices[0].message
                r_text = extract_reasoning(msg_obj)
                c_text = msg_obj.content or ""
                if r_text and not c_text.strip().startswith(("<think>", "<thinking>")):
                    output = f"<think>\n{r_text}\n</think>\n{c_text}"
                else:
                    output = c_text
        except Exception as e:
            trace_exception(e)
            return {"status": "error", "message": f"vLLM API error: {e}"}

        return output

    def tokenize(self, text: str) -> List[int]:
        if not text:
            return []
        try:
            if self.active_server_info:
                url = f"http://127.0.0.1:{self.active_server_info['port']}/tokenize"
                r = requests.post(url, json={"model": self.model_name, "prompt": text}, timeout=10)
                if r.status_code == 200:
                    return r.json().get("tokens", [])
        except Exception:
            pass
        return list(tiktoken.get_encoding("cl100k_base").encode(text))

    def detokenize(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        try:
            if self.active_server_info:
                url = f"http://127.0.0.1:{self.active_server_info['port']}/detokenize"
                r = requests.post(url, json={"model": self.model_name, "tokens": tokens}, timeout=10)
                if r.status_code == 200:
                    return r.json().get("prompt", "")
        except Exception:
            pass
        return tiktoken.get_encoding("cl100k_base").decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def embed(self, text: str | list[str], **kwargs) -> list:
        if not self.client:
            raise RuntimeError("vLLM server is not loaded.")
        input_texts = [text] if isinstance(text, str) else text
        res = self.client.embeddings.create(model=self.model_name, input=input_texts)
        embeddings = [item.embedding for item in res.data]
        return embeddings[0] if isinstance(text, str) else embeddings

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        if self.client:
            try:
                res = self.client.models.list()
                for m in res.data:
                    models.append({
                        "model_name": m.id,
                        "owned_by": "vllm",
                        "running": True,
                        "port": self.active_server_info.get("port") if self.active_server_info else None
                    })
            except Exception:
                pass
        return models

    def get_model_info(self) -> dict:
        return {
            "name": "vLLM",
            "model_name": self.model_name,
            "host": self.host,
            "port": self.port,
            "active_server": self.active_server_info,
        }

    def ps(self) -> List[Dict[str, Any]]:
        results = []
        for rf in list(self.servers_dir.glob("*.json")):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pid = data.get("pid")
                if not pid or not psutil.pid_exists(pid):
                    rf.unlink(missing_ok=True)
                    continue
                proc = psutil.Process(pid)
                mem = proc.memory_info()
                results.append({
                    "model_name": data.get("model_name"),
                    "pid": pid,
                    "port": data.get("port"),
                    "url": data.get("url"),
                    "clients": data.get("clients", []),
                    "rss_mb": round(mem.rss / 1024 / 1024, 1),
                    "started_at": data.get("started_at"),
                })
            except Exception:
                pass
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Installation & Update Commands
    # ──────────────────────────────────────────────────────────────────────────

    def install_vllm(
        self,
        force: bool = False,
        cuda_version: str = "auto",
        progress_callback: Optional[Callable[[dict], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Installs vLLM into the Python environment.
        Supports automatic CUDA backend selection using uv or pip.
        """
        def report(status: str, msg: str, percent: int = 0):
            ASCIIColors.info(f"[vLLM Installer] {msg}")
            if progress_callback:
                progress_callback({"status": status, "message": msg, "completed": percent, "total": 100})

        report("starting", "Detecting platform environment...", 10)
        sys_plat = platform.system()

        if sys_plat == "Windows":
            msg = (
                "Native Windows is not officially supported by vLLM. "
                "Please run vLLM inside WSL 2 (Ubuntu) with 'uv pip install vllm --torch-backend=auto', "
                "or connect to a remote vLLM instance using the host_address parameter."
            )
            report("error", msg, 0)
            return {"status": False, "message": msg}

        # Check if already installed and not forced
        if not force:
            try:
                import vllm
                v_ver = getattr(vllm, "__version__", "unknown")
                msg = f"vLLM is already installed (version {v_ver}). Use force=True to re-install."
                report("success", msg, 100)
                return {"status": True, "message": msg}
            except ImportError:
                pass

        has_uv = shutil.which("uv") is not None
        installer_cmd: List[str] = []

        if has_uv:
            report("working", "Using uv package manager for installation...", 25)
            installer_cmd = ["uv", "pip", "install", "vllm"]
            if cuda_version:
                installer_cmd.append(f"--torch-backend={cuda_version}")
        else:
            report("working", "Using standard pip installer...", 25)
            installer_cmd = [sys.executable, "-m", "pip", "install", "vllm"]
            if force:
                installer_cmd.append("--force-reinstall")

        report("working", f"Executing: {' '.join(installer_cmd)}", 40)
        try:
            proc = subprocess.Popen(
                installer_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line_str = line.strip()
                if line_str and progress_callback:
                    progress_callback({"status": "installing", "message": line_str, "completed": 60, "total": 100})

            proc.wait()
            if proc.returncode == 0:
                report("success", "vLLM successfully installed.", 100)
                return {"status": True, "message": "vLLM successfully installed."}
            else:
                err_msg = f"Installation command failed with exit code {proc.returncode}."
                report("error", err_msg, 0)
                return {"status": False, "message": err_msg}
        except Exception as e:
            trace_exception(e)
            report("error", f"Installation failed: {e}", 0)
            return {"status": False, "message": str(e)}

    def update(self, progress_callback: Optional[Callable[[dict], None]] = None, **kwargs) -> Dict[str, Any]:
        """
        Terminates running vLLM daemon servers and upgrades vLLM to the latest release.
        """
        def report(status: str, msg: str, percent: int = 0):
            ASCIIColors.info(f"[vLLM Update] {msg}")
            if progress_callback:
                progress_callback({"status": status, "message": msg, "completed": percent, "total": 100})

        report("stopping", "Stopping all active vLLM daemons before update...", 10)
        if self.model_name:
            self.unload_model(self.model_name)

        # Evict all running servers
        for rf in list(self.servers_dir.glob("*.json")):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                m_name = data.get("model_name")
                if m_name:
                    self.unload_model(m_name)
            except Exception:
                rf.unlink(missing_ok=True)

        time.sleep(1.5)

        has_uv = shutil.which("uv") is not None
        cmd = ["uv", "pip", "install", "-U", "vllm"] if has_uv else [sys.executable, "-m", "pip", "install", "--upgrade", "vllm"]

        report("working", f"Executing upgrade: {' '.join(cmd)}", 40)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                line_str = line.strip()
                if line_str and progress_callback:
                    progress_callback({"status": "updating", "message": line_str, "completed": 75, "total": 100})

            proc.wait()
            if proc.returncode == 0:
                report("success", "vLLM successfully updated.", 100)
                return {"status": True, "message": "vLLM updated successfully."}
            else:
                msg = f"Update command failed with exit code {proc.returncode}."
                report("error", msg, 0)
                return {"status": False, "message": msg}
        except Exception as e:
            trace_exception(e)
            report("error", f"Update failed: {e}", 0)
            return {"status": False, "message": str(e)}

    def update_vllm(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for update command."""
        return self.update(*args, **kwargs)