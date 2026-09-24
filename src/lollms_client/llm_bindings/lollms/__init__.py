# llm_bindings/lollms/__init__.py
from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import ssl
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import openai
import pipmaster as pm
import requests
import tiktoken
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE
from lollms_client.lollms_utilities import encode_image

pm.ensure_packages(["openai", "tiktoken"])

BindingName = "LollmsBinding"

_NIM_FUNCTION_NAME_PLACEHOLDER = "23d4f03a-b8a6-4adb-a183-7daa083a09cc"


def _read_file_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_markdown_path(s):
    s = s.strip()
    if s.startswith("[") and s.endswith(")"):
        lb, rb = s.find("["), s.find("]")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb + 1 : rb].strip()
    return s


def _guess_mime_from_name(name, default="image/jpeg"):
    mime, _ = mimetypes.guess_type(name)
    return mime or default


def _to_data_url(b64_str, mime):
    return f"data:{mime};base64,{b64_str}"


def normalize_image_input(img: Any, default_mime: str = "image/jpeg", glm_format: bool = False) -> Dict[str, Any]:
    """
    Returns an OpenAI Chat Completions-compatible content block:
      { "type": "image_url", "image_url": { "url": "data:<mime>;base64,<...>" } }
    Accepts:
      - dict {'data': '<base64>', 'mime': 'image/png'}
      - dict {'path': 'path/to/img.png'}
      - dict {'url': 'https://...'}
      - string URL ('http://' or 'https://')
      - string data URL ('data:image/...')
      - string local file path
      - string raw base64
    """
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
    """
    Returns an OpenAI / vLLM Chat Completions API-compliant video content block:
      { "type": "video_url", "video_url": { "url": "data:<mime>;base64,<...>" } }
    Accepts URLs, local video files, data URLs, dicts, and raw base64 strings.
    """
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
    """
    Extracts reasoning/thinking content from an OpenAI delta or message object.
    Checks direct attributes, Pydantic v2 model_extra dictionaries, and dict representations.
    """
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
    """
    Manages streaming chunks for OpenAI-compatible endpoints to guarantee that:
    1. Dedicated reasoning fields (reasoning_content, reasoning, etc.) are wrapped
       in visible <think>...</think> tags and dispatched as MSG_TYPE_THOUGHT_CHUNK.
    2. In-content <think>...</think> tags are detected, streaming thoughts as
       MSG_TYPE_THOUGHT_CHUNK and answer text as MSG_TYPE_CHUNK, while preserving
       the <think> and </think> tags in the final output.
    """

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


class LollmsBinding(LollmsLLMBinding):
    def __init__(self, **kwargs):
        super().__init__(BindingName, **kwargs)

        host = kwargs.get("host_address", "http://localhost:9642").rstrip("/")
        if host.endswith("/lollms/v1"):
            host = host[: -len("/lollms/v1")].rstrip("/")
        elif host.endswith("/v1"):
            host = host[: -len("/v1")].rstrip("/")

        self.base_address = host
        self.open_ai_host_address = f"{self.base_address}/v1"
        self.lollms_host_address = f"{self.base_address}/lollms/v1"

        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key") or os.getenv("LOLLMS_API_KEY")
        raw_verify_ssl = kwargs.get("verify_ssl_certificate", True)
        if isinstance(raw_verify_ssl, str):
            self.verify_ssl_certificate = raw_verify_ssl.lower().strip() not in ("false", "0", "no", "off", "")
        else:
            self.verify_ssl_certificate = bool(raw_verify_ssl)
        self.certificate_file_path = kwargs.get("certificate_file_path")

        self.default_completion_format = kwargs.get(
            "default_completion_format", ELF_COMPLETION_FORMAT.Chat
        )
        self.glm_image_embedding = kwargs.get("glm_image_embedding", False)
        self.video_enabled = kwargs.get("video_enabled", False)

        raw_efforts = kwargs.get("supported_reasoning_efforts")
        if isinstance(raw_efforts, str) and raw_efforts.strip():
            self.supported_reasoning_efforts = [s.strip() for s in raw_efforts.split(",") if s.strip()]
        elif isinstance(raw_efforts, list):
            self.supported_reasoning_efforts = raw_efforts
        else:
            default_efforts = ["low", "high", "max"] if self.glm_image_embedding else ["low", "medium", "high"]
            self.supported_reasoning_efforts = default_efforts

        self.is_vllm = kwargs.get("is_vllm", False)
        self.send_thinking_parameter = kwargs.get("send_thinking_parameter", True)
        self.thinking_effort_keyword = kwargs.get("thinking_effort_keyword", "enable_thinking")

        self.verify = True
        verify = True

        if not self.verify_ssl_certificate or os.getenv("LOLLMS_SKIP_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
            self.verify = False
            verify = False
        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)
            if not cert_path.exists():
                raise FileNotFoundError(f"Certificate file not found: {cert_path}")
            ssl_context = ssl.create_default_context(cafile=str(cert_path))
            self.verify = str(cert_path)
            verify = ssl_context

        self._verify_obj = verify
        self._init_client()
        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    def _init_client(self) -> None:
        try:
            if hasattr(self, "_http_client") and self._http_client and not self._http_client.is_closed:
                self._http_client.close()
        except Exception:
            pass
        verify = getattr(self, "_verify_obj", True)
        self._http_client = httpx.Client(verify=verify, timeout=300.0)
        self.client = openai.OpenAI(
            api_key=self.service_key or "nokey",
            base_url=self.open_ai_host_address,
            http_client=self._http_client,
        )

    def _ensure_client(self) -> None:
        if not hasattr(self, "_http_client") or self._http_client is None or self._http_client.is_closed:
            self._init_client()

    def cancel(self) -> None:
        try:
            if hasattr(self, "_http_client") and self._http_client and not self._http_client.is_closed:
                self._http_client.close()
        except Exception as e:
            ASCIIColors.warning(f"[LollmsBinding] Error closing HTTP client during cancel: {e}")
        super().cancel()

    def reset_cancel(self) -> None:
        super().reset_cancel()
        self._ensure_client()

    def close(self) -> None:
        self.cancel()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _lollms_headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
        return headers

    def _lollms_get(self, path: str, timeout: int = 10) -> dict:
        url = f"{self.lollms_host_address}{path}"
        response = requests.get(url, headers=self._lollms_headers(), timeout=timeout, verify=self.verify)
        if response.status_code == 401:
            ASCIIColors.error(f"[LollmsBinding] 401 Unauthorized on GET {path}.")
        response.raise_for_status()
        return response.json()

    def _lollms_post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        url = f"{self.lollms_host_address}{path}"
        response = requests.post(url, json=payload, headers=self._lollms_headers(), timeout=timeout, verify=self.verify)
        if response.status_code == 401:
            ASCIIColors.error(f"[LollmsBinding] 401 Unauthorized on POST {path}.")
        response.raise_for_status()
        return response.json()

    def _apply_vllm_thinking_kwargs(self, params: dict, effort: Optional[str]) -> dict:
        if not self.is_vllm or not self.send_thinking_parameter or getattr(self, "glm_image_embedding", False):
            return params
        params.setdefault("extra_body", {}).setdefault(
            "chat_template_kwargs", {}
        )[self.thinking_effort_keyword] = effort is not None
        return params

    def get_capabilities(self) -> Dict:
        try:
            return self._lollms_get("/capabilities", timeout=10)
        except Exception as e:
            ASCIIColors.warning(f"Failed to fetch capabilities: {e}")
            return {"capabilities": [], "active_bindings": {}}

    def lollms_listMountedPersonalities(self, host_address: str | None = None):
        base = self.lollms_host_address
        if host_address:
            h = host_address.rstrip("/")
            if h.endswith("/lollms/v1"):
                base = h
            elif h.endswith("/v1"):
                base = h[: -len("/v1")] + "/lollms/v1"
            else:
                base = h + "/lollms/v1"

        url = f"{base}/personalities"
        try:
            response = requests.get(url, headers=self._lollms_headers(), timeout=10, verify=self.verify)
            if response.status_code == 200:
                return response.json()
        except Exception as ex:
            return {"status": False, "error": str(ex)}

        return {"status": False, "error": f"Failed to list personalities: HTTP {response.status_code}"}

    def tokenize(self, text: str) -> list:
        if text is None:
            return []

        if not getattr(self, "_remote_tokenizer_healthy", True):
            try:
                return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)
            except Exception:
                return []

        try:
            data = self._lollms_post("/tokenize", {"model": self.model_name, "text": text}, timeout=10)
            self._remote_tokenizer_healthy = True
            if "tokens" in data:
                return data["tokens"]
        except Exception as e:
            self._remote_tokenizer_healthy = False
            ASCIIColors.warning(f"Remote tokenization failed: {e}. Falling back to local tiktoken.")
        try:
            return tiktoken.model.encoding_for_model(self.model_name).encode(text)
        except Exception:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text)

    def detokenize(self, tokens: list) -> str:
        if not tokens:
            return ""
        try:
            data = self._lollms_post("/detokenize", {"model": self.model_name, "tokens": tokens}, timeout=10)
            if "text" in data:
                return data["text"]
        except Exception as e:
            ASCIIColors.warning(f"Remote detokenization failed: {e}. Falling back to local tiktoken.")
        try:
            return tiktoken.model.encoding_for_model(self.model_name).decode(tokens)
        except Exception:
            return tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens)

    def count_tokens(self, text: str) -> int:
        if text is None:
            return 0

        if not getattr(self, "_remote_tokenizer_healthy", True):
            try:
                return len(tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text))
            except Exception:
                return len(text) // 4

        try:
            data = self._lollms_post("/tokenize", {"model": self.model_name, "text": text}, timeout=10)
            self._remote_tokenizer_healthy = True
            if "count" in data:
                return int(data["count"])
            elif "tokens" in data:
                return len(data["tokens"])
        except Exception as e:
            self._remote_tokenizer_healthy = False
            ASCIIColors.warning(f"Remote token count failed: {e}. Falling back to local count.")
        try:
            return len(tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(text))
        except Exception:
            return len(text) // 4

    def _get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        target_model = model_name or self.model_name
        if not target_model:
            return self.default_ctx_size

        if not hasattr(self, "_ctx_size_failures"):
            self._ctx_size_failures = 0

        if self._ctx_size_failures >= 2:
            return 32000

        try:
            data = self._lollms_post("/context_size", {"model": target_model}, timeout=10)
            if "context_size" in data:
                size = int(data["context_size"])
                self._ctx_size_failures = 0
                return size
        except Exception as e:
            self._ctx_size_failures += 1
            if self._ctx_size_failures == 1:
                ASCIIColors.warning(f"Could not retrieve remote context size for '{target_model}': {e}.")
            return 4096

        return 4096

    def long_context_process(
        self,
        text: str,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_generation_tokens: int = 4096,
    ) -> str:
        payload = {
            "text": text,
            "prompt": prompt,
            "model": model or self.model_name,
            "max_generation_tokens": max_generation_tokens,
        }
        try:
            data = self._lollms_post("/long_context_process", payload, timeout=600)
            return data.get("result", "")
        except Exception as e:
            trace_exception(e)
            raise RuntimeError(f"Long context processing failed: {e}")

    def rag_list_databases(self) -> List[Dict]:
        try:
            data = self._lollms_get("/rag/databases", timeout=10)
            return data.get("data", [])
        except Exception as e:
            ASCIIColors.warning(f"Failed to list RAG databases: {e}")
            return []

    def rag_query(
        self,
        datastore_id: str,
        query: str,
        top_k: int = 10,
        min_similarity: float = 50.0,
    ) -> List[Dict]:
        payload = {
            "datastore_id": datastore_id,
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
        try:
            return self._lollms_post("/rag/query", payload, timeout=60)
        except Exception as e:
            trace_exception(e)
            raise RuntimeError(f"RAG query failed: {e}")

    def list_models_by_type(self, binding_type: str = "llm", binding_alias: Optional[str] = None) -> List[Dict]:
        params = {}
        if binding_alias:
            params["binding_alias"] = binding_alias
        try:
            url = f"{self.lollms_host_address}/{binding_type}/models"
            response = requests.get(
                url, headers=self._lollms_headers(), params=params, timeout=15, verify=self.verify
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            ASCIIColors.warning(f"Failed to list {binding_type} models: {e}")
            return []

    def tts_generate(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        language: Optional[str] = None,
        audio_sample: Optional[str] = None,
    ) -> bytes:
        payload: Dict = {
            "input": text,
            "response_format": response_format,
            "speed": speed,
        }
        if voice:
            payload["voice"] = voice
        if model:
            payload["model"] = model
        if language:
            payload["language"] = language
        if audio_sample:
            payload["audio_sample"] = audio_sample

        try:
            url = f"{self.open_ai_host_address}/audio/speech"
            response = requests.post(
                url, json=payload, headers=self._lollms_headers(), timeout=300, verify=self.verify
            )
            if response.status_code == 200:
                return response.content
        except Exception:
            pass

        url = f"{self.lollms_host_address}/audio/speech"
        response = requests.post(
            url, json=payload, headers=self._lollms_headers(), timeout=300, verify=self.verify
        )
        response.raise_for_status()
        return response.content

    def tts_list_voices(self) -> List[Dict]:
        try:
            data = self._lollms_get("/audio/voices", timeout=10)
            return data.get("data", [])
        except Exception as e:
            ASCIIColors.warning(f"Failed to list voices: {e}")
            return []

    def extract_text(self, file_b64: str, filename: str) -> str:
        payload = {"file": file_b64, "filename": filename}
        url = f"{self.open_ai_host_address}/extract_text"
        response = requests.post(
            url, json=payload, headers=self._lollms_headers(), timeout=120, verify=self.verify
        )
        response.raise_for_status()
        return response.json().get("text", "")

    def create_response(
        self,
        input_data,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        stream: bool = False,
        reasoning: Optional[Dict] = None,
    ) -> Dict:
        payload: Dict = {
            "model": self.model_name,
            "input": input_data,
            "stream": stream,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if reasoning:
            payload["reasoning"] = reasoning

        url = f"{self.open_ai_host_address}/responses"
        response = requests.post(
            url, json=payload, headers=self._lollms_headers(), timeout=300, verify=self.verify
        )
        response.raise_for_status()
        return response.json()

    def _build_openai_params(self, messages: list = None, prompt: str = None, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        restricted_families = ["gpt-5", "gpt-4o", "o1", "o3", "o4"]

        allowed_params = {
            "model", "messages", "temperature", "top_p", "n",
            "stop", "max_tokens", "presence_penalty", "frequency_penalty",
            "logit_bias", "stream", "user", "max_completion_tokens", "reasoning_effort",
        }

        if not model:
            raise ValueError("[LollmsBinding] No model name resolved.")

        params: Dict = {"model": model}
        if messages is not None:
            params["messages"] = messages
        if prompt is not None:
            params["prompt"] = prompt

        for k, v in kwargs.items():
            if k in allowed_params and v is not None:
                params[k] = v
            elif v is not None and kwargs.get("debug", False):
                ASCIIColors.warning(f"Removed unsupported OpenAI param '{k}'")

        model_lower = (model or "").lower()
        if any(fam in model_lower for fam in restricted_families):
            if "temperature" in params and params["temperature"] != 1:
                ASCIIColors.warning(f"{model} does not support temperature != 1. Overriding to 1.")
                params["temperature"] = 1
            if "top_p" in params:
                ASCIIColors.warning(f"{model} does not support top_p. Removing it.")
                params.pop("top_p")

        return params

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
        self._ensure_client()
        count = 0
        output = ""

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
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + media_blocks,
                })
        else:
            if split:
                messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

        try:
            if self.completion_format == ELF_COMPLETION_FORMAT.Chat:
                params = self._build_openai_params(
                    messages=messages,
                    n_predict=n_predict,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                )

                if effort is not None:
                    if self.is_vllm:
                        self._apply_vllm_thinking_kwargs(params, effort)
                    else:
                        params["reasoning_effort"] = effort
                        if reasoning_summary and reasoning_summary != "auto":
                            params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary
                        params.pop("temperature", None)
                        params.pop("top_p", None)
                else:
                    if self.is_vllm:
                        self._apply_vllm_thinking_kwargs(params, None)

                try:
                    chat_completion = self.client.chat.completions.create(**params)
                except Exception as ex:
                    trace_exception(ex)
                    if "max_tokens" in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")
                    params.pop("top_p", None)
                    params.pop("frequency_penalty", None)
                    params.pop("reasoning_effort", None)
                    if effort is None:
                        params["temperature"] = 1
                    if "extra_body" in params:
                        params["extra_body"].pop("chat_template_kwargs", None)
                    chat_completion = self.client.chat.completions.create(**params)

                if stream:
                    handler = _StreamThinkingHandler(streaming_callback)
                    for resp in chat_completion:
                        if self.is_cancelled():
                            break
                        if count >= (n_predict or float("inf")):
                            break
                        if not resp.choices:
                            continue

                        delta = resp.choices[0].delta
                        reasoning = extract_reasoning(delta)
                        content = getattr(delta, "content", None)

                        if reasoning:
                            if not handler.process_reasoning(reasoning):
                                break
                            count += 1
                            continue

                        if content:
                            if not handler.process_content(content):
                                break
                            count += 1

                    output = handler.flush()
                else:
                    message_obj = chat_completion.choices[0].message
                    reasoning = extract_reasoning(message_obj)
                    content = message_obj.content or ""
                    if reasoning and not content.strip().startswith(("<think>", "<thinking>")):
                        output = f"<think>\n{reasoning}\n</think>\n{content}"
                    else:
                        output = content
            else:
                params = self._build_openai_params(
                    prompt=prompt,
                    n_predict=n_predict,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                )
                try:
                    completion = self.client.completions.create(**params)
                except Exception as ex:
                    trace_exception(ex)
                    if "max_tokens" in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")
                    params["temperature"] = 1
                    params.pop("top_p", None)
                    params.pop("frequency_penalty", None)
                    completion = self.client.completions.create(**params)

                if stream:
                    for resp in completion:
                        if self.is_cancelled():
                            break
                        if count >= (n_predict or float("inf")):
                            break
                        word = getattr(resp.choices[0], "text", "") or ""
                        if streaming_callback and not streaming_callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
                        if word:
                            output += word
                            count += 1
                else:
                    output = completion.choices[0].text

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

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
        self._ensure_client()
        _OPENAI_ROLE_MAP = {
            "system": "system", "developer": "developer", "user": "user",
            "assistant": "assistant", "tool": "tool", "function": "function",
            "admin": "system", "root": "system", "manager": "system",
            "supervisor": "system", "controller": "system", "orchestrator": "system",
            "planner": "system", "critic": "assistant", "refiner": "assistant",
            "reviewer": "assistant", "validator": "assistant", "executor": "assistant",
            "worker": "assistant", "agent": "assistant", "bot": "assistant",
            "ai": "assistant", "human": "user", "guest": "user",
            "client": "user", "customer": "user", "operator": "user",
        }

        def normalize_message(msg: Dict) -> Dict:
            raw_role = msg.get("role", "user") or "user"
            role = _OPENAI_ROLE_MAP.get(raw_role.lower(), "user")
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
                    elif item.get("type") in ("input_image", "image_url"):
                        val = item.get("image_url")
                        if isinstance(val, dict):
                            val = val.get("url") or val.get("base64")
                        if isinstance(val, str) and val:
                            images.append(val)
                    elif item.get("type") in ("video_url", "video", "input_video"):
                        val = item.get("video_url") or item.get("video")
                        if isinstance(val, dict):
                            val = val.get("url") or val.get("base64")
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
                img_block = normalize_image_input(img, glm_format=self.glm_image_embedding)
                openai_content.append(img_block)
            for vid in videos:
                vid_block = normalize_video_input(vid)
                openai_content.append(vid_block)
            return {"role": role, "content": openai_content}

        openai_messages = [normalize_message(m) for m in messages]

        raw_tools = kwargs.get("tools")
        sanitized_tools = None
        if raw_tools and isinstance(raw_tools, list):
            sanitized_tools = []
            for tool in raw_tools:
                if not isinstance(tool, dict):
                    continue
                if "id" in tool and len(str(tool["id"])) == 36 and "-" in str(tool["id"]):
                    tool.pop("id", None)
                if "function" in tool:
                    func_def = tool["function"]
                    func_def["strict"] = False
                    if "name" in func_def and isinstance(func_def["name"], str):
                        func_def["name"] = func_def["name"].replace(
                            _NIM_FUNCTION_NAME_PLACEHOLDER, "lcp_tool"
                        )
                sanitized_tools.append(tool)

        params: Dict = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": n_predict,
            "n": 1,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": repeat_penalty,
            "stream": stream,
        }
        if seed is not None:
            params["seed"] = seed

        if sanitized_tools:
            params["tools"] = sanitized_tools
            params["tool_choice"] = "auto"

        params = {k: v for k, v in params.items() if v is not None}

        effort = self.get_effective_reasoning_effort(think=think, reasoning_effort=reasoning_effort)
        if effort is not None:
            if self.is_vllm:
                self._apply_vllm_thinking_kwargs(params, effort)
            else:
                params["reasoning_effort"] = effort
                if reasoning_summary and reasoning_summary != "auto":
                    params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary
                params.pop("temperature", None)
                params.pop("top_p", None)
        else:
            if self.is_vllm:
                self._apply_vllm_thinking_kwargs(params, None)

        output = ""

        try:
            try:
                completion = self.client.chat.completions.create(**params)
            except Exception as ex:
                trace_exception(ex)
                if (
                    isinstance(ex, openai.NotFoundError)
                    and "Function" in str(ex)
                    and "Not found for account" in str(ex)
                ):
                    ASCIIColors.warning(
                        "[NIM Strict Validation] Intercepted 404 Function Not Found. Retrying without tools array."
                    )
                    params.pop("tools", None)
                    params.pop("tool_choice", None)
                    completion = self.client.chat.completions.create(**params)
                else:
                    if "max_tokens" in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")
                    params.pop("top_p", None)
                    params.pop("frequency_penalty", None)
                    params.pop("presence_penalty", None)
                    params.pop("reasoning_effort", None)
                    if effort is None:
                        params["temperature"] = 1
                    if "extra_body" in params:
                        params["extra_body"].pop("chat_template_kwargs", None)
                    completion = self.client.chat.completions.create(**params)

            if stream:
                handler = _StreamThinkingHandler(streaming_callback)
                for chunk in completion:
                    if self.is_cancelled():
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = extract_reasoning(delta)
                    content = getattr(delta, "content", None)

                    if reasoning:
                        if not handler.process_reasoning(reasoning):
                            break
                        continue

                    if content:
                        if not handler.process_content(content):
                            break

                output = handler.flush()
            else:
                message_obj = completion.choices[0].message
                reasoning = extract_reasoning(message_obj)
                content = message_obj.content or ""
                if reasoning and not content.strip().startswith(("<think>", "<thinking>")):
                    output = f"<think>\n{reasoning}\n</think>\n{content}"
                else:
                    output = content

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    def embed(self, text: str | list[str], normalize: bool = False, **kwargs) -> list:
        """
        Get embeddings for input text(s) using the OpenAI embeddings endpoint.
        Supports single strings or list of strings, with optional vector normalization.
        """
        embedding_model = kwargs.get("model", self.model_name)
        if not embedding_model or not embedding_model.startswith("text-embedding"):
            embedding_model = "text-embedding-3-small"

        is_single_input = isinstance(text, str)
        input_texts = [text] if is_single_input else text

        max_tokens_map = {
            "text-embedding-3-small": 8191,
            "text-embedding-3-large": 8191,
            "text-embedding-ada-002": 8191,
        }
        max_tokens = max_tokens_map.get(embedding_model, None)
        if max_tokens is not None:
            input_texts = [
                self.detokenize(self.tokenize(t)[:max_tokens]) for t in input_texts
            ]

        try:
            self._ensure_client()
            response = self.client.embeddings.create(model=embedding_model, input=input_texts)
            if not response.data:
                ASCIIColors.warning(f"OpenAI API returned no data for embedding (model: {embedding_model}).")
                return []

            embeddings = [item.embedding for item in response.data]
            if normalize:
                embeddings = [
                    [v / math.sqrt(sum(x * x for x in emb)) for v in emb]
                    for emb in embeddings
                ]
            return embeddings[0] if is_single_input else embeddings
        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using model '{embedding_model}': {e}")
            trace_exception(e)
            return []

    def transcribe_audio(self, audio_file_path: Union[str, Path], language: Optional[str] = None, prompt: Optional[str] = None, **kwargs) -> str:
        """
        Transcribe audio using OpenAI Whisper API endpoint with fallback to LoLLMS native STT.
        """
        p = Path(audio_file_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        try:
            self._ensure_client()
            with open(p, "rb") as f:
                res = self.client.audio.transcriptions.create(
                    model=kwargs.get("model", "whisper-1"),
                    file=f,
                    language=language,
                    prompt=prompt
                )
                return getattr(res, "text", str(res))
        except Exception as e:
            ASCIIColors.warning(f"[LollmsBinding] SDK audio transcription failed: {e}. Trying native endpoint.")

        url = f"{self.lollms_host_address}/audio/transcriptions"
        with open(p, "rb") as f:
            files = {"file": f}
            data = {}
            if language:
                data["language"] = language
            resp = requests.post(url, files=files, data=data, headers=self._lollms_headers(), timeout=300, verify=self.verify)
            resp.raise_for_status()
            return resp.json().get("text", "")

    def get_input_tokens_price(self, model_name: str | None = None) -> float:
        m = (model_name or self.model_name or "").lower()
        price_map = {
            "gpt-4o": 5e-6,
            "gpt-4o-mini": 1.5e-6,
            "gpt-3.5-turbo": 1.5e-6,
            "o1": 15e-6,
            "o3": 15e-6,
        }
        for key, price in price_map.items():
            if m.startswith(key):
                return price
        return 0.0

    def get_output_tokens_price(self, model_name: str | None = None) -> float:
        m = (model_name or self.model_name or "").lower()
        price_map = {
            "gpt-4o": 15e-6,
            "gpt-4o-mini": 6e-6,
            "gpt-3.5-turbo": 2e-6,
            "o1": 60e-6,
            "o3": 60e-6,
        }
        for key, price in price_map.items():
            if m.startswith(key):
                return price
        return 0.0

    def get_model_info(self) -> dict:
        return {
            "name": "LoLLMs",
            "version": "2.0",
            "host_address": self.open_ai_host_address,
            "model_name": self.model_name,
        }

    def _extract_models_from_payload(self, payload: Any) -> List[str]:
        if not payload:
            return []

        items = []
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                items = payload["data"]
            elif "models" in payload and isinstance(payload["models"], list):
                items = payload["models"]
            elif "model_names" in payload and isinstance(payload["model_names"], list):
                items = payload["model_names"]

        model_names = []
        for item in items:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    model_names.append(name)
            elif isinstance(item, dict):
                name = item.get("id") or item.get("model_name") or item.get("name") or item.get("model")
                if name and isinstance(name, str):
                    model_names.append(name.strip())
            else:
                name = getattr(item, "id", None) or getattr(item, "model_name", None) or getattr(item, "name", None)
                if name and isinstance(name, str):
                    model_names.append(name.strip())

        return list(dict.fromkeys(model_names))

    def list_models(self) -> List[Dict]:
        known_context_lengths = {
            "gpt-4o": 128000, "gpt-4": 8192, "gpt-4-0613": 8192,
            "gpt-4-1106-preview": 128000, "gpt-4-0125-preview": 128000,
            "gpt-4-turbo": 128000, "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16000, "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-0125": 16385, "text-davinci-003": 4097,
            "text-davinci-002": 4097, "davinci": 2049, "curie": 2049,
            "babbage": 2049, "ada": 2049,
        }
        generation_prefixes = ("gpt-", "text-davinci", "davinci", "curie", "babbage", "ada")
        prompt_buffer = 500

        model_names: List[str] = []

        try:
            self._ensure_client()
            entries = self.client.models.list()
            model_names = self._extract_models_from_payload(entries.data if hasattr(entries, "data") else entries)
        except Exception as e:
            ASCIIColors.warning(f"[LollmsBinding] SDK models.list failed: {e}. Falling back to direct endpoints.")

        if not model_names:
            candidate_endpoints = [
                f"{self.open_ai_host_address}/models",
                f"{self.base_address}/models",
                f"{self.base_address}/v1/models",
                f"{self.base_address}/list_models",
                f"{self.lollms_host_address}/models",
                f"{self.base_address}/api/models",
            ]
            for endpoint in candidate_endpoints:
                try:
                    resp = requests.get(
                        endpoint,
                        headers=self._lollms_headers(),
                        timeout=10,
                        verify=self.verify
                    )
                    if resp.status_code == 200:
                        parsed = self._extract_models_from_payload(resp.json())
                        if parsed:
                            model_names = parsed
                            break
                except Exception:
                    continue

        models_info = []
        for model_id in model_names:
            if model_id.startswith(generation_prefixes):
                context_length = known_context_lengths.get(model_id, "unknown")
                max_generation = (
                    context_length - prompt_buffer if isinstance(context_length, int) else "unknown"
                )
                models_info.append({
                    "model_name": model_id,
                    "owned_by": "lollms",
                    "created": "N/A",
                    "context_length": context_length,
                    "max_generation": max_generation,
                })
            else:
                models_info.append({
                    "model_name": model_id,
                    "owned_by": "lollms",
                    "created": "N/A",
                    "context_length": None,
                    "max_generation": None,
                })

        return models_info

    def load_model(self, model_name: str) -> bool:
        self.model = model_name
        self.model_name = model_name
        return True

    def ps(self):
        models = self.list_models()
        standardized_models = []
        for m in models:
            standardized_models.append({
                "model_name": m.get("model_name"),
                "size": None, "vram_size": None,
                "gpu_usage_percent": None, "cpu_usage_percent": None,
                "expires_at": None, "parameters_size": None,
                "quantization_level": None, "parent_model": None,
                "context_size": m.get("context_length"),
                "owned_by": m.get("owned_by"), "created": m.get("created"),
            })
        return standardized_models