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
import pipmaster as pm
pm.ensure_packages(["openai", "tiktoken"])

import httpx
import openai
import requests
import tiktoken
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.llm_bindings._tool_loop import (
    MAX_TOOL_ROUNDS,
    accumulate_streamed_tool_deltas,
    build_assistant_tool_calls_message,
    build_tool_call_callback_payloads,
    build_tool_result_messages,
    emit_callback_safe,
    extract_native_tool_calls,
    normalize_callback_result,
)

BindingName = "LollmsBinding"


def _wire_shape(native: List[Any]) -> List[Dict[str, Any]]:
    """Normalize native tool_calls objects into OpenAI wire shape.

    Local helper used by ``LollmsBinding._generate_one_round`` when persisting
    the assistant tool-call turn for the next round. Kept private to this
    module to avoid importing the OpenAIBinding internals.
    """
    out: List[Dict[str, Any]] = []
    for tc in native or []:
        if isinstance(tc, dict):
            out.append(tc)
            continue
        try:
            out.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", ""),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                    },
                }
            )
        except Exception:
            continue
    return out

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


def _sanitize_system_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enforces the OpenAI-compatible invariant: image content may only appear on
    user-role messages.

    Servers (OpenAI, vLLM, llama.cpp) reject the whole request with HTTP 400
    'System message cannot contain images.' when the system message carries
    image content. This guard extracts any system-role image blocks and moves
    them to the first user message (synthesizing one if none exists) so the
    vision context survives and the payload stays valid. No-op when clean.
    """
    def _extract_media(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        media: List[Dict[str, Any]] = []
        content = msg.get("content")
        if isinstance(content, list):
            texts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") in (
                    "image_url", "input_image", "video_url", "input_video",
                ):
                    media.append(part)
            if media:
                joined = "\n".join(t for t in texts if t.strip())
                msg["content"] = joined if joined else ""
        legacy_images = msg.pop("images", None)
        if legacy_images:
            media.extend(
                {"type": "image_url", "image_url": {"url": img}}
                if isinstance(img, str) else img
                for img in legacy_images
            )
        legacy_videos = msg.pop("videos", None)
        if legacy_videos:
            media.extend(
                {"type": "video_url", "video_url": {"url": vid}}
                if isinstance(vid, str) else vid
                for vid in legacy_videos
            )
        return media

    rescued: List[Dict[str, Any]] = []
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict) or str(msg.get("role", "")).lower() != "system":
            sanitized.append(msg)
            continue
        extracted = _extract_media(msg)
        if extracted:
            rescued.extend(extracted)
        sanitized.append(msg)

    if not rescued:
        return sanitized

    first_user_idx = next(
        (i for i, m in enumerate(sanitized) if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"),
        None,
    )
    if first_user_idx is None:
        sanitized.append({"role": "user", "content": rescued})
        return sanitized

    target = sanitized[first_user_idx]
    content = target.get("content")
    if isinstance(content, str):
        target["content"] = [{"type": "text", "text": content}] + rescued
    elif isinstance(content, list):
        target["content"] = list(content) + rescued
    else:
        target["content"] = rescued
    return sanitized


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
    Manages streaming chunks for OpenAI/LoLLMs-compatible endpoints to guarantee that:
    1. Dedicated reasoning fields (reasoning_content, reasoning, etc.) are wrapped
       in visible <think>...</think> tags and dispatched as MSG_TYPE_THOUGHT_CHUNK.
    2. In-content <think>...</think> tags are detected, streaming thoughts as
       MSG_TYPE_THOUGHT_CHUNK and answer text as MSG_TYPE_CHUNK, while preserving
       the <think> and </think> tags in the final output.
    3. When suppress_thinking=True, all thinking chunks and <think> blocks are
       completely silenced and excluded from output.
    """

    def __init__(
        self,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        suppress_thinking: bool = False,
    ):
        self.callback = streaming_callback
        self.suppress_thinking = suppress_thinking
        self.in_dedicated_reasoning = False
        self.dedicated_reasoning_opened = False
        self.in_content_thinking = False
        self.buffer = ""
        self.output = ""
        # Reasoning is captured even when suppressed so the caller can rescue
        # a reasoning-only response (e.g. deepseek-reasoner with think=False)
        # instead of returning an empty string to the user.
        self._reasoning_capture: List[str] = []

    def process_reasoning(self, reasoning: str) -> bool:
        if not reasoning:
            return True

        self._reasoning_capture.append(reasoning)

        if self.suppress_thinking:
            return True

        if not self.in_dedicated_reasoning:
            self.in_dedicated_reasoning = True
            self.dedicated_reasoning_opened = True
            self.output += "<think>\n"

        self.output += reasoning
        if self.callback:
            from lollms_client.llm_bindings._tool_loop import emit_callback_safe
            if emit_callback_safe(
                self.callback, reasoning, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, {}
            ) is False:
                return False

        return True

    def _close_dedicated_reasoning(self) -> bool:
        if self.in_dedicated_reasoning:
            self.in_dedicated_reasoning = False
            self.output += "\n</think>\n"
        return True

    def reasoning_snapshot(self) -> str:
        """Returns all reasoning text captured, including suppressed chunks."""
        return "".join(self._reasoning_capture)

    def process_content(self, content: str) -> bool:
        if not self._close_dedicated_reasoning():
            return False

        if not content:
            return True

        if self.dedicated_reasoning_opened:
            self.output += content
            if self.callback:
                from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                return emit_callback_safe(
                    self.callback, content, MSG_TYPE.MSG_TYPE_CHUNK, {}
                ) is not False
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
                        from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                        if self.callback and emit_callback_safe(
                            self.callback, pre, MSG_TYPE.MSG_TYPE_CHUNK, {}
                        ) is False:
                            return False

                    tag = m.group(0)
                    if not self.suppress_thinking:
                        self.output += tag
                    self.in_content_thinking = True
                    if not self.suppress_thinking and self.callback:
                        from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                        if emit_callback_safe(
                            self.callback, tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, {}
                        ) is False:
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
                        from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                        if self.callback and emit_callback_safe(
                            self.callback, text, MSG_TYPE.MSG_TYPE_CHUNK, {}
                        ) is False:
                            return False
                    break
            else:
                m = close_tag_re.search(text)
                if m:
                    thought_part = text[:m.start()]
                    if thought_part and not self.suppress_thinking:
                        self.output += thought_part
                        from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                        if self.callback and emit_callback_safe(
                            self.callback, thought_part, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, {}
                        ) is False:
                            return False

                    tag = m.group(0)
                    if not self.suppress_thinking:
                        self.output += tag
                    self.in_content_thinking = False
                    if not self.suppress_thinking and self.callback:
                        from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                        if emit_callback_safe(
                            self.callback, tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, {}
                        ) is False:
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
                        from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                        if self.callback and emit_callback_safe(
                            self.callback, text, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, {}
                        ) is False:
                            return False
                    break

        return True

    def flush(self) -> str:
        self._close_dedicated_reasoning()

        if self.buffer:
            msg_type = MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK if self.in_content_thinking else MSG_TYPE.MSG_TYPE_CHUNK
            self.output += self.buffer
            if self.callback:
                from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                emit_callback_safe(self.callback, self.buffer, msg_type, {})
            self.buffer = ""

        if self.in_content_thinking:
            self.in_content_thinking = False
            if not self.suppress_thinking:
                close_tag = "\n</think>\n"
                self.output += close_tag
                if self.callback:
                    from lollms_client.llm_bindings._tool_loop import emit_callback_safe
                    emit_callback_safe(
                        self.callback, close_tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK, {}
                    )

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

        self.service_key = kwargs.get("service_key") or os.getenv("LOLLMS_API_KEY")

        raw_model_name = kwargs.get("model_name")
        if raw_model_name:
            self.model_name = raw_model_name
        else:
            self.model_name = self._auto_detect_model_name()
            if not self.model_name:
                ASCIIColors.warning(
                    "[LollmsBinding] No model_name provided and auto-detection failed for "
                    f"host: {self.open_ai_host_address}. Generation methods will return an "
                    "error until a model is configured or the server becomes reachable."
                )
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

    def _auto_detect_model_name(self) -> Optional[str]:
        """Attempts to discover the first available model from the server."""
        try:
            self._init_client()
            models = self.list_models()
            if models:
                # Prefer models that don't look like embeddings or system models
                generation_models = [m for m in models if not m["model_name"].startswith("text-embedding")]
                if generation_models:
                    return generation_models[0]["model_name"]
                return models[0]["model_name"]
        except Exception as e:
            ASCIIColors.warning(f"[LollmsBinding] Failed to auto-detect model: {e}")
        return None

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

    def check_is_vllm(self) -> bool:
        """
        Determines whether the target server is a vLLM server.
        """
        if getattr(self, "is_vllm", False):
            return True
        if getattr(self, "_vllm_probed", False):
            return getattr(self, "_is_vllm_detected", False)

        self._vllm_probed = True
        self._is_vllm_detected = False

        probe_url = self.base_address or (
            self.open_ai_host_address[:-3] if self.open_ai_host_address and self.open_ai_host_address.endswith("/v1") else self.open_ai_host_address
        )
        if probe_url:
            try:
                endpoint = f"{probe_url.rstrip('/')}/version"
                r = requests.get(endpoint, timeout=1.0, verify=self.verify)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, dict) and "version" in data:
                        ASCIIColors.info(f"[LollmsBinding] Auto-detected vLLM server (version {data.get('version')}).")
                        self._is_vllm_detected = True
                        self.is_vllm = True
                        return True
            except Exception:
                pass

        return False

    def _apply_thinking_params(
        self,
        params: dict,
        effort: Optional[str],
        is_deactivated: bool
    ) -> dict:
        """
        Applies thinking / reasoning controls across LoLLMs, OpenAI, vLLM,
        Z.AI/GLM, and OpenAI-compatible engines.

        When thinking is deactivated (``think`` not True, or an explicit
        "off" effort), all reasoning / thinking controls are *omitted*
        entirely rather than set to ``False`` / ``"none"``. Several
        OpenAI-compatible endpoints (notably DeepSeek V4.1) forward
        ``chat_template_kwargs`` and validate them server-side — any
        non-conforming key there triggers a 400. Omission is the portable
        signal.
        """
        model_name_lower = (self.model_name or "").lower()
        is_glm = "glm" in model_name_lower
        is_reasoner_family = any(fam in model_name_lower for fam in ("o1", "o3", "o4", "gpt-5"))
        is_vllm = self.check_is_vllm()
        extra_body = params.setdefault("extra_body", {})

        normalized_effort = None
        if effort is not None:
            candidate = str(effort).strip().lower()
            if candidate not in ("none", "off", "disabled", "false", "0", ""):
                normalized_effort = "high" if candidate == "max" else candidate

        thinking_active = (not is_deactivated) and (normalized_effort is not None)

        if not thinking_active:
            # Omit every reasoning / thinking control. Servers use their
            # default behaviour, which is exactly what "no thinking controls"
            # means. Explicitly setting "none"/False breaks DeepSeek V4.1.
            params.pop("reasoning_effort", None)
            extra_body.pop("reasoning_effort", None)
            extra_body.pop("thinking", None)
            extra_body.pop("chat_template_kwargs", None)
        else:
            # Only OpenAI-native reasoning families and vLLM accept a
            # top-level reasoning_effort; forwarding it to other
            # OpenAI-compatible endpoints 400s.
            if is_vllm or is_reasoner_family:
                params["reasoning_effort"] = normalized_effort
                extra_body["reasoning_effort"] = normalized_effort
            else:
                params.pop("reasoning_effort", None)
                extra_body.pop("reasoning_effort", None)

            if self.send_thinking_parameter and not getattr(self, "glm_image_embedding", False):
                ctk = extra_body.setdefault("chat_template_kwargs", {})
                if self.thinking_effort_keyword == "reasoning_effort":
                    ctk["reasoning_effort"] = normalized_effort
                    ctk["enable_thinking"] = True
                else:
                    ctk[self.thinking_effort_keyword] = True
                    ctk["enable_thinking"] = True
                    ctk["thinking"] = True

            if is_glm:
                extra_body["thinking"] = {"type": "enabled"}
            else:
                extra_body["thinking"] = True

            if is_reasoner_family:
                params.pop("temperature", None)
                params.pop("top_p", None)

        # Never send an empty extra_body: some servers 400 on it and every
        # path below assumes the dict is meaningful only when non-empty.
        if not extra_body:
            params.pop("extra_body", None)

        return params

    def _apply_vllm_thinking_kwargs(self, params: dict, effort: Optional[str]) -> dict:
        is_deact = effort is None or str(effort).strip().lower() in ("none", "off", "disabled", "false", "0")
        return self._apply_thinking_params(params, effort, is_deact)

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
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> Union[str, dict]:
        """
        Generate text from a single prompt (chat-completions endpoint).

        ``tools`` follows the OpenAI Chat Completions schema::

            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]

        ``tool_choice`` may be ``"auto"``, ``"none"``, ``"required"``, or a
        ``{"type": "function", "function": {"name": "..."}}`` dict.

        When ``tools`` is provided, the call is routed through the shared
        tool-calling loop which:
          1. Forwards any ``tool_calls`` (native or DSML-leaked) to the
             callback as ``<tool>{"name":..., "parameters":{...}}</tool>`` XML.
          2. If the callback returns ``{"tool_results": [...]}``, appends the
             assistant tool-call turn + matching ``role="tool"`` messages and
             re-invokes the model.
          3. Repeats until no more tool calls are requested, the callback
             returns ``False``, or ``MAX_TOOL_ROUNDS`` (8) is reached.

        Callback contract::

            True / None                          → continue
            False                                → abort
            {"tool_results": [                   → append tool results and
                {"tool_call_id": "call_abc",        re-invoke the model
                 "content": "{\\"temp\\":22}"},
                ...
            ]}

        Each ``tool_results`` entry MUST carry a ``tool_call_id`` matching a
        ``tool_call.id`` from the previous assistant turn.
        """
        if not self.model_name:
            return {"status": "error", "message": "[LollmsBinding] No model_name resolved. Please configure a model profile or set model_name in binding config."}

        self._ensure_client()
        count = 0
        output = ""

        if think is True:
            effort = self.get_effective_reasoning_effort(think=True, reasoning_effort=reasoning_effort)
            if effort is None or str(effort).strip().lower() in ("none", "off", "disabled", "false", "0", ""):
                is_thinking_deactivated = True
                effort = None
            else:
                is_thinking_deactivated = False
        else:
            is_thinking_deactivated = True
            effort = None

        ASCIIColors.info(
            f"[LollmsBinding.generate_text] think={think}, reasoning_effort={reasoning_effort} "
            f"-> effective_effort={effort}, deactivated={is_thinking_deactivated}"
        )

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

        # ── Tool-aware path: route through the shared loop so the callback
        # can return tool results and expect the binding to re-invoke.
        if tools:
            return self._run_tool_loop(
                messages,
                n_predict=n_predict,
                stream=stream,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                streaming_callback=streaming_callback,
                think=think,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                tools=tools,
                tool_choice=tool_choice,
                extra_kwargs=kwargs,
            )

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

                self._apply_thinking_params(params, effort, is_thinking_deactivated)
                if reasoning_summary and reasoning_summary != "auto" and not is_thinking_deactivated:
                    params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary

                try:
                    chat_completion = self.client.chat.completions.create(**params)
                except Exception as ex:
                    trace_exception(ex)
                    if "max_tokens" in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")
                    params.pop("top_p", None)
                    params.pop("frequency_penalty", None)
                    params.pop("reasoning_effort", None)
                    if "extra_body" in params and isinstance(params["extra_body"], dict):
                        params["extra_body"].pop("chat_template_kwargs", None)
                        params["extra_body"].pop("reasoning_effort", None)
                        params["extra_body"].pop("thinking", None)
                    if effort is None or is_thinking_deactivated:
                        params["temperature"] = 1
                    chat_completion = self.client.chat.completions.create(**params)

                if stream:
                    handler = _StreamThinkingHandler(streaming_callback, suppress_thinking=is_thinking_deactivated)

                    for resp in chat_completion:
                        if self.is_cancelled():
                            break

                        if count >= (n_predict or float("inf")):
                            break

                        if not resp.choices:
                            continue

                        delta = resp.choices[0].delta
                        reasoning = extract_reasoning(delta)
                        content = getattr(delta, "content", "") or ""

                        if reasoning:
                            if not handler.process_reasoning(reasoning):
                                break
                            count += 1
                            continue

                        if content:
                            if not handler.process_content(content):
                                break
                            count += 1

                    output = self._finalize_stream_output(handler, streaming_callback)
                else:
                    message_obj = chat_completion.choices[0].message
                    reasoning = extract_reasoning(message_obj)
                    content = message_obj.content or ""

                    if reasoning and not content.strip().startswith((" thinking", "<thinking>")):
                        output = f"<think>\n{reasoning}\n</think>\n\n{content}"
                    else:
                        output = content

                    has_native_tool_calls = bool(
                        getattr(message_obj, "tool_calls", None)
                    )
                    if not output.strip() and not has_native_tool_calls:
                        try:
                            fallback_text, fallback_acc = self._stream_fallback_non_streaming(
                                params, streaming_callback, is_thinking_deactivated
                            )
                            if fallback_text.strip():
                                ASCIIColors.warning(
                                    "[LollmsBinding.generate_text] Non-streaming "
                                    "response was empty; streaming fallback "
                                    "produced content."
                                )
                                output = fallback_text
                            if fallback_acc:
                                promoted = []
                                for idx in sorted(fallback_acc.keys()):
                                    tc = fallback_acc[idx]
                                    name = (tc.get("name") or "").strip()
                                    if not name:
                                        continue
                                    tc_id = tc.get("id") or f"call_{idx}"
                                    raw_args = tc.get("arguments") or ""
                                    try:
                                        tool_args = json.loads(raw_args) if raw_args else {}
                                    except (json.JSONDecodeError, TypeError):
                                        tool_args = {"raw": raw_args}
                                    promoted.append(
                                        {
                                            "id": tc_id,
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(tool_args),
                                            },
                                        }
                                    )
                                if promoted:
                                    native_tool_xml = LollmsLLMBinding.tool_calls_to_lollms_xml(promoted)
                                    if native_tool_xml:
                                        if output:
                                            output += "\n" + native_tool_xml
                                        else:
                                            output = native_tool_xml
                        except Exception as fb_err:
                            ASCIIColors.warning(
                                f"[LollmsBinding.generate_text] Non-streaming "
                                f"fallback failed: {fb_err}"
                            )
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

        except openai.BadRequestError as e:
            ASCIIColors.error(f"[LollmsBinding] Request rejected 400: {e}")
            # Surface the actionable remediation for known server-side
            # tool-calling misconfigurations (vLLM missing --tool-call-parser,
            # etc.) so the caller never sees a bare "model does not support
            # tools" message when the real cause is a server launch flag.
            hint = self._classify_tool_server_error(e)
            if hint:
                return {
                    "status": "error",
                    "message": f"Server misconfiguration detected: {hint}",
                    "server_body": getattr(e, "body", None),
                }
            raise
        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            hint = self._classify_tool_server_error(e)
            if hint:
                err_msg = f"{err_msg}\n\n{hint}"
            if streaming_callback:
                emit_callback_safe(streaming_callback, err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION, {})
            return {"status": "error", "message": err_msg, "server_body": getattr(e, "body", None)}

        return output

    def _run_generate_once(
        self,
        params: dict,
        *,
        stream: Optional[bool],
        n_predict: Optional[int],
        is_thinking_deactivated: bool,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
        count_ref: Optional[List[int]] = None,
    ) -> Union[str, dict]:
        """
        Execute a single chat-completions request (as used by
        ``generate_text``'s thinking ladder) and drain its response. Never
        raises; returns the answer string (possibly ``""``) or an error dict.
        """
        count_ref = count_ref if count_ref is not None else [0]
        try:
            try:
                chat_completion = self.client.chat.completions.create(**params)
            except Exception:
                if "max_tokens" in params:
                    params["max_completion_tokens"] = params.pop("max_tokens")
                params.pop("top_p", None)
                params.pop("frequency_penalty", None)
                params.pop("reasoning_effort", None)
                if "extra_body" in params and isinstance(params["extra_body"], dict):
                    params["extra_body"].pop("chat_template_kwargs", None)
                    params["extra_body"].pop("reasoning_effort", None)
                    params["extra_body"].pop("thinking", None)
                    if not params["extra_body"]:
                        params.pop("extra_body", None)
                if is_thinking_deactivated:
                    params["temperature"] = 1
                chat_completion = self.client.chat.completions.create(**params)

            if stream:
                handler = _StreamThinkingHandler(
                    streaming_callback, suppress_thinking=is_thinking_deactivated
                )
                for resp in chat_completion:
                    if self.is_cancelled():
                        break
                    if count_ref[0] >= (n_predict or float("inf")):
                        break
                    if not resp.choices:
                        continue
                    delta = resp.choices[0].delta
                    reasoning = extract_reasoning(delta)
                    content = getattr(delta, "content", "") or ""
                    if reasoning:
                        if not handler.process_reasoning(reasoning):
                            break
                        count_ref[0] += 1
                        continue
                    if content:
                        if not handler.process_content(content):
                            break
                        count_ref[0] += 1
                return self._finalize_stream_output(handler, streaming_callback)

            message_obj = chat_completion.choices[0].message
            reasoning = extract_reasoning(message_obj)
            content = message_obj.content or ""
            output = self.finalize_non_stream_output(
                content, reasoning, is_thinking_deactivated
            )

            # Some OpenAI-compatible deployments only populate streaming
            # responses; when the non-streaming twin yields an empty body we
            # retry the exact same payload in streaming mode.
            if not output.strip():
                try:
                    fallback_text, _ = self._stream_fallback_non_streaming(
                        params, streaming_callback, is_thinking_deactivated
                    )
                    if fallback_text.strip():
                        ASCIIColors.warning(
                            "[LollmsBinding._run_generate_once] Non-streaming "
                            "response was empty; streaming fallback produced "
                            "content."
                        )
                        output = fallback_text
                except Exception as fb_err:
                    ASCIIColors.warning(
                        f"[LollmsBinding._run_generate_once] Non-streaming "
                        f"fallback failed: {fb_err}"
                    )

            if not output.strip():
                return {
                    "status": "error",
                    "message": (
                        "Model returned an empty non-streaming response and "
                        "the streaming fallback also produced no content. "
                        "The endpoint may be rejecting this request shape."
                    ),
                }
            return output
        except openai.BadRequestError as e:
            ASCIIColors.error(f"[LollmsBinding] Request rejected 400: {e}")
            return {"status": "error", "message": str(e), "server_body": getattr(e, "body", None)}
        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                emit_callback_safe(streaming_callback, err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION, {})
            return {"status": "error", "message": err_msg, "server_body": getattr(e, "body", None)}

    def generate_from_messages(
        self,
        messages: List[Dict],
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> Union[str, dict]:
        """
        Generate text from a full OpenAI-style message list, using the
        OpenAI Chat Completions tool-calling lifecycle.

        ``tools`` follows the OpenAI Chat Completions schema::

            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["location"],
                        },
                        "strict": True,
                    },
                }
            ]

        ``tool_choice`` may be ``"auto"``, ``"none"``, ``"required"``, or a
        ``{"type": "function", "function": {"name": "..."}}`` dict. When
        ``tools`` is provided and ``tool_choice`` is omitted, ``"auto"`` is
        used.

        Lifecycle (complying with the documented OpenAI protocol):

        1. Model evaluates the conversation and may return ``tool_calls``.
           DSML-leaked blocks (``<｜DSML｜ invoke>...</｜DSML｜>``) are
           translated to ``<tool>{"name":..., "parameters":{...}}</tool>``
           XML and treated identically to native tool calls.
        2. Each call is forwarded to the callback as
           ``<tool>{"name":..., "parameters":{...}}</tool>`` XML.
        3. If the callback returns ``{"tool_results": [...]}``, the binding
           appends the assistant tool-call turn + a matching ``role="tool"``
           message per result, then re-invokes the model for the next round.
        4. The loop terminates when the model emits no tool calls, the
           callback returns ``False``, or ``MAX_TOOL_ROUNDS`` (8) is reached.

        Callback contract::

            True / None                          → continue; output unchanged
            False                                → abort the call
            {"tool_results": [                   → binding appends the tool
                {"tool_call_id": "call_abc",        results and re-invokes
                 "content": "{\\"temp\\":22}"},            the model
                ...
            ]}

        Tool results MUST carry a ``tool_call_id`` matching one of the
        ``tool_calls`` returned by the previous assistant turn. Entries
        without a ``tool_call_id`` are dropped. ``content`` may be any
        serializable value; it is coerced to a JSON string when not already
        a string.

        Backward compatible: callbacks returning ``bool`` keep working.
        """
        if not self.model_name:
            return {"status": "error", "message": "[LollmsBinding] No model_name resolved. Please configure a model profile or set model_name in binding config."}

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
        # ── 🛡️ SYSTEM-IMAGE ROLE GUARD (400 defense-in-depth) ──
        # normalize_message can leave image content on the system role when
        # the caller attached images to a system message; OpenAI-compatible
        # servers 400-reject such payloads. Move system images to the first
        # user message before the wire call.
        openai_messages = _sanitize_system_images(openai_messages)

        raw_tools = tools
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

        if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
            params.setdefault("extra_body", {}).update(kwargs["extra_body"])

        if self.check_is_vllm():
            extra_body = params.setdefault("extra_body", {})
            effective_top_k = top_k if top_k is not None else kwargs.get("top_k")
            if effective_top_k is not None:
                extra_body.setdefault("top_k", effective_top_k)
            if repeat_penalty is not None:
                extra_body.setdefault("repetition_penalty", repeat_penalty)
            if "min_tokens" in kwargs and kwargs["min_tokens"] is not None:
                extra_body.setdefault("min_tokens", kwargs["min_tokens"])

        params = {k: v for k, v in params.items() if v is not None}

        if think is True:
            effort = self.get_effective_reasoning_effort(think=True, reasoning_effort=reasoning_effort)
            if effort is None or str(effort).strip().lower() in ("none", "off", "disabled", "false", "0", ""):
                is_thinking_deactivated = True
                effort = None
            else:
                is_thinking_deactivated = False
        else:
            is_thinking_deactivated = True
            effort = None

        ASCIIColors.debug(
            f"[LollmsBinding.generate_from_messages] think={think}, reasoning_effort={reasoning_effort} "
            f"-> effective_effort={effort}, deactivated={is_thinking_deactivated}"
        )

        self._apply_thinking_params(params, effort, is_thinking_deactivated)
        if reasoning_summary and reasoning_summary != "auto" and not is_thinking_deactivated:
            params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary

        # ── Tool-aware path: route through the shared loop so the callback
        # can return tool results and expect the binding to re-invoke.
        if tools:
            return self._run_tool_loop(
                messages,
                n_predict=n_predict,
                stream=stream,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                streaming_callback=streaming_callback,
                think=think,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                tools=tools,
                tool_choice=tool_choice,
                extra_kwargs=kwargs,
            )

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
                    if "extra_body" in params and isinstance(params["extra_body"], dict):
                        params["extra_body"].pop("chat_template_kwargs", None)
                        params["extra_body"].pop("reasoning_effort", None)
                        params["extra_body"].pop("thinking", None)
                        if not params["extra_body"]:
                            params.pop("extra_body", None)
                    if effort is None or is_thinking_deactivated:
                        params["temperature"] = 1
                    completion = self.client.chat.completions.create(**params)

            if stream:
                handler = _StreamThinkingHandler(streaming_callback, suppress_thinking=is_thinking_deactivated)

                for chunk in completion:
                    if self.is_cancelled():
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = extract_reasoning(delta)
                    content = getattr(delta, "content", "") or ""

                    if reasoning:
                        if not handler.process_reasoning(reasoning):
                            break
                        continue

                    if content:
                        if not handler.process_content(content):
                            break

                output = self._finalize_stream_output(handler, streaming_callback)
            else:
                message_obj = completion.choices[0].message
                reasoning = extract_reasoning(message_obj)
                content = message_obj.content or ""

                if is_thinking_deactivated:
                    output = re.sub(r'<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>', '', content, flags=re.IGNORECASE).strip()
                elif reasoning and not content.strip().startswith((" thinking", "<thinking>")):
                    output = f"<think>\n{reasoning}\n</think>\n\n{content}"
                else:
                    output = content

                if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
                    native_tool_xml = LollmsLLMBinding.tool_calls_to_lollms_xml(message_obj.tool_calls)
                    if native_tool_xml:
                        if output:
                            output += "\n" + native_tool_xml
                        else:
                            output = native_tool_xml

                if not output.strip() and not (
                    hasattr(message_obj, "tool_calls") and message_obj.tool_calls
                ):
                    try:
                        fallback_text, fallback_acc = self._stream_fallback_non_streaming(
                            params, streaming_callback, is_thinking_deactivated
                        )
                        if fallback_text.strip():
                            ASCIIColors.warning(
                                "[LollmsBinding.generate_from_messages] Non-streaming "
                                "response was empty; streaming fallback produced content."
                            )
                            output = fallback_text
                        if fallback_acc:
                            promoted = []
                            for idx in sorted(fallback_acc.keys()):
                                tc = fallback_acc[idx]
                                name = (tc.get("name") or "").strip()
                                if not name:
                                    continue
                                tc_id = tc.get("id") or f"call_{idx}"
                                raw_args = tc.get("arguments") or ""
                                try:
                                    tool_args = json.loads(raw_args) if raw_args else {}
                                except (json.JSONDecodeError, TypeError):
                                    tool_args = {"raw": raw_args}
                                promoted.append(
                                    {
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(tool_args),
                                        },
                                    }
                                )
                            if promoted:
                                native_tool_xml = LollmsLLMBinding.tool_calls_to_lollms_xml(promoted)
                                if native_tool_xml:
                                    if output:
                                        output += "\n" + native_tool_xml
                                    else:
                                        output = native_tool_xml
                    except Exception as fb_err:
                        ASCIIColors.warning(
                            f"[LollmsBinding.generate_from_messages] Non-streaming "
                            f"fallback failed: {fb_err}"
                        )

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg, "server_body": getattr(e, "body", None)}

        return output

    def _finalize_stream_output(
        self,
        handler: "_StreamThinkingHandler",
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
    ) -> str:
        """
        Returns the accumulated streaming output. When the model produced only
        reasoning while thinking was deactivated (e.g. deepseek-reasoner with
        think=False), the visible content chunk stream is empty; in that case
        the captured reasoning is surfaced as a CHUNK so the turn never dies
        silently.
        """
        output = handler.flush()
        if output and output.strip():
            return output

        reasoning = handler.reasoning_snapshot()
        if reasoning and reasoning.strip():
            ASCIIColors.warning(
                "[LollmsBinding] Model produced reasoning-only output while "
                "thinking was deactivated. Surfacing reasoning to avoid an "
                "empty response. Enable think=True for proper thought streaming."
            )
            if streaming_callback:
                try:
                    streaming_callback(reasoning, MSG_TYPE.MSG_TYPE_CHUNK)
                except Exception:
                    pass
            return reasoning

        return output

    def _non_stream_fallback_for_streamed_tools(
        self,
        params: dict,
        is_thinking_deactivated: bool,
    ) -> Optional[Tuple[str, List[Any], Optional[str]]]:
        """
        Retry a streamed + tools round in NON-streaming mode.

        Some OpenAI-compatible proxies silently return an empty stream when a
        request mixes ``stream=true`` with a ``tools`` array. The symmetric
        non-streaming request frequently returns the answer — either as
        native ``tool_calls`` (the intended path) or as plain content (when
        the model chose not to invoke a tool, which is valid under
        ``tool_choice="auto"``).

        Returns ``(content, native_tool_calls, reasoning)`` on success, or
        ``None`` when the retry was rejected/raised and the caller must fall
        back to its own empty handling. Never raises.
        """
        retry_params = dict(params)
        retry_params["stream"] = False
        retry_params.pop("stream_options", None)

        try:
            completion = self.client.chat.completions.create(**retry_params)
            message_obj = completion.choices[0].message
            content = message_obj.content or ""
            native_calls = extract_native_tool_calls(message_obj)
            reasoning = extract_reasoning(message_obj)

            if is_thinking_deactivated:
                content = re.sub(
                    r'<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>',
                    '', content, flags=re.IGNORECASE,
                ).strip()

            return content, native_calls, reasoning
        except Exception as e:
            ASCIIColors.warning(
                f"[LollmsBinding._non_stream_fallback_for_streamed_tools] "
                f"Retry rejected: {e}"
            )
            return None

    def _stream_fallback_non_streaming(
        self,
        params: dict,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
        is_thinking_deactivated: bool,
    ) -> Tuple[str, Dict[int, Dict[str, Any]]]:
        """
        Retry an identical request in streaming mode when its non-streaming
        twin produced no visible content. Some LoLLMS proxy deployments only
        populate streaming responses (verified against
        ``DeepSeekV4.1Flash`` behind vLLM). Returns
        ``(text, accumulated_tool_calls)``.
        """
        stream_params = dict(params)
        stream_params["stream"] = True
        stream_params.pop("stream_options", None)

        handler = _StreamThinkingHandler(
            streaming_callback, suppress_thinking=is_thinking_deactivated
        )
        accumulated: Dict[int, Dict[str, Any]] = {}
        completion = self.client.chat.completions.create(**stream_params)
        for chunk in completion:
            if self.is_cancelled():
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = extract_reasoning(delta)
            content = getattr(delta, "content", "") or ""
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                accumulate_streamed_tool_deltas(accumulated, delta_tool_calls)
            if reasoning:
                if not handler.process_reasoning(reasoning):
                    break
                continue
            if content:
                if not handler.process_content(content):
                    break
        return self._finalize_stream_output(handler, streaming_callback), accumulated

    def _run_tool_loop(
        self,
        messages: List[Dict],
        *,
        n_predict: Optional[int],
        stream: Optional[bool],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repeat_penalty: Optional[float],
        seed: Optional[int],
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
        think: Optional[bool],
        reasoning_effort: Optional[str],
        reasoning_summary: Optional[str],
        tools: Optional[List[Dict]],
        tool_choice: Optional[Union[str, Dict]],
        extra_kwargs: Dict[str, Any],
    ) -> Union[str, dict]:
        """
        Execute one chat round, and (when a callback returns a structured tool
        result) re-invoke the model with the tool outputs until either no more
        tool calls are requested, the callback returns ``False``, or the
        iteration cap (``MAX_TOOL_ROUNDS``) is reached.
        """
        conversation = list(messages)
        accumulated_outputs: List[str] = []
        last_output: Union[str, dict] = ""

        for round_index in range(MAX_TOOL_ROUNDS):
            result, tool_results = self._generate_one_round(
                conversation,
                n_predict=n_predict,
                stream=stream,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                streaming_callback=streaming_callback,
                think=think,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                tools=tools,
                tool_choice=tool_choice,
                extra_kwargs=extra_kwargs,
            )

            if isinstance(result, dict) and result.get("status") == "error":
                return result

            last_output = result
            if isinstance(result, str) and result:
                accumulated_outputs.append(result)

            if not tool_results:
                # When tools were requested but the round produced NO tool
                # results (callback never returned {"tool_results": [...]}),
                # there is nothing to continue: appending nothing and
                # re-sending the identical payload cannot make progress.
                # Every plausible cause (server missing --tool-call-parser,
                # tool-capable chat template not loaded, model family does
                # not emit wire tool_calls, reasoning-only response under
                # think=False) requires a SERVER-side fix surfaced by the
                # diagnostic at the end of this method. Bail NOW, on every
                # round, unconditionally. Re-iterating is pure latency/cost
                # waste: the model can never advance from a round that
                # emits neither content nor tool_calls.
                if tools:
                    if not isinstance(result, str) or not result.strip():
                        ASCIIColors.warning(
                            "[LollmsBinding] Tool round returned no content "
                            "and no tool_calls. Aborting tool loop — the "
                            "server must be configured to emit native "
                            "tool_calls (e.g. `--tool-call-parser <family>` "
                            "for vLLM)."
                        )
                break

            assistant_msg = getattr(self, "_last_assistant_tool_message", None)
            self._last_assistant_tool_message = None
            if not assistant_msg:
                ASCIIColors.warning(
                    "[LollmsBinding] Callback returned tool_results but no "
                    "assistant tool-call message was captured for this round. "
                    "Discarding the tool results and terminating the loop."
                )
                break

            requested_ids = {
                tc.get("id") for tc in assistant_msg.get("tool_calls", []) if tc.get("id")
            }
            matched = [r for r in tool_results if r.get("tool_call_id") in requested_ids]
            if not matched:
                ASCIIColors.warning(
                    "[LollmsBinding] All provided tool_results had tool_call_id "
                    "values that did not match any tool_calls in the previous "
                    "assistant turn. Terminating the loop."
                )
                break
            if len(matched) != len(tool_results):
                ASCIIColors.warning(
                    f"[LollmsBinding] Dropped {len(tool_results) - len(matched)} "
                    f"tool_results with unmatched tool_call_id."
                )

            conversation.append(assistant_msg)
            conversation.extend(build_tool_result_messages(matched))
        else:
            ASCIIColors.warning(
                f"[LollmsBinding] Tool-call loop reached MAX_TOOL_ROUNDS="
                f"{MAX_TOOL_ROUNDS}; returning last partial result."
            )

        if len(accumulated_outputs) > 1:
            return "\n".join(accumulated_outputs)
        if accumulated_outputs:
            return accumulated_outputs[-1]

        # Never return a bare empty string: a silent "" leaves the caller
        # unable to distinguish "endpoint returned nothing" from "loop
        # exited cleanly". Surface a diagnostic error so the root cause is
        # visible instead of swallowed by the example's final print.
        if isinstance(last_output, str) and not last_output.strip():
            return {
                "status": "error",
                "message": (
                    "[LollmsBinding] Tool loop completed with an EMPTY final "
                    "answer (no content, no reasoning, and no native "
                    "tool_calls were emitted). Two likely causes:\n"
                    "  1. The server does not emit OpenAI-format "
                    "`tool_calls`. For vLLM, launch with "
                    "`--enable-auto-tool-choice --tool-call-parser <family>` "
                    "(e.g. `deepseek_v3`, `hermes`, `internlm`, `mistral`, "
                    "`llama3_json`); for llama.cpp / Ollama use a chat "
                    "template that exposes the tools array to the model. "
                    "Check the server log for the actual tool-parsing "
                    "messages.\n"
                    "  2. The model never emits visible content (reasoning-"
                    "only responses under think=False); try think=True."
                ),
            }
        return last_output

    def _generate_one_round(
        self,
        conversation: List[Dict],
        *,
        n_predict: Optional[int],
        stream: Optional[bool],
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repeat_penalty: Optional[float],
        seed: Optional[int],
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]],
        think: Optional[bool],
        reasoning_effort: Optional[str],
        reasoning_summary: Optional[str],
        tools: Optional[List[Dict]],
        tool_choice: Optional[Union[str, Dict]],
        extra_kwargs: Dict[str, Any],
    ) -> Tuple[Union[str, dict], Optional[List[Dict[str, str]]]]:
        """
        Single round of the tool loop. Sends the conversation to
        /v1/chat/completions (with or without tools), parses the response
        (streaming or not), translates any DSML blocks or native tool_calls
        to Lollms ``<tool>`` XML, forwards to the callback, and returns
        ``(output, tool_results)`` where ``tool_results`` is ``None`` unless
        the callback opted into the structured contract.
        """
        _ROLE_MAP = {
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
            role = _ROLE_MAP.get(raw_role.lower(), "user")
            content = msg.get("content", "")
            text_parts: List[str] = []
            images: List[str] = []
            videos: List[str] = []

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

            # OpenAI native tool wire state preservation
            if role == "assistant" and msg.get("tool_calls"):
                return {
                    "role": "assistant",
                    "content": text_content if text_content else None,
                    "tool_calls": msg["tool_calls"],
                }
            if role == "tool":
                tool_entry: Dict[str, Any] = {"role": "tool", "content": text_content}
                if msg.get("tool_call_id"):
                    tool_entry["tool_call_id"] = msg["tool_call_id"]
                if msg.get("name"):
                    tool_entry["name"] = msg["name"]
                return tool_entry

            if not images and not videos:
                return {"role": role, "content": text_content}

            openai_content: List[Dict[str, Any]] = []
            if text_content:
                openai_content.append({"type": "text", "text": text_content})
            for img in images:
                openai_content.append(normalize_image_input(img, glm_format=self.glm_image_embedding))
            for vid in videos:
                openai_content.append(normalize_video_input(vid))
            return {"role": role, "content": openai_content}

        openai_messages = [normalize_message(m) for m in conversation]
        openai_messages = _sanitize_system_images(openai_messages)

        # Sanitize caller-provided tools (drop stray UUID ids, force strict=False
        # for NIM compatibility, and neutralize the placeholder name).
        sanitized_tools: Optional[List[Dict]] = None
        if tools and isinstance(tools, list):
            sanitized_tools = []
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                if "id" in tool and len(str(tool["id"])) == 36 and "-" in str(tool["id"]):
                    tool.pop("id", None)
                if "function" in tool and isinstance(tool["function"], dict):
                    func_def = tool["function"]
                    func_def.setdefault("strict", False)
                    name = func_def.get("name")
                    if isinstance(name, str):
                        func_def["name"] = name.replace(
                            _NIM_FUNCTION_NAME_PLACEHOLDER, "lcp_tool"
                        )
                sanitized_tools.append(tool)

        # Resolve thinking
        if think is True:
            effort = self.get_effective_reasoning_effort(think=True, reasoning_effort=reasoning_effort)
            if effort is None or str(effort).strip().lower() in ("none", "off", "disabled", "false", "0", ""):
                is_thinking_deactivated = True
                effort = None
            else:
                is_thinking_deactivated = False
        else:
            is_thinking_deactivated = True
            effort = None

        params: Dict[str, Any] = {
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
            params["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        if extra_kwargs.get("extra_body") and isinstance(extra_kwargs["extra_body"], dict):
            params.setdefault("extra_body", {}).update(extra_kwargs["extra_body"])

        if self.check_is_vllm():
            extra_body = params.setdefault("extra_body", {})
            effective_top_k = top_k if top_k is not None else extra_kwargs.get("top_k")
            if effective_top_k is not None:
                extra_body.setdefault("top_k", effective_top_k)
            if repeat_penalty is not None:
                extra_body.setdefault("repetition_penalty", repeat_penalty)
            if extra_kwargs.get("min_tokens") is not None:
                extra_body.setdefault("min_tokens", extra_kwargs["min_tokens"])

        params = {k: v for k, v in params.items() if v is not None}

        self._apply_thinking_params(params, effort, is_thinking_deactivated)
        if reasoning_summary and reasoning_summary != "auto" and not is_thinking_deactivated:
            params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary

        output = ""
        tool_results: Optional[List[Dict[str, str]]] = None

        # ── Single attempt (no double-create) ─────────────────────────────
        # The request is issued exactly ONCE. If the caller's thinking
        # configuration is rejected by the server we retry once without
        # reasoning params — never more. Earlier revisions issued two
        # back-to-back ``create(**params)`` calls (one whose generator was
        # discarded and one whose generator was drained), wasting one full
        # HTTP round-trip per tool round and inflating payload traces.
        last_err_dict: Optional[dict] = None
        accumulated_any = False

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
                        "[NIM Strict Validation] Intercepted 404 Function Not Found. "
                        "Retrying without tools array."
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
                    if "extra_body" in params and isinstance(params["extra_body"], dict):
                        params["extra_body"].pop("chat_template_kwargs", None)
                        params["extra_body"].pop("reasoning_effort", None)
                        params["extra_body"].pop("thinking", None)
                        if not params["extra_body"]:
                            params.pop("extra_body", None)
                    if effort is None or is_thinking_deactivated:
                        params["temperature"] = 1
                    completion = self.client.chat.completions.create(**params)

            attempt_deactivated = is_thinking_deactivated
            attempt_eff = effort

            if stream:
                    handler = _StreamThinkingHandler(
                        streaming_callback, suppress_thinking=attempt_deactivated
                    )
                    accumulated: Dict[int, Dict[str, Any]] = {}
                    for chunk in completion:
                        if self.is_cancelled():
                            break
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        reasoning = extract_reasoning(delta)
                        content = getattr(delta, "content", "") or ""
                        delta_tool_calls = getattr(delta, "tool_calls", None)
                        if delta_tool_calls:
                            accumulate_streamed_tool_deltas(accumulated, delta_tool_calls)
                        if reasoning:
                            if not handler.process_reasoning(reasoning):
                                break
                            continue
                        if content:
                            if not handler.process_content(content):
                                break

                    if sanitized_tools and not accumulated and not handler.output.strip():
                        try:
                            retry_fallback = self._non_stream_fallback_for_streamed_tools(
                                params, attempt_deactivated
                            )
                        except Exception as fb_err:
                            ASCIIColors.warning(
                                f"[LollmsBinding._generate_one_round] Streamed-tool "
                                f"non-stream retry failed: {fb_err}"
                            )
                            retry_fallback = None

                        if retry_fallback is not None:
                            (retry_content, retry_native_calls,
                             retry_reasoning) = retry_fallback

                            # ── 1. Native tool_calls returned non-streamed ──
                            # Promote to the standard native-tool-call path
                            # so the round emits <tool> XML through the
                            # documented callback lifecycle.
                            if retry_native_calls:
                                output = retry_content or ""
                                native_tool_calls = retry_native_calls
                                self._last_assistant_tool_message = {
                                    "role": "assistant",
                                    "content": retry_content or None,
                                    "tool_calls": _wire_shape(native_tool_calls),
                                }
                                native_payloads: List[Tuple[str, str, str]] = []
                                for tc in native_tool_calls:
                                    if isinstance(tc, dict):
                                        fn = tc.get("function") or {}
                                        name = fn.get("name", "") or ""
                                        raw_args = fn.get("arguments", "") or ""
                                        tc_id = tc.get("id") or f"call_{len(native_payloads)}"
                                    else:
                                        name = getattr(getattr(tc, "function", None), "name", "") or ""
                                        raw_args = getattr(getattr(tc, "function", None), "arguments", "") or ""
                                        tc_id = getattr(tc, "id", None) or f"call_{len(native_payloads)}"
                                    if not name:
                                        continue
                                    try:
                                        tool_args = json.loads(raw_args) if raw_args else {}
                                    except (json.JSONDecodeError, TypeError):
                                        tool_args = {"raw": raw_args}
                                    xml = f"<tool>{json.dumps({'name': name, 'parameters': tool_args})}</tool>"
                                    native_payloads.append((tc_id, name, xml))

                                collected: List[Dict[str, str]] = []
                                for tc_id, tc_name, tc_xml in native_payloads:
                                    cb_continue_call = True
                                    cb_payload_call: Optional[List[Dict[str, str]]] = None
                                    if streaming_callback:
                                        cb_result = emit_callback_safe(
                                            streaming_callback, tc_xml,
                                            MSG_TYPE.MSG_TYPE_CHUNK,
                                            {"tool_call_id": tc_id, "tool_name": tc_name},
                                        )
                                        cb_continue_call, cb_payload_call = normalize_callback_result(cb_result)
                                    if cb_continue_call:
                                        if output:
                                            output += "\n" + tc_xml
                                        else:
                                            output = tc_xml
                                    if cb_payload_call:
                                        collected.extend(cb_payload_call)
                                    if not cb_continue_call:
                                        break
                                if collected:
                                    tool_results = collected
                                ASCIIColors.warning(
                                    "[LollmsBinding._generate_one_round] Streamed "
                                    "tool round was empty; non-streaming retry "
                                    "produced native tool_calls."
                                )
                                return output, tool_results

                            # ── 2. Plain content returned non-streamed ──
                            # The model answered without tools (perfectly
                            # valid — the tool_choice is "auto"). Accept it.
                            if retry_content and retry_content.strip():
                                ASCIIColors.warning(
                                    "[LollmsBinding._generate_one_round] Streamed "
                                    "tool round was empty; non-streaming retry "
                                    "produced plain content (no tool_calls)."
                                )
                                output = retry_content
                                return output, tool_results

                            # ── 3. Reasoning-only non-stream rescue ──
                            if retry_reasoning and retry_reasoning.strip():
                                # Reasoning-only with tools requested cannot
                                # advance the loop; the diagnostic is far more
                                # actionable than a silently-mangled turn.
                                ASCIIColors.warning(
                                    "[LollmsBinding._generate_one_round] Streamed "
                                    "tool round was empty AND non-streaming "
                                    "retry produced reasoning only (no tool_calls). "
                                    "Bailing out of the tool round."
                                )

                        return "", None

                    output = self._finalize_stream_output(handler, streaming_callback)

                    if accumulated:
                        accumulated_any = True
                        self._last_assistant_tool_message = build_assistant_tool_calls_message(output, accumulated)
                        collected: List[Dict[str, str]] = []
                        for tc_id, tc_name, tc_xml in build_tool_call_callback_payloads(accumulated):
                            cb_continue_call = True
                            cb_payload_call: Optional[List[Dict[str, str]]] = None
                            if streaming_callback:
                                cb_result = emit_callback_safe(
                                    streaming_callback,
                                    tc_xml,
                                    MSG_TYPE.MSG_TYPE_CHUNK,
                                    {"tool_call_id": tc_id, "tool_name": tc_name},
                                )
                                cb_continue_call, cb_payload_call = normalize_callback_result(cb_result)
                            if cb_continue_call:
                                if output and tc_xml not in output:
                                    output += "\n" + tc_xml
                                elif not output:
                                    output = tc_xml
                            if cb_payload_call:
                                collected.extend(cb_payload_call)
                            if not cb_continue_call:
                                break
                        if collected:
                            tool_results = collected
                        return output, tool_results

                    if output.strip():
                        return output, tool_results
                    # Empty: no further escalation options for the streaming
                    # path (the tool-aware guard above already bailed).

            else:
                message_obj = completion.choices[0].message
                reasoning = extract_reasoning(message_obj)
                content = message_obj.content or ""
                output = self.finalize_non_stream_output(
                    content, reasoning, attempt_deactivated
                )
                native_tool_calls = extract_native_tool_calls(message_obj)

                if not output.strip() and not native_tool_calls:
                    try:
                        fallback_text, fallback_acc = self._stream_fallback_non_streaming(
                            params, streaming_callback, attempt_deactivated
                        )
                        if fallback_text.strip():
                            ASCIIColors.warning(
                                "[LollmsBinding._generate_one_round] Non-streaming "
                                "round returned no content; streaming fallback "
                                "produced content."
                            )
                            output = fallback_text
                        if fallback_acc:
                            promoted: List[Dict[str, Any]] = []
                            for idx in sorted(fallback_acc.keys()):
                                tc = fallback_acc[idx]
                                name = (tc.get("name") or "").strip()
                                if not name:
                                    continue
                                tc_id = tc.get("id") or f"call_{idx}"
                                raw_args = tc.get("arguments") or ""
                                try:
                                    tool_args = json.loads(raw_args) if raw_args else {}
                                except (json.JSONDecodeError, TypeError):
                                    tool_args = {"raw": raw_args}
                                promoted.append(
                                    {
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(tool_args),
                                        },
                                    }
                                )
                            if promoted:
                                native_tool_calls = promoted
                    except Exception as fb_err:
                        ASCIIColors.warning(
                            f"[LollmsBinding._generate_one_round] Non-streaming "
                            f"fallback failed: {fb_err}"
                        )

                # When tools were requested but the round returned no
                # native tool_calls AND no visible content, do NOT accept
                # the reasoning rescue (or stream fallback) as a valid
                # answer. Re-issuing the identical payload cannot make
                # the model suddenly emit tool_calls; the diagnostic is
                # surfaced by _run_tool_loop. Bail now.
                if sanitized_tools and not native_tool_calls and not output.strip():
                    return "", None

                if native_tool_calls:
                    accumulated_any = True
                    self._last_assistant_tool_message = {
                        "role": "assistant",
                        "content": message_obj.content or None,
                        "tool_calls": _wire_shape(native_tool_calls),
                    }
                    native_payloads: List[Tuple[str, str, str]] = []
                    for tc in native_tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function") or {}
                            name = fn.get("name", "") or ""
                            raw_args = fn.get("arguments", "") or ""
                            tc_id = tc.get("id") or f"call_{len(native_payloads)}"
                        else:
                            name = getattr(getattr(tc, "function", None), "name", "") or ""
                            raw_args = getattr(getattr(tc, "function", None), "arguments", "") or ""
                            tc_id = getattr(tc, "id", None) or f"call_{len(native_payloads)}"
                        if not name:
                            continue
                        try:
                            tool_args = json.loads(raw_args) if raw_args else {}
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {"raw": raw_args}
                        xml = f"<tool>{json.dumps({'name': name, 'parameters': tool_args})}</tool>"
                        native_payloads.append((tc_id, name, xml))

                    collected: List[Dict[str, str]] = []
                    for tc_id, tc_name, tc_xml in native_payloads:
                        cb_continue_call = True
                        cb_payload_call: Optional[List[Dict[str, str]]] = None
                        if streaming_callback:
                            cb_result = emit_callback_safe(
                                streaming_callback,
                                tc_xml,
                                MSG_TYPE.MSG_TYPE_CHUNK,
                                {"tool_call_id": tc_id, "tool_name": tc_name},
                            )
                            cb_continue_call, cb_payload_call = normalize_callback_result(cb_result)
                        if cb_continue_call:
                            if output:
                                output += "\n" + tc_xml
                            else:
                                output = tc_xml
                        if cb_payload_call:
                            collected.extend(cb_payload_call)
                        if not cb_continue_call:
                            break
                    if collected:
                        tool_results = collected
                    return output, tool_results

                if output.strip():
                    return output, tool_results
                # Empty: no further escalation options.

            if last_err_dict is not None and not accumulated_any:
                return last_err_dict, None

            if accumulated_any:
                return output, tool_results
            return "", None

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            # vLLM rejects tool_choice="auto" with a 400 when launched
            # without --enable-auto-tool-choice / --tool-call-parser. Surface
            # the exact remediation instead of the generic message.
            hint = self._classify_tool_server_error(e)
            if hint:
                err_msg = f"{err_msg}\n\n{hint}"
            if streaming_callback:
                emit_callback_safe(streaming_callback, err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION, {})
            return {"status": "error", "message": err_msg, "server_body": getattr(e, "body", None)}, None

        return output, tool_results

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
            "name": "lollms",
            "model_name": self.model_name,
            "host": self.host_address,
            "vision_enabled": self.vision_enabled,
            "video_enabled": self.video_enabled,
            "supported_reasoning_efforts": self.supported_reasoning_efforts,
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