# bindings/OpenAI/binding.py
from __future__ import annotations

import base64
import math
import mimetypes
import os
import re
import ssl
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import httpx
import openai
import pipmaster as pm
import tiktoken
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE

pm.ensure_packages(["openai", "tiktoken"])

BindingName = "OpenAIBinding"

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
    Returns a Chat Completions API-compliant content block:
      { "type": "image_url", "image_url": { "url": "data:<mime>;base64,<...>" } }
    Supports dictionaries (with data, path, or url), local file paths, raw base64 strings,
    and HTTP/HTTPS URLs.
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
    Supports dictionaries (with url, path, or data), local file paths,
    raw base64 strings, and HTTP/HTTPS URLs.
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
    Extract reasoning/thinking text from an OpenAI delta or message object.
    Supports standard and provider-specific fields (DeepSeek, vLLM, Groq, Together, etc.),
    checking direct attributes, Pydantic v2 model_extra dictionaries, and dict lookups.
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
            self.output += "<think>\n"

        self.output += reasoning
        if self.callback:
            if self.callback(reasoning, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                return False

        return True

    def _close_dedicated_reasoning(self) -> bool:
        if self.in_dedicated_reasoning:
            self.in_dedicated_reasoning = False
            self.output += "\n</think>\n"
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


class OpenAIBinding(LollmsLLMBinding):
    """OpenAI-specific binding implementation"""

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the OpenAI binding.

        Args:
            host_address (str): Host address for the OpenAI service. Defaults to DEFAULT_HOST_ADDRESS.
            model_name (str): Name of the model to use. Defaults to empty string.
            service_key (str): Authentication key for the service. Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            personality (Optional[int]): Ignored parameter for compatibility with LollmsLLMBinding.
        """
        super().__init__(BindingName, **kwargs)

        self.host_address = kwargs.get("host_address")
        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path = kwargs.get("certificate_file_path", None)
        self.default_completion_format = kwargs.get(
            "default_completion_format", ELF_COMPLETION_FORMAT.Chat
        )
        self.is_vllm = kwargs.get("is_vllm", False)
        self.send_thinking_parameter = kwargs.get("send_thinking_parameter", True)
        self.thinking_effort_keyword = kwargs.get("thinking_effort_keyword", "enable_thinking")
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
                    
        self.base_address = self.host_address
        if self.base_address:
            self.open_ai_host_address = (
                f"{self.base_address}"
                if self.base_address.endswith("/v1")
                else f"{self.base_address}/v1"
            )
        else:
            self.open_ai_host_address = None

        if not self.service_key:
            self.service_key = os.getenv("OPENAI_API_KEY", self.service_key) or "EMPTY"

        self.verify = True
        verify = True

        if not self.verify_ssl_certificate:
            self.verify = False
            verify = False

        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)

            if not cert_path.exists():
                raise FileNotFoundError(f"Certificate file not found: {cert_path}")

            ssl_context = ssl.create_default_context(cafile=str(cert_path))
            self.verify = cert_path
            verify = ssl_context

        self.client = openai.OpenAI(
            api_key=self.service_key or "EMPTY",
            base_url=self.open_ai_host_address,
            http_client=httpx.Client(
                verify=verify,
                timeout=300.0,
            ),
        )
        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    def _build_openai_params(self, messages: list, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        restricted_families = [
            "gpt-5",
            "gpt-4o",
            "o1",
            "o3",
            "o4",
        ]

        allowed_params = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "n",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "stream",
            "user",
            "max_completion_tokens",
            "reasoning_effort",
        }

        params = {
            "model": model,
            "messages": messages,
        }

        for k, v in kwargs.items():
            if k in allowed_params and v is not None:
                params[k] = v
            else:
                if v is not None and kwargs.get("debug", False):
                    ASCIIColors.warning(f"Removed unsupported OpenAI param '{k}'")

        model_lower = model.lower() if model else ""
        if any(fam in model_lower for fam in restricted_families):
            if "temperature" in params and params["temperature"] != 1:
                ASCIIColors.warning(
                    f"{model} does not support temperature != 1. Overriding to 1."
                )
                params["temperature"] = 1
            if "top_p" in params:
                ASCIIColors.warning(f"{model} does not support top_p. Removing it.")
                params.pop("top_p")

        return params

    def _apply_vllm_thinking_kwargs(self, params: dict, effort: Optional[str]) -> dict:
        if not self.is_vllm:
            return params
        if not self.send_thinking_parameter or getattr(self, "glm_image_embedding", False):
            return params
        params.setdefault("extra_body", {}).setdefault(
            "chat_template_kwargs", {}
        )[self.thinking_effort_keyword] = effort is not None
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
        count = 0
        output = ""

        effort = self.get_effective_reasoning_effort(think=think, reasoning_effort=reasoning_effort)

        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant.",
            }
        ]

        media_blocks = []
        if images:
            media_blocks.extend([normalize_image_input(img, glm_format=self.glm_image_embedding) for img in images])
        if videos:
            media_blocks.extend([normalize_video_input(vid) for vid in videos])

        if media_blocks:
            if split:
                messages += self.split_discussion(
                    prompt,
                    user_keyword=user_keyword,
                    ai_keyword=ai_keyword,
                )
                last = messages[-1]
                last["content"] = [{"type": "text", "text": last["content"]}] + media_blocks
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] + media_blocks,
                    }
                )
        else:
            if split:
                messages += self.split_discussion(
                    prompt,
                    user_keyword=user_keyword,
                    ai_keyword=ai_keyword,
                )
            else:
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
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

                if effort is not None:
                    if self.is_vllm:
                        self._apply_vllm_thinking_kwargs(params, effort)
                    else:
                        params["reasoning_effort"] = (
                            effort if effort != "max" else "high"
                        )
                        if reasoning_summary and reasoning_summary != "auto":
                            params.setdefault("extra_body", {})[
                                "reasoning_summary"
                            ] = reasoning_summary
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
                        if not resp.choices:
                            continue
                        word = getattr(resp.choices[0], "text", "") or ""
                        if word:
                            output += word
                            count += 1
                        if streaming_callback and not streaming_callback(
                            word, MSG_TYPE.MSG_TYPE_CHUNK
                        ):
                            break
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
        _OPENAI_ROLE_MAP = {
            "system": "system",
            "developer": "developer",
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "function": "function",
            "admin": "system",
            "root": "system",
            "manager": "system",
            "supervisor": "system",
            "controller": "system",
            "orchestrator": "system",
            "planner": "system",
            "critic": "assistant",
            "refiner": "assistant",
            "reviewer": "assistant",
            "validator": "assistant",
            "executor": "assistant",
            "worker": "assistant",
            "agent": "assistant",
            "bot": "assistant",
            "ai": "assistant",
            "human": "user",
            "guest": "user",
            "client": "user",
            "customer": "user",
            "operator": "user",
        }

        def normalize_message(msg: Dict) -> Dict:
            raw_role = msg.get("role", "user") or "user"
            role = _OPENAI_ROLE_MAP.get(raw_role.lower(), "user")
            content = msg.get("content", "")
            text_parts = []
            images = []

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

            videos = []
            if isinstance(content, list):
                for item in content:
                    if item.get("type") in ("video_url", "video", "input_video"):
                        val = item.get("video_url") or item.get("video")
                        if isinstance(val, dict):
                            val = val.get("url") or val.get("base64")
                        if isinstance(val, str) and val:
                            videos.append(val)

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

        params = {
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

        effort = self.normalize_reasoning_effort(think=think, reasoning_effort=reasoning_effort)
        if effort is not None:
            if self.is_vllm:
                self._apply_vllm_thinking_kwargs(params, effort)
            else:
                params["reasoning_effort"] = effort if effort != "max" else "high"
                if reasoning_summary and reasoning_summary != "auto":
                    params.setdefault("extra_body", {})[
                        "reasoning_summary"
                    ] = reasoning_summary
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

    def _get_encoding(self, model_name: str | None = None):
        """
        Get the tiktoken encoding for a given model.
        Falls back to 'cl100k_base' if model is unknown.
        """
        if model_name is None:
            model_name = self.model_name
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text into a list of token IDs.

        Args:
            text (str): The text to tokenize.

        Returns:
            list[int]: List of token IDs.
        """
        encoding = self._get_encoding()
        return encoding.encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert a list of token IDs back to text.

        Args:
            tokens (list[int]): List of tokens.

        Returns:
            str: The decoded text.
        """
        encoding = self._get_encoding()
        return encoding.decode(tokens)

    def get_input_tokens_price(self, model_name: str | None = None) -> float:
        """
        Get the price per input token for a given model (USD).

        Args:
            model_name (str | None): Model name. Defaults to self.model_name.

        Returns:
            float: Price per input token in USD.
        """
        if model_name is None:
            model_name = self.model_name

        price_map = {
            "gpt-4o": 5e-6,
            "gpt-4o-mini": 1.5e-6,
            "gpt-3.5-turbo": 1.5e-6,
            "o1": 15e-6,
            "o3": 15e-6,
        }

        for key, price in price_map.items():
            if model_name.lower().startswith(key):
                return price
        return 0.0

    def get_output_tokens_price(self, model_name: str | None = None) -> float:
        """
        Get the price per output token for a given model (USD).

        Args:
            model_name (str | None): Model name. Defaults to self.model_name.

        Returns:
            float: Price per output token in USD.
        """
        if model_name is None:
            model_name = self.model_name

        price_map = {
            "gpt-4o": 15e-6,
            "gpt-4o-mini": 6e-6,
            "gpt-3.5-turbo": 2e-6,
            "o1": 60e-6,
            "o3": 60e-6,
        }

        for key, price in price_map.items():
            if model_name.lower().startswith(key):
                return price
        return 0.0

    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            int: Number of tokens in text.
        """
        return len(self.tokenize(text))

    def embed(self, text: str | list[str], normalize: bool = False, **kwargs) -> list:
        """
        Get embeddings for input text(s) using OpenAI API.

        Args:
            text (str | list[str]): Input text or list of texts to embed.
            normalize (bool): Whether to normalize the resulting vector(s) to unit length.
            **kwargs: Additional arguments. The 'model' argument can be used
                    to specify the embedding model (e.g., "text-embedding-3-small").
                    Defaults to "text-embedding-3-small".

        Returns:
            list: A single embedding vector (list of floats) if input is str,
                or a list of embedding vectors if input is list[str].
                Returns empty list on failure.
        """
        embedding_model = kwargs.get("model", self.model_name)
        if not embedding_model.startswith("text-embedding"):
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
            response = self.client.embeddings.create(
                model=embedding_model,
                input=input_texts,
            )

            if not response.data:
                ASCIIColors.warning(
                    f"OpenAI API returned no data for the embedding request (model: {embedding_model})."
                )
                return []

            embeddings = [item.embedding for item in response.data]

            if normalize:
                embeddings = [
                    [v / math.sqrt(sum(x * x for x in emb)) for v in emb]
                    for emb in embeddings
                ]

            return embeddings[0] if is_single_input else embeddings

        except Exception as e:
            ASCIIColors.error(
                f"Failed to generate embeddings using model '{embedding_model}': {e}"
            )
            trace_exception(e)
            return []

    def _get_ctx_size(self, model_name: str | None = None) -> int:
        """
        Get the context size for a given model.
        If model_name is None, use the instance's model_name.

        Args:
            model_name (str | None): The model name to check.

        Returns:
            int: The context window size in tokens.
        """
        if model_name is None:
            model_name = self.model_name
            if model_name is None:
                return 0

        context_map = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-5": 200000,
            "o1": 200000,
            "o3": 200000,
            "o4": 200000,
        }

        model_name_lower = model_name.lower()
        for key, size in context_map.items():
            if model_name_lower.startswith(key):
                return size

        return None

    def get_model_info(self) -> dict:
        """
        Return information about the current OpenAI model.

        Returns:
            dict: Dictionary containing model name, version, and host address.
        """
        return {
            "name": "OpenAI",
            "version": "2.0",
            "host_address": self.host_address,
            "model_name": self.model_name,
        }

    def list_models(self) -> List[Dict]:
        known_context_lengths = {
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-1106-preview": 128000,
            "gpt-4-0125-preview": 128000,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16000,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-0125": 16385,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "davinci": 2049,
            "curie": 2049,
            "babbage": 2049,
            "ada": 2049,
        }

        generation_prefixes = (
            "gpt-",
            "text-davinci",
            "davinci",
            "curie",
            "babbage",
            "ada",
        )

        models_info = []
        prompt_buffer = 500

        try:
            models = self.client.models.list()
            for model in models.data:
                model_id = model.id
                if model_id.startswith(generation_prefixes):
                    context_length = known_context_lengths.get(model_id, "unknown")
                    max_generation = (
                        context_length - prompt_buffer
                        if isinstance(context_length, int)
                        else "unknown"
                    )
                    models_info.append(
                        {
                            "model_name": model_id,
                            "owned_by": getattr(model, "owned_by", "N/A"),
                            "created": getattr(model, "created", "N/A"),
                            "context_length": context_length,
                            "max_generation": max_generation,
                        }
                    )
                else:
                    models_info.append(
                        {
                            "model_name": model_id,
                            "owned_by": getattr(model, "owned_by", "N/A"),
                            "created": getattr(model, "created", "N/A"),
                            "context_length": None,
                            "max_generation": None,
                        }
                    )

        except Exception as e:
            print(f"Failed to list models: {e}")

        return models_info

    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model into the OpenAI binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully.
        """
        self.model = model_name
        self.model_name = model_name
        return True