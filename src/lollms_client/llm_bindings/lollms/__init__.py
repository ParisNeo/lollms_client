# llm_bindings/lollms/__init__.py
import requests
import json
from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_utilities import encode_image
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
from lollms_client.lollms_discussion import LollmsDiscussion
from typing import Optional, Callable, List, Union
from ascii_colors import ASCIIColors, trace_exception
from typing import List, Dict
import httpx
import pipmaster as pm
import mimetypes
import base64
from pathlib import Path
import ssl

pm.ensure_packages(["openai", "tiktoken"])

import openai
import tiktoken
import os

BindingName = "LollmsBinding"


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


def normalize_image_input(img, default_mime="image/jpeg"):
    if isinstance(img, dict):
        if "data" in img and isinstance(img["data"], str):
            mime = img.get("mime", default_mime)
            return {"type": "input_image", "image_url": _to_data_url(img["data"], mime)}
        if "path" in img and isinstance(img["path"], str):
            p = _extract_markdown_path(img["path"])
            b64 = _read_file_as_base64(p)
            mime = _guess_mime_from_name(p, default_mime)
            return {"type": "input_image", "image_url": _to_data_url(b64, mime)}
        if "url" in img:
            raise ValueError("URL inputs not allowed here; provide base64 or local path")
        raise ValueError("Unsupported dict format for image input")

    if isinstance(img, str):
        s = _extract_markdown_path(img)
        if s.startswith("data:"):
            return {"type": "input_image", "image_url": s}
        if os.path.exists(s) or (":" in s and "\\" in s) or s.startswith("/") or s.startswith("."):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            return {"type": "input_image", "image_url": _to_data_url(b64, mime)}
        return {"type": "input_image", "image_url": _to_data_url(s, default_mime)}

    raise ValueError("Unsupported image input type")


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
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path = kwargs.get("certificate_file_path")
        self.default_completion_format = kwargs.get(
            "default_completion_format", ELF_COMPLETION_FORMAT.Chat
        )

        if not self.service_key:
            ASCIIColors.warning(
                "[LollmsBinding] No service_key provided and LOLLMS_API_KEY env var not set. "
                "Requests will be sent without Authorization header and will fail with 401 "
                "if the server requires authentication."
            )

        self.verify = True
        verify = True

        if not self.verify_ssl_certificate:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            self.verify = False
            verify = ssl_context
        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)
            if not cert_path.exists():
                raise FileNotFoundError(f"Certificate file not found: {cert_path}")
            ssl_context = ssl.create_default_context(cafile=str(cert_path))
            self.verify = str(cert_path)
            verify = ssl_context

        self._http_client = httpx.Client(verify=verify, timeout=300.0)
        self.client = openai.OpenAI(
            api_key=self.service_key or "nokey",
            base_url=self.open_ai_host_address,
            http_client=self._http_client,
        )

        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    # ── Cancellation ──────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Close the httpx connection to abort in-flight HTTP requests, then set the cancel event."""
        try:
            if self._http_client and not self._http_client.is_closed:
                self._http_client.close()
                ASCIIColors.yellow("[LollmsBinding] HTTP client closed for cancellation.")
        except Exception as e:
            ASCIIColors.warning(f"[LollmsBinding] Error closing HTTP client during cancel: {e}")
        super().cancel()

    def close(self) -> None:
        """Clean resource disposal."""
        self.cancel()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────

    def _lollms_headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
        return headers

    def _lollms_get(self, path: str, timeout: int = 10) -> dict:
        url = f"{self.lollms_host_address}{path}"
        response = requests.get(url, headers=self._lollms_headers(), timeout=timeout, verify=self.verify)
        if response.status_code == 401:
            ASCIIColors.error(
                f"[LollmsBinding] 401 Unauthorized on GET {path}. "
                f"Key present: {bool(self.service_key)}. "
                f"Key prefix: {self.service_key[:8] + '...' if self.service_key else 'N/A'}"
            )
        response.raise_for_status()
        return response.json()

    def _lollms_post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        url = f"{self.lollms_host_address}{path}"
        response = requests.post(url, json=payload, headers=self._lollms_headers(), timeout=timeout, verify=self.verify)
        if response.status_code == 401:
            ASCIIColors.error(
                f"[LollmsBinding] 401 Unauthorized on POST {path}. "
                f"Key present: {bool(self.service_key)}. "
                f"Key prefix: {self.service_key[:8] + '...' if self.service_key else 'N/A'}"
            )
        response.raise_for_status()
        return response.json()

    # ── Capabilities ──────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict:
        try:
            return self._lollms_get("/capabilities", timeout=10)
        except Exception as e:
            ASCIIColors.warning(f"Failed to fetch capabilities: {e}")
            return {"capabilities": [], "active_bindings": {}}

    # ── Personalities ─────────────────────────────────────────────────────

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
        headers = self._lollms_headers()

        try:
            response = requests.get(url, headers=headers, timeout=10, verify=self.verify)
            if response.status_code == 200:
                return response.json()
        except Exception as ex:
            return {"status": False, "error": str(ex)}

        return {"status": False, "error": f"Failed to list personalities: HTTP {response.status_code}"}

    # ── Tokenize / Detokenize / Count ─────────────────────────────────────

    def tokenize(self, text: str) -> list:
        if text is None:
            return []
        try:
            data = self._lollms_post("/tokenize", {"model": self.model_name, "text": text}, timeout=10)
            if "tokens" in data:
                return data["tokens"]
        except Exception as e:
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
        try:
            data = self._lollms_post("/tokenize", {"model": self.model_name, "text": text}, timeout=10)
            if "count" in data:
                return int(data["count"])
            elif "tokens" in data:
                return len(data["tokens"])
        except Exception as e:
            ASCIIColors.warning(f"Remote token count failed: {e}. Falling back to local count.")
        return len(self.tokenize(text))

    # ── Context Size ──────────────────────────────────────────────────────

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
                ASCIIColors.warning(
                    f"Could not retrieve remote context size for '{target_model}': {e}. "
                    "Falling back to default. Further failures will be silent."
                )
            return 4096

        return 4096

    # ── Long Context Processing ───────────────────────────────────────────

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

    # ── RAG ───────────────────────────────────────────────────────────────

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

    # ── Per-Binding Model Listing ─────────────────────────────────────────

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

    # ── TTS via LoLLMS ───────────────────────────────────────────────────

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

    # ── File Text Extraction ──────────────────────────────────────────────

    def extract_text(self, file_b64: str, filename: str) -> str:
        payload = {"file": file_b64, "filename": filename}
        url = f"{self.open_ai_host_address}/extract_text"
        response = requests.post(
            url, json=payload, headers=self._lollms_headers(), timeout=120, verify=self.verify
        )
        response.raise_for_status()
        return response.json().get("text", "")

    # ── Responses API ─────────────────────────────────────────────────────

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

    # ── OpenAI Params Builder ─────────────────────────────────────────────

    def _build_openai_params(self, messages: list = None, prompt: str = None, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        restricted_families = ["gpt-5", "gpt-4o", "o1", "o3", "o4"]

        allowed_params = {
            "model", "messages", "temperature", "top_p", "n",
            "stop", "max_tokens", "presence_penalty", "frequency_penalty",
            "logit_bias", "stream", "user", "max_completion_tokens",
        }

        think = kwargs.pop("think", False)
        reasoning_effort = kwargs.pop("reasoning_effort", "low")
        reasoning_summary = kwargs.pop("reasoning_summary", "auto")

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

        if think:
            params["reasoning_effort"] = reasoning_effort or "low"
            if reasoning_summary and reasoning_summary != "auto":
                params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary
            params.pop("temperature", None)
            params.pop("top_p", None)

        model_lower = (model or "").lower()
        if any(fam in model_lower for fam in restricted_families):
            if "temperature" in params and params["temperature"] != 1:
                ASCIIColors.warning(f"{model} does not support temperature != 1. Overriding to 1.")
                params["temperature"] = 1
            if "top_p" in params:
                ASCIIColors.warning(f"{model} does not support top_p. Removing it.")
                params.pop("top_p")

        return params

    # ── Generate Text ─────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: str = "",
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
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
        messages = [{"role": "system", "content": system_prompt or "You are a helpful assistant."}]

        if images:
            if split:
                messages += self.split_discussion(prompt, user_keyword=user_keyword, ai_keyword=ai_keyword)
                last = messages[-1]
                text_block = {"type": "text", "text": last["content"]}
                image_blocks = [normalize_image_input(img) for img in images]
                last["content"] = [text_block] + image_blocks
            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                    + [normalize_image_input(img) for img in images],
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
                    think=think,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                )
                try:
                    chat_completion = self.client.chat.completions.create(**params)
                except Exception:
                    params["max_completion_tokens"] = (
                        params.get("max_tokens") or params.get("max_completion_tokens") or self.default_ctx_size
                    )
                    params["temperature"] = 1
                    for k in ("max_tokens", "top_p", "frequency_penalty"):
                        params.pop(k, None)
                    chat_completion = self.client.chat.completions.create(**params)

                if stream:
                    in_thinking = False
                    for resp in chat_completion:
                        if self.is_cancelled():
                            ASCIIColors.warning("[LollmsBinding] Generation cancelled mid-stream.")
                            break
                        if count >= (n_predict or float("inf")):
                            break
                        delta = resp.choices[0].delta
                        reasoning_content = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                        content = getattr(delta, "content", None)

                        if reasoning_content:
                            if not in_thinking:
                                in_thinking = True
                                if streaming_callback:
                                    streaming_callback("<think>\n", MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                                output += "<think>\n"
                            if streaming_callback:
                                streaming_callback(reasoning_content, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                            output += reasoning_content
                            count += 1
                            continue

                        if content:
                            if in_thinking:
                                in_thinking = False
                                if streaming_callback:
                                    streaming_callback("\n</think>\n", MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                                output += "\n</think>\n"
                            if streaming_callback and not streaming_callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                            output += content
                            count += 1

                    if in_thinking:
                        if streaming_callback:
                            streaming_callback("\n</think>\n", MSG_TYPE.MSG_TYPE_CHUNK)
                        output += "\n</think>\n"
                else:
                    message_obj = chat_completion.choices[0].message
                    reasoning_content = getattr(message_obj, "reasoning_content", None) or getattr(message_obj, "reasoning", None)
                    content = message_obj.content or ""
                    if reasoning_content:
                        output = f"<think>\n{reasoning_content}\n</think>\n{content}"
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
                    think=think,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                )
                try:
                    completion = self.client.completions.create(**params)
                except Exception:
                    params["max_completion_tokens"] = params.get("max_tokens")
                    params["temperature"] = 1
                    for k in ("max_tokens", "top_p", "frequency_penalty"):
                        params.pop(k, None)
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

    # ── Generate From Messages ────────────────────────────────────────────

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

            text_content = "\n".join(p for p in text_parts if p.strip())

            if not images:
                return {"role": role, "content": text_content}

            openai_content = []
            if text_content:
                openai_content.append({"type": "text", "text": text_content})
            for img in images:
                img_url = img
                if not img.startswith("http") and not img.startswith("data:"):
                    img_url = f"data:image/jpeg;base64,{img}"
                openai_content.append({"type": "image_url", "image_url": {"url": img_url}})
            return {"role": role, "content": openai_content}

        def extract_reasoning(obj):
            for attr in ("reasoning_content", "reasoning", "thinking", "reasoning_text"):
                value = getattr(obj, attr, None)
                if value:
                    return value
            return None

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
                            "23d4f03a-b8a6-4adb-a183-7daa083a09cc", "lcp_tool"
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

        if think:
            params["reasoning_effort"] = reasoning_effort or "low"
            if reasoning_summary and reasoning_summary != "auto":
                params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary
            params.pop("temperature", None)
            params.pop("top_p", None)

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
                    if not think:
                        params["temperature"] = 1
                    if "extra_body" in params:
                        params["extra_body"].pop("chat_template_kwargs", None)
                    completion = self.client.chat.completions.create(**params)

            if stream:
                in_reasoning = False
                for chunk in completion:
                    if self.is_cancelled():
                        ASCIIColors.warning("[LollmsBinding] Generation cancelled mid-stream.")
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = extract_reasoning(delta)
                    content = getattr(delta, "content", None)

                    if reasoning:
                        if not in_reasoning:
                            in_reasoning = True
                            opening = "<think>\n"
                            output += opening
                            if streaming_callback:
                                streaming_callback(opening, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                        output += reasoning
                        if streaming_callback:
                            streaming_callback(reasoning, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                        continue

                    if content:
                        if in_reasoning:
                            in_reasoning = False
                            closing = "\n</think>\n"
                            output += closing
                            if streaming_callback:
                                streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                        output += content
                        if streaming_callback:
                            if not streaming_callback(content, MSG_TYPE.MSG_TYPE_CHUNK):
                                break

                if in_reasoning:
                    closing = "\n</think>\n"
                    output += closing
                    if streaming_callback:
                        streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
            else:
                message_obj = completion.choices[0].message
                reasoning = extract_reasoning(message_obj)
                content = message_obj.content or ""
                if reasoning:
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

    # ── Embeddings ────────────────────────────────────────────────────────

    def embed(self, text: str, **kwargs) -> list:
        embedding_model = kwargs.get("model", self.model_name)
        try:
            response = self.client.embeddings.create(model=embedding_model, input=[text])
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                ASCIIColors.warning("OpenAI API returned no data for the embedding request.")
                return []
        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using OpenAI API: {e}")
            trace_exception(e)
            return []

    # ── Model Info ────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        return {
            "name": "OpenAI",
            "version": "2.0",
            "host_address": self.open_ai_host_address,
            "model_name": self.model_name,
        }

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
        models_info = []
        prompt_buffer = 500

        try:
            models = self.client.models.list()
            for model in models.data:
                model_id = model.id
                if model_id.startswith(generation_prefixes):
                    context_length = known_context_lengths.get(model_id, "unknown")
                    max_generation = (
                        context_length - prompt_buffer if isinstance(context_length, int) else "unknown"
                    )
                    models_info.append({
                        "model_name": model_id,
                        "owned_by": getattr(model, "owned_by", "N/A"),
                        "created": getattr(model, "created", "N/A"),
                        "context_length": context_length,
                        "max_generation": max_generation,
                    })
                else:
                    models_info.append({
                        "model_name": model_id,
                        "owned_by": getattr(model, "owned_by", "N/A"),
                        "created": getattr(model, "created", "N/A"),
                        "context_length": None,
                        "max_generation": None,
                    })
        except Exception as e:
            trace_exception(e)
            print(f"Failed to list models: {e}")

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