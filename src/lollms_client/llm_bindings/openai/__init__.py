# bindings/OpenAI/binding.py
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
import math
import httpx
import pipmaster as pm
import mimetypes
pm.ensure_packages(["openai","tiktoken"])
from pathlib import Path
import ssl

import openai
import tiktoken
import os
import base64
BindingName = "OpenAIBinding"


def _read_file_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _extract_markdown_path(s):
    s = s.strip()
    if s.startswith("[") and s.endswith(")"):
        lb, rb = s.find("["), s.find("]")
        if lb != -1 and rb != -1 and rb > lb:
            return s[lb+1:rb].strip()
    return s

def _guess_mime_from_name(name, default="image/jpeg"):
    mime, _ = mimetypes.guess_type(name)
    return mime or default

def _to_data_url(b64_str, mime):
    return f"data:{mime};base64,{b64_str}"

def normalize_image_input(img, default_mime="image/jpeg"):
    """
    Returns a Responses API-ready content block:
      { "type": "input_image", "image_url": "data:<mime>;base64,<...>" }
    Accepts:
      - dict {'data': '<base64>', 'mime': 'image/png'}
      - dict {'path': 'E:\\images\\x.png'}
      - string raw base64
      - string local path (Windows/POSIX), including markdown-like "[E:\\path\\img.png]()"
    URLs are intentionally not supported (base64 only).
    """
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
        # Accept already-correct data URLs as-is
        if s.startswith("data:"):
            return {"type": "input_image", "image_url": s}
        # Local path heuristics: exists on disk or looks like a path
        if os.path.exists(s) or (":" in s and "\\" in s) or s.startswith("/") or s.startswith("."):
            b64 = _read_file_as_base64(s)
            mime = _guess_mime_from_name(s, default_mime)
            return {"type": "input_image", "image_url": _to_data_url(b64, mime)}
        # Otherwise, treat as raw base64 payload
        return {"type": "input_image", "image_url": _to_data_url(s, default_mime)}

    raise ValueError("Unsupported image input type")
class OpenAIBinding(LollmsLLMBinding):
    """OpenAI-specific binding implementation"""
    
    
    def __init__(self,
                 **kwargs):
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
        self.host_address=kwargs.get("host_address")
        self.model_name=kwargs.get("model_name")
        self.service_key=kwargs.get("service_key")
        self.verify_ssl_certificate=kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path=kwargs.get("certificate_file_path", None)
        self.default_completion_format=kwargs.get("default_completion_format", ELF_COMPLETION_FORMAT.Chat)
        self.is_vllm = kwargs.get("is_vllm", False)

        if not self.service_key:
            self.service_key = os.getenv("OPENAI_API_KEY", self.service_key)


        self.verify = True
        verify = True

        if not self.verify_ssl_certificate:
            self.verify = False
            verify = False

        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)

            if not cert_path.exists():
                raise FileNotFoundError(
                    f"Certificate file not found: {cert_path}"
                )

            ssl_context = ssl.create_default_context(
                cafile=str(cert_path)
            )
            self.verify = cert_path
            verify = ssl_context

        self.client = openai.OpenAI(
            api_key=self.service_key,
            base_url=self.open_ai_host_address,
            http_client=httpx.Client(
                verify=verify,
                timeout=300.0
            )
        )

    def _build_openai_params(self, messages: list, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        restricted_families = [
            "gpt-5",
            "gpt-4o",
            "o1",
            "o3",
            "o4"
        ]

        allowed_params = {
            "model", "messages", "temperature", "top_p", "n",
            "stop", "max_tokens", "presence_penalty", "frequency_penalty",
            "logit_bias", "stream", "user", "max_completion_tokens",
            "reasoning_effort"
        }

        params = {
            "model": model,
            "messages": messages,
        }

        for k, v in kwargs.items():
            if k in allowed_params and v is not None:
                params[k] = v
            else:
                if v is not None and kwargs.get("debug",False):
                    ASCIIColors.warning(f"Removed unsupported OpenAI param '{k}'")

        model_lower = model.lower()
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
        system_prompt: str = "",
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: int | None = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        split: Optional[bool] = False,
        user_keyword: Optional[str] = "!@>user:",
        ai_keyword: Optional[str] = "!@>assistant:",
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",
        reasoning_summary: Optional[str] = "auto",
        **kwargs
    ) -> Union[str, dict]:

        count = 0
        output = ""

        # ── Build message list ────────────────────────────────────────────────
        messages = [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant."
            }
        ]

        if images:
            if split:
                messages += self.split_discussion(
                    prompt,
                    user_keyword=user_keyword,
                    ai_keyword=ai_keyword
                )
                last = messages[-1]
                last["content"] = (
                    [{"type": "text", "text": last["content"]}]
                    + [normalize_image_input(img) for img in images]
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + [normalize_image_input(img) for img in images]
                        )
                    }
                )
        else:
            if split:
                messages += self.split_discussion(
                    prompt,
                    user_keyword=user_keyword,
                    ai_keyword=ai_keyword
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                )

        # ── Helper: extract reasoning from any delta / message object ─────────
        def extract_reasoning(obj):
            for attr in ("reasoning_content", "reasoning", "thinking", "reasoning_text"):
                value = getattr(obj, attr, None)
                if value:
                    return value
            return None

        try:

            # ══════════════════════════════════════════════════════════════════
            # Chat-completion path
            # ══════════════════════════════════════════════════════════════════
            if self.completion_format == ELF_COMPLETION_FORMAT.Chat:

                params = self._build_openai_params(
                    messages=messages,
                    n_predict=n_predict,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed
                )

                # ── Inject think / reasoning params ───────────────────────────
                if think:
                    if self.is_vllm:
                        # vLLM: enable thinking via chat_template_kwargs
                        params.setdefault("extra_body", {}).setdefault(
                            "chat_template_kwargs", {}
                        )["enable_thinking"] = True
                    else:
                        # OpenAI extended-thinking models (o3, o4-mini, gpt-5 …)
                        # Chat Completions uses flat reasoning_effort, not the
                        # nested `reasoning` dict (that is Responses API only).
                        params["reasoning_effort"] = reasoning_effort or "low"
                        # Some providers extend the API with a summary field;
                        # pass it via extra_body so the SDK doesn't reject it.
                        if reasoning_summary and reasoning_summary != "auto":
                            params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary
                        # These models reject temperature / top_p
                        params.pop("temperature", None)
                        params.pop("top_p", None)
                else:
                    if self.is_vllm:
                        # Explicitly disable thinking so vLLM doesn't carry it
                        # over from a cached template state
                        params.setdefault("extra_body", {}).setdefault(
                            "chat_template_kwargs", {}
                        )["enable_thinking"] = False

                # ── First attempt ─────────────────────────────────────────────
                try:
                    chat_completion = self.client.chat.completions.create(**params)

                except Exception as ex:
                    trace_exception(ex)

                    # Retry: adapt params for servers that reject certain fields
                    if "max_tokens" in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")

                    params.pop("top_p", None)
                    params.pop("frequency_penalty", None)
                    params.pop("reasoning_effort", None)

                    if not think:
                        params["temperature"] = 1

                    # Strip vLLM-specific extras so the retry is clean
                    if "extra_body" in params:
                        params["extra_body"].pop("chat_template_kwargs", None)

                    chat_completion = self.client.chat.completions.create(**params)

                # ── Streaming ─────────────────────────────────────────────────
                if stream:
                    in_reasoning = False

                    for resp in chat_completion:
                        if count >= (n_predict or float("inf")):
                            break

                        if not resp.choices:
                            continue

                        delta = resp.choices[0].delta
                        reasoning = extract_reasoning(delta)
                        content = getattr(delta, "content", None)

                        if reasoning:
                            if not in_reasoning:
                                in_reasoning = True
                                chunk = "<think>\n"
                                output += chunk
                                if streaming_callback:
                                    streaming_callback(chunk, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                            output += reasoning
                            if streaming_callback:
                                streaming_callback(reasoning, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)
                            count += 1
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
                            count += 1

                    # Close any dangling <think> block
                    if in_reasoning:
                        closing = "\n</think>\n"
                        output += closing
                        if streaming_callback:
                            streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)

                # ── Non-streaming ─────────────────────────────────────────────
                else:
                    message_obj = chat_completion.choices[0].message
                    reasoning = extract_reasoning(message_obj)
                    content = message_obj.content or ""

                    if reasoning:
                        # Server returned reasoning separately — wrap it
                        output = f"<think>\n{reasoning}\n</think>\n{content}"
                    else:
                        # vLLM (and some others) embed <think>…</think> directly
                        # in content, or thinking was disabled — pass through as-is
                        output = content

            # ══════════════════════════════════════════════════════════════════
            # Legacy completion path  (think / reasoning not supported here)
            # ══════════════════════════════════════════════════════════════════
            else:
                params = self._build_openai_params(
                    prompt=prompt,
                    n_predict=n_predict,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed
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
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None,
        ctx_size: int | None = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = "low",   # low, medium, high
        reasoning_summary: Optional[str] = "auto",  # auto
        **kwargs
    ) -> Union[str, dict]:

        # ── Normalize messages to the OpenAI wire format ──────────────────────
        def normalize_message(msg: Dict) -> Dict:
            role = msg.get("role", "user")
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
                openai_content.append(
                    {"type": "image_url", "image_url": {"url": img_url}}
                )
            return {"role": role, "content": openai_content}

        # ── Helper: extract reasoning from any delta / message object ─────────
        def extract_reasoning(obj):
            for attr in ("reasoning_content", "reasoning", "thinking", "reasoning_text"):
                value = getattr(obj, attr, None)
                if value:
                    return value
            return None

        openai_messages = [normalize_message(m) for m in messages]

        # ── Build base params ─────────────────────────────────────────────────
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

        # Drop None values
        params = {k: v for k, v in params.items() if v is not None}

        # ── Inject think / reasoning params ───────────────────────────────────
        if think:
            if self.is_vllm:
                params.setdefault("extra_body", {}).setdefault(
                    "chat_template_kwargs", {}
                )["enable_thinking"] = True
            else:
                # OpenAI extended-thinking models (o3, o4-mini, gpt-5 …)
                # Chat Completions uses flat reasoning_effort, not the
                # nested `reasoning` dict (that is Responses API only).
                params["reasoning_effort"] = reasoning_effort or "low"
                # Some providers extend the API with a summary field;
                # pass it via extra_body so the SDK doesn't reject it.
                if reasoning_summary and reasoning_summary != "auto":
                    params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary
                # These models reject temperature / top_p
                params.pop("temperature", None)
                params.pop("top_p", None)
        else:
            if self.is_vllm:
                # Explicitly disable so vLLM doesn't carry over a cached state
                params.setdefault("extra_body", {}).setdefault(
                    "chat_template_kwargs", {}
                )["enable_thinking"] = False

        output = ""

        try:
            # ── First attempt ─────────────────────────────────────────────────
            try:
                completion = self.client.chat.completions.create(**params)

            except Exception as ex:
                trace_exception(ex)

                # Retry: adapt for servers that reject certain fields
                if "max_tokens" in params:
                    params["max_completion_tokens"] = params.pop("max_tokens")

                params.pop("top_p", None)
                params.pop("frequency_penalty", None)
                params.pop("presence_penalty", None)
                params.pop("reasoning_effort", None)

                if not think:
                    params["temperature"] = 1

                # Strip vLLM-specific extras so the retry is clean
                if "extra_body" in params:
                    params["extra_body"].pop("chat_template_kwargs", None)

                completion = self.client.chat.completions.create(**params)

            # ── Streaming ─────────────────────────────────────────────────────
            if stream:
                in_reasoning = False

                for chunk in completion:
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

                # Close any dangling <think> block
                if in_reasoning:
                    closing = "\n</think>\n"
                    output += closing
                    if streaming_callback:
                        streaming_callback(closing, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)

            # ── Non-streaming ─────────────────────────────────────────────────
            else:
                message_obj = completion.choices[0].message
                reasoning = extract_reasoning(message_obj)
                content = message_obj.content or ""

                if reasoning:
                    # Server returned reasoning separately — wrap it
                    output = f"<think>\n{reasoning}\n</think>\n{content}"
                else:
                    # vLLM (and some others) embed <think>…</think> in content,
                    # or thinking was disabled — pass through as-is
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
        return 0.0  # Unknown → treat as free

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
        # Determine the embedding model
        embedding_model = kwargs.get("model", self.model_name)
        if not embedding_model.startswith("text-embedding"):
            embedding_model = "text-embedding-3-small"

        # Ensure input is a list of strings
        is_single_input = isinstance(text, str)
        input_texts = [text] if is_single_input else text

        # Optional safety: truncate if too many tokens for embedding model
        max_tokens_map = {
            "text-embedding-3-small": 8191,
            "text-embedding-3-large": 8191,
            "text-embedding-ada-002": 8191
        }
        max_tokens = max_tokens_map.get(embedding_model, None)
        if max_tokens is not None:
            input_texts = [
                self.detokenize(self.tokenize(t)[:max_tokens])
                for t in input_texts
            ]

        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=input_texts
            )

            if not response.data:
                ASCIIColors.warning(f"OpenAI API returned no data for the embedding request (model: {embedding_model}).")
                return []

            embeddings = [item.embedding for item in response.data]

            # Normalize if requested
            if normalize:
                embeddings = [
                    [v / math.sqrt(sum(x*x for x in emb)) for v in emb]
                    for emb in embeddings
                ]

            return embeddings[0] if is_single_input else embeddings

        except Exception as e:
            ASCIIColors.error(f"Failed to generate embeddings using model '{embedding_model}': {e}")
            trace_exception(e)
            return []


    def get_ctx_size(self, model_name: str | None = None) -> int:
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

        # Default context sizes (update as needed)
        context_map = {
            # GPT-4 family
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            # GPT-3.5 family
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            # GPT-5 and o-series
            "gpt-5": 200000,
            "o1": 200000,
            "o3": 200000,
            "o4": 200000,
        }

        # Try to find the best match
        model_name_lower = model_name.lower()
        for key, size in context_map.items():
            if model_name_lower.startswith(key):
                return size

        # Fallback: default safe value
        return 8192


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
            "model_name": self.model_name
        }

    def list_models(self) -> List[Dict]:
        # Known context lengths
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
            "ada"
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
