# bindings/OpenAI/binding.py
from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import http.server
import json
import math
import mimetypes
import os
import re
import secrets
import ssl
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import openai
import requests
import pipmaster as pm
import tiktoken
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_llm_binding import LollmsLLMBinding
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT, MSG_TYPE
from lollms_client.llm_bindings._tool_loop import (
    MAX_TOOL_ROUNDS,
    accumulate_streamed_tool_deltas,
    build_assistant_tool_calls_message,
    build_tool_call_callback_payloads,
    build_tool_result_messages,
    emit_callback_safe,
    extract_native_tool_calls,
    normalize_callback_result,
    resolve_max_tool_rounds,
)

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


def _sanitize_tools(raw_tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Sanitize a caller-provided tools array into OpenAI wire shape.

    - Drops unknown entries.
    - Removes accidental UUID `id` fields (some providers reject them).
    - Replaces the NIM placeholder function name.
    - Forces ``strict: False`` (broadest compatibility; NIM rejects strict).
    """
    if not raw_tools or not isinstance(raw_tools, list):
        return None
    sanitized: List[Dict] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        if "id" in tool and len(str(tool["id"])) == 36 and "-" in str(tool["id"]):
            tool.pop("id", None)
        if "function" in tool and isinstance(tool["function"], dict):
            func_def = tool["function"]
            func_def["strict"] = False
            name = func_def.get("name")
            if isinstance(name, str):
                func_def["name"] = name.replace(_NIM_FUNCTION_NAME_PLACEHOLDER, "lcp_tool")
        sanitized.append(tool)
    return sanitized


def _wire_shape_tool_calls(native: List[Any]) -> List[Dict[str, Any]]:
    """Normalize native tool_calls objects into OpenAI wire shape (list of dicts)."""
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
        "reasoning_details",
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
            self.output += " thinking\n"

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
                    if not self.suppress_thinking:
                        self.output += tag
                    self.in_content_thinking = True
                    if not self.suppress_thinking and self.callback:
                        if self.callback(tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
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
                    if thought_part and not self.suppress_thinking:
                        self.output += thought_part
                        if self.callback and self.callback(thought_part, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
                            return False

                    tag = m.group(0)
                    if not self.suppress_thinking:
                        self.output += tag
                    self.in_content_thinking = False
                    # Once answer content has started, a late dedicated-reasoning
                    # field must not reopen a  thinking block mid-answer.
                    self.dedicated_reasoning_opened = True
                    if not self.suppress_thinking and self.callback:
                        if self.callback(tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK) is False:
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
            if not self.suppress_thinking:
                close_tag = "\n</think>\n"
                self.output += close_tag
                if self.callback:
                    self.callback(close_tag, MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK)

        return self.output


# ═══════════════════════════════════════════════════════════════════════════
#  ChatGPT-plan OAuth  ("Sign in with ChatGPT", OpenAI's open-source
#  token-sharing flow)
#
#  Official docs: https://developers.openai.com/siwc/token-sharing-open-source
#
#  - Authorization-code + PKCE against auth.openai.com, public client,
#    no client secret and no partner key (dynamic client registration).
#  - The resulting OAuth access token is used as the Bearer credential on
#    the PUBLIC endpoint https://api.openai.com/v1/responses, billed to the
#    user's ChatGPT plan instead of API credits.
#  - Requirements imposed by that flow on every inference request:
#    store=false, stream=true, no `system` role items (use `instructions`).
# ═══════════════════════════════════════════════════════════════════════════


class OAuthError(RuntimeError):
    """OAuth failure (network, protocol or configuration)."""

    def __init__(self, message: str, code: Optional[str] = None, status: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status = status


class OAuthReauthRequired(OAuthError):
    """The stored session is unusable: the user must sign in again."""


@contextlib.contextmanager
def _interprocess_lock(path: Path):
    """
    Best-effort cross-process lock around a credentials file. Refresh tokens
    rotate, so two processes refreshing at once would invalidate each other.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass  # degrade to in-process locking only
        yield
    finally:
        fh.close()  # closing releases the lock on both platforms


class _OAuthCallbackServer(http.server.HTTPServer):
    callback_params: Optional[Dict[str, str]] = None


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != ChatGPTOAuth.CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        self.server.callback_params = {
            k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()
        }
        # Static page on purpose: never reflect callback parameters into HTML.
        body = (
            "<html><body style='font-family:sans-serif'>"
            "<h3>LoLLMS</h3><p>Sign-in finished. You can close this tab and return to the application.</p>"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence default stderr logging
        pass


class ChatGPTOAuth:
    """
    Credential manager for OpenAI's "Sign in with ChatGPT" plan-usage flow.

    Handles first sign-in (browser + loopback callback + PKCE + dynamic client
    registration), ID-token validation, protected local storage, and rotating
    refresh. Tokens are never logged.
    """

    ISSUER = "https://auth.openai.com"
    RESOURCE = "https://api.openai.com/v1"
    DYNAMIC_CLIENT_ID = "dynamic_agent_client"
    PLAN_SCOPE = "chatgpt.tokens.use.direct"
    SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    CALLBACK_PATH = "/auth/callback"
    DEFAULT_PORT = 1455
    REFRESH_MARGIN_S = 300
    TERMINAL_REFRESH_ERRORS = frozenset(
        {
            "invalid_grant",
            "invalid_refresh_token",
            "token_expired",
            "refresh_token_expired",
            "refresh_token_invalidated",
            "refresh_token_reused",
            "invalid_client",
        }
    )

    def __init__(
        self,
        credentials_path: Optional[Union[str, Path]] = None,
        agent_name: str = "LoLLMS",
        verify: Any = True,
        timeout: float = 30.0,
    ):
        self.path = Path(credentials_path).expanduser() if credentials_path else self.default_path()
        self.agent_name = agent_name
        self.verify = verify
        self.timeout = timeout
        self.issuer = self.ISSUER
        self.authorize_url = f"{self.issuer}/api/accounts/authorize"
        self.token_url = f"{self.issuer}/api/accounts/oauth/token"
        self.jwks_url = f"{self.issuer}/.well-known/jwks.json"
        self._lock = threading.RLock()

    # ── storage ────────────────────────────────────────────────────────────
    @staticmethod
    def default_path() -> Path:
        home = os.environ.get("LOLLMS_HOME")
        base = Path(home).expanduser() if home else Path.home() / ".lollms"
        return base / "openai_chatgpt_oauth.json"

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def status(self) -> Dict[str, Any]:
        rec = self._read()
        scopes = rec.get("scopes") or []
        return {
            "signed_in": bool(rec.get("access_token") and rec.get("refresh_token")),
            "email": rec.get("email"),
            "client_id": rec.get("client_id"),
            "scopes": scopes,
            "plan_usage_enabled": self.PLAN_SCOPE in scopes,
            "expires_at": rec.get("expires_at"),
            "credentials_path": str(self.path),
        }

    def is_signed_in(self) -> bool:
        return self.status()["signed_in"]

    def logout(self) -> None:
        """
        Delete the local credentials. This does NOT revoke the grant on
        OpenAI's side: remove the app in ChatGPT settings to do that.
        """
        with self._lock, _interprocess_lock(self.path):
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()

    def _ensure_host_id(self) -> str:
        with _interprocess_lock(self.path):
            rec = self._read()
            if not rec.get("ext_agent_host_id"):
                rec["ext_agent_host_id"] = f"urn:uuid:{uuid.uuid4()}"
                self._write(rec)
            return rec["ext_agent_host_id"]

    # ── token endpoint ─────────────────────────────────────────────────────
    @staticmethod
    def _error_code(body: Any) -> Optional[str]:
        if not isinstance(body, dict):
            return None
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("code") or err.get("type")
        if isinstance(err, str):
            return err
        return body.get("code")

    def _token_request(self, data: Dict[str, str]) -> Dict[str, Any]:
        try:
            r = httpx.post(
                self.token_url,
                data=data,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
                verify=self.verify,
            )
        except httpx.HTTPError as e:
            raise OAuthError(f"Network error contacting the OpenAI token endpoint: {e}") from e
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code != 200:
            code = self._error_code(body)
            rid = r.headers.get("openai-request-id")
            raise OAuthError(
                f"Token endpoint returned HTTP {r.status_code}"
                + (f" ({code})" if code else "")
                + (f" [request {rid}]" if rid else ""),
                code=code,
                status=r.status_code,
            )
        if not isinstance(body, dict) or not body.get("access_token"):
            raise OAuthError("Token endpoint response did not contain an access_token")
        return body

    @staticmethod
    def _parse_time(value: Any) -> Optional[float]:
        """earliest_refresh_at may come as epoch seconds/ms or ISO-8601."""
        if value in (None, ""):
            return None
        try:
            num = float(value)
            return num / 1000.0 if num > 1e12 else num
        except (TypeError, ValueError):
            pass
        try:
            return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def _record_from_token_response(self, tok: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        rec = dict(base)
        rec["access_token"] = tok["access_token"]
        if tok.get("refresh_token"):
            rec["refresh_token"] = tok["refresh_token"]
        if tok.get("id_token"):
            rec["id_token"] = tok["id_token"]
        rec["token_type"] = tok.get("token_type", "Bearer")
        expires_in = int(tok.get("expires_in") or 3600)
        rec["expires_in"] = expires_in
        rec["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rec["expires_at"] = now + expires_in
        rec["earliest_refresh_at"] = self._parse_time(tok.get("earliest_refresh_at"))
        if tok.get("scope"):
            rec["scopes"] = sorted(str(tok["scope"]).split())
        return rec

    # ── id-token validation ────────────────────────────────────────────────
    def _validate_id_token(self, id_token: str, client_id: str, nonce: str) -> Dict[str, Any]:
        try:
            import jwt  # PyJWT
        except ImportError:
            pm.ensure_packages(["PyJWT", "cryptography"])
            import jwt
        try:
            signing_key = jwt.PyJWKClient(self.jwks_url).get_signing_key_from_jwt(id_token)
            claims = jwt.decode(
                id_token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "PS256"],
                audience=client_id,
                issuer=self.issuer,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except Exception as e:
            raise OAuthError(f"ID token validation failed: {e}") from e
        if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise OAuthError("ID token validation failed: nonce mismatch")
        return claims

    # ── sign-in ────────────────────────────────────────────────────────────
    def _start_callback_server(self, port: Optional[int]) -> _OAuthCallbackServer:
        for p in ([port] if port else [self.DEFAULT_PORT, 0]):
            try:
                srv = _OAuthCallbackServer(("127.0.0.1", p), _OAuthCallbackHandler)
                srv.timeout = 1.0
                return srv
            except OSError:
                continue
        raise OAuthError("Could not open a loopback port for the OAuth callback")

    @staticmethod
    def _params_from_pasted(text: str) -> Dict[str, str]:
        text = text.strip()
        query = urllib.parse.urlparse(text).query if "?" in text or "://" in text else text
        return {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}

    def login(
        self,
        open_browser: bool = True,
        manual: bool = False,
        timeout: float = 300.0,
        port: Optional[int] = None,
        fresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Interactive sign-in.

        open_browser: open the system browser automatically.
        manual: headless mode. Prints the URL and asks you to paste the final
                redirect URL (the 127.0.0.1 page that fails to load) back.
        fresh: ignore the saved registration and sign in as a new account.
        """
        with self._lock:
            host_id = self._ensure_host_id()
            saved = {} if fresh else self._read()
            issued_client_id = saved.get("client_id")

            verifier = secrets.token_urlsafe(64)
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
                .rstrip(b"=")
                .decode("ascii")
            )
            state = secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(32)

            server = None if manual else self._start_callback_server(port)
            bound_port = server.server_address[1] if server else (port or self.DEFAULT_PORT)
            redirect_uri = f"http://127.0.0.1:{bound_port}{self.CALLBACK_PATH}"

            params: Dict[str, str] = {
                "response_type": "code",
                "client_id": issued_client_id or self.DYNAMIC_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": self.SCOPES,
                "resource": self.RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
                "ext_agent_host_id": host_id,
            }
            if not issued_client_id:
                params["agent_name_hint"] = self.agent_name  # first registration only
            else:
                if saved.get("email"):
                    params["login_hint"] = saved["email"]

            def _url(with_id_token_hint: bool) -> str:
                p = dict(params)
                if with_id_token_hint and issued_client_id and saved.get("id_token"):
                    p["id_token_hint"] = saved["id_token"]
                return f"{self.authorize_url}?{urllib.parse.urlencode(p, quote_via=urllib.parse.quote)}"

            try:
                cb: Optional[Dict[str, str]] = None
                if manual:
                    print("\nOpen this URL in a browser and approve access:\n")
                    print(_url(False))  # never print id_token_hint
                    print(
                        "\nThe browser will end on a 127.0.0.1 page that fails to load. "
                        "Copy that page's full URL and paste it below."
                    )
                    cb = self._params_from_pasted(input("Redirect URL: "))
                else:
                    opened = False
                    if open_browser:
                        with contextlib.suppress(Exception):
                            opened = bool(webbrowser.open(_url(True)))
                    if not opened:
                        ASCIIColors.info("Open this URL in your browser to sign in with ChatGPT:")
                        print(_url(False))
                    deadline = time.time() + timeout
                    while server.callback_params is None and time.time() < deadline:
                        server.handle_request()
                    cb = server.callback_params
                    if cb is None:
                        raise OAuthError(f"Timed out after {int(timeout)}s waiting for the sign-in callback")
            finally:
                if server is not None:
                    server.server_close()

            # ---- callback handling -----------------------------------------
            if cb.get("error"):
                if cb["error"] == "access_denied":
                    raise OAuthError(
                        "Access denied: ChatGPT plan usage was not authorised. "
                        "Run the sign-in again and approve token sharing.",
                        code="access_denied",
                    )
                raise OAuthError(f"Authorization failed: {cb['error']}", code=cb["error"])
            if not secrets.compare_digest(cb.get("state", ""), state):
                raise OAuthError("State mismatch in the sign-in callback; aborting")
            code = cb.get("code")
            if not code:
                raise OAuthError("The sign-in callback did not contain an authorization code")

            cb_client_id = cb.get("client_id")
            if issued_client_id:
                if cb_client_id and cb_client_id != issued_client_id:
                    raise OAuthError("The callback returned a different client_id than the saved registration")
                client_id = issued_client_id
            else:
                if not cb_client_id or cb_client_id == self.DYNAMIC_CLIENT_ID:
                    raise OAuthError("Registration incomplete: the callback did not include an issued client_id")
                client_id = cb_client_id

            tok = self._token_request(
                {
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect_uri,
                    "resource": self.RESOURCE,
                }
            )
            if not tok.get("id_token"):
                raise OAuthError("The token response did not contain an ID token")
            claims = self._validate_id_token(tok["id_token"], client_id, nonce)

            if issued_client_id and saved.get("subject") and saved["subject"] != claims.get("sub"):
                raise OAuthError("Signed in as a different ChatGPT account than the saved one; use fresh=True")
            if not tok.get("refresh_token"):
                raise OAuthError("No refresh_token was returned (offline_access not granted); cannot keep a session")

            base = {
                "issuer": self.issuer,
                "subject": claims.get("sub"),
                "email": claims.get("email"),
                "client_id": client_id,
                "ext_agent_host_id": host_id,
            }
            with _interprocess_lock(self.path):
                rec = self._record_from_token_response(tok, base)
                self._write(rec)

            if self.PLAN_SCOPE not in rec.get("scopes", []):
                raise OAuthError(
                    "Signed in, but ChatGPT plan usage was not granted "
                    f"(missing scope {self.PLAN_SCOPE}). Your plan/workspace may not allow it, "
                    "or the consent was not approved.",
                    code="plan_usage_not_granted",
                )
            return self.status()

    # ── token access / refresh ─────────────────────────────────────────────
    def _require_usable(self, rec: Dict[str, Any]) -> None:
        if not (rec.get("access_token") and rec.get("refresh_token") and rec.get("client_id")):
            raise OAuthReauthRequired(
                "Not signed in with ChatGPT. Call oauth_login() (or set oauth_auto_login=True)."
            )
        if self.PLAN_SCOPE not in (rec.get("scopes") or []):
            raise OAuthReauthRequired(
                "This sign-in does not include ChatGPT plan usage. Run oauth_login() and approve token sharing."
            )

    def _needs_refresh(self, rec: Dict[str, Any], now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        exp = rec.get("expires_at")
        if exp is None:
            return True
        if now < exp - self.REFRESH_MARGIN_S:
            return False
        earliest = rec.get("earliest_refresh_at")
        if earliest and now < earliest and now < exp:
            return False  # server asked us not to refresh yet and the token is still valid
        return True

    def _do_refresh(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        last: Optional[OAuthError] = None
        for attempt in range(3):
            try:
                tok = self._token_request(
                    {
                        "grant_type": "refresh_token",
                        "client_id": rec["client_id"],
                        "refresh_token": rec["refresh_token"],
                        "resource": self.RESOURCE,
                    }
                )
                return self._record_from_token_response(tok, rec)
            except OAuthError as e:
                transient = e.status is None or e.status >= 500 or e.status == 429
                if transient:
                    last = e
                    time.sleep(2**attempt)
                    continue
                if e.code in self.TERMINAL_REFRESH_ERRORS or e.status in (400, 401, 403):
                    dead = {
                        k: v
                        for k, v in rec.items()
                        if k not in ("access_token", "refresh_token", "expires_at", "earliest_refresh_at")
                    }
                    self._write(dead)  # keep identity + id_token for a hinted re-login
                    raise OAuthReauthRequired(
                        f"The ChatGPT session can no longer be refreshed ({e.code or e.status}). "
                        "Sign in again with oauth_login().",
                        code=e.code,
                        status=e.status,
                    ) from e
                raise
        raise last  # type: ignore[misc]

    def _refresh_locked(self, stale_access_token: Optional[str] = None) -> Dict[str, Any]:
        with _interprocess_lock(self.path):
            rec = self._read()  # another process may have refreshed meanwhile
            self._require_usable(rec)
            if stale_access_token is not None:
                if rec["access_token"] != stale_access_token and not self._needs_refresh(rec):
                    return rec
            elif not self._needs_refresh(rec):
                return rec
            new = self._do_refresh(rec)
            self._write(new)
            return new

    def get_access_token(self) -> str:
        with self._lock:
            rec = self._read()
            self._require_usable(rec)
            if self._needs_refresh(rec):
                rec = self._refresh_locked()
            return rec["access_token"]

    def force_refresh(self, stale_access_token: Optional[str] = None) -> str:
        """Refresh after a 401. Reuses a newer token if another process already refreshed."""
        with self._lock:
            return self._refresh_locked(stale_access_token=stale_access_token or "")["access_token"]


# ── Chat-format → Responses-format translation ─────────────────────────────
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)>[\s\S]*?</\1>", re.IGNORECASE)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text")
        )
    return "" if content is None else str(content)


def chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert Chat-Completions-style messages to (instructions, input items).

    The ChatGPT-plan flow rejects explicit `system` items, so system/developer
    messages are folded into `instructions`. Assistant history has its
    <think> blocks stripped before being replayed.
    """
    instructions_parts: List[str] = []
    items: List[Dict[str, Any]] = []

    for m in messages:
        role = str(m.get("role") or "user").lower()
        content = m.get("content", "")

        if role in ("system", "developer"):
            text = _content_to_text(content).strip()
            if text:
                instructions_parts.append(text)
            continue

        if role == "assistant":
            text = _THINK_BLOCK_RE.sub("", _content_to_text(content)).strip()
            if text:
                items.append({"role": "assistant", "content": text})
            continue

        # user / anything else
        if isinstance(content, str) or content is None:
            text = (content or "").strip()
            if text:
                items.append({"role": "user", "content": text})
            continue

        parts: List[Dict[str, Any]] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype in ("text", "input_text"):
                if p.get("text"):
                    parts.append({"type": "input_text", "text": p["text"]})
            elif ptype in ("image_url", "input_image"):
                url = p.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if isinstance(url, str) and url:
                    parts.append({"type": "input_image", "image_url": url})
            elif ptype in ("video_url", "video", "input_video"):
                ASCIIColors.warning("[OpenAIBinding] Video input is not supported with ChatGPT OAuth; dropped.")
        if parts:
            items.append({"role": "user", "content": parts})

    if not items:
        items.append({"role": "user", "content": " "})

    instructions = "\n\n".join(instructions_parts).strip() or "You are a helpful assistant."
    return instructions, items


_OAUTH_ERROR_HINTS = {
    "subscription_sharing_usage_limit_exceeded": (
        "ChatGPT plan usage limit reached. Check ChatGPT settings → Usage and retry later."
    ),
    "subscription_sharing_usage_unavailable": (
        "ChatGPT plan usage is currently unavailable. Check ChatGPT settings → Usage."
    ),
    "subscription_sharing_user_not_eligible": (
        "ChatGPT plan usage is not available for this user, workspace or policy."
    ),
}


def _describe_api_error(ex: Exception) -> str:
    code = None
    msg = str(ex)
    body = getattr(ex, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            code = err.get("code") or err.get("type")
            msg = err.get("message") or msg
        elif isinstance(err, str):
            msg = err
        if isinstance(body.get("detail"), str):
            msg = body["detail"]
    out = msg + (f" [{code}]" if code else "")
    hint = _OAUTH_ERROR_HINTS.get(str(code))
    rid = getattr(ex, "request_id", None)
    if hint:
        out += f" — {hint}"
    if rid:
        out += f" (request {rid})"
    return out


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
            auth_method (str): "api_key" (default), "oauth" (Sign in with ChatGPT, billed to the
                ChatGPT plan) or "auto" (api_key if a key/host is configured, else oauth if signed in).
            oauth_credentials_path (str): Where OAuth tokens are stored (default ~/.lollms/openai_chatgpt_oauth.json).
            oauth_agent_name (str): App name shown on the consent screen. Defaults to "LoLLMS".
            oauth_auto_login (bool): Start the browser sign-in automatically when not signed in.
            oauth_api_base (str): Responses endpoint used in oauth mode. Defaults to https://api.openai.com/v1.
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
        raw_is_vllm = kwargs.get("is_vllm", False)
        if isinstance(raw_is_vllm, str):
            self.is_vllm = raw_is_vllm.lower().strip() in ("true", "1", "yes", "on")
        else:
            self.is_vllm = bool(raw_is_vllm)
        self._vllm_probed = False
        self._is_vllm_detected = False

        raw_send_thinking = kwargs.get("send_thinking_parameter", True)
        if isinstance(raw_send_thinking, str):
            self.send_thinking_parameter = raw_send_thinking.lower().strip() not in ("false", "0", "no", "off", "")
        else:
            self.send_thinking_parameter = bool(raw_send_thinking)

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

        explicit_key = self.service_key or os.getenv("OPENAI_API_KEY")

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

        self._httpx_verify = verify

        # ── authentication method: API key (default) or ChatGPT-plan OAuth ──
        self._oauth = ChatGPTOAuth(
            credentials_path=kwargs.get("oauth_credentials_path") or None,
            agent_name=kwargs.get("oauth_agent_name") or "LoLLMS",
            verify=verify,
        )
        raw_auto_login = kwargs.get("oauth_auto_login", False)
        if isinstance(raw_auto_login, str):
            self.oauth_auto_login = raw_auto_login.lower().strip() in ("true", "1", "yes", "on")
        else:
            self.oauth_auto_login = bool(raw_auto_login)
        self.oauth_api_base = (kwargs.get("oauth_api_base") or ChatGPTOAuth.RESOURCE).rstrip("/")
        self._oauth_client_cache: Optional[Tuple[str, openai.OpenAI]] = None
        self._oauth_warned = False

        raw_method = str(kwargs.get("auth_method") or "api_key").lower().strip()
        if raw_method in ("oauth", "chatgpt", "chatgpt_oauth", "siwc"):
            self.auth_method = "oauth"
        elif raw_method in ("api_key", "apikey", "key"):
            self.auth_method = "api_key"
        elif raw_method == "auto":
            if explicit_key or self.host_address:
                self.auth_method = "api_key"
            elif self._oauth.is_signed_in():
                self.auth_method = "oauth"
            else:
                self.auth_method = "api_key"
        else:
            raise ValueError(f"Unknown auth_method '{raw_method}'. Use 'api_key', 'oauth' or 'auto'.")

        if self.auth_method == "oauth" and self.host_address:
            ASCIIColors.warning(
                "[OpenAIBinding] host_address is ignored in oauth mode (inference goes to "
                f"{self.oauth_api_base}). Use oauth_api_base to override."
            )

        if self.auth_method == "oauth" and not explicit_key:
            # No API key available: embeddings etc. are unavailable, inference uses OAuth.
            self.client = None
        else:
            self.client = openai.OpenAI(
                api_key=self.service_key or "EMPTY",
                base_url=self.open_ai_host_address,
                http_client=httpx.Client(
                    verify=verify,
                    timeout=300.0,
                ),
            )
        self.completion_format = ELF_COMPLETION_FORMAT.Chat

    # ── client lifecycle ───────────────────────────────────────────────────
    def _ensure_client(self) -> None:
        """
        Lazily (re)create the OpenAI SDK client.

        ``cancel()`` / ``close()`` dispose the underlying ``httpx.Client``;
        long-lived daemons that reuse the binding must be able to re-issue
        requests afterwards. This mirrors ``LollmsBinding._ensure_client``.

        In OAuth mode ``self.client`` is intentionally ``None`` (inference
        goes through ``_oauth_openai_client()``); nothing to rebuild here.

        Defensive: uses ``getattr`` for every attribute set during
        ``__init__`` so a partially-initialized instance (e.g. ``__init__``
        raised after ``super().__init__`` but before ``_httpx_verify`` was
        assigned) can still recover its client instead of cascading an
        ``AttributeError``.
        """
        if getattr(self, "auth_method", "api_key") == "oauth":
            return
        client = getattr(self, "client", None)
        http_client = getattr(client, "_client", None) if client is not None else None
        if client is None or (http_client is not None and getattr(http_client, "is_closed", False)):
            self.client = openai.OpenAI(
                api_key=getattr(self, "service_key", None) or "EMPTY",
                base_url=getattr(self, "open_ai_host_address", None),
                http_client=httpx.Client(
                    verify=getattr(self, "_httpx_verify", True),
                    timeout=300.0,
                ),
            )

    def cancel(self) -> None:
        """Dispose the SDK / OAuth HTTP clients and signal cancellation."""
        try:
            client = getattr(self, "client", None)
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()
        except Exception:
            pass
        cache = getattr(self, "_oauth_client_cache", None)
        if cache is not None:
            with contextlib.suppress(Exception):
                cache[1].close()
            self._oauth_client_cache = None
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

    # ── OAuth public helpers ───────────────────────────────────────────────
    def oauth_login(
        self,
        open_browser: bool = True,
        manual: bool = False,
        timeout: float = 300.0,
        fresh: bool = False,
    ) -> dict:
        """Sign in with ChatGPT (browser + loopback). Use manual=True on headless machines."""
        return self._oauth.login(open_browser=open_browser, manual=manual, timeout=timeout, fresh=fresh)

    def oauth_status(self) -> dict:
        return self._oauth.status()

    def oauth_logout(self) -> None:
        """Delete local OAuth credentials (revoke the app in ChatGPT settings to cut access)."""
        self._oauth.logout()
        self._oauth_client_cache = None

    # ── OAuth internals ────────────────────────────────────────────────────
    def _oauth_access_token(self) -> str:
        try:
            return self._oauth.get_access_token()
        except OAuthReauthRequired:
            if self.oauth_auto_login and not self._oauth.is_signed_in():
                ASCIIColors.info("[OpenAIBinding] Not signed in; starting Sign in with ChatGPT...")
                self._oauth.login()
                return self._oauth.get_access_token()
            raise

    def _oauth_openai_client(self) -> openai.OpenAI:
        token = self._oauth_access_token()
        if self._oauth_client_cache is None or self._oauth_client_cache[0] != token:
            if self._oauth_client_cache is not None:
                with contextlib.suppress(Exception):
                    self._oauth_client_cache[1].close()
            client = openai.OpenAI(
                api_key=token,
                base_url=self.oauth_api_base,
                max_retries=0,
                http_client=httpx.Client(verify=self._httpx_verify, timeout=300.0),
            )
            self._oauth_client_cache = (token, client)
        return self._oauth_client_cache[1]

    def _stream_responses_once(
        self,
        client: openai.OpenAI,
        instructions: str,
        input_items: List[Dict[str, Any]],
        reasoning: Optional[Dict[str, Any]],
        n_predict: Optional[int],
        handler: _StreamThinkingHandler,
    ) -> None:
        if not hasattr(client, "responses"):
            raise RuntimeError("The installed 'openai' package is too old for the Responses API: pip install -U openai")

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_items,
            "instructions": instructions,
            "store": False,  # required by the ChatGPT-plan flow
            "stream": True,  # required by the ChatGPT-plan flow
        }
        if reasoning:
            create_kwargs["reasoning"] = reasoning

        events = client.responses.create(**create_kwargs)
        completed = cancelled = limited = False
        count = 0
        summary_parts = 0
        try:
            for event in events:
                if self.is_cancelled():
                    cancelled = True
                    break
                etype = getattr(event, "type", "")

                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        if not handler.process_content(delta):
                            cancelled = True
                            break
                        count += 1
                        if n_predict and count >= n_predict:
                            limited = True
                            break

                elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    delta = getattr(event, "delta", "") or ""
                    if delta and not handler.process_reasoning(delta):
                        cancelled = True
                        break

                elif etype == "response.reasoning_summary_part.added":
                    if summary_parts and not handler.process_reasoning("\n\n"):
                        cancelled = True
                        break
                    summary_parts += 1

                elif etype == "response.completed":
                    completed = True

                elif etype == "response.incomplete":
                    details = getattr(getattr(event, "response", None), "incomplete_details", None)
                    ASCIIColors.warning(f"[OpenAIBinding] Response incomplete: {details}")
                    completed = True

                elif etype == "response.failed":
                    err = getattr(getattr(event, "response", None), "error", None)
                    code = getattr(err, "code", None) or "unknown_error"
                    message = getattr(err, "message", None) or ""
                    text = f"{message} [{code}]".strip()
                    hint = _OAUTH_ERROR_HINTS.get(str(code))
                    raise RuntimeError(text + (f" — {hint}" if hint else ""))

                elif etype == "error":
                    code = getattr(event, "code", None)
                    message = getattr(event, "message", None) or "stream error"
                    raise RuntimeError(f"{message}" + (f" [{code}]" if code else ""))
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        if not (completed or cancelled or limited):
            raise RuntimeError("The stream ended before response.completed")

    def _generate_via_chatgpt_oauth(
        self,
        messages: List[Dict[str, Any]],
        n_predict: Optional[int] = None,
        stream: Optional[bool] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        effort: Optional[str] = None,
        is_thinking_deactivated: bool = True,
        reasoning_summary: Optional[str] = None,
    ) -> Union[str, dict]:
        """
        Run one generation through the Responses API using the ChatGPT-plan OAuth token.

        Notes: the flow always streams upstream (a non-streaming call just doesn't
        invoke the callback). temperature / top_p / seed / penalties are not sent
        (reasoning models reject them), and tools are not forwarded.
        """
        if not self._oauth_warned:
            self._oauth_warned = True
            ASCIIColors.info(
                "[OpenAIBinding] oauth mode: sampling parameters (temperature, top_p, seed, penalties) "
                "and tools are not sent."
            )

        handler = _StreamThinkingHandler(
            streaming_callback if stream else None,
            suppress_thinking=is_thinking_deactivated,
        )
        try:
            instructions, input_items = chat_messages_to_responses_input(messages)

            if is_thinking_deactivated:
                reasoning: Optional[Dict[str, Any]] = {"effort": "none"}
            else:
                projected = self._project_effort(effort) or "medium"
                reasoning = {
                    "effort": projected,
                    "summary": reasoning_summary or "auto",
                }

            refreshed = False
            while True:
                client = self._oauth_openai_client()
                try:
                    self._stream_responses_once(client, instructions, input_items, reasoning, n_predict, handler)
                    break
                except openai.AuthenticationError:
                    if refreshed:
                        raise
                    refreshed = True
                    self._oauth.force_refresh(stale_access_token=client.api_key)
                except openai.BadRequestError as ex:
                    msg = str(ex).lower()
                    if reasoning is not None and ("reasoning" in msg or "effort" in msg):
                        ASCIIColors.warning(
                            "[OpenAIBinding] Model rejected the reasoning setting; retrying without it."
                        )
                        reasoning = None
                        continue
                    raise
            output = handler.flush()

        except OAuthError as e:
            err_msg = f"ChatGPT OAuth error: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}
        except Exception as e:
            trace_exception(e)
            detail = _describe_api_error(e) if isinstance(e, openai.APIStatusError) else str(e)
            err_msg = f"An error occurred with the OpenAI API (ChatGPT OAuth): {detail}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}

        return output

    def check_is_vllm(self) -> bool:
        """
        Determines whether the target server is a vLLM server.
        Uses configured is_vllm flag, or runs a fast cached probe against /version.
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
                        ASCIIColors.info(f"[OpenAIBinding] Auto-detected vLLM server (version {data.get('version')}).")
                        self._is_vllm_detected = True
                        self.is_vllm = True
                        return True
            except Exception:
                pass

        return False

    def _build_openai_params(self, messages: Optional[list] = None, prompt: Optional[str] = None, **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        if "n_predict" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("n_predict")

        restricted_families = [
            "gpt-5",
            "o1",
            "o3",
            "o4",
        ]

        allowed_params = {
            "model",
            "messages",
            "prompt",
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
            "extra_body",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
        }

        params: Dict[str, Any] = {
            "model": model,
        }
        if messages is not None:
            params["messages"] = messages
        if prompt is not None:
            params["prompt"] = prompt

        for k, v in kwargs.items():
            if k in allowed_params and v is not None:
                params[k] = v
            else:
                if v is not None and kwargs.get("debug", False):
                    ASCIIColors.warning(f"Removed unsupported OpenAI param '{k}'")

        is_vllm = self.check_is_vllm()
        if is_vllm:
            extra_body = params.setdefault("extra_body", {})
            if "top_k" in kwargs and kwargs["top_k"] is not None:
                extra_body.setdefault("top_k", kwargs["top_k"])
            if "repeat_penalty" in kwargs and kwargs["repeat_penalty"] is not None:
                extra_body.setdefault("repetition_penalty", kwargs["repeat_penalty"])
            if "min_tokens" in kwargs and kwargs["min_tokens"] is not None:
                extra_body.setdefault("min_tokens", kwargs["min_tokens"])

        model_lower = model.lower() if model else ""
        if not is_vllm and any(fam in model_lower for fam in restricted_families):
            if "temperature" in params and params["temperature"] != 1:
                ASCIIColors.warning(
                    f"{model} does not support temperature != 1. Overriding to 1."
                )
                params["temperature"] = 1
            if "top_p" in params:
                ASCIIColors.warning(f"{model} does not support top_p. Removing it.")
                params.pop("top_p")

        return params

    # ── thinking / reasoning helpers ───────────────────────────────────────
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")
    _EFFORT_OFF_VALUES = frozenset({"none", "off", "disabled", "false", "0", ""})

    def _is_reasoning_model(self) -> bool:
        """True for OpenAI reasoning families that accept `reasoning_effort`."""
        model_lower = (self.model_name or "").lower()
        return any(model_lower.startswith(p) for p in self._REASONING_MODEL_PREFIXES)

    @staticmethod
    def _normalize_effort(effort: Optional[str]) -> Optional[str]:
        """Normalize an effort string; returns None when thinking is off."""
        if effort is None:
            return None
        s = str(effort).strip().lower()
        if s in OpenAIBinding._EFFORT_OFF_VALUES:
            return None
        return "high" if s == "max" else s

    def _project_effort(self, effort: Optional[str]) -> Optional[str]:
        """
        Project an incoming effort onto the model's supported_reasoning_efforts list
        by ordinal position (closest match). Honors the description.yaml contract.
        """
        norm = self._normalize_effort(effort)
        if norm is None:
            return None
        supported = [str(e).strip().lower() for e in (self.supported_reasoning_efforts or []) if str(e).strip()]
        if not supported:
            return norm
        if norm in supported:
            return norm
        # Map by ordinal position: low→first, medium→middle, high→last, max→last
        ordinal = {"low": 0, "medium": 1, "high": 2, "max": 2}
        idx = ordinal.get(norm, 1)
        if idx >= len(supported):
            idx = len(supported) - 1
        return supported[idx]

    def _apply_thinking_params(
        self,
        params: dict,
        effort: Optional[str],
        is_deactivated: bool,
    ) -> dict:
        """
        Applies thinking / reasoning controls across OpenAI, vLLM, Z.AI/GLM,
        and OpenAI-compatible engines. Delegates to small helpers so each
        backend's quirks stay isolated.
        """
        is_glm = "glm" in (self.model_name or "").lower()
        is_vllm = self.check_is_vllm()
        extra_body = params.setdefault("extra_body", {})

        projected = None if is_deactivated else self._project_effort(effort)
        thinking_on = projected is not None

        # 1) Top-level reasoning_effort — only for models that accept it.
        if self._is_reasoning_model():
            params["reasoning_effort"] = projected if thinking_on else "none"
        elif thinking_on:
            # Non-reasoning models: don't send reasoning_effort at all.
            params.pop("reasoning_effort", None)

        # 2) extra_body.reasoning_effort — for OpenAI-compatible backends.
        extra_body["reasoning_effort"] = projected if thinking_on else "none"

        # 3) Backend-specific thinking flags.
        if is_vllm:
            self._apply_vllm_thinking_flags(extra_body, projected, is_glm)
        else:
            self._apply_generic_thinking_flags(extra_body, projected, is_glm)

        # 4) Reasoning models reject temperature/top_p when thinking is on.
        if thinking_on and self._is_reasoning_model():
            params.pop("temperature", None)
            params.pop("top_p", None)

        return params

    def _apply_vllm_thinking_flags(
        self, extra_body: dict, projected: Optional[str], is_glm: bool
    ) -> None:
        """vLLM-specific chat_template_kwargs + thinking flag."""
        thinking_on = projected is not None
        if self.send_thinking_parameter and not getattr(self, "glm_image_embedding", False):
            ctk = extra_body.setdefault("chat_template_kwargs", {})
            if self.thinking_effort_keyword == "reasoning_effort":
                ctk["reasoning_effort"] = projected if thinking_on else "none"
                ctk["enable_thinking"] = thinking_on
            else:
                ctk[self.thinking_effort_keyword] = thinking_on
                ctk["enable_thinking"] = thinking_on
                ctk["thinking"] = thinking_on
        if is_glm:
            extra_body["thinking"] = {"type": "enabled" if thinking_on else "disabled"}
        else:
            extra_body["thinking"] = thinking_on

    def _apply_generic_thinking_flags(
        self, extra_body: dict, projected: Optional[str], is_glm: bool
    ) -> None:
        """Generic OpenAI-compatible backend thinking flags."""
        thinking_on = projected is not None
        ctk = extra_body.setdefault("chat_template_kwargs", {})
        ctk["enable_thinking"] = thinking_on
        ctk["thinking"] = thinking_on
        if is_glm:
            extra_body["thinking"] = {"type": "enabled" if thinking_on else "disabled"}
        else:
            extra_body["thinking"] = thinking_on

    def _apply_vllm_thinking_kwargs(self, params: dict, effort: Optional[str]) -> dict:
        is_deact = self._normalize_effort(effort) is None
        return self._apply_thinking_params(params, effort, is_deact)

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

        ``tools`` accepts the OpenAI Chat Completions tools schema::

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather for a location",
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
        ``{"type": "function", "function": {"name": "..."}}`` dict. When
        ``tools`` is provided and ``tool_choice`` is omitted, ``"auto"`` is
        used.

        When ``tools`` is provided, the call is routed through
        :meth:`_run_tool_loop`, which implements the documented OpenAI
        lifecycle: (1) the model emits ``tool_calls``; (2) the binding
        forwards each call to the callback as ``<tool>{"name":...,
        "parameters":{...}}</tool>`` XML; (3) if the callback returns
        ``{"tool_results": [...]}``, the binding appends the assistant
        tool-call turn and the ``tool`` messages, then re-invokes the model
        until no more tool calls are requested or ``MAX_TOOL_ROUNDS`` is
        reached.

        Callback contract (streaming AND non-streaming when tools are used)::

            True / None                          → continue; output unchanged
            False                                → abort the call
            {"tool_results": [                   → binding appends these tool
                {"tool_call_id": "call_abc",          results to the conversation
                 "content": "{\\"temp\\":22}"},          and re-invokes the model
                ...                              (bounded by MAX_TOOL_ROUNDS)
            ]}

        Backward compatible: callbacks returning ``bool`` keep working.
        """
        if not self.model_name:
            return {"status": "error", "message": "[LollmsBinding] No model_name resolved. Please configure a model profile or set model_name in binding config."}

        self._ensure_client()

        if think is True:
            effort = self.get_effective_reasoning_effort(think=True, reasoning_effort=reasoning_effort)
            effort = self._normalize_effort(effort)
            is_thinking_deactivated = effort is None
        else:
            is_thinking_deactivated = True
            effort = None

        ASCIIColors.info(
            f"[OpenAIBinding.generate_text] think={think}, reasoning_effort={reasoning_effort} "
            f"-> effective_effort={effort}, deactivated={is_thinking_deactivated}"
        )

        messages: List[Dict[str, Any]] = [
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

        if self.auth_method == "oauth":
            return self._generate_via_chatgpt_oauth(
                messages,
                n_predict=n_predict,
                stream=stream,
                streaming_callback=streaming_callback,
                effort=effort,
                is_thinking_deactivated=is_thinking_deactivated,
                reasoning_summary=reasoning_summary,
            )

        count = 0
        output = ""

        try:
            if self.completion_format == ELF_COMPLETION_FORMAT.Chat:
                params = self._build_openai_params(
                    messages=messages,
                    n_predict=n_predict,
                    stream=stream,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                    **kwargs,
                )

                if self.check_is_vllm():
                    extra_body = params.setdefault("extra_body", {})
                    if top_k is not None:
                        extra_body.setdefault("top_k", top_k)
                    if repeat_penalty is not None:
                        extra_body.setdefault("repetition_penalty", repeat_penalty)
                    if "min_tokens" in kwargs and kwargs["min_tokens"] is not None:
                        extra_body.setdefault("min_tokens", kwargs["min_tokens"])

                self._apply_thinking_params(params, effort, is_thinking_deactivated)
                if reasoning_summary and reasoning_summary != "auto" and not is_thinking_deactivated:
                    params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary

                if "extra_body" in params and not params["extra_body"]:
                    params.pop("extra_body", None)

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
                        if not params["extra_body"]:
                            params.pop("extra_body", None)

                    if not self.check_is_vllm() and (effort is None or is_thinking_deactivated):
                        params["temperature"] = 1

                    chat_completion = self.client.chat.completions.create(**params)

                if stream:
                    handler = _StreamThinkingHandler(streaming_callback, suppress_thinking=is_thinking_deactivated)
                    accumulated_stream_tool_calls: Dict[int, Dict[str, Any]] = {}

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

                        # Accumulate tool-call deltas even when tools were not
                        # requested: some OpenAI-compatible backends emit
                        # unsolicited tool_calls. They are synthesized into
                        # <tool> XML below when native tools are disabled.
                        delta_tool_calls = getattr(delta, "tool_calls", None)
                        if delta_tool_calls:
                            accumulate_streamed_tool_deltas(
                                accumulated_stream_tool_calls, delta_tool_calls
                            )

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

                    if accumulated_stream_tool_calls:
                        synthesized = []
                        for idx in sorted(accumulated_stream_tool_calls.keys()):
                            tc = accumulated_stream_tool_calls[idx]
                            name = (tc.get("name") or "").strip()
                            if not name:
                                continue
                            tc_id = tc.get("id") or f"call_{idx}"
                            raw_args = tc.get("arguments") or ""
                            try:
                                tool_args = json.loads(raw_args) if raw_args else {}
                            except (json.JSONDecodeError, TypeError):
                                tool_args = {"raw": raw_args}
                            synthesized.append(
                                {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(tool_args),
                                    },
                                }
                            )
                        if synthesized:
                            output = self._synthesize_tool_xml_if_needed(
                                output, synthesized
                            )

                else:
                    message_obj = chat_completion.choices[0].message
                    reasoning = extract_reasoning(message_obj)
                    content = message_obj.content or ""

                    if reasoning and not content.strip().startswith((" thinking", "<thinking>")):
                        output = f" thinking\n{reasoning}\n</think>\n\n{content}"
                    else:
                        output = content

                    # Synthesize <tool> XML when native tool calling is not
                    # advertised but the backend still returned tool_calls.
                    output = self._synthesize_tool_xml_if_needed(
                        output, getattr(message_obj, "tool_calls", None)
                    )

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

    @staticmethod
    def _is_auto_tool_choice_unsupported(ex: Exception) -> bool:
        """
        True when the server rejected the request because it was launched
        without vLLM's tool-call parser flags. vLLM returns HTTP 400 with a
        message like::

            '"auto" tool choice requires --enable-auto-tool-choice and
             --tool-call-parser to be set'

        In that case the correct client-side fallback is to retry the SAME
        payload with ``tools`` / ``tool_choice`` stripped: the model will
        answer normally, tool calling is simply unavailable on this server.
        """
        if not isinstance(ex, openai.BadRequestError):
            return False
        text = str(ex).lower()
        return (
            "tool choice requires" in text
            and (
                "--enable-auto-tool-choice" in text
                or "--tool-call-parser" in text
            )
        )

    def _warn_auto_tool_choice_once(self) -> None:
        if getattr(self, "_auto_tool_choice_warned", False):
            return
        self._auto_tool_choice_warned = True
        ASCIIColors.error(
            "[OpenAIBinding] Server rejected tool_choice='auto': vLLM was "
            "launched without --enable-auto-tool-choice / --tool-call-parser. "
            "Native tool calling is DISABLED for this session; the request "
            "will be retried without the tools array. To enable tool calling, "
            "relaunch vLLM with e.g. "
            "`--enable-auto-tool-choice --tool-call-parser hermes` "
            "(or `deepseek_v3`, `internlm`, `mistral`, `llama3_json`)."
        )

    def _synthesize_tool_xml_if_needed(
        self,
        content: str,
        native_tool_calls: Optional[List[Any]],
    ) -> str:
        """
        When native tool calling is NOT advertised by this binding
        (``native_tools_enabled is False``) but the backend still emits
        native ``tool_calls`` (some OpenAI-compatible servers do this
        unsolicited), translate them into the Lollms ``<tool>{json}</tool>``
        XML protocol so the XML tooling layer can pick them up.

        The XML block is appended on a NEW LINE after the visible content.
        When ``native_tools_enabled`` is True the native lifecycle already
        owns the tool calls, so this is a no-op.
        """
        if getattr(self, "native_tools_enabled", False):
            return content
        if not native_tool_calls:
            return content

        try:
            xml = LollmsLLMBinding.tool_calls_to_lollms_xml(native_tool_calls)
        except Exception as e:
            ASCIIColors.warning(
                f"[LollmsBinding] Failed to synthesize <tool> XML from native "
                f"tool_calls: {e}"
            )
            return content

        if not xml:
            return content

        if content and content.strip():
            return f"{content}\n{xml}"
        return xml

    def _synthesize_tool_xml_if_needed(
        self,
        content: str,
        native_tool_calls: Optional[List[Any]],
    ) -> str:
        """
        When native tool calling is NOT advertised by this binding
        (``native_tools_enabled is False``) but the backend still emits
        native ``tool_calls`` (some OpenAI-compatible servers do this
        unsolicited), translate them into the Lollms ``<tool>{json}</tool>``
        XML protocol so the XML tooling layer can pick them up.

        The XML block is appended on a NEW LINE after the visible content.
        When ``native_tools_enabled`` is True the native lifecycle already
        owns the tool calls, so this is a no-op.
        """
        if getattr(self, "native_tools_enabled", False):
            return content
        if not native_tool_calls:
            return content

        try:
            xml = LollmsLLMBinding.tool_calls_to_lollms_xml(native_tool_calls)
        except Exception as e:
            ASCIIColors.warning(
                f"[OpenAIBinding] Failed to synthesize <tool> XML from native "
                f"tool_calls: {e}"
            )
            return content

        if not xml:
            return content

        if content and content.strip():
            return f"{content}\n{xml}"
        return xml

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
                "[OpenAIBinding] Model produced reasoning-only output while "
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
                f"[OpenAIBinding._non_stream_fallback_for_streamed_tools] "
                f"Retry rejected: {e}"
            )
            return None

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

        The callback may return:
          - ``True`` / ``None`` → keep streaming (backward compat)
          - ``False``           → abort the stream and stop the loop
          - ``{"tool_results": [{"tool_call_id": "...", "content": "..."}]}``
                                 → feed results back into the next round

        Both streaming and non-streaming modes are supported. In streaming
        mode, tool-call deltas are accumulated across the stream and their
        translated ``<tool>`` XML is dispatched to the callback BEFORE the
        structured-return check, so a caller that wants the legacy behaviour
        (parse the XML itself and drive their own loop) still works.
        """
        conversation = list(messages)
        last_output: Union[str, dict] = ""
        accumulated_outputs: List[str] = []

        # Resolve the effective round budget: an explicit ``max_tool_rounds``
        # kwarg wins, then the ambient contextvar set by the chat mixin via
        # ``tool_rounds_budget(...)``, then the module default (8). This is
        # what lets a 50-round agentic budget flow through the binding
        # without every provider re-declaring it.
        effective_max_rounds = resolve_max_tool_rounds(
            extra_kwargs.get("max_tool_rounds")
        )

        for round_index in range(effective_max_rounds):
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
                break

            # A round that produced tool results MUST also have produced a
            # matching assistant tool-call turn. If it didn't (edge case: the
            # callback returned tool_results without the model requesting any
            # tool call — impossible on a well-behaved endpoint, but a broken
            # server could produce it), skip the round rather than corrupt
            # the wire state.
            assistant_msg = getattr(self, "_last_assistant_tool_message", None)
            self._last_assistant_tool_message = None
            if not assistant_msg:
                ASCIIColors.warning(
                    "[OpenAIBinding] Callback returned tool_results but no "
                    "assistant tool-call message was captured for this round. "
                    "Discarding the tool results and terminating the loop."
                )
                break

            # Validate that every tool_result's tool_call_id matches one of the
            # assistant's tool_calls. Drop orphans; if all are orphans, stop.
            requested_ids = {
                tc.get("id") for tc in assistant_msg.get("tool_calls", []) if tc.get("id")
            }
            matched = [r for r in tool_results if r.get("tool_call_id") in requested_ids]
            if not matched:
                ASCIIColors.warning(
                    "[OpenAIBinding] All provided tool_results had tool_call_id "
                    "values that did not match any tool_calls in the previous "
                    "assistant turn. Terminating the loop to avoid corrupting "
                    "the conversation state."
                )
                break
            if len(matched) != len(tool_results):
                ASCIIColors.warning(
                    f"[OpenAIBinding] Dropped {len(tool_results) - len(matched)} "
                    f"tool_results with unmatched tool_call_id."
                )

            conversation.append(assistant_msg)
            conversation.extend(build_tool_result_messages(matched))
        else:
            ASCIIColors.warning(
                f"[OpenAIBinding] Tool-call loop reached the effective round "
                f"budget ({effective_max_rounds}); returning last partial "
                f"result. Pass a larger `max_tool_rounds` kwarg or set the "
                f"ambient `tool_rounds_budget(...)` context to raise it."
            )

        if len(accumulated_outputs) > 1:
            return "\n".join(accumulated_outputs)
        if accumulated_outputs:
            return accumulated_outputs[-1]

        # Never return a bare empty string: a silent "" leaves the caller
        # unable to distinguish "endpoint returned nothing" from "loop
        # exited cleanly". Surface a diagnostic error so the root cause is
        # visible instead of swallowed by the example's final print.
        #
        # The diagnostic is TOOL-AWARE: when the caller never supplied a
        # tools array (plain chat through the managed loop), the misleading
        # "tool loop" message would send operators chasing server-side
        # --tool-call-parser flags that are irrelevant. Emit an honest
        # "empty model response" hint instead, including the most common
        # root cause for reasoning-heavy families (DeepSeek V4.1, QwQ,
        # o-series): a reasoning-only completion that was suppressed by
        # think=False.
        if isinstance(last_output, str) and not last_output.strip():
            if tools:
                return {
                    "status": "error",
                    "message": (
                        "[OpenAIBinding] Tool loop completed with an EMPTY final "
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
            return {
                "status": "error",
                "message": (
                    "[OpenAIBinding] The model returned an EMPTY answer — no "
                    "visible content arrived from the endpoint across the "
                    "non-streaming request and its streaming fallback. This "
                    "is not a tool-calling failure: no tools were requested "
                    "on this call.\n"
                    "Most common causes:\n"
                    "  1. Reasoning-only model under think=False. DeepSeek "
                    "V4.1/R1, QwQ, o-series, and similar families can emit "
                    "only `reasoning_content` for certain prompts; the "
                    "binding strips it because thinking is deactivated. "
                    "Retry with `think=True` (or an explicit reasoning_effort) "
                    "so the reasoning is surfaced instead of discarded.\n"
                    "  2. The prompt was truncated below the model's reasoning "
                    "budget — raise n_predict.\n"
                    "  3. vLLM's chat template produced an empty assistant turn "
                    "for this request shape — check the server log for "
                    "template warnings."
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
        One round of the tool loop. Returns ``(output, tool_results)`` where
        ``tool_results`` is ``None`` unless the callback opted into the
        structured contract.
        """
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

            # ── OpenAI native tool wire state preservation ────────────────
            # An assistant turn that emits tool_calls must carry them back
            # verbatim on the next round; a tool turn must preserve its
            # tool_call_id so the provider can correlate results. Both are
            # REQUIRED by the OpenAI Chat Completions contract.
            if role == "assistant" and msg.get("tool_calls"):
                preserved: Dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": msg["tool_calls"],
                    # OpenAI allows (and prefers) content=null when only
                    # tool calls are present.
                    "content": text_content if text_content else None,
                }
                return preserved
            if role == "tool":
                tool_entry: Dict[str, Any] = {"role": "tool", "content": text_content}
                if msg.get("tool_call_id"):
                    tool_entry["tool_call_id"] = msg["tool_call_id"]
                if msg.get("name"):
                    tool_entry["name"] = msg["name"]
                return tool_entry

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

        openai_messages = [normalize_message(m) for m in conversation]

        sanitized_tools = _sanitize_tools(tools)

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

        if "extra_body" in extra_kwargs and isinstance(extra_kwargs["extra_body"], dict):
            params.setdefault("extra_body", {}).update(extra_kwargs["extra_body"])

        if self.check_is_vllm():
            extra_body = params.setdefault("extra_body", {})
            effective_top_k = top_k if top_k is not None else extra_kwargs.get("top_k")
            if effective_top_k is not None:
                extra_body.setdefault("top_k", effective_top_k)
            if repeat_penalty is not None:
                extra_body.setdefault("repetition_penalty", repeat_penalty)
            if "min_tokens" in extra_kwargs and extra_kwargs["min_tokens"] is not None:
                extra_body.setdefault("min_tokens", extra_kwargs["min_tokens"])

        params = {k: v for k, v in params.items() if v is not None}

        if think is True:
            effort = self.get_effective_reasoning_effort(think=True, reasoning_effort=reasoning_effort)
            effort = self._normalize_effort(effort)
            is_thinking_deactivated = effort is None
        else:
            is_thinking_deactivated = True
            effort = None

        ASCIIColors.info(
            f"[OpenAIBinding.generate_from_messages] think={think}, reasoning_effort={reasoning_effort} "
            f"-> effective_effort={effort}, deactivated={is_thinking_deactivated}"
        )

        if self.auth_method == "oauth":
            return self._generate_via_chatgpt_oauth(
                openai_messages,
                n_predict=n_predict,
                stream=stream,
                streaming_callback=streaming_callback,
                effort=effort,
                is_thinking_deactivated=is_thinking_deactivated,
                reasoning_summary=reasoning_summary,
            ), None

        self._apply_thinking_params(params, effort, is_thinking_deactivated)
        if reasoning_summary and reasoning_summary != "auto" and not is_thinking_deactivated:
            params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary

        if "extra_body" in params and not params["extra_body"]:
            params.pop("extra_body", None)

        output = ""
        tool_results: Optional[List[Dict[str, str]]] = None

        try:
            try:
                completion = self.client.chat.completions.create(**params)

            except Exception as ex:
                trace_exception(ex)

                if self._is_auto_tool_choice_unsupported(ex):
                    # vLLM without --enable-auto-tool-choice: native tool
                    # calling is unavailable. Retry the SAME payload without
                    # tools/tool_choice so the model answers normally.
                    self._warn_auto_tool_choice_once()
                    params.pop("tools", None)
                    params.pop("tool_choice", None)
                    sanitized_tools = None
                    completion = self.client.chat.completions.create(**params)
                elif (
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

                    if not self.check_is_vllm() and (effort is None or is_thinking_deactivated):
                        params["temperature"] = 1

                    completion = self.client.chat.completions.create(**params)

            if stream:
                handler = _StreamThinkingHandler(streaming_callback, suppress_thinking=is_thinking_deactivated)
                accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

                for chunk in completion:
                    if self.is_cancelled():
                        break

                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = extract_reasoning(delta)
                    content = getattr(delta, "content", "") or ""

                    # Accumulate tool-call deltas BEFORE any `continue` path.
                    # A provider may interleave reasoning/content and tool_call
                    # fragments inside the same chunk; skipping accumulation
                    # when reasoning is present would silently drop the tool
                    # call and break the managed loop.
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        accumulate_streamed_tool_deltas(accumulated_tool_calls, delta_tool_calls)

                    if reasoning:
                        if not handler.process_reasoning(reasoning):
                            break
                        continue

                    if content:
                        if not handler.process_content(content):
                            break

                # ── Streaming+tools empty-stream rescue ──────────────────
                # Some OpenAI-compatible proxies return an empty stream when
                # `stream=true` is combined with a `tools` array. Retry the
                # identical payload in NON-streaming mode; the model either
                # emits native tool_calls (the intended path) or plain
                # content (valid under tool_choice="auto").
                if sanitized_tools and not accumulated_tool_calls and not handler.output.strip():
                    try:
                        retry_fallback = self._non_stream_fallback_for_streamed_tools(
                            params, is_thinking_deactivated
                        )
                    except Exception as fb_err:
                        ASCIIColors.warning(
                            f"[OpenAIBinding._generate_one_round] Streamed-tool "
                            f"non-stream retry failed: {fb_err}"
                        )
                        retry_fallback = None

                    if retry_fallback is not None:
                        (retry_content, retry_native_calls,
                         retry_reasoning) = retry_fallback

                        if retry_native_calls:
                            output = retry_content or ""
                            self._last_assistant_tool_message = {
                                "role": "assistant",
                                "content": retry_content or None,
                                "tool_calls": _wire_shape_tool_calls(retry_native_calls),
                            }
                            native_payloads: List[Tuple[str, str, str]] = []
                            for idx, tc in enumerate(retry_native_calls):
                                if isinstance(tc, dict):
                                    fn = tc.get("function") or {}
                                    name = fn.get("name", "") or ""
                                    raw_args = fn.get("arguments", "") or ""
                                    tc_id = tc.get("id") or f"call_{idx}"
                                else:
                                    name = getattr(getattr(tc, "function", None), "name", "") or ""
                                    raw_args = getattr(getattr(tc, "function", None), "arguments", "") or ""
                                    tc_id = getattr(tc, "id", None) or f"call_{idx}"
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
                                "[OpenAIBinding._generate_one_round] Streamed "
                                "tool round was empty; non-streaming retry "
                                "produced native tool_calls."
                            )
                            return output, tool_results

                        if retry_content and retry_content.strip():
                            ASCIIColors.warning(
                                "[OpenAIBinding._generate_one_round] Streamed "
                                "tool round was empty; non-streaming retry "
                                "produced plain content (no tool_calls)."
                            )
                            return retry_content, tool_results

                        if retry_reasoning and retry_reasoning.strip():
                            ASCIIColors.warning(
                                "[OpenAIBinding._generate_one_round] Streamed "
                                "tool round was empty AND non-streaming "
                                "retry produced reasoning only (no tool_calls). "
                                "Bailing out of the tool round."
                            )

                    return "", None

                output = self._finalize_stream_output(handler, streaming_callback)

                if accumulated_tool_calls:
                    self._last_assistant_tool_message = build_assistant_tool_calls_message(
                        output, accumulated_tool_calls
                    )
                    collected: List[Dict[str, str]] = []
                    for tc_id, tc_name, tc_xml in build_tool_call_callback_payloads(accumulated_tool_calls):
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

            else:
                message_obj = completion.choices[0].message
                reasoning = extract_reasoning(message_obj)
                content = message_obj.content or ""

                if is_thinking_deactivated:
                    output = re.sub(r'<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>', '', content, flags=re.IGNORECASE).strip()
                elif reasoning and not content.strip().startswith((" thinking", "<thinking>")):
                    output = f" thinking\n{reasoning}\n</think>\n{content}"
                else:
                    output = content

                native_tool_calls = extract_native_tool_calls(message_obj)
                native_payloads: List[Tuple[str, str, str]] = []
                if native_tool_calls:
                    self._last_assistant_tool_message = {
                        "role": "assistant",
                        "content": message_obj.content or None,
                        "tool_calls": _wire_shape_tool_calls(native_tool_calls),
                    }
                    for idx, tc in enumerate(native_tool_calls):
                        name = getattr(getattr(tc, "function", None), "name", "") or ""
                        raw_args = getattr(getattr(tc, "function", None), "arguments", "") or ""
                        tc_id = getattr(tc, "id", None) or f"call_{idx}"
                        if not name:
                            continue
                        try:
                            params = json.loads(raw_args) if raw_args else {}
                        except (json.JSONDecodeError, TypeError):
                            params = {"raw": raw_args}
                        xml = f"<tool>{json.dumps({'name': name, 'parameters': params})}</tool>"
                        native_payloads.append((tc_id, name, xml))

                if native_payloads:
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

        except Exception as e:
            trace_exception(e)
            err_msg = f"An error occurred with the OpenAI API: {e}"
            if streaming_callback:
                streaming_callback(err_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return {"status": "error", "message": err_msg}, None

        return output, tool_results

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
        Generate text from a full OpenAI-style messages list, using the
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
        2. Each call is forwarded to the callback as
           ``<tool>{"name":..., "parameters":{...}}</tool>`` XML (so
           runtimes that don't want a managed loop can parse it manually).
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
                ...                              (bounded by MAX_TOOL_ROUNDS)
            ]}

        Tool results MUST carry a ``tool_call_id`` matching one of the
        ``tool_calls`` returned by the previous assistant turn. Entries
        without a ``tool_call_id`` are dropped. ``content`` may be any
        serializable value; it is coerced to a JSON string when not already
        a string.

        Backward compatible: callbacks returning ``bool`` keep working.
        """
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
        if self.client is None:
            ASCIIColors.warning(
                "Embeddings are not available through ChatGPT OAuth (plan usage only covers eligible "
                "Responses requests). Set service_key / OPENAI_API_KEY to use an API key for embeddings."
            )
            return []

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
            "auth_method": self.auth_method,
        }

    def _list_models_oauth(self) -> List[Dict]:
        """
        Model catalog for the signed-in ChatGPT account. The plan-usage flow returns a
        `models` array (slug / display_name / visibility), not the classic `data` array.
        """
        models_info: List[Dict] = []
        try:
            token = self._oauth_access_token()
            r = httpx.get(
                f"{self.oauth_api_base}/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
                verify=self._httpx_verify,
            )
            r.raise_for_status()
            body = r.json()
            entries = body.get("models") or body.get("data") or []
            for m in entries:
                if not isinstance(m, dict):
                    continue
                slug = m.get("slug") or m.get("id")
                if not slug:
                    continue
                if m.get("visibility") not in (None, "list"):
                    continue  # hidden catalog entries
                models_info.append(
                    {
                        "model_name": slug,
                        "display_name": m.get("display_name", slug),
                        "owned_by": m.get("owned_by", "openai"),
                        "created": m.get("created", "N/A"),
                        "context_length": m.get("context_window") or m.get("context_length"),
                        "max_generation": None,
                    }
                )
        except Exception as e:
            print(f"Failed to list models (ChatGPT OAuth): {e}")
        return models_info

    def list_models(self) -> List[Dict]:
        if self.auth_method == "oauth":
            return self._list_models_oauth()

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