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

    def process_reasoning(self, reasoning: str) -> bool:
        if not reasoning:
            return True

        if self.suppress_thinking:
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
                reasoning = {
                    "effort": effort if effort != "max" else "high",
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

    def _apply_thinking_params(
        self,
        params: dict,
        effort: Optional[str],
        is_deactivated: bool
    ) -> dict:
        """
        Applies thinking / reasoning controls across OpenAI, vLLM, Z.AI/GLM, and OpenAI-compatible engines.
        """
        model_name_lower = (self.model_name or "").lower()
        is_glm = "glm" in model_name_lower
        is_vllm = self.check_is_vllm()
        extra_body = params.setdefault("extra_body", {})

        if is_deactivated:
            params["reasoning_effort"] = "none"
            extra_body["reasoning_effort"] = "none"

            if is_vllm:
                if self.send_thinking_parameter and not getattr(self, "glm_image_embedding", False):
                    ctk = extra_body.setdefault("chat_template_kwargs", {})
                    if self.thinking_effort_keyword == "reasoning_effort":
                        ctk["reasoning_effort"] = "none"
                        ctk["enable_thinking"] = False
                    else:
                        ctk[self.thinking_effort_keyword] = False
                        ctk["enable_thinking"] = False
                        ctk["thinking"] = False

                if is_glm:
                    extra_body["thinking"] = {"type": "disabled"}
                else:
                    extra_body["thinking"] = False
            else:
                extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                extra_body.setdefault("chat_template_kwargs", {})["thinking"] = False
                if is_glm:
                    extra_body["thinking"] = {"type": "disabled"}
                else:
                    extra_body["thinking"] = False
        else:
            if effort is not None and str(effort).strip().lower() not in ("none", "off", "disabled", "false", "0"):
                norm_effort = effort if effort != "max" else "high"
                params["reasoning_effort"] = norm_effort
                extra_body["reasoning_effort"] = norm_effort

                if is_vllm:
                    if self.send_thinking_parameter and not getattr(self, "glm_image_embedding", False):
                        ctk = extra_body.setdefault("chat_template_kwargs", {})
                        if self.thinking_effort_keyword == "reasoning_effort":
                            ctk["reasoning_effort"] = norm_effort
                            ctk["enable_thinking"] = True
                        else:
                            ctk[self.thinking_effort_keyword] = True
                            ctk["enable_thinking"] = True
                            ctk["thinking"] = True

                    if is_glm:
                        extra_body["thinking"] = {"type": "enabled"}
                    else:
                        extra_body["thinking"] = True
                else:
                    extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
                    extra_body.setdefault("chat_template_kwargs", {})["thinking"] = True
                    if is_glm:
                        extra_body["thinking"] = {"type": "enabled"}
                    else:
                        extra_body["thinking"] = True

                    restricted = ["o1", "o3", "o4", "gpt-5"]
                    if any(r in model_name_lower for r in restricted):
                        params.pop("temperature", None)
                        params.pop("top_p", None)
            else:
                params["reasoning_effort"] = "none"
                extra_body["reasoning_effort"] = "none"

                if is_vllm:
                    if self.send_thinking_parameter and not getattr(self, "glm_image_embedding", False):
                        ctk = extra_body.setdefault("chat_template_kwargs", {})
                        if self.thinking_effort_keyword == "reasoning_effort":
                            ctk["reasoning_effort"] = "none"
                            ctk["enable_thinking"] = False
                        else:
                            ctk[self.thinking_effort_keyword] = False
                            ctk["enable_thinking"] = False
                            ctk["thinking"] = False

                    if is_glm:
                        extra_body["thinking"] = {"type": "disabled"}
                    else:
                        extra_body["thinking"] = False
                else:
                    extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                    extra_body.setdefault("chat_template_kwargs", {})["thinking"] = False
                    if is_glm:
                        extra_body["thinking"] = {"type": "disabled"}
                    else:
                        extra_body["thinking"] = False

        return params

    def _apply_vllm_thinking_kwargs(self, params: dict, effort: Optional[str]) -> dict:
        is_deact = effort is None or str(effort).strip().lower() in ("none", "off", "disabled", "false", "0")
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
        **kwargs,
    ) -> Union[str, dict]:
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
            f"[OpenAIBinding.generate_text] think={think}, reasoning_effort={reasoning_effort} "
            f"-> effective_effort={effort}, deactivated={is_thinking_deactivated}"
        )

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
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
        think: Optional[bool] = False,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
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
            )

        self._apply_thinking_params(params, effort, is_thinking_deactivated)
        if reasoning_summary and reasoning_summary != "auto" and not is_thinking_deactivated:
            params.setdefault("extra_body", {})["reasoning_summary"] = reasoning_summary

        if "extra_body" in params and not params["extra_body"]:
            params.pop("extra_body", None)

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

                    if not self.check_is_vllm() and (effort is None or is_thinking_deactivated):
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

                if is_thinking_deactivated:
                    output = re.sub(r'<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>', '', content, flags=re.IGNORECASE).strip()
                elif reasoning and not content.strip().startswith(("<think>", "<thinking>")):
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