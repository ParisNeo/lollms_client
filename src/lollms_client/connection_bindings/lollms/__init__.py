"""
LoLLMs CONNECTION binding for lollms_client.

Bridges a local LoLLMs client to the lollms.com web gateway using a bot
token. The binding polls the gateway for pending prompts, generates the
answer with the attached LollmsClient's LLM, and posts the response back.
Credentials come from the binding config (service_key = bot token).
"""
import asyncio
import base64
from typing import Any, Dict, List, Optional, Tuple

import requests
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, ConnectionSendResult

BindingName = "LollmsWebConnectionBinding"

DEFAULT_GATEWAY_URL = "https://lollms.com/wp-json/lollms/v1"
DEFAULT_POLL_INTERVAL_S = 1.5
INVALID_TOKEN_BACKOFF_S = 10.0
CONNECTION_ERROR_BACKOFF_S = 3.0
POLL_TIMEOUT_S = 10
RESPOND_TIMEOUT_S = 15


class LollmsWebConnectionBinding(LollmsConnectionBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="lollms", **kwargs)
        self.config = kwargs
        self.token = kwargs.get("service_key") or ""
        gateway = str(kwargs.get("host_address") or DEFAULT_GATEWAY_URL).rstrip("/")
        self.gateway_url = gateway if gateway else DEFAULT_GATEWAY_URL
        self.poll_interval = float(kwargs.get("poll_interval", DEFAULT_POLL_INTERVAL_S))
        if self.poll_interval <= 0:
            self.poll_interval = DEFAULT_POLL_INTERVAL_S
        self.lollms_client = kwargs.get("lollms_client")
        self._running = False

    def connect(self) -> bool:
        if not self.token:
            ASCIIColors.warning("lollms binding: cannot connect without a bot token.")
            return False
        self._connected = True
        return True

    def list_channels(self) -> List[str]:
        return ["lollms:*"]

    def send_message(self, content: str, channel_alias: Optional[str] = None, artefacts: Optional[List[dict]] = None, **kwargs) -> ConnectionSendResult:
        if not self._running:
            return ConnectionSendResult(success=False, message="LoLLMs web bridge is not running.", channel=channel_alias)
        if artefacts:
            encoded = self._encode_artefacts_for_gateway(artefacts)
            return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="lollms", artefacts=encoded)
        return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="lollms")

    async def start(self) -> None:
        if not self.token:
            ASCIIColors.red("lollms binding: missing 'service_key' (bot token). Skipping.")
            return

        self._running = True
        ASCIIColors.info("🌐 Starting LoLLMs web bridge connection binding...")
        try:
            await self._poll_loop()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.red(f"lollms binding error: {e}")
        finally:
            self._running = False

    async def _poll_loop(self) -> None:
        ASCIIColors.info(f"Gateway: {self.gateway_url}")
        ASCIIColors.info("🟢 Bot online! Listening for web prompts... (Ctrl+C to stop)")
        while self._running:
            backoff = await self._poll_once()
            await asyncio.sleep(backoff)

    async def _poll_once(self) -> float:
        try:
            response = requests.get(
                f"{self.gateway_url}/bot/poll",
                params={"token": self.token},
                timeout=POLL_TIMEOUT_S,
            )
            if response.status_code == 401:
                ASCIIColors.red("❌ Invalid bot token! Please verify your token in the web UI.")
                return INVALID_TOKEN_BACKOFF_S
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            ASCIIColors.warning(f"lollms binding: gateway unreachable ({e}). Retrying...")
            return CONNECTION_ERROR_BACKOFF_S

        if data.get("status") != "pending_task":
            return self.poll_interval

        msg_id = data.get("msg_id")
        prompt = data.get("prompt", "")
        history = data.get("history") or []

        if msg_id is None:
            ASCIIColors.warning("lollms binding: pending task without msg_id. Skipping.")
            return self.poll_interval

        ASCIIColors.info(f"📥 [Prompt #{msg_id}] {prompt}")
        reply, artefacts = await self._reply_async(prompt, history)
        summary = (reply or "")[:60]
        ASCIIColors.info(f"📤 [Response] {summary}...")
        if artefacts:
            ASCIIColors.info(f"📎 Attaching {len(artefacts)} artefact(s)")

        if self._respond(msg_id, reply, artefacts):
            return self.poll_interval
        return CONNECTION_ERROR_BACKOFF_S

    def _respond(self, msg_id: Any, reply: str, artefacts: Optional[List[dict]] = None) -> bool:
        payload: Dict[str, Any] = {
            "token": self.token,
            "msg_id": msg_id,
            "response": reply,
        }
        encoded = self._encode_artefacts_for_gateway(artefacts)
        if encoded:
            payload["artefacts"] = encoded
        try:
            response = requests.post(
                f"{self.gateway_url}/bot/respond",
                json=payload,
                timeout=RESPOND_TIMEOUT_S,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            ASCIIColors.warning(f"lollms binding: failed to post response for #{msg_id}: {e}")
            return False

    def _encode_artefacts_for_gateway(self, artefacts: Optional[List[dict]]) -> List[dict]:
        if not artefacts:
            return []
        encoded: List[dict] = []
        for artefact in artefacts:
            data, mime, title = self._load_artefact_payload(artefact)
            if data is None:
                continue
            encoded.append(
                {
                    "title": title,
                    "type": artefact.get("type", "document"),
                    "mime": mime,
                    "content": base64.b64encode(data).decode("utf-8"),
                }
            )
        return encoded

    async def _reply_async(self, prompt: str, history: List[Dict[str, Any]]) -> Tuple[str, List[dict]]:
        return await asyncio.to_thread(self._reply, prompt, history)

    def _reply(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[dict]]:
        client = self.lollms_client
        if client is not None and getattr(client, "llm", None) is not None:
            try:
                messages = [self._normalize_message(m) for m in (history or [])]
                messages.append({"role": "user", "content": prompt})
                out = client.llm.generate_from_messages(messages)
                if isinstance(out, dict):
                    text = out.get("message") or out.get("error") or prompt
                else:
                    text = out or prompt
                return self._extract_artefacts_from_text(str(text))
            except Exception:
                return prompt, []
        return prompt, []

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            role = str(message.get("role", "user")).lower()
            if role not in ("system", "user", "assistant"):
                role = "user"
            return {"role": role, "content": str(message.get("content", ""))}
        return {"role": "user", "content": str(message)}