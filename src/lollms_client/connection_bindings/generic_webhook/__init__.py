"""
Generic Webhook Connection Binding.

Allows sending messages to any webhook endpoint (Slack Incoming Webhooks,
Discord Webhooks, Telegram Bot API sendMessage endpoint, n8n webhooks, etc.).
The webhook URL is the service_key. POSTs a JSON payload with a "text" field.
"""

import json
import requests
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

from lollms_client.lollms_connection_binding import (
    LollmsConnectionBinding,
    ConnectionSendResult,
    ConnectionReceiveResult,
)

BindingName = "GenericWebhookConnectionBinding"


class GenericWebhookConnectionBinding(LollmsConnectionBinding):
    """
    Sends messages to a generic webhook URL via HTTP POST.
    Payload: {"text": "...", "sender": "...", "timestamp": ...}
    """

    def __init__(self, **kwargs):
        super().__init__(binding_name="generic_webhook", **kwargs)
        self.webhook_url = kwargs.get("service_key", "")
        self.instance_name = kwargs.get("instance_name", "default")
        self.timeout = kwargs.get("timeout", 30)

    def connect(self) -> bool:
        if not self.webhook_url:
            ASCIIColors.warning(
                "[GenericWebhook] No webhook URL provided (service_key kwarg). "
                "Connection is idle."
            )
            self._connected = False
            return False
        self._connected = True
        ASCIIColors.info(
            f"[GenericWebhook] Connected to webhook: {self.webhook_url[:60]}..."
        )
        return True

    def send_message(
        self,
        content: str,
        sender_name: Optional[str] = None,
        **kwargs,
    ) -> ConnectionSendResult:
        if not self._connected or not self.webhook_url:
            return ConnectionSendResult(
                sent=False,
                error="Not connected. Call connect() first with a valid service_key.",
            )

        payload = {
            "text": content,
            "sender": sender_name or "LoLLMS",
            "instance": self.instance_name,
        }

        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code < 400:
                try:
                    raw = resp.json()
                    msg_id = raw.get("id") or raw.get("message_id")
                except Exception:
                    raw = resp.text
                    msg_id = None

                return ConnectionSendResult(
                    sent=True,
                    message_id=msg_id,
                    channel=self.instance_name,
                    raw=raw,
                )
            else:
                return ConnectionSendResult(
                    sent=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    channel=self.instance_name,
                )

        except requests.exceptions.Timeout:
            return ConnectionSendResult(
                sent=False,
                error=f"Request timed out after {self.timeout}s",
                channel=self.instance_name,
            )
        except Exception as e:
            return ConnectionSendResult(
                sent=False,
                error=str(e),
                channel=self.instance_name,
            )

    def receive_message(
        self,
        timeout: int = 0,
        **kwargs,
    ) -> Optional[ConnectionReceiveResult]:
        return None

    def list_channels(self) -> List[Dict[str, Any]]:
        return [{"id": "webhook", "name": self.webhook_url[:60] + "...", "type": "webhook"}]

    def list_models(self) -> List[str]:
        return ["webhook"]