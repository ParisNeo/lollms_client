"""
LoLLMs CONNECTION binding for lollms_client.

Bridges a local LoLLMs client to the lollms.com web gateway using a bot
token. The binding polls the gateway for pending prompts, generates the
answer with the attached LollmsClient's LLM, and posts the response back.
Credentials come from the binding config (service_key = bot token).
"""
import asyncio
import base64
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        self._chat_lock = threading.Lock()

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
        self._prewarm_personality()
        try:
            await self._poll_loop()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.red(f"lollms binding error: {e}")
        finally:
            self._cancel_personality()
            self._running = False

    def _cancel_personality(self) -> None:
        personality = getattr(self, "lollms_personality", None)
        if personality is not None and bool(personality):
            try:
                personality.cancel_generation()
            except Exception:
                pass

    def _handle_command(self, prompt: str) -> Optional[str]:
        """
        Dispatches slash commands (/personas, /persona, /skills, /skill).
        Returns a reply string for commands, or None to fall through to the
        normal personality chat path. Never raises.
        """
        text = (prompt or "").strip()
        if not text.startswith("/"):
            return None
        try:
            command, _, argument = text.partition(" ")
            command = command.lower()
            argument = argument.strip()

            if command == "/personas":
                return self._cmd_personas()
            if command == "/persona":
                return self._cmd_persona_switch(argument)
            if command == "/skills":
                return self._cmd_skills(argument)
            if command == "/skill":
                return self._cmd_skill_load(argument)

            return (
                "Unknown command. Available commands:\n"
                "/personas - list available personas\n"
                "/persona <name> - switch the active persona\n"
                "/skills - list skills of the active persona\n"
                "/skill <name> - load a skill into the conversation"
            )
        except Exception as e:
            trace_exception(e)
            return f"[LollmsBot] Command failed: {e}"

    def _cmd_personas(self) -> str:
        personality = getattr(self, "lollms_personality", None)
        active_name = ""
        if personality is not None and bool(personality):
            try:
                active_name = str(personality.display_name or "").lower()
            except Exception:
                active_name = ""

        registry = getattr(self, "persona_registry", None) or {}
        if not registry:
            return "No personas registered. Add handbag-shaped folders under ~/.lollms-bot/personas/ to create one."

        lines = ["Available personas:"]
        for index, (name, info) in enumerate(sorted(registry.items()), start=1):
            display_name = str(info.get("display_name") or name)
            marker = " ✅ (active)" if name.lower() == active_name else ""
            lines.append(f"{index}. {display_name} — switch with: /persona {name}{marker}")
        return "\n".join(lines)

    def _cmd_persona_switch(self, name: str) -> str:
        if not name:
            return "Usage: /persona <name> — use /personas to list available personas."

        registry = getattr(self, "persona_registry", None) or {}
        match = next(
            (key for key in registry if key.lower() == name.lower()),
            None,
        )
        if match is None:
            return f"Persona '{name}' not found. Use /personas to list available personas."

        factory = getattr(self, "personality_factory", None)
        if factory is None:
            return "Persona switching is unavailable (no personality factory attached)."

        try:
            with self._chat_lock:
                new_personality = factory(match)
                self.lollms_personality = new_personality
        except Exception as e:
            trace_exception(e)
            return f"[LollmsBot] Failed to switch to persona '{match}': {e}"

        display_name = getattr(new_personality, "display_name", match)
        persist = getattr(self, "on_persona_selected", None)
        if callable(persist):
            try:
                persist(match)
            except Exception:
                pass
        ASCIIColors.success(f"lollms binding: persona switched to '{match}'.")
        return f"Persona switched to '{display_name}'."

    def _cmd_skills(self, argument: str) -> str:
        personality = getattr(self, "lollms_personality", None)
        if personality is None or not bool(personality):
            return "No personality attached. Cannot list skills."

        skills_manager = getattr(personality, "skills_manager", None)
        if skills_manager is None:
            return "Skills are not enabled for the active persona."

        query = argument.partition("search ")[2].strip() if argument.lower().startswith("search ") else ""
        if query:
            matches = skills_manager.search_skills(query)
            if not matches:
                return f"No skills matching '{query}'."
            lines = [f"Skills matching '{query}':"]
            for skill in matches:
                lines.append(f"• {skill.get('title')} — {skill.get('description', '')}")
            return "\n".join(lines)

        skills = skills_manager.list_skills()
        if not skills:
            return "The active persona has no skills. Add SKILL.md files to the handbag's skills/ folder."

        lines = ["Skills of the active persona:"]
        for skill in skills:
            visibility = skill.get("visibility", "")
            lines.append(f"• {skill.get('title')} ({visibility}) — {skill.get('description', '')}")
        lines.append("Load one with: /skill <name>")
        return "\n".join(lines)

    def _cmd_skill_load(self, name: str) -> str:
        if not name:
            return "Usage: /skill <name> — use /skills to list available skills."

        personality = getattr(self, "lollms_personality", None)
        if personality is None or not bool(personality):
            return "No personality attached. Cannot load skills."

        skills_manager = getattr(personality, "skills_manager", None)
        if skills_manager is None:
            return "Skills are not enabled for the active persona."

        skill = skills_manager.get_skill(name)
        if skill is None:
            matches = skills_manager.search_skills(name)
            suggestion = ""
            if matches:
                titles = ", ".join(str(m.get("title")) for m in matches[:5])
                suggestion = f" Did you mean: {titles}?"
            return f"Skill '{name}' not found.{suggestion} Use /skills to list them."

        try:
            loaded = skills_manager.load_skill(skill.get("key"))
        except Exception as e:
            trace_exception(e)
            return f"[LollmsBot] Failed to load skill '{name}': {e}"

        title = (loaded or {}).get("title") or skill.get("title") or name
        return f"Skill '{title}' loaded. It will now be used in this conversation."

    def _prewarm_personality(self) -> None:
        personality = getattr(self, "lollms_personality", None)
        if personality is None or not bool(personality):
            return
        try:
            _ = personality.tools
            ASCIIColors.info("lollms binding: personality tool surface pre-warmed.")
        except Exception as e:
            ASCIIColors.warning(f"lollms binding: personality tool pre-warm failed (non-fatal): {e}")

    def _chat_locked(self, prompt: str) -> Tuple[str, Optional[str]]:
        personality = getattr(self, "lollms_personality", None)
        if personality is None or not bool(personality):
            return "", "no_personality"
        if not self._chat_lock.acquire(blocking=False):
            ASCIIColors.warning("lollms binding: personality.chat re-entry blocked (turn already in flight).")
            return "", "busy"
        try:
            result = personality.chat(prompt=prompt, enforce_end_tag=False)
        except Exception as e:
            trace_exception(e)
            return "", f"chat_error: {e}"
        finally:
            self._chat_lock.release()
        if isinstance(result, dict):
            return str(result.get("response", "") or ""), None
        return str(result or ""), None

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
        command_reply = self._handle_command(prompt)
        if command_reply is not None:
            return command_reply, []
        reply, failure = await asyncio.to_thread(self._chat_locked, prompt)
        if failure == "no_personality":
            return self._reply_raw(prompt, history)
        if failure:
            ASCIIColors.warning(f"lollms binding: personality chat failed ({failure}). No raw fallback.")
            return f"[LollmsBot] Personality chat failed: {failure}", []
        if not reply.strip():
            ASCIIColors.warning("lollms binding: personality chat returned an empty response. No raw fallback.")
            return "[LollmsBot] The personality produced an empty response for this prompt.", []
        return self._extract_artefacts_from_text(reply)

    def _reply(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[dict]]:
        command_reply = self._handle_command(prompt)
        if command_reply is not None:
            return command_reply, []
        reply, failure = self._chat_locked(prompt)
        if failure == "no_personality":
            return self._reply_raw(prompt, history)
        if failure:
            ASCIIColors.warning(f"lollms binding: personality chat failed ({failure}). No raw fallback.")
            return f"[LollmsBot] Personality chat failed: {failure}", []
        if not reply.strip():
            ASCIIColors.warning("lollms binding: personality chat returned an empty response. No raw fallback.")
            return "[LollmsBot] The personality produced an empty response for this prompt.", []
        return self._extract_artefacts_from_text(reply)

    def _reply_raw(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[dict]]:
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