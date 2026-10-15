"""
Discord CONNECTION binding for lollms_client.
"""
from typing import List, Optional

from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, ConnectionSendResult

BindingName = "DiscordConnectionBinding"


class DiscordConnectionBinding(LollmsConnectionBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="discord", **kwargs)
        self.config = kwargs
        self.token = kwargs.get("service_key") or ""
        self.allowed_guild_ids = self._parse_ids(kwargs.get("allowed_guild_ids"))
        self.lollms_client = kwargs.get("lollms_client")
        self._running = False

    @staticmethod
    def _parse_ids(raw) -> List[int]:
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            return [int(x) for x in raw if str(x).strip()]
        return [int(x) for x in str(raw).split(",") if x.strip()]

    def list_channels(self) -> List[str]:
        return [f"discord:{gid}" for gid in self.allowed_guild_ids] or ["discord:*"]

    def send_message(self, content: str, channel_alias: Optional[str] = None, **kwargs) -> ConnectionSendResult:
        if not self._running:
            return ConnectionSendResult(success=False, message="Discord bot is not running.", channel=channel_alias)
        return ConnectionSendResult(success=True, message="queued", channel=channel_alias, platform="discord")

    async def start(self) -> None:
        if not self.token:
            ASCIIColors.red("Discord binding: missing 'service_key' (bot token). Skipping.")
            return
        try:
            import discord
        except ImportError:
            ASCIIColors.red("discord.py not installed. Run: pip install discord.py")
            return

        self._running = True
        ASCIIColors.info("🎮 Starting Discord connection binding...")
        try:
            client = discord.Client(intents=discord.Intents.default())

            @client.event
            async def on_ready():
                ASCIIColors.info(f"Discord bot logged in as {client.user}")

            @client.event
            async def on_message(message):
                if message.author == client.user:
                    return
                if self.allowed_guild_ids and (message.guild is None or message.guild.id not in self.allowed_guild_ids):
                    return
                await message.reply(self._reply(message.content))

            client.run(self.token)
        except Exception as e:
            trace_exception(e)
            ASCIIColors.red(f"Discord binding error: {e}")
        finally:
            self._running = False

    def _reply(self, text: str) -> str:
        client = self.lollms_client
        if client is not None and getattr(client, "llm", None) is not None:
            try:
                out = client.llm.generate_text(prompt=text)
                if isinstance(out, dict):
                    return out.get("message", text)
                return out or text
            except Exception:
                return text
        return text