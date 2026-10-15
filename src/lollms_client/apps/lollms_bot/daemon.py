import asyncio
from typing import Optional, List

from ascii_colors import ASCIIColors
from lollms_client.apps.lollms_bot.config import BotConfig


def _build_client() -> Optional["LollmsClient"]:
    """
    Builds the shared LollmsClient from the global configuration, reusing the
    exact same resolution path as lollms-code (config.yaml + .env + two-tier
    profiles). Requests both the LLM and the CONNECTION profiles so the client
    populates its connection bindings. Returns None when no LLM is configured.
    """
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        return get_client_from_env(
            create_llm=True,
            create_connection=True,
            run_wizard_if_fail=False,
        )
    except Exception as e:
        ASCIIColors.red(f"Could not build the LollmsClient: {e}")
        ASCIIColors.info("Configure one first: python -m lollms_client.lollms_config_cli_env")
        return None


def _attach_client(bindings: List, client) -> None:
    """Attaches the shared client to each binding so they can generate replies."""
    for binding in bindings:
        try:
            binding.lollms_client = client
        except Exception:
            pass


async def _start_connections(bindings: List) -> None:
    tasks = [b.start() for b in bindings if hasattr(b, "start")]
    if not tasks:
        ASCIIColors.yellow("No connection bindings to start.")
        return
    ASCIIColors.info(f"Starting {len(tasks)} connection binding(s)...")
    await asyncio.gather(*tasks)


class BotDaemon:
    """
    Thin host that starts the configured native connection bindings.
    The bot no longer defines its own connections: it reuses the global
    LollmsClient's connection-binding profiles (the same system as lollms-code).
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.client = _build_client()

    async def start(self):
        ASCIIColors.info("🚀 Starting Lollms-Bot Daemon...")

        if self.client is None:
            ASCIIColors.red("No LLM client available. The bot cannot respond.")
            return

        llm = self.client.llm
        ASCIIColors.info(
            f"Using LLM → binding: {getattr(llm, 'binding_name', None)}, "
            f"model: {getattr(llm, 'model_name', None)}"
        )

        bindings = list(getattr(self.client, "connections", {}).values())
        if not bindings:
            active = getattr(self.client, "connection", None)
            if active is not None:
                bindings = [active]

        if not bindings:
            ASCIIColors.yellow(
                "No connection bindings configured. Add one in the wizard:\n"
                "   python -m lollms_client.lollms_config_cli_env   (CONNECTION section)"
            )
            return

        _attach_client(bindings, self.client)
        await _start_connections(bindings)