"""
Connection runtime for lollms-bot.

Fuses the bot with lollms-client's native connection-binding architecture:
the bot builds the shared LollmsClient (LLM + CONNECTION profiles) from the
global config and starts the configured connection bindings from
`client.connections` / `client.connection`.
"""
import asyncio
from typing import List, Optional

from ascii_colors import ASCIIColors
from lollms_client.lollms_config_cli_env import get_client_from_env


def build_bot_client() -> Optional["LollmsClient"]:
    """Builds the shared LollmsClient (LLM + CONNECTION) from global config."""
    try:
        client = get_client_from_env(
            create_llm=True,
            create_connection=True,
            run_wizard_if_fail=False,
        )
    except Exception as e:
        ASCIIColors.red(f"Could not build the bot client from global config: {e}")
        ASCIIColors.info("Configure one first: python -m lollms_client.lollms_config_cli_env")
        return None

    if getattr(client, "llm", None) is None:
        ASCIIColors.red("No LLM client available. The bot cannot respond.")
        return None
    return client


def _active_connection_bindings(client) -> List:
    """Returns the connection bindings to start, attaching the client to each."""
    bindings = []
    registry = getattr(client, "connections", None)
    if isinstance(registry, dict) and registry:
        bindings = list(registry.values())
    elif getattr(client, "connection", None) is not None:
        bindings = [client.connection]
    for b in bindings:
        try:
            b.lollms_client = client
        except Exception:
            pass
    return bindings


async def run_connections(client) -> None:
    """Starts all enabled connection bindings and blocks until interrupted."""
    bindings = _active_connection_bindings(client)
    if not bindings:
        ASCIIColors.yellow(
            "No connection bindings configured. Add one in the wizard:\n"
            "   python -m lollms_client.lollms_config_cli_env   (CONNECTION section)"
        )
        return

    llm = client.llm
    ASCIIColors.info(
        f"Using LLM → binding: {getattr(llm, 'binding_name', None)}, "
        f"model: {getattr(llm, 'model_name', None)}"
    )
    ASCIIColors.info(f"Starting {len(bindings)} connection binding(s)...")

    tasks = [b.start() for b in bindings if hasattr(b, "start")]
    if not tasks:
        ASCIIColors.yellow("None of the configured connection bindings expose a start() method.")
        return
    await asyncio.gather(*tasks)