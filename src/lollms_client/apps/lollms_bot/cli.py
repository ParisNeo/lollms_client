import argparse
import asyncio
import os
import sys
from pathlib import Path

from ascii_colors import ASCIIColors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lollms-Bot: Handbag-based Multi-platform AI Bot Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Run the interactive configuration wizard"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to a specific configuration file (default: ~/.lollms-bot/config.yaml)"
    )
    parser.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override the lollms-bot home directory (default: ~/.lollms-bot)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    from lollms_client.apps.lollms_bot.config import BotConfig, ensure_bot_home

    if args.home:
        os.environ["LOLLMS_BOT_HOME"] = str(Path(args.home).expanduser())

    ensure_bot_home()

    config = BotConfig(config_path=args.config_path)

    if args.config:
        config.run_wizard()
        return 0

    from lollms_client.apps.lollms_bot.daemon import BotDaemon
    daemon = BotDaemon(config)

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        ASCIIColors.yellow("\n🛑 Bot stopped by user.")
        return 0
    except Exception as e:
        ASCIIColors.red(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())