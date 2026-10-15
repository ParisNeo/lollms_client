import sys
import argparse
import asyncio
from lollms_client.apps.lollms_bot.config import BotConfig
from ascii_colors import ASCIIColors

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lollms-Bot: Multi-platform AI Bot Daemon",
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
        help="Path to the specific configuration file (default: global)"
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
    
    config = BotConfig(config_path=args.config_path)
    
    if args.config:
        config.run_wizard()
        return 0
    
    # Start Daemon
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

if __name__ == "__main__":
    sys.exit(main())