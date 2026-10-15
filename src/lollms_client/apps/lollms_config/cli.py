import sys
from typing import Optional

from lollms_client.lollms_config_cli_env import run_wizard_and_save


def main() -> int:
    """
    Console entry point for the lollms-client configuration wizard.

    Launches the interactive wizard (bindings + profiles) which saves to
    ~/.lollms_client/config.yaml. Optional: --env-file to target a specific
    config file.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Lollms-Client Configuration Wizard"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Target a specific configuration file (.env/.yaml/.json/.ini) "
             "instead of the default ~/.lollms_client/config.yaml",
    )
    args = parser.parse_args()

    target: Optional[str] = args.env_file

    try:
        run_wizard_and_save(cli_env_path=target)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"lollms-config failed: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()