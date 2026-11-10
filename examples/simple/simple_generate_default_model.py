"""Minimal one-shot text generation. Usage: python examples/simple/simple_generate.py [--model MODEL] [--prompt TEXT]"""
import argparse
from lollms_client import LollmsClient
from ascii_colors import rich_print
from lollms_client.lollms_config_cli_env import get_client_from_env

def main():
    parser = argparse.ArgumentParser(description="lollms_client hello world")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    args = parser.parse_args()

    lc: LollmsClient = get_client_from_env()

    rich_print(f"[cyan]Prompt: {args.prompt}[/cyan]")
    response = lc.generate_text(args.prompt, n_predict=128)
    rich_print(f"[green]Response: {response}[/green]")

if __name__ == "__main__":
    main()