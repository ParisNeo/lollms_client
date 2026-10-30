"""Minimal one-shot text generation. Usage: python examples/simple/simple_generate.py [--model MODEL] [--prompt TEXT]"""
import argparse
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors

def main():
    parser = argparse.ArgumentParser(description="lollms_client hello world")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    args = parser.parse_args()

    client = LollmsClient(
        binding_profiles=[{"profile_name": "local", "binding_name": "ollama",
                           "config": {"host_address": args.host}}],
        model_profiles=[{"profile_name": "default", "binding_profile": "local",
                         "model_name": args.model}],
        default_binding_profile="local", default_model_profile="default")

    print(f"{ASCIIColors.cyan}Prompt: {args.prompt}{ASCIIColors.color_reset}")
    response = client.generate_text(args.prompt, n_predict=128)
    print(f"{ASCIIColors.green}Response: {response}{ASCIIColors.color_reset}")

if __name__ == "__main__":
    main()