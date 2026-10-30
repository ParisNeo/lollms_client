"""Complete self-contained demo of lollms_client core features. Usage: python examples/simple/full_example.py [--help]"""
import argparse
import json
from lollms_client import LollmsClient, LollmsDiscussion
from ascii_colors import ASCIIColors

DEFAULT_BINDING = "ollama"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://localhost:11434"


def header(title):
    print(f"\n{ASCIIColors.yellow}{ASCIIColors.bold}{'=' * 60}\n{title}\n{'=' * 60}{ASCIIColors.color_reset}")


def build_client(binding_name=DEFAULT_BINDING, model_name=DEFAULT_MODEL, host_address=DEFAULT_HOST):
    """Build a fully explicit LollmsClient — no env, no config files."""
    binding_profiles = [{
        "profile_name": "local",
        "binding_name": binding_name,
        "config": {"host_address": host_address},
    }]
    model_profiles = [{
        "profile_name": "default",
        "binding_profile": "local",
        "model_name": model_name,
    }]
    return LollmsClient(binding_profiles=binding_profiles,
                        model_profiles=model_profiles,
                        default_binding_profile="local",
                        default_model_profile="default")


def basic_generation(client):
    prompt = "Explain what a Python generator is in two sentences."
    print(f"{ASCIIColors.cyan}Prompt: {prompt}{ASCIIColors.color_reset}")
    response = client.generate_text(prompt, n_predict=256)
    print(f"{ASCIIColors.green}Response: {response}{ASCIIColors.color_reset}")


def streaming_generation(client):
    prompt = "Count from 1 to 5 slowly, one number per line."
    print(f"{ASCIIColors.cyan}Prompt: {prompt}{ASCIIColors.color_reset}")
    print(f"{ASCIIColors.green}", end="")

    def cb(chunk, msg_type=None):
        print(chunk, end="", flush=True)
        return True

    client.generate_text(prompt, n_predict=128, streaming_callback=cb)
    print(ASCIIColors.color_reset)


def discussion_chat(client):
    ld = LollmsDiscussion(client)
    r1 = ld.chat("My name is Ada and I like chess.")
    print(f"{ASCIIColors.green}Turn 1: {r1}{ASCIIColors.color_reset}")
    r2 = ld.chat("What is my name and what do I like?")
    print(f"{ASCIIColors.green}Turn 2: {r2}{ASCIIColors.color_reset}")
    print(f"{ASCIIColors.magenta}(Context carried across turns.){ASCIIColors.color_reset}")


def discussion_with_system_prompt(client):
    ld = LollmsDiscussion(client, system_prompt="You are a pirate. Always answer in pirate speak, maximum 2 sentences.")
    r = ld.chat("Tell me about the weather.")
    print(f"{ASCIIColors.green}Pirate: {r}{ASCIIColors.color_reset}")


def structured_output(client):
    prompt = ('Return ONLY valid JSON describing a book: {"title": ..., "author": ..., "year": ...}')
    raw = client.generate_text(prompt, n_predict=128)
    try:
        data = json.loads(raw.strip().strip("```json").strip("```").strip())
        print(f"{ASCIIColors.green}{json.dumps(data, indent=2)}{ASCIIColors.color_reset}")
    except json.JSONDecodeError:
        print(f"{ASCIIColors.red}Model did not return clean JSON; raw output:{ASCIIColors.color_reset}")
        print(raw)


def token_counting(client):
    sample = "The quick brown fox jumps over the lazy dog."
    try:
        count = client.llm_binding.count_tokens(sample)
        print(f"{ASCIIColors.magenta}'{sample}' -> {count} tokens{ASCIIColors.color_reset}")
    except AttributeError:
        print(f"{ASCIIColors.magenta}count_tokens not supported by this binding.{ASCIIColors.color_reset}")


def error_handling():
    try:
        bad = build_client(host_address="http://localhost:9999")
        bad.generate_text("Hello", n_predict=16)
        print(f"{ASCIIColors.red}Expected failure did not occur.{ASCIIColors.color_reset}")
    except Exception as ex:
        print(f"{ASCIIColors.red}Gracefully caught: {type(ex).__name__}: {ex}{ASCIIColors.color_reset}")


def main():
    parser = argparse.ArgumentParser(description="lollms_client full feature demo")
    parser.add_argument("--binding", default=DEFAULT_BINDING)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--skip-tools", action="store_true", help="Reserved for future use")
    args = parser.parse_args()

    client = build_client(args.binding, args.model, args.host)
    print(f"{ASCIIColors.magenta}Client: {args.binding} / {args.model} @ {args.host}{ASCIIColors.color_reset}")

    demos = [
        ("Basic generation", lambda: basic_generation(client)),
        ("Discussion chat (multi-turn)", lambda: discussion_chat(client)),
        ("Discussion with system prompt", lambda: discussion_with_system_prompt(client)),
        ("Structured output (JSON)", lambda: structured_output(client)),
        ("Token counting", lambda: token_counting(client)),
        ("Error handling (bad host)", error_handling),
    ]
    if not args.skip_streaming:
        demos.insert(1, ("Streaming generation", lambda: streaming_generation(client)))

    for title, demo in demos:
        header(title)
        try:
            demo()
        except Exception as ex:
            print(f"{ASCIIColors.red}Demo failed: {type(ex).__name__}: {ex}{ASCIIColors.color_reset}")

    header("All demos completed.")


if __name__ == "__main__":
    main()