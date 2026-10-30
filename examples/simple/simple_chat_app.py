"""Interactive chat REPL with streaming. Usage: python examples/simple/simple_chat_app.py [--help]"""
import argparse
from lollms_client import LollmsClient, LollmsDiscussion
from ascii_colors import ASCIIColors

def build_client(model, host):
    return LollmsClient(
        binding_profiles=[{"profile_name": "local", "binding_name": "ollama",
                           "config": {"host_address": host}}],
        model_profiles=[{"profile_name": "default", "binding_profile": "local",
                         "model_name": model}],
        default_binding_profile="local", default_model_profile="default")

def cb(chunk, msg_type=None):
    print(chunk, end="", flush=True)
    return True

def main():
    parser = argparse.ArgumentParser(description="lollms_client interactive chat")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    client = build_client(args.model, args.host)
    ld = LollmsDiscussion(client)

    print(f"{ASCIIColors.yellow}{ASCIIColors.bold}lollms chat — {args.model}")
    print("Commands: /exit  /clear  /system <prompt>")
    print(f"{'=' * 50}{ASCIIColors.color_reset}")

    while True:
        try:
            user_input = input(f"{ASCIIColors.cyan}You > {ASCIIColors.color_reset}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/clear":
            ld = LollmsDiscussion(client)
            print(f"{ASCIIColors.magenta}Context cleared.{ASCIIColors.color_reset}")
            continue
        if user_input.startswith("/system "):
            ld = LollmsDiscussion(client, system_prompt=user_input[len("/system "):])
            print(f"{ASCIIColors.magenta}System prompt set (context reset).{ASCIIColors.color_reset}")
            continue
        try:
            print(f"{ASCIIColors.green}AI  > ", end="")
            ld.chat(user_input, streaming_callback=cb)
            print(f"{ASCIIColors.color_reset}\n---")
        except Exception as ex:
            print(f"\n{ASCIIColors.red}Error: {type(ex).__name__}: {ex}{ASCIIColors.color_reset}\n---")

if __name__ == "__main__":
    main()