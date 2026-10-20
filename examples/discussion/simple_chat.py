"""
Simple Chat Example using LollmsDiscussion.
Demonstrates stateful multi-turn conversation with streaming responses.
"""
from typing import Optional, Dict, Any
from lollms_client import LollmsClient
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage


def cb(chunk: str, type: int, metadata: Optional[Dict[str, Any]]) -> bool:
    """Streaming callback: prints chunks in real-time."""
    print(chunk, end="", flush=True)
    return True


def main():
    # Initialize client from environment configuration
    lc: Optional[LollmsClient] = get_client_from_env()
    if lc is None:
        print("Error: Could not create LollmsClient. Check your configuration.")
        return

    # Create in-memory discussion (no database persistence)
    ld = LollmsDiscussion(lc)
    
    print("\n" + "="*50)
    print("Simple Chat — Type '/exit' to quit, Ctrl+C to interrupt")
    print("="*50 + "\n")

    try:
        while True:
            print("\nYou> ", end="", flush=True)
            user_input = input().strip()
            
            if user_input.lower() == "/exit":
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            # Generate response with streaming
            ld.chat(user_input, streaming_callback=cb)
            print()  # Newline after response completes
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")


if __name__ == "__main__":
    main()