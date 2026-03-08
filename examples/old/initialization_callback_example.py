import sys
from pathlib import Path

# Ensure we can import from src if running directly from the repository
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors

def progress_callback(content: str, msg_type: MSG_TYPE, metadata: dict = None) -> bool:
    """
    Callback function to handle messages from LollmsClient.
    
    Args:
        content (str): The message content (often markdown).
        msg_type (MSG_TYPE): The type of message (INIT_PROGRESS, INFO, ERROR, etc.).
        metadata (dict): Additional metadata associated with the message.
    
    Returns:
        bool: True to continue, False to stop (if supported by the caller).
    """
    
    if msg_type == MSG_TYPE.MSG_TYPE_INIT_PROGRESS:
        # Use success color (usually green) for progress updates
        ASCIIColors.success(f"[INIT] {content}")
    elif msg_type == MSG_TYPE.MSG_TYPE_ERROR:
        ASCIIColors.error(f"[ERROR] {content}")
    elif msg_type == MSG_TYPE.MSG_TYPE_WARNING:
        ASCIIColors.warning(f"[WARN] {content}")
    elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
        ASCIIColors.info(f"[INFO] {content}")
    else:
        # Catch-all for other types
        print(f"[{msg_type.name}] {content}")
    
    return True

def main():
    ASCIIColors.cyan("--- Starting Lollms Client Initialization Example ---")
    
    # Initialize the client with the callback
    # We specify a few bindings to demonstrate the progress reporting
    # Note: If you don't have these specific backends installed/configured, 
    # you might see error messages in the callback, which is also part of the demonstration.
    client = LollmsClient(
        # Attempt to initialize an LLM binding (e.g., 'ollama' or 'litellm')
        llm_binding_name="ollama", 
        llm_binding_config={"model_name": "llama3"}, # Example config
        
        # Attempt to initialize a TTI binding
        tti_binding_name="lollms", 
        
        # Pass our custom callback
        callback=progress_callback
    )

    ASCIIColors.cyan("\n--- Initialization Sequence Complete ---")
    
    if client.llm:
        ASCIIColors.green("LLM is ready.")
    else:
        ASCIIColors.red("LLM binding could not be loaded (check configuration).")

if __name__ == "__main__":
    main()
