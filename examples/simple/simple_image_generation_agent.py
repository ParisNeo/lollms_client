import os
from lollms_client import LollmsClient, MSG_TYPE
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion
from ascii_colors import ASCIIColors
from pathlib import Path

# Initialize client from environment/config
lc: LollmsClient = get_client_from_env()
# Create discussion
images_path = Path(__file__).parent/"images"
images_path.mkdir(parents=True, exist_ok=True)
ld = LollmsDiscussion(lc, workspace_path=images_path)

def cb(chunk, msg_type, metadata):
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True

print("\n--- Chat Session Started ---")
while True:
    user_input = input("you> ")
    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        out = ld.chat(
            user_input,
            streaming_callback=cb,
            enable_image_generation=True,
            enforce_end_tag=True,
        )
    except Exception as e:
        ASCIIColors.error(f"Error during chat: {e}")