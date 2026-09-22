from lollms_client import LollmsClient
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
lc: LollmsClient = get_client_from_env()

ld = LollmsDiscussion(lc)
def cb(chunk, type, metadata):
    print(chunk,end="")
    return True

while True:
    print("you>", end="")
    user_input = input()
    if user_input.strip().lower() == "/exit":
        print("Goodbye!")
        break
    out = ld.chat(user_input, streaming_callback=cb)
