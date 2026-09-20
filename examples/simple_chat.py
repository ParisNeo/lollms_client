from lollms_client import LollmsClient
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
lc: LollmsClient = get_client_from_env()

ld = LollmsDiscussion(lc)
out = ld.chat("hi")
ai_message:LollmsMessage= out["ai_message"]

print(ai_message.content)