from lollms_client import LollmsClient
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
lc: LollmsClient = get_client_from_env()

print(lc.generate_text("hello"))