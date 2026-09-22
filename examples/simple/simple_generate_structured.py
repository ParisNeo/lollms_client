from lollms_client import LollmsClient
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsMessage
lc: LollmsClient = get_client_from_env()
schema={

}

document ="some text about animals"
structured:dict=lc.generate_structured_content(
    system_prompt ="you are a set of triplets to populate a knowledge graph, using the following schema and this document pacage:\n"+document,
    prompt="create a small animals graph as triplets", 
    schema=schema)
print(structured)