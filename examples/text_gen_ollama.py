#lollms client with ollama as server
from lollms_client import LollmsClient, MSG_TYPE

# Initialize the LollmsClient instance
# two possible ways: 
# Secure using my ollama proxy Server Fortress (https://github.com/ParisNeo/ollama_proxy_server)
lc = LollmsClient(llm_binding_name="ollama", llm_binding_config={
        "host_address": "http://localhost:8080",
        "model_name": "gpt-oss:20b",
        "service_key": "op_85f467a44ba90bec_02a106ef01d83e9d14eeb7dfcd32634b1bef9d160e3f47d9",# change this to your actual service key
    })
#Not secure
# lc = LollmsClient(llm_binding_name="ollama", llm_binding_config={
#         "host_address": "http://localhost:11434",
#         "model_name": "gpt-oss:20b",
#     })

def cb(data:str|dict, type:MSG_TYPE)->bool:
    if type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(data,end="",flush=True)
    return True
print("Streaming:")
response = lc.generate_text(prompt="What is the capital of France?", stream=True, temperature=0.5, streaming_callback=cb, split=True)
print()
print("Full answer:")
print(response)
print()

# List Models
response = lc.listModels()
print(response)