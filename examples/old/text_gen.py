from lollms_client import LollmsClient, MSG_TYPE

# Initialize the LollmsClient instance
lc = LollmsClient(llm_binding_name="lollms", llm_binding_config={
        "model_name": "ollama/gemma3:12b",
        "service_key": "lollms_m1fOU6eS_HXTc09wA9CtCl-yyJBGpaqcvPtOvMqANKPZL9_PEn18",# change this to your actual service key
    })

def cb(chunk, type):
    if type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk,end="",flush=True)
    return True
response = lc.generate_text(prompt="!@>user: Hi there\n!@>assistant: Hi there, how can I help you?!@>user: what is 1+1?\n!@>assistant: ", stream=True, temperature=0.5, streaming_callback=cb, split=True)
print()
print(response)
print()


# List Mounted Personalities
response = lc.listMountedPersonalities()
print(response)

# List Models
response = lc.list_models()
print(response)