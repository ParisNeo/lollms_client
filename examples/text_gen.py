from lollms_client import LollmsClient

# Initialize the LollmsClient instance
lc = LollmsClient(llm_binding_name="lollms", llm_binding_config={
        "model_name": "ollama/gemma3:12b",
        "service_key": "lollms_m1fOU6eS_HXTc09wA9CtCl-yyJBGpaqcvPtOvMqANKPZL9_PEn18",
    })
#lc = LollmsClient("ollama", model_name="mistral-nemo:latest")
#lc = LollmsClient("llamacpp", models_path=r"E:\drumber", model_name="llava-v1.6-mistral-7b.Q3_K_XS.gguf")
# Generate Text
# response = lc.generate_text(prompt="Once upon a time", stream=False, temperature=0.5)
# print(response)

# # Generate Completion
# response = lc.generate_completion(prompt="What is the capital of France", stream=False, temperature=0.5)
# print(response)

def cb(chunk, type):
    print(chunk,end="",flush=True)
    
response = lc.generate_text(prompt="!@>user: Hi there\n!@>assistant: Hi there, how can I help you?!@>user: what is 1+1?\n!@>assistant: ", stream=False, temperature=0.5, streaming_callback=cb, split=True)
print()
print(response)
print()


# List Mounted Personalities
response = lc.listMountedPersonalities()
print(response)

# List Models
response = lc.listModels()
print(response)