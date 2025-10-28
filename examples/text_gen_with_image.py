from lollms_client import LollmsClient, MSG_TYPE
from PIL import Image

image_path = ""
# Initialize the LollmsClient instance
lc = LollmsClient(llm_binding_name="lollms", llm_binding_config={
        "model_name": "ollama/gemma3:12b",
        "service_key": "lollms_0vWhQYNV__q6GMMfZ4_Ib2Crw8wnPTAXDu2KWrsKvsulpptxzfiY",# change this to your actual service key
    })

def cb(chunk, type):
    if type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk,end="",flush=True)
    
response = lc.generate_text(prompt="describe the image", images=[r"E:\images\parisneo2.png"], stream=False, temperature=0.5, streaming_callback=cb, split=True)
print()
print(response)
print()


# List Mounted Personalities
response = lc.listMountedPersonalities()
print(response)

# List Models
response = lc.list_models()
print(response)