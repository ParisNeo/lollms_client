from lollms_client import LollmsClient, MSG_TYPE
from base64 import b64encode
from io import BytesIO
from PIL import Image

def img2b64(png_file_path: str) -> str:
    """Read an image from a specified file path and return a data URL in base64."""
    # Open the image file
    img = Image.open(png_file_path)
    
    # Convert the image to bytes
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    
    # Encode the image bytes to base64
    b64 = b64encode(img_bytes).decode("utf-8")
    
    # Return the data URL
    return f"data:image/png;base64,{b64}"

image_path = ""
# Initialize the LollmsClient instance
lc = LollmsClient(llm_binding_name="lollms", llm_binding_config={
        "model_name": "ollama/gemma3:12b",
        "service_key": "lollms_m1fOU6eS_HXTc09wA9CtCl-yyJBGpaqcvPtOvMqANKPZL9_PEn18",# change this to your actual service key
    })

def cb(chunk, type):
    if type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk,end="",flush=True)
    
response = lc.generate_text(prompt="describe the image", images=[img2b64(r"E:\images\parisneo2.png")], stream=False, temperature=0.5, streaming_callback=cb, split=True)
print()
print(response)
print()


# List Mounted Personalities
response = lc.listMountedPersonalities()
print(response)

# List Models
response = lc.listModels()
print(response)
