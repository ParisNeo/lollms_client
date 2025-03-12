from lollms_client import LollmsClient

# Initialize the LollmsClient instance please select a different model to test with
lc = LollmsClient("transformers", model_name= r"E:\drumber\Llama-3.2-1B-Instruct-lollms-smart-router")
def cb(text, msg_type=0):
    print(text,end='', flush=True)
    return True
out = lc.generate_text(f"{lc.system_full_header} Act as lollms, a helpful assistant.\n!@>user:Write a poem about love.\n!@>lollms:",streaming_callback=cb)
print(out)
