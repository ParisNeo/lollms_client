from lollms_client import LollmsClient, ELF_GENERATION_FORMAT

# Initialize the LollmsClient instance please select a different model to test with
lc = LollmsClient(model_name= r"E:\drumber\LOLLMS_AWARE_LLAMA_mi_lord",default_generation_mode=ELF_GENERATION_FORMAT.TRANSFORMERS)
def cb(text, msg_type=0):
    print(text,end='', flush=True)
    return True
out = lc.generate("Once apon a time",streaming_callback=cb)
print(out)
