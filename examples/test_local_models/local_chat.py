from lollms_client import LollmsClient, ELF_GENERATION_FORMAT, TasksLibrary, FunctionCalling_Library
import random

# Initialize the LollmsClient instance
lc = LollmsClient(model_name= r"E:\drumber\LOLLMS_AWARE_LLAMA_mi_lord",default_generation_mode=ELF_GENERATION_FORMAT.TRANSFORMERS)
def cb(text):
    print(text,end='', flush=True)
lc.generate("Once apon a time",streaming_callback=cb)
