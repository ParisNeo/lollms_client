from lollms_client import LollmsClient
#make sure you use your key
lc = LollmsClient("openai","http://localhost:9642/v1/", service_key="lollms_zXQdyvrP_ecMXm3UZ0D004x979aHpyF8iq4ki_b52q0WdFuiEfMo")
models = lc.list_models()
print(f"Found models:\n{models}")

lc.set_model_name("ollama/gemma3:27b")

res = lc.generate_text("Describe this image",images=[
    r"C:\Users\parisneo\Pictures\me.jpg"
])
print(res)
