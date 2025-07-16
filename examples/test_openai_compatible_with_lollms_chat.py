from lollms_client import LollmsClient
#make sure you use your key
lc = LollmsClient("openai","http://localhost:9642/v1/", service_key="lollms_zXQdyvrP_ecMXm3UZ0D004x979aHpyF8iq4ki_b52q0WdFuiEfMo")
models = lc.listModels()
print(f"Found models:\n{models}")

# format is: binding name/model name.
lc.set_model_name("litellm/mistralsmall-22b")

res = lc.generate_text("Describe this image",images=[
    r"C:\Users\Pictures\Charles_de_Gaulle-1963.jpg"
])
print(res)
