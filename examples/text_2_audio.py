from lollms_client import LollmsClient, ELF_GENERATION_FORMAT, LollmsTTS
import random

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600",default_generation_mode=ELF_GENERATION_FORMAT.LOLLMS)
tts = LollmsTTS(lc)
voices = tts.get_voices()

# Pick a voice randomly
random_voice = random.choice(voices)

print(f"Selected voice: {random_voice}")

# Generate Text
# response = lc.generate_text(prompt="Once upon a time", stream=False, temperature=0.5)
# print(response)

# # Generate Completion
# response = lc.generate_completion(prompt="What is the capital of France", stream=False, temperature=0.5)
# print(response)



def cb(chunk, type):
    print(chunk,end="",flush=True)
    
response = lc.generate_text(prompt="One plus one equals ", stream=False, temperature=0.5, streaming_callback=cb)
print()
print(response)
print()
tts.text2Audio(response, random_voice)

# List Mounted Personalities
response = lc.listMountedPersonalities()
print(response)

# List Models
response = lc.listModels()
print(response)