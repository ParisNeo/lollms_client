import cv2
from lollms_client import LollmsClient, ELF_GENERATION_FORMAT, LollmsXTTS, TasksLibrary, FunctionCalling_Library
import random

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600", default_generation_mode=ELF_GENERATION_FORMAT.LOLLMS)
tl = TasksLibrary(lc)
tts = LollmsXTTS(lc)
fcl = FunctionCalling_Library(tl)
voices = tts.get_voices()
# Pick a voice randomly
random_voice = random.choice(voices)
print(f"Selected voice: {random_voice}")

# File path to save the captured image
file_path = "captured_image.jpg"
images = []
# Capture image from webcam and save it to a file
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image")
    images.clear()
    images.append(file_path)
    cv2.imwrite(file_path, frame)
    cap.release()
    return "Image captured successfully"


fcl.register_function("capture_image",capture_image,"Captures an image from the user webcam",[])



# Function to handle streaming callback
def cb(chunk, type):
    print(chunk, end="", flush=True)

# Generate text with image
response, function_calls = fcl.generate_with_functions_and_images(prompt="user: take a look at me then tell ma how i look.\nassistant: ", images=images, stream=False, temperature=0.5, streaming_callback=cb)
print(f"response: {response}")
if len(function_calls)>0:
    results = fcl.execute_function_calls(function_calls)
    result = "\n".join(results)
    prompt="user: take a look at me then tell ma how i look.\nassistant: "+response + f"\nfunction execution result: {result}\nassistant: "
    response, function_calls = fcl.generate_with_functions_and_images(prompt, images=images, stream=False, temperature=0.5, streaming_callback=cb)
print(f"response: {response}")
tts.text2Audio(response, random_voice)