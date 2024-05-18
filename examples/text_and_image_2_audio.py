"""
Author: ParisNeo, a computer geek passionate about AI

This example code demonstrates how to use the LoLLMs (Lord of Large Language Models) system to capture an image from a webcam, send it to the LollmsClient for analysis, and receive a descriptive response. The response is then converted to audio using the LollmsXTTS service.

Requirements:
- LoLLMs should be up and running.
- The XTTS service within LoLLMs must be working.

Steps:
1. Initialize the LollmsClient instance.
2. Fetch available voices and randomly select one.
3. Capture an image from the webcam and save it to a file.
4. Generate a descriptive text for the captured image using the LollmsClient.
5. Convert the generated text to audio using the selected voice.

Make sure you have the necessary dependencies installed and your webcam is accessible.
"""
import cv2
from lollms_client import LollmsClient, ELF_GENERATION_FORMAT, LollmsXTTS
import random

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600", default_generation_mode=ELF_GENERATION_FORMAT.LOLLMS)
tts = LollmsXTTS(lc)
voices = tts.get_voices()

# Pick a voice randomly
random_voice = random.choice(voices)
print(f"Selected voice: {random_voice}")

# Capture image from webcam and save it to a file
def capture_image(file_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image")

    cv2.imwrite(file_path, frame)
    cap.release()

# File path to save the captured image
image_path = "captured_image.jpg"

# Capture and save the image
capture_image(image_path)

# Function to handle streaming callback
def cb(chunk, type):
    print(chunk, end="", flush=True)

# Generate text with image
response = lc.generate_with_images(prompt="user: describe the content of the image.\nassistant: ", images=[image_path], stream=False, temperature=0.5, streaming_callback=cb)
print(f"response: {response}")
tts.text2Audio(response, random_voice)