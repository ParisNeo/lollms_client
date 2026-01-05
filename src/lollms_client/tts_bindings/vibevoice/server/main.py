import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import argparse
import sys
import os
from typing import Optional, List, Dict
import numpy as np
import io
import wave
from ascii_colors import ASCIIColors

# --- API Models ---
class GenerationRequest(BaseModel):
    text: str
    voice: Optional[str] = "default_female"
    speed: Optional[float] = 1.0

class StartSessionRequest(BaseModel):
    session_id: str

class EmotionRequest(BaseModel):
    emotion: str

class VibeVoiceServer:
    def __init__(self):
        self.active_session = None
        self.current_emotion = "neutral"
        # In a real scenario, load your model here
        ASCIIColors.info("VibeVoice Engine Initialized")

    def generate_dummy_audio(self, duration_sec=1.0) -> bytes:
        """Generates dummy sine wave audio for testing"""
        sample_rate = 24000
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        # Change freq based on emotion
        freq = 440
        if self.current_emotion == "excited": freq = 880
        if self.current_emotion == "sad": freq = 220
        
        note = np.sin(freq * t * 2 * np.pi)
        audio = (note * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buffer.getvalue()

# --- Server Setup ---
app = FastAPI(title="VibeVoice Server")
router = APIRouter()
engine = VibeVoiceServer()

@router.get("/status")
async def status():
    return {
        "status": "running", 
        "session": engine.active_session, 
        "emotion": engine.current_emotion
    }

@router.get("/list_voices")
async def list_voices():
    return {"voices": ["default_female", "default_male", "robot"]}

@router.get("/list_models")
async def list_models():
    return {"models": ["vibevoice_v1_streaming"]}

@router.post("/generate_audio")
async def generate_audio(req: GenerationRequest):
    try:
        ASCIIColors.info(f"Generating audio for: {req.text} (Emotion: {engine.current_emotion})")
        # Replace this with real TTS generation logic
        audio_data = engine.generate_dummy_audio(duration_sec=len(req.text)*0.1) 
        return Response(content=audio_data, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Action Endpoints ---

@router.post("/actions/start_conversation")
async def start_conversation(req: StartSessionRequest):
    engine.active_session = req.session_id
    ASCIIColors.success(f"Session started: {req.session_id}")
    return {"status": True, "message": "Session initialized and buffer warmed up."}

@router.post("/actions/stop_conversation")
async def stop_conversation():
    prev = engine.active_session
    engine.active_session = None
    ASCIIColors.warning(f"Session stopped: {prev}")
    return {"status": True}

@router.post("/actions/change_voice_tone")
async def change_voice_tone(req: EmotionRequest):
    if req.emotion in ["neutral", "happy", "sad", "angry", "excited"]:
        engine.current_emotion = req.emotion
        ASCIIColors.info(f"Voice tone changed to: {req.emotion}")
        return {"success": True}
    return {"success": False, "error": "Invalid emotion"}

app.include_router(router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9634)
    args = parser.parse_args()

    ASCIIColors.cyan("--- VibeVoice Server ---")
    uvicorn.run(app, host=args.host, port=args.port)
