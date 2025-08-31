# File: lollms_client/tts_bindings/bark/server/main.py

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import argparse
import sys
from pathlib import Path
import asyncio
import traceback
import os
from typing import Optional, List
import io
import wave
import numpy as np

# --- Bark TTS Implementation ---
try:
    print("Server: Loading Bark dependencies...")
    import torch
    import torchaudio
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.generation import set_seed
    print("Server: Bark dependencies loaded successfully")
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Server: Using device: {device}")
    
    # Set environment variable for Bark to use GPU if available
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    bark_available = True
    
except Exception as e:
    print(f"Server: Failed to load Bark dependencies: {e}")
    print(f"Server: Traceback:\n{traceback.format_exc()}")
    bark_available = False
    SAMPLE_RATE = 24000  # Bark's default sample rate

# --- API Models ---
class GenerationRequest(BaseModel):
    text: str
    voice: Optional[str] = "v2/en_speaker_6"  # Default voice
    temperature: Optional[float] = 0.7
    silence_duration: Optional[float] = 0.25
    seed: Optional[int] = None

class VoiceRequest(BaseModel):
    voice: str

class BarkServer:
    def __init__(self):
        self.model_loaded = False
        self.current_voice = "v2/en_speaker_6"
        self.available_voices = self._get_available_voices()
        self.available_models = ["bark"]
        
        if bark_available:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Bark model"""
        try:
            print("Server: Initializing Bark model (this may take a few minutes on first run)...")
            
            # Preload models for faster generation
            preload_models()
            
            self.model_loaded = True
            print("Server: Bark model loaded successfully")
            
        except Exception as e:
            print(f"Server: Error initializing Bark model: {e}")
            print(f"Server: Traceback:\n{traceback.format_exc()}")
            self.model_loaded = False
    
    def _get_available_voices(self) -> List[str]:
        """Return list of available Bark voices"""
        # Bark voice presets - these are the built-in speaker voices
        voices = [
            # English speakers
            "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
            "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
            "v2/en_speaker_8", "v2/en_speaker_9",
            
            # Chinese speakers
            "v2/zh_speaker_0", "v2/zh_speaker_1", "v2/zh_speaker_2", "v2/zh_speaker_3",
            "v2/zh_speaker_4", "v2/zh_speaker_5", "v2/zh_speaker_6", "v2/zh_speaker_7",
            "v2/zh_speaker_8", "v2/zh_speaker_9",
            
            # French speakers
            "v2/fr_speaker_0", "v2/fr_speaker_1", "v2/fr_speaker_2", "v2/fr_speaker_3",
            "v2/fr_speaker_4", "v2/fr_speaker_5", "v2/fr_speaker_6", "v2/fr_speaker_7",
            "v2/fr_speaker_8", "v2/fr_speaker_9",
            
            # German speakers
            "v2/de_speaker_0", "v2/de_speaker_1", "v2/de_speaker_2", "v2/de_speaker_3",
            "v2/de_speaker_4", "v2/de_speaker_5", "v2/de_speaker_6", "v2/de_speaker_7",
            "v2/de_speaker_8", "v2/de_speaker_9",
            
            # Hindi speakers
            "v2/hi_speaker_0", "v2/hi_speaker_1", "v2/hi_speaker_2", "v2/hi_speaker_3",
            "v2/hi_speaker_4", "v2/hi_speaker_5", "v2/hi_speaker_6", "v2/hi_speaker_7",
            "v2/hi_speaker_8", "v2/hi_speaker_9",
            
            # Italian speakers
            "v2/it_speaker_0", "v2/it_speaker_1", "v2/it_speaker_2", "v2/it_speaker_3",
            "v2/it_speaker_4", "v2/it_speaker_5", "v2/it_speaker_6", "v2/it_speaker_7",
            "v2/it_speaker_8", "v2/it_speaker_9",
            
            # Japanese speakers
            "v2/ja_speaker_0", "v2/ja_speaker_1", "v2/ja_speaker_2", "v2/ja_speaker_3",
            "v2/ja_speaker_4", "v2/ja_speaker_5", "v2/ja_speaker_6", "v2/ja_speaker_7",
            "v2/ja_speaker_8", "v2/ja_speaker_9",
            
            # Korean speakers
            "v2/ko_speaker_0", "v2/ko_speaker_1", "v2/ko_speaker_2", "v2/ko_speaker_3",
            "v2/ko_speaker_4", "v2/ko_speaker_5", "v2/ko_speaker_6", "v2/ko_speaker_7",
            "v2/ko_speaker_8", "v2/ko_speaker_9",
            
            # Polish speakers
            "v2/pl_speaker_0", "v2/pl_speaker_1", "v2/pl_speaker_2", "v2/pl_speaker_3",
            "v2/pl_speaker_4", "v2/pl_speaker_5", "v2/pl_speaker_6", "v2/pl_speaker_7",
            "v2/pl_speaker_8", "v2/pl_speaker_9",
            
            # Portuguese speakers
            "v2/pt_speaker_0", "v2/pt_speaker_1", "v2/pt_speaker_2", "v2/pt_speaker_3",
            "v2/pt_speaker_4", "v2/pt_speaker_5", "v2/pt_speaker_6", "v2/pt_speaker_7",
            "v2/pt_speaker_8", "v2/pt_speaker_9",
            
            # Russian speakers
            "v2/ru_speaker_0", "v2/ru_speaker_1", "v2/ru_speaker_2", "v2/ru_speaker_3",
            "v2/ru_speaker_4", "v2/ru_speaker_5", "v2/ru_speaker_6", "v2/ru_speaker_7",
            "v2/ru_speaker_8", "v2/ru_speaker_9",
            
            # Spanish speakers
            "v2/es_speaker_0", "v2/es_speaker_1", "v2/es_speaker_2", "v2/es_speaker_3",
            "v2/es_speaker_4", "v2/es_speaker_5", "v2/es_speaker_6", "v2/es_speaker_7",
            "v2/es_speaker_8", "v2/es_speaker_9",
            
            # Turkish speakers
            "v2/tr_speaker_0", "v2/tr_speaker_1", "v2/tr_speaker_2", "v2/tr_speaker_3",
            "v2/tr_speaker_4", "v2/tr_speaker_5", "v2/tr_speaker_6", "v2/tr_speaker_7",
            "v2/tr_speaker_8", "v2/tr_speaker_9",
        ]
        
        return voices
    
    def generate_audio(self, text: str, voice: Optional[str] = None, temperature: float = 0.7, 
                      silence_duration: float = 0.25, seed: Optional[int] = None) -> bytes:
        """Generate audio from text using Bark"""
        if not bark_available:
            raise RuntimeError("Bark library not available")
        
        if not self.model_loaded:
            raise RuntimeError("Bark model not initialized")
        
        try:
            # Use provided voice or current default
            speaker_voice = voice or self.current_voice
            
            print(f"Server: Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"Server: Using voice: {speaker_voice}")
            print(f"Server: Temperature: {temperature}, Seed: {seed}")
            
            # Set seed for reproducibility if provided
            if seed is not None:
                set_seed(seed)
            
            # Generate audio using Bark
            # Bark expects text prompts that can include special tokens for emotions, etc.
            audio_array = generate_audio(
                text, 
                history_prompt=speaker_voice,
                text_temp=temperature,
                waveform_temp=temperature
            )
            
            # Add silence at the end if requested
            if silence_duration > 0:
                silence_samples = int(SAMPLE_RATE * silence_duration)
                silence = np.zeros(silence_samples, dtype=audio_array.dtype)
                audio_array = np.concatenate([audio_array, silence])
            
            # Convert to 16-bit PCM
            audio_array = (audio_array * 32767).astype(np.int16)
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_array.tobytes())
            
            audio_bytes = buffer.getvalue()
            print(f"Server: Generated {len(audio_bytes)} bytes of audio")
            return audio_bytes
            
        except Exception as e:
            print(f"Server: Error generating audio: {e}")
            print(f"Server: Traceback:\n{traceback.format_exc()}")
            raise
    
    def set_voice(self, voice: str) -> bool:
        """Set the current default voice"""
        if voice in self.available_voices:
            self.current_voice = voice
            print(f"Server: Voice changed to: {voice}")
            return True
        else:
            print(f"Server: Voice '{voice}' not found in available voices")
            return False
    
    def list_voices(self) -> List[str]:
        """Return list of available voices"""
        return self.available_voices
    
    def list_models(self) -> List[str]:
        """Return list of available models"""
        return self.available_models

# --- Globals ---
app = FastAPI(title="Bark TTS Server")
router = APIRouter()
bark_server = BarkServer()
model_lock = asyncio.Lock()  # Ensure thread-safe access

# --- API Endpoints ---
@router.post("/generate_audio")
async def generate_audio(request: GenerationRequest):
    async with model_lock:
        try:
            audio_bytes = bark_server.generate_audio(
                text=request.text,
                voice=request.voice,
                temperature=request.temperature,
                silence_duration=request.silence_duration,
                seed=request.seed
            )
            from fastapi.responses import Response
            return Response(content=audio_bytes, media_type="audio/wav")
        except Exception as e:
            print(f"Server: ERROR in generate_audio endpoint: {e}")
            print(f"Server: ERROR traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/set_voice")
async def set_voice(request: VoiceRequest):
    try:
        success = bark_server.set_voice(request.voice)
        if success:
            return {"success": True, "message": f"Voice set to {request.voice}"}
        else:
            return {"success": False, "message": f"Voice {request.voice} not found"}
    except Exception as e:
        print(f"Server: ERROR in set_voice endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_voices")
async def list_voices():
    try:
        voices = bark_server.list_voices()
        print(f"Server: Returning {len(voices)} voices")
        return {"voices": voices}
    except Exception as e:
        print(f"Server: ERROR in list_voices endpoint: {e}")
        print(f"Server: ERROR traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_models")
async def list_models():
    try:
        models = bark_server.list_models()
        print(f"Server: Returning {len(models)} models: {models}")
        return {"models": models}
    except Exception as e:
        print(f"Server: ERROR in list_models endpoint: {e}")
        print(f"Server: ERROR traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def status():
    return {
        "status": "running",
        "bark_available": bark_available,
        "model_loaded": bark_server.model_loaded,
        "current_voice": bark_server.current_voice,
        "voices_count": len(bark_server.available_voices),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

app.include_router(router)

# --- Server Startup ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bark TTS Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind the server to.")
    
    args = parser.parse_args()

    print(f"Server: Starting Bark TTS server on {args.host}:{args.port}")
    print(f"Server: Bark available: {bark_available}")
    print(f"Server: Model loaded: {bark_server.model_loaded}")
    print(f"Server: Available voices: {len(bark_server.available_voices)}")
    print(f"Server: Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    uvicorn.run(app, host=args.host, port=args.port)
