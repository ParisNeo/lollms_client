try:
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
    import tempfile

    # --- XTTS Implementation ---
    try:
        print("Server: Loading XTTS dependencies...")
        import torch
        import torchaudio
        from TTS.api import TTS
        print("Server: XTTS dependencies loaded successfully")
        
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Server: Using device: {device}")
        
        xtts_available = True
        
    except Exception as e:
        print(f"Server: Failed to load XTTS dependencies: {e}")
        print(f"Server: Traceback:\n{traceback.format_exc()}")
        xtts_available = False

    # --- API Models ---
    class GenerationRequest(BaseModel):
        text: str
        voice: Optional[str] = None
        language: Optional[str] = "en"
        speaker_wav: Optional[str] = None

    class XTTSServer:
        def __init__(self):
            self.model = None
            self.model_loaded = False
            self.model_loading = False  # Flag to prevent concurrent loading
            self.available_voices = self._load_available_voices()
            self.available_models = ["xtts_v2"]
            
            # Don't initialize model here - do it lazily on first request
            print("Server: XTTS server initialized (model will be loaded on first request)")
        
        async def _ensure_model_loaded(self):
            """Ensure the XTTS model is loaded (lazy loading)"""
            if self.model_loaded:
                return
                
            if self.model_loading:
                # Another request is already loading the model, wait for it
                while self.model_loading and not self.model_loaded:
                    await asyncio.sleep(0.1)
                return
                
            if not xtts_available:
                raise RuntimeError("XTTS library not available")
                
            try:
                self.model_loading = True
                print("Server: Loading XTTS model for the first time (this may take a few minutes)...")
                
                # Initialize XTTS model
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                
                self.model_loaded = True
                print("Server: XTTS model loaded successfully")
                
            except Exception as e:
                print(f"Server: Error loading XTTS model: {e}")
                print(f"Server: Traceback:\n{traceback.format_exc()}")
                self.model_loaded = False
                raise
            finally:
                self.model_loading = False
        
        def _load_available_voices(self) -> List[str]:
            """Load and return available voices"""
            try:
                # Look for voice files in voices directory
                voices_dir = Path(__file__).parent / "voices"
                voices = []
                
                if voices_dir.exists():
                    # Look for WAV files in voices directory
                    for voice_file in voices_dir.glob("*.wav"):
                        voices.append(voice_file.stem)
                
                # If no custom voices found, provide some default names
                if not voices:
                    voices = ["default", "female", "male"]
                    
                return voices
                
            except Exception as e:
                print(f"Server: Error loading voices: {e}")
                return ["default"]
        
        async def generate_audio(self, text: str, voice: Optional[str] = None, 
                        language: str = "en", speaker_wav: Optional[str] = None) -> bytes:
            """Generate audio from text using XTTS"""
            # Ensure model is loaded before proceeding
            await self._ensure_model_loaded()
            
            if not self.model_loaded or self.model is None:
                raise RuntimeError("XTTS model failed to load")
            
            try:
                print(f"Server: Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"Server: Using voice: {voice}, language: {language}")
                
                # Handle voice/speaker selection
                speaker_wav_path = None
                
                # First priority: use provided speaker_wav parameter
                if speaker_wav:
                    speaker_wav_path = speaker_wav
                    print(f"Server: Using provided speaker_wav: {speaker_wav_path}")
                
                # Second priority: check if voice parameter is a file path
                elif voice and voice != "default":
                    if os.path.exists(voice):
                        # Voice parameter is a full file path
                        speaker_wav_path = voice
                        print(f"Server: Using voice as file path: {speaker_wav_path}")
                    else:
                        # Look for voice file in voices directory
                        voices_dir = Path(__file__).parent / "voices"
                        potential_voice_path = voices_dir / f"{voice}.wav"
                        if potential_voice_path.exists():
                            speaker_wav_path = str(potential_voice_path)
                            print(f"Server: Using custom voice file: {speaker_wav_path}")
                        else:
                            print(f"Server: Voice '{voice}' not found in voices directory")
                else:
                    voice = "default_voice"
                    # Look for voice file in voices directory
                    voices_dir = Path(__file__).parent / "voices"
                    potential_voice_path = voices_dir / f"{voice}.mp3"
                    if potential_voice_path.exists():
                        speaker_wav_path = str(potential_voice_path)
                        print(f"Server: Using custom voice file: {speaker_wav_path}")
                    else:
                        print(f"Server: Voice '{voice}' not found in voices directory")
                # Create a temporary file for output
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_output_path = temp_file.name
                
                try:
                    # Generate audio using XTTS
                    if speaker_wav_path and os.path.exists(speaker_wav_path):
                        print(f"Server: Generating with speaker reference: {speaker_wav_path}")
                        self.model.tts_to_file(
                            text=text,
                            speaker_wav=speaker_wav_path,
                            language=language,
                            file_path=temp_output_path
                        )
                    else:
                        print("Server: No valid speaker reference found, trying default")
                        # For XTTS without speaker reference, try to find a default
                        default_speaker = self._get_default_speaker_file()
                        if default_speaker and os.path.exists(default_speaker):
                            print(f"Server: Using default speaker: {default_speaker}")
                            self.model.tts_to_file(
                                text=text,
                                speaker_wav=default_speaker,
                                language=language,
                                file_path=temp_output_path
                            )
                        else:
                            # Create a more helpful error message
                            available_voices = self._get_all_available_voice_files()
                            error_msg = f"No speaker reference available. XTTS requires a speaker reference file.\n"
                            error_msg += f"Attempted to use: {speaker_wav_path if speaker_wav_path else 'None'}\n"
                            error_msg += f"Available voice files: {available_voices}"
                            raise RuntimeError(error_msg)
                    
                    # Read the generated audio file
                    with open(temp_output_path, 'rb') as f:
                        audio_bytes = f.read()
                    
                    print(f"Server: Generated {len(audio_bytes)} bytes of audio")
                    return audio_bytes
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_output_path):
                        os.unlink(temp_output_path)
                
            except Exception as e:
                print(f"Server: Error generating audio: {e}")
                print(f"Server: Traceback:\n{traceback.format_exc()}")
                raise
        
        def _get_all_available_voice_files(self) -> List[str]:
            """Get list of all available voice files for debugging"""
            voices_dir = Path(__file__).parent / "voices"
            voice_files = []
            
            if voices_dir.exists():
                voice_files = [str(f) for f in voices_dir.glob("*.wav")]
                
            return voice_files
        
        def _get_default_speaker_file(self) -> Optional[str]:
            """Get path to default speaker file"""
            voices_dir = Path(__file__).parent / "voices"
            
            # Look for a default speaker file
            for filename in ["default.wav", "speaker.wav", "reference.wav"]:
                potential_path = voices_dir / filename
                if potential_path.exists():
                    return str(potential_path)
            
            # If no default found, look for any wav file
            wav_files = list(voices_dir.glob("*.wav"))
            if wav_files:
                return str(wav_files[0])
            
            return None
        
        def list_voices(self) -> List[str]:
            """Return list of available voices"""
            return self.available_voices
        
        def list_models(self) -> List[str]:
            """Return list of available models"""
            return self.available_models

    # --- Globals ---
    app = FastAPI(title="XTTS Server")
    router = APIRouter()
    xtts_server = XTTSServer()
    model_lock = asyncio.Lock()  # Ensure thread-safe access

    # --- API Endpoints ---
    @router.post("/generate_audio")
    async def generate_audio(request: GenerationRequest):
        async with model_lock:
            try:
                print(f"request.language:{request.language}")
                audio_bytes = await xtts_server.generate_audio(
                    text=request.text,
                    voice=request.voice,
                    language=request.language,
                    speaker_wav=request.speaker_wav
                )
                from fastapi.responses import Response
                return Response(content=audio_bytes, media_type="audio/wav")
            except Exception as e:
                print(f"Server: ERROR in generate_audio endpoint: {e}")
                print(f"Server: ERROR traceback:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

    @router.get("/list_voices")
    async def list_voices():
        try:
            voices = xtts_server.list_voices()
            print(f"Server: Returning {len(voices)} voices: {voices}")
            return {"voices": voices}
        except Exception as e:
            print(f"Server: ERROR in list_voices endpoint: {e}")
            print(f"Server: ERROR traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/list_models")
    async def list_models():
        try:
            models = xtts_server.list_models()
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
            "xtts_available": xtts_available,
            "model_loaded": xtts_server.model_loaded,
            "model_loading": xtts_server.model_loading,
            "voices_count": len(xtts_server.available_voices),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    # Add a health check endpoint that responds immediately
    @router.get("/health")
    async def health_check():
        return {"status": "healthy", "ready": True}

    app.include_router(router)

    # --- Server Startup ---
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="XTTS TTS Server")
        parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to.")
        parser.add_argument("--port", type=int, default="96", help="Port to bind the server to.")
        
        args = parser.parse_args()

        print(f"Server: Starting XTTS server on {args.host}:{args.port}")
        print(f"Server: XTTS available: {xtts_available}")
        print(f"Server: Model will be loaded on first audio generation request")
        print(f"Server: Available voices: {len(xtts_server.available_voices)}")
        if xtts_available:
            print(f"Server: Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Create voices directory if it doesn't exist
        voices_dir = Path(__file__).parent / "voices"
        voices_dir.mkdir(exist_ok=True)
        print(f"Server: Voices directory: {voices_dir}")
        try:
            uvicorn.run(app, host=args.host, port=args.port)
        except Exception as e:
            print(f"Server: CRITICAL ERROR running server: {e}")
            print(f"Server: Traceback:\n{traceback.format_exc()}")  
except Exception as e:
    print(f"Server: CRITICAL ERROR during startup: {e}")