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
    
    # Use ascii_colors for logging
    from ascii_colors import ASCIIColors

    # --- XTTS Implementation ---
    try:
        ASCIIColors.info("Server: Loading XTTS dependencies...")
        import torch
        from TTS.api import TTS
        ASCIIColors.green("Server: XTTS dependencies loaded successfully")
        
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ASCIIColors.info(f"Server: Using device: {device}")
        
        xtts_available = True
        
    except Exception as e:
        ASCIIColors.error(f"Server: Failed to load XTTS dependencies: {e}")
        ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
        xtts_available = False

    # --- API Models ---
    class GenerationRequest(BaseModel):
        text: str
        voice: Optional[str] = None
        language: Optional[str] = "en"
        # speaker_wav is kept for backward compatibility but voice is preferred
        speaker_wav: Optional[str] = None 
        split_sentences: Optional[bool] = True

    class XTTSServer:
        def __init__(self):
            self.model = None
            self.model_loaded = False
            self.model_loading = False  # Flag to prevent concurrent loading
            self.available_models = ["tts_models/multilingual/multi-dataset/xtts_v2"]
            self.voices_dir = Path(__file__).parent / "voices"
            self.voices_dir.mkdir(exist_ok=True)
            self.available_voices = self._load_available_voices()
            
            ASCIIColors.info("Server: XTTS server initialized (model will be loaded on first request)")
        
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
                raise RuntimeError("XTTS library not available. Please ensure all dependencies are installed correctly in the venv.")
                
            try:
                self.model_loading = True
                ASCIIColors.yellow("Server: Loading XTTS model for the first time (this may take a few minutes)...")
                
                # Initialize XTTS model
                self.model = TTS(self.available_models[0]).to(device)
                
                self.model_loaded = True
                ASCIIColors.green("Server: XTTS model loaded successfully")
                
            except Exception as e:
                ASCIIColors.error(f"Server: Error loading XTTS model: {e}")
                ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
                self.model_loaded = False
                raise
            finally:
                self.model_loading = False
        
        def _load_available_voices(self) -> List[str]:
            """Load and return available voices, ensuring 'default_voice' is always present."""
            try:
                self.voices_dir.mkdir(exist_ok=True)
                
                # Scan for case-insensitive .wav and .mp3 files and get their stems
                found_voices = {p.stem for p in self.voices_dir.glob("*.[wW][aA][vV]")}
                found_voices.update({p.stem for p in self.voices_dir.glob("*.[mM][pP]3")})
                
                # GUARANTEE 'default_voice' is in the list for UI consistency.
                all_voices = {"default_voice"}.union(found_voices)
                
                sorted_voices = sorted(list(all_voices))
                ASCIIColors.info(f"Discovered voices: {sorted_voices}")
                return sorted_voices
                
            except Exception as e:
                ASCIIColors.error(f"Server: Error scanning voices directory: {e}")
                # If scanning fails, it's crucial to still return the default.
                return ["default_voice"]
        
        def _get_speaker_wav_path(self, voice_name: str) -> Optional[str]:
            """Find the path to a speaker wav/mp3 file from its name."""
            if not voice_name:
                return None
            
            # Case 1: voice_name is an absolute path that exists
            if os.path.isabs(voice_name) and os.path.exists(voice_name):
                return voice_name
            
            # Case 2: voice_name is a name in the voices directory (check for .mp3 then .wav)
            mp3_path = self.voices_dir / f"{voice_name}.mp3"
            if mp3_path.exists():
                return str(mp3_path)

            wav_path = self.voices_dir / f"{voice_name}.wav"
            if wav_path.exists():
                return str(wav_path)
            
            return None

        async def generate_audio(self, req: GenerationRequest) -> bytes:
            """Generate audio from text using XTTS"""
            await self._ensure_model_loaded()
            
            if not self.model_loaded or self.model is None:
                raise RuntimeError("XTTS model failed to load or is not available.")
            
            try:
                text_to_generate = req.text
                ASCIIColors.info(f"Server: Generating audio for: '{text_to_generate[:50]}{'...' if len(text_to_generate) > 50 else ''}'")
                ASCIIColors.info(f"Server: Language: {req.language}, Requested Voice: {req.voice}")

                # Determine which voice name to use. Priority: speaker_wav > voice > 'default_voice'
                voice_to_find = req.speaker_wav or req.voice or "default_voice"
                speaker_wav_path = self._get_speaker_wav_path(voice_to_find)

                # If the chosen voice wasn't found and it wasn't the default, try the default as a fallback.
                if not speaker_wav_path and voice_to_find != "default_voice":
                    ASCIIColors.warning(f"Voice '{voice_to_find}' not found. Falling back to 'default_voice'.")
                    speaker_wav_path = self._get_speaker_wav_path("default_voice")

                # If still no path, it's a critical error because even the default is missing.
                if not speaker_wav_path:
                    available = self._get_all_available_voice_files()
                    raise RuntimeError(
                        f"XTTS requires a speaker reference file, but none could be found.\n"
                        f"Attempted to use '{voice_to_find}' but it was not found, and the fallback 'default_voice.mp3' is also missing from the voices folder.\n"
                        f"Please add audio files to the '{self.voices_dir.resolve()}' directory. Available files: {available or 'None'}"
                    )
                
                ASCIIColors.info(f"Server: Using speaker reference: {speaker_wav_path}")

                # Generate audio using XTTS
                wav_chunks = self.model.tts(
                    text=text_to_generate,
                    speaker_wav=speaker_wav_path,
                    language=req.language,
                    split_sentences=req.split_sentences
                )
                
                # Combine chunks into a single audio stream
                audio_data = np.array(wav_chunks, dtype=np.float32)
                
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(self.model.synthesizer.output_sample_rate)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                audio_bytes = buffer.getvalue()

                ASCIIColors.green(f"Server: Generated {len(audio_bytes)} bytes of audio.")
                return audio_bytes
                
            except Exception as e:
                ASCIIColors.error(f"Server: Error generating audio: {e}")
                ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
                raise
        
        def _get_all_available_voice_files(self) -> List[str]:
            """Get list of all available voice files for debugging"""
            return [f.name for f in self.voices_dir.glob("*.*")]
        
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
    model_lock = asyncio.Lock()  # Ensure only one generation happens at a time on the model

    # --- API Endpoints ---
    @router.post("/generate_audio")
    async def api_generate_audio(request: GenerationRequest):
        async with model_lock:
            try:
                from fastapi.responses import Response
                audio_bytes = await xtts_server.generate_audio(request)
                return Response(content=audio_bytes, media_type="audio/wav")
            except Exception as e:
                ASCIIColors.error(f"Server: ERROR in generate_audio endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    @router.get("/list_voices")
    async def api_list_voices():
        try:
            voices = xtts_server.list_voices()
            return {"voices": voices}
        except Exception as e:
            ASCIIColors.error(f"Server: ERROR in list_voices endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/list_models")
    async def api_list_models():
        try:
            models = xtts_server.list_models()
            return {"models": models}
        except Exception as e:
            ASCIIColors.error(f"Server: ERROR in list_models endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/status")
    async def status():
        return {
            "status": "running",
            "xtts_available": xtts_available,
            "model_loaded": xtts_server.model_loaded,
            "device": device if xtts_available else "N/A"
        }

    app.include_router(router)

    # --- Server Startup ---
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="LoLLMs XTTS Server")
        parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to.")
        parser.add_argument("--port", type=int, default=8081, help="Port to bind the server to.")
        
        args = parser.parse_args()

        ASCIIColors.cyan("--- LoLLMs XTTS Server ---")
        ASCIIColors.green(f"Starting server on http://{args.host}:{args.port}")
        ASCIIColors.info(f"Voices directory: {xtts_server.voices_dir.resolve()}")
        
        if not xtts_available:
            ASCIIColors.red("Warning: XTTS dependencies not found. Server will run but generation will fail.")
        else:
            ASCIIColors.info(f"Detected device: {device}")
        
        uvicorn.run(app, host=args.host, port=args.port)

except Exception as e:
    # This will catch errors during initial imports
    from ascii_colors import ASCIIColors
    ASCIIColors.red(f"Server: CRITICAL ERROR during startup: {e}")
    import traceback
    ASCIIColors.red(f"Server: Traceback:\n{traceback.format_exc()}")```