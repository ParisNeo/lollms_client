try:
    import uvicorn
    import fastapi
    from fastapi import FastAPI, APIRouter, HTTPException, File, Form, UploadFile
    from pydantic import BaseModel
    import argparse
    import sys
    from pathlib import Path
    import asyncio
    import traceback
    import os
    import re
    from typing import Optional, List
    import io
    import wave
    import numpy as np
    import tempfile
    import warnings

    # Suppress transformers warnings about generation mixin and attention masks to clean up logs
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    
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

    class VoiceUploadResponse(BaseModel):
        success: bool
        voice_name: str
        message: str

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

        def _chunk_text(self, text: str, max_chunk_length: int = 200) -> List[str]:
            """Split text into chunks respecting sentence boundaries."""
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Split into sentences (handles . ! ? followed by space or end)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # If a single sentence exceeds max length, we must split it
                if len(sentence) > max_chunk_length:
                    # First, save any accumulated chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Split long sentence by phrases (commas, semicolons) or force split
                    phrases = re.split(r'(?<=[,;])\s+', sentence)
                    current_split = ""
                    
                    for phrase in phrases:
                        if len(current_split) + len(phrase) + 1 <= max_chunk_length:
                            current_split = (current_split + " " + phrase).strip() if current_split else phrase
                        else:
                            if current_split:
                                chunks.append(current_split)
                            # If single phrase still too long, force split by words
                            if len(phrase) > max_chunk_length:
                                words = phrase.split()
                                current_split = ""
                                for word in words:
                                    if len(current_split) + len(word) + 1 <= max_chunk_length:
                                        current_split = (current_split + " " + word).strip() if current_split else word
                                    else:
                                        if current_split:
                                            chunks.append(current_split)
                                        current_split = word
                            else:
                                current_split = phrase
                    
                    if current_split:
                        chunks.append(current_split.strip())
                        current_split = ""
                        
                else:
                    # Normal case: try to add sentence to current chunk
                    if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                        current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                    else:
                        # Current chunk is full, start a new one
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            
            # Don't forget the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks

        async def generate_audio(self, req: GenerationRequest) -> bytes:
            """Generate audio from text using XTTS with chunking for long texts."""
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

                # Chunk text if it's long (threshold: 250 chars as per XTTS warning)
                CHUNK_THRESHOLD = 250
                MAX_CHUNK_LENGTH = 200  # Slightly under to be safe
                
                if len(text_to_generate) > CHUNK_THRESHOLD:
                    chunks = self._chunk_text(text_to_generate, MAX_CHUNK_LENGTH)
                    ASCIIColors.info(f"Server: Text split into {len(chunks)} chunks for synthesis")
                    
                    all_audio_parts = []
                    sample_rate = None
                    
                    for i, chunk in enumerate(chunks):
                        ASCIIColors.info(f"Server: Synthesizing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                        
                        wav_chunks = self.model.tts(
                            text=chunk,
                            speaker_wav=speaker_wav_path,
                            language=req.language,
                            split_sentences=False  # We already split manually
                        )
                        
                        audio_part = np.array(wav_chunks, dtype=np.float32)
                        all_audio_parts.append(audio_part)
                        
                        # Capture sample rate from first chunk
                        if sample_rate is None:
                            sample_rate = self.model.synthesizer.output_sample_rate
                    
                    # Concatenate all audio parts
                    audio_data = np.concatenate(all_audio_parts)
                    ASCIIColors.info(f"Server: Concatenated {len(all_audio_parts)} audio chunks")
                    
                else:
                    # Short text: single synthesis call
                    wav_chunks = self.model.tts(
                        text=text_to_generate,
                        speaker_wav=speaker_wav_path,
                        language=req.language,
                        split_sentences=req.split_sentences
                    )
                    
                    audio_data = np.array(wav_chunks, dtype=np.float32)
                    sample_rate = self.model.synthesizer.output_sample_rate
                
                # Convert to WAV format
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(sample_rate)
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

    @router.post("/upload_voice")
    async def api_upload_voice(
        voice_file: fastapi.UploadFile = fastapi.File(...),
        voice_name: Optional[str] = fastapi.Form(None)
    ):
        """Upload a voice file to the server for use with XTTS"""
        try:
            from fastapi.responses import JSONResponse
            
            # Validate file type
            allowed_extensions = {'.wav', '.mp3'}
            file_ext = Path(voice_file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type. Only {allowed_extensions} are supported."
                )

            # Determine voice name
            if voice_name:
                # Sanitize voice name
                safe_voice_name = re.sub(r'[^\w\-_]', '_', voice_name)
            else:
                # Use filename without extension
                safe_voice_name = Path(voice_file.filename).stem
                safe_voice_name = re.sub(r'[^\w\-_]', '_', safe_voice_name)

            # Ensure unique name if file exists
            target_path = xtts_server.voices_dir / f"{safe_voice_name}{file_ext}"
            counter = 1
            original_name = safe_voice_name
            while target_path.exists():
                safe_voice_name = f"{original_name}_{counter}"
                target_path = xtts_server.voices_dir / f"{safe_voice_name}{file_ext}"
                counter += 1

            # Save the file
            content = await voice_file.read()
            with open(target_path, "wb") as f:
                f.write(content)

            # Refresh available voices list
            xtts_server.available_voices = xtts_server._load_available_voices()

            ASCIIColors.green(f"Server: Voice uploaded successfully: {safe_voice_name} ({len(content)} bytes)")
            
            return VoiceUploadResponse(
                success=True,
                voice_name=safe_voice_name,
                message=f"Voice '{safe_voice_name}' uploaded successfully. Available as voice='{safe_voice_name}' in generation requests."
            )

        except HTTPException:
            raise
        except Exception as e:
            ASCIIColors.error(f"Server: ERROR in upload_voice endpoint: {e}")
            ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")

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
    ASCIIColors.red(f"Server: Traceback:\n{traceback.format_exc()}")
