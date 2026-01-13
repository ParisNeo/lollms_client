try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel
    import argparse
    import sys
    import os
    from pathlib import Path
    import asyncio
    import traceback
    import base64
    import io
    import wave
    import numpy as np
    from typing import Optional, List
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    from ascii_colors import ASCIIColors

    # Fish Speech imports
    try:
        ASCIIColors.info("Server: Loading Fish Speech dependencies...")
        import torch
        from fish_speech.models.text2semantic.inference import InferenceBuilder as Text2SemanticInference
        from fish_speech.models.dac.inference import AudioCodecInference
        ASCIIColors.green("Server: Fish Speech dependencies loaded")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ASCIIColors.info(f"Server: Using device: {device}")
        fish_speech_available = True
        
    except Exception as e:
        ASCIIColors.error(f"Server: Failed to load Fish Speech: {e}")
        ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
        fish_speech_available = False

    # API Models
    class TTSRequest(BaseModel):
        text: str
        reference_audio: Optional[str] = None  # base64 encoded
        reference_text: Optional[str] = None
        format: str = "wav"
        top_p: float = 0.9
        temperature: float = 0.9
        repetition_penalty: float = 1.2
        normalize: bool = True
        chunk_length: int = 200

    class FishSpeechServer:
        def __init__(self, model_path: str, device: str = "auto", compile: bool = False):
            self.model_path = Path(model_path)
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.compile = compile and self.device == "cuda"
            self.model_loaded = False
            self.model_loading = False
            
            self.text2semantic_model = None
            self.codec_model = None
            
            self.references_dir = Path(__file__).parent / "references"
            self.references_dir.mkdir(exist_ok=True)
            
            ASCIIColors.info(f"Server: Fish Speech server initialized (model will load on first request)")
            ASCIIColors.info(f"Server: Model path: {self.model_path}")
            ASCIIColors.info(f"Server: Device: {self.device}, Compile: {self.compile}")
        
        async def _ensure_model_loaded(self):
            """Lazy load Fish Speech models."""
            if self.model_loaded:
                return
            
            if self.model_loading:
                while self.model_loading and not self.model_loaded:
                    await asyncio.sleep(0.1)
                return
            
            if not fish_speech_available:
                raise RuntimeError("Fish Speech not available. Check dependencies.")
            
            try:
                self.model_loading = True
                ASCIIColors.yellow("Server: Loading Fish Speech models (first run may take time)...")
                
                # Load text2semantic model
                self.text2semantic_model = Text2SemanticInference(
                    checkpoint_path=str(self.model_path),
                    device=self.device,
                    compile=self.compile
                )
                
                # Load codec model
                codec_path = self.model_path / "codec.pth"
                if not codec_path.exists():
                    # Try alternative names
                    codec_path = self.model_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
                
                self.codec_model = AudioCodecInference(
                    checkpoint_path=str(codec_path),
                    device=self.device
                )
                
                self.model_loaded = True
                ASCIIColors.green("Server: Fish Speech models loaded successfully")
                
            except Exception as e:
                ASCIIColors.error(f"Server: Error loading models: {e}")
                ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
                self.model_loaded = False
                raise
            finally:
                self.model_loading = False
        
        async def generate_audio(self, request: TTSRequest) -> bytes:
            """Generate audio from text using Fish Speech."""
            await self._ensure_model_loaded()
            
            if not self.model_loaded:
                raise RuntimeError("Fish Speech models not loaded")
            
            try:
                ASCIIColors.info(f"Server: Generating audio for: '{request.text[:50]}...'")
                
                # Prepare reference audio if provided
                reference_tokens = None
                if request.reference_audio:
                    audio_bytes = base64.b64decode(request.reference_audio)
                    # Encode reference audio
                    reference_tokens = self._encode_reference_audio(
                        audio_bytes,
                        request.reference_text
                    )
                
                # Generate semantic tokens from text
                codes = self.text2semantic_model.generate(
                    text=request.text,
                    prompt_tokens=reference_tokens,
                    prompt_text=request.reference_text,
                    top_p=request.top_p,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    max_new_tokens=2048
                )
                
                # Generate audio from semantic tokens
                audio_data = self.codec_model.decode(codes)
                
                # Convert to bytes
                if request.format == "wav":
                    audio_bytes = self._to_wav_bytes(audio_data)
                elif request.format == "mp3":
                    audio_bytes = self._to_mp3_bytes(audio_data)
                else:  # pcm
                    audio_bytes = audio_data.tobytes()
                
                ASCIIColors.green(f"Server: Generated {len(audio_bytes)} bytes")
                return audio_bytes
                
            except Exception as e:
                ASCIIColors.error(f"Server: Error generating audio: {e}")
                ASCIIColors.error(f"Server: Traceback:\n{traceback.format_exc()}")
                raise
        
        def _encode_reference_audio(self, audio_bytes: bytes, transcript: Optional[str]) -> np.ndarray:
            """Encode reference audio to semantic tokens."""
            # Save temporarily
            temp_path = self.references_dir / "temp_reference.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            try:
                tokens = self.codec_model.encode(str(temp_path))
                return tokens
            finally:
                temp_path.unlink(missing_ok=True)
        
        def _to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int = 44100) -> bytes:
            """Convert audio array to WAV bytes."""
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            return buffer.getvalue()
        
        def _to_mp3_bytes(self, audio_data: np.ndarray) -> bytes:
            """Convert audio array to MP3 bytes."""
            # Requires pydub - fallback to WAV
            return self._to_wav_bytes(audio_data)
        
        def list_voices(self) -> List[str]:
            """List available reference voices."""
            return [f.stem for f in self.references_dir.glob("*.[wW][aA][vV]")]

    # FastAPI app
    app = FastAPI(title="Fish Speech Server")
    fish_server = None

    @app.post("/v1/tts")
    async def tts_endpoint(request: TTSRequest):
        try:
            audio_bytes = await fish_server.generate_audio(request)
            
            media_type = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "pcm": "audio/pcm"
            }.get(request.format, "audio/wav")
            
            return Response(content=audio_bytes, media_type=media_type)
        except Exception as e:
            ASCIIColors.error(f"Server: TTS endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/list_voices")
    async def list_voices_endpoint():
        try:
            voices = fish_server.list_voices()
            return {"voices": voices}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {
            "status": "running",
            "fish_speech_available": fish_speech_available,
            "model_loaded": fish_server.model_loaded if fish_server else False
        }

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Fish Speech TTS Server")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8080)
        parser.add_argument("--model-path", type=str, required=True)
        parser.add_argument("--device", type=str, default="auto")
        parser.add_argument("--compile", action="store_true")
        
        args = parser.parse_args()

        fish_server = FishSpeechServer(
            model_path=args.model_path,
            device=args.device,
            compile=args.compile
        )

        ASCIIColors.cyan("--- Fish Speech TTS Server ---")
        ASCIIColors.green(f"Starting server on http://{args.host}:{args.port}")
        
        uvicorn.run(app, host=args.host, port=args.port)

except Exception as e:
    from ascii_colors import ASCIIColors
    ASCIIColors.red(f"Server: CRITICAL ERROR: {e}")
    import traceback
    ASCIIColors.red(f"Server: Traceback:\n{traceback.format_exc()}")