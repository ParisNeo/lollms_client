import os
import argparse
import uvicorn
import gc
import torch
import io
import tempfile
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Diffusers imports
from diffusers import (
    LTXVideoPipeline,
    CogVideoXPipeline,
    VideoPipelineOutput
)
from diffusers.utils import export_to_video
from ascii_colors import ASCIIColors

# Global State
class ServerState:
    def __init__(self):
        self.model_name: str = ""
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

state = ServerState()

app = FastAPI(title="Diffusers TTV Server")

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    params: Dict[str, Any] = Field(default_factory=dict)

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(model_name: str):
    """Loads the TTV pipeline based on the model name."""
    if state.pipeline and state.model_name == model_name:
        return

    # Unload previous
    if state.pipeline:
        del state.pipeline
        state.pipeline = None
        flush()

    ASCIIColors.info(f"Loading TTV model: {model_name} on {state.device} with {state.dtype}")

    try:
        if "LTX" in model_name:
            # LTX-Video
            state.pipeline = LTXVideoPipeline.from_pretrained(
                model_name,
                torch_dtype=state.dtype
            )
        elif "CogVideoX" in model_name:
            # CogVideoX
            state.pipeline = CogVideoXPipeline.from_pretrained(
                model_name,
                torch_dtype=state.dtype
            )
        else:
            # Fallback attempt for generic pipelines
            ASCIIColors.warning(f"Unknown model type for {model_name}, trying LTXVideoPipeline default...")
            state.pipeline = LTXVideoPipeline.from_pretrained(
                model_name,
                torch_dtype=state.dtype
            )

        # Move to device and optimizations
        state.pipeline.to(state.device)
        
        # Enable slicing/offloading for VRAM efficiency if possible
        if hasattr(state.pipeline, "enable_vae_slicing"):
            state.pipeline.enable_vae_slicing()
            
        # Optional: enable_model_cpu_offload() for low VRAM 
        # state.pipeline.enable_model_cpu_offload() 

        state.model_name = model_name
        ASCIIColors.green(f"Model {model_name} loaded successfully.")

    except Exception as e:
        ASCIIColors.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.post("/generate_video")
async def generate_video(req: GenerateRequest):
    params = req.params
    model_name = params.get("model_name", "Lightricks/LTX-Video")
    
    # Load model if needed
    load_model(model_name)
    
    # Extract params with defaults
    width = int(params.get("width", 768))
    height = int(params.get("height", 512))
    num_frames = int(params.get("num_frames", 24))
    steps = int(params.get("num_inference_steps", 30))
    guidance = float(params.get("guidance_scale", 7.5))
    seed = int(params.get("seed", -1))

    generator = None
    if seed != -1:
        generator = torch.Generator(device="cpu").manual_seed(seed) # Generator often needs to be CPU for reproducibility across devices in diffusers
    
    ASCIIColors.info(f"Generating video: '{req.prompt}' ({width}x{height}, {num_frames} frames)")

    try:
        # LTX and CogVideo specifics
        # LTX usually requires width/height to be multiples of 32
        
        with torch.no_grad():
            output = state.pipeline(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            )

        # Output is VideoPipelineOutput with .frames
        frames = output.frames[0] # frames is List[List[PIL.Image]] or List[np.array]
        
        # Export to temp file then read bytes
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        
        export_to_video(frames, tmp_path, fps=24) # 24fps default
        
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
            
        os.remove(tmp_path)
        
        return io.BytesIO(video_bytes).getvalue() # Return bytes directly via starlette/fastapi default response handling for binary? 
        # Actually FastAPI returns JSON by default. We should return Response object.
        from fastapi.responses import Response
        return Response(content=video_bytes, media_type="video/mp4")

    except Exception as e:
        ASCIIColors.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        flush()

@app.get("/list_models")
def list_models():
    # Return valid models we know + currently loaded
    models = [
        "Lightricks/LTX-Video",
        "THUDM/CogVideoX-2b",
        "THUDM/CogVideoX-5b"
    ]
    if state.model_name and state.model_name not in models:
        models.append(state.model_name)
    return models

@app.get("/status")
def status():
    return {
        "status": "running",
        "model_loaded": bool(state.pipeline),
        "current_model": state.model_name,
        "device": state.device
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9634)
    parser.add_argument("--models-path", default="./models")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    uvicorn.run(app, host=args.host, port=args.port)
