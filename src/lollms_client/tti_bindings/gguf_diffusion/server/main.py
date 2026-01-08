import io
import threading
import queue
import logging
import torch
import gguf
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
import uvicorn
from diffusers import FluxPipeline
from ops import GGMLLinear, GGMLConv2d, GGMLEmbedding, GGMLLayerNorm, GGMLGroupNorm, GGMLTensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GGUFServer")

app = FastAPI()
task_queue = queue.Queue()

# --- GGUF Loading Logic ---

def load_gguf_tensors(path):
    """Loads GGUF tensors into a dictionary of GGMLTensors."""
    reader = gguf.GGUFReader(path)
    state_dict = {}
    
    for tensor in reader.tensors:
        # Map raw data to torch tensor
        torch_tensor = torch.from_numpy(tensor.data) 
        
        # Determine shape
        # GGUF stores shapes in reverse order compared to Torch usually
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        
        # Reshape if necessary
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            
        # Wrap in GGMLTensor
        ggml_tensor = GGMLTensor(
            torch_tensor, 
            tensor_type=tensor.tensor_type, 
            tensor_shape=shape
        )
        state_dict[tensor.name] = ggml_tensor
        
    return state_dict

def patch_model(model, gguf_state_dict):
    """
    Recursively replaces layers in the model with GGUF-aware layers 
    if keys exist in the gguf_state_dict.
    """
    normalized_sd = {}
    for k, v in gguf_state_dict.items():
        # Clean keys
        clean_k = k
        for p in ["model.diffusion_model.", "diffusion_model.", "model."]:
            if clean_k.startswith(p):
                clean_k = clean_k[len(p):]
                break
        normalized_sd[clean_k] = v

    modules_to_replace = {}
    
    for name, module in model.named_modules():
        if name == "": continue
        
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        
        if weight_key in normalized_sd:
            if isinstance(module, torch.nn.Linear):
                modules_to_replace[name] = GGMLLinear(
                    module.in_features, module.out_features, 
                    bias=module.bias is not None, 
                    device=module.weight.device, dtype=module.weight.dtype
                )
            elif isinstance(module, torch.nn.Conv2d):
                modules_to_replace[name] = GGMLConv2d(
                    module.in_channels, module.out_channels, module.kernel_size,
                    stride=module.stride, padding=module.padding, 
                    dilation=module.dilation, groups=module.groups,
                    bias=module.bias is not None
                )
            elif isinstance(module, torch.nn.Embedding):
                 modules_to_replace[name] = GGMLEmbedding(
                     module.num_embeddings, module.embedding_dim,
                     padding_idx=module.padding_idx, max_norm=module.max_norm,
                     norm_type=module.norm_type, scale_grad_by_freq=module.scale_grad_by_freq,
                     sparse=module.sparse
                 )
            elif isinstance(module, torch.nn.LayerNorm):
                modules_to_replace[name] = GGMLLayerNorm(
                    module.normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine
                )
            elif isinstance(module, torch.nn.GroupNorm):
                modules_to_replace[name] = GGMLGroupNorm(
                    module.num_groups, module.num_channels, eps=module.eps, affine=module.affine
                )

    # Apply replacements
    for name, new_module in modules_to_replace.items():
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        setattr(parent, child_name, new_module)
        
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        
        if weight_key in normalized_sd:
            # We set requires_grad=False to ensure safety
            new_module.weight = torch.nn.Parameter(normalized_sd[weight_key], requires_grad=False)
        
        if bias_key in normalized_sd and new_module.bias is not None:
            new_module.bias = torch.nn.Parameter(normalized_sd[bias_key], requires_grad=False)

    logger.info(f"Patched {len(modules_to_replace)} layers with GGUF tensors.")
    return model

# --- Worker Logic ---

class InferenceWorker:
    def __init__(self):
        self.pipeline = None
        self.current_base_model = None
        self.current_gguf_path = None
        self.current_device = None
        self.current_policy = None

    def resolve_device(self, device_request):
        if device_request == "auto" or not device_request:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device_request == "cpu":
            return "cpu"
        return device_request

    def load_model(self, base_model_id, gguf_path, device_request, vram_policy):
        target_device = self.resolve_device(device_request)
        
        if (self.pipeline and 
            self.current_base_model == base_model_id and 
            self.current_gguf_path == gguf_path and
            self.current_device == target_device and
            self.current_policy == vram_policy):
            logger.info("Model already loaded with correct settings.")
            return

        logger.info(f"Loading base model: {base_model_id} on {target_device} with policy {vram_policy}")
        
        if self.pipeline:
            del self.pipeline
            torch.cuda.empty_cache()
            self.pipeline = None

        if "flux" in base_model_id.lower():
            # Initial load on CPU
            pipe = FluxPipeline.from_pretrained(
                base_model_id, 
                torch_dtype=torch.bfloat16 if target_device != "cpu" else torch.float32,
                transformer=None 
            )
            from diffusers import FluxTransformer2DModel
            transformer_config = FluxTransformer2DModel.load_config(base_model_id, subfolder="transformer")
            with torch.device("cpu"): 
                pipe.transformer = FluxTransformer2DModel.from_config(transformer_config)
                pipe.transformer.to(dtype=torch.bfloat16 if target_device != "cpu" else torch.float32)
        else:
            raise ValueError("Only Flux models supported in this MVP.")

        logger.info(f"Loading GGUF tensors from: {gguf_path}")
        gguf_tensors = load_gguf_tensors(gguf_path)
        
        logger.info("Patching model...")
        pipe.transformer = patch_model(pipe.transformer, gguf_tensors)
        
        if vram_policy == "cpu_mode":
            target_device = "cpu"
            pipe.to("cpu")
        elif vram_policy == "offload":
            logger.info(f"Enabling Model CPU Offload to {target_device}")
            pipe.enable_model_cpu_offload(device=target_device)
        elif vram_policy == "sequential":
            logger.info(f"Enabling Sequential CPU Offload to {target_device}")
            pipe.enable_sequential_cpu_offload(device=target_device)
        else: 
            logger.info(f"Moving full model to {target_device}")
            pipe.to(target_device)

        self.pipeline = pipe
        self.current_base_model = base_model_id
        self.current_gguf_path = gguf_path
        self.current_device = target_device
        self.current_policy = vram_policy
        
        logger.info("Model loaded successfully.")

    def generate(self, params):
        self.load_model(
            params["base_model_id"], 
            params["gguf_path"], 
            params.get("device", "auto"),
            params.get("vram_policy", "regular")
        )
        
        steps = params.get("steps", 20)
        
        output = self.pipeline(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", ""),
            width=params["width"],
            height=params["height"],
            num_inference_steps=steps,
            guidance_scale=params["guidance_scale"],
            output_type="pil"
        )
        
        return output.images[0]

worker = InferenceWorker()

def process_queue():
    while True:
        # Unpack tuple with action type
        future, params, action = task_queue.get()
        try:
            if action == "load":
                worker.load_model(
                    params["base_model_id"], 
                    params["gguf_path"], 
                    params.get("device", "auto"),
                    params.get("vram_policy", "regular")
                )
                future["result"] = "ok"
            elif action == "generate":
                result = worker.generate(params)
                future["result"] = result
            
            future["event"].set()
        except Exception as e:
            logger.error(f"Worker error: {e}")
            future["error"] = str(e)
            future["event"].set()
        task_queue.task_done()

threading.Thread(target=process_queue, daemon=True).start()

# --- API ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/load")
def load_endpoint(request: Request, params: dict):
    future = {"event": threading.Event()}
    # Enqueue with "load" action
    task_queue.put((future, params, "load"))
    
    # Wait for completion (model loading is synchronous in worker)
    future["event"].wait()
    
    if "error" in future:
        return JSONResponse(status_code=500, content={"error": future["error"]})
        
    return {"status": "loaded"}

@app.post("/generate")
def generate_endpoint(request: Request, params: dict):
    future = {"event": threading.Event()}
    # Enqueue with "generate" action
    task_queue.put((future, params, "generate"))
    
    future["event"].wait()
    
    if "error" in future:
        return JSONResponse(status_code=500, content={"error": future["error"]})
        
    image = future["result"]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8182)
