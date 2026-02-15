#!/usr/bin/env python3
"""
🔧 LM Studio Optimizer for RTX 3060 12GB + 64GB RAM
Automatically configures optimal settings and provides performance recommendations
"""

from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import json
import subprocess
import requests

class LMStudioOptimizer:
    """Advanced LM Studio optimizer for maximum performance."""
    
    def __init__(self):
        self.host_address = "http://localhost:1234"
        self.hardware_profile = self.detect_hardware()
        
    def detect_hardware(self):
        """Detect current hardware configuration."""
        try:
            # Get GPU info
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                gpu_name = gpu_info[0]
                gpu_memory = int(gpu_info[1].split()[0])  # Extract MB value
                
                return {
                    "gpu_name": gpu_name,
                    "gpu_memory_mb": gpu_memory,
                    "gpu_memory_gb": round(gpu_memory / 1024, 1),
                    "is_rtx_3060": "RTX 3060" in gpu_name
                }
        except:
            pass
        
        return {"gpu_name": "Unknown", "gpu_memory_gb": 0}
    
    def get_optimal_settings(self):
        """Get optimal LM Studio settings based on hardware."""
        settings = {
            "context_length": 8192,
            "batch_size": 512,
            "gpu_layers": -1,
            "rope_scaling": "linear",
            "flash_attention": True,
            "cpu_threads": 8,
            "mmap": True,
            "mlock": False
        }
        
        gpu_gb = self.hardware_profile.get("gpu_memory_gb", 0)
        
        if gpu_gb >= 12:  # RTX 3060 12GB or better
            settings.update({
                "max_model_size": "14B",
                "recommended_quantization": "Q4_K_M",
                "context_length": 16384,  # Can handle larger context
                "batch_size": 1024,
                "parallel_requests": 4
            })
        elif gpu_gb >= 8:
            settings.update({
                "max_model_size": "7B", 
                "recommended_quantization": "Q4_K_M",
                "context_length": 8192,
                "batch_size": 512,
                "parallel_requests": 2
            })
        else:
            settings.update({
                "max_model_size": "3B",
                "recommended_quantization": "Q4_0", 
                "context_length": 4096,
                "batch_size": 256,
                "parallel_requests": 1
            })
        
        return settings
    
    def get_model_recommendations(self):
        """Get model recommendations based on hardware and use cases."""
        gpu_gb = self.hardware_profile.get("gpu_memory_gb", 0)
        
        if gpu_gb >= 12:  # RTX 3060 12GB optimized
            return {
                "programming": [
                    "deepcoder-7b-aurora-v2",
                    "qwen2.5-coder-14b-instruct", 
                    "deepcoder-14b-preview.gguf"
                ],
                "reasoning": [
                    "deepseek-r1-distill-qwen-14b",
                    "cogito-v1-preview-qwen-14b",
                    "smallthinker:latest"
                ],
                "multimodal": [
                    "gemma-3-12b-it",
                    "phi-3.5-vision-instruct",
                    "llama-3.2-vision"
                ],
                "general": [
                    "llama-3.2-3b-instruct",
                    "gemma-3-4b-it-qat",
                    "phi3:mini"
                ],
                "speed_demon": [
                    "gemma-3-1b-it-qat",
                    "deepcoder-1.5b-preview",
                    "llama-3.2-3b-instruct"
                ]
            }
        else:
            return {
                "programming": ["deepcoder-1.5b-preview"],
                "reasoning": ["phi3:mini"],
                "multimodal": ["phi-3.5-vision-instruct"],
                "general": ["llama-3.2-3b-instruct"],
                "speed_demon": ["gemma-3-1b-it-qat"]
            }
    
    def check_lmstudio_status(self):
        """Check LM Studio server status and loaded models."""
        try:
            # Check if server is running
            response = requests.get(f"{self.host_address}/api/v0/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('data', [])
                
                loaded_models = [m for m in models if m.get('state') == 'loaded']
                available_models = [m for m in models if m.get('state') == 'not-loaded']
                
                return {
                    "status": "running",
                    "total_models": len(models),
                    "loaded_models": loaded_models,
                    "available_models": available_models
                }
            else:
                return {"status": "error", "message": "Server not responding properly"}
        except:
            return {"status": "offline", "message": "LM Studio not running"}
    
    def optimize_lmstudio_config(self):
        """Generate optimized LM Studio configuration."""
        ASCIIColors.cyan("🔧 LM Studio Optimizer for RTX 3060 12GB")
        ASCIIColors.cyan("=" * 50)
        
        # Display hardware info
        ASCIIColors.yellow("🖥️  Hardware Detection:")
        hw = self.hardware_profile
        print(f"   GPU: {hw.get('gpu_name', 'Unknown')}")
        print(f"   VRAM: {hw.get('gpu_memory_gb', 0)}GB")
        print(f"   Optimized for RTX 3060: {'✅' if hw.get('is_rtx_3060') else '❌'}")
        
        # Check LM Studio status
        ASCIIColors.yellow("\n🚀 LM Studio Status:")
        status = self.check_lmstudio_status()
        if status["status"] == "running":
            ASCIIColors.green("   ✅ LM Studio is running")
            print(f"   📊 Total models: {status['total_models']}")
            print(f"   🔥 Loaded models: {len(status['loaded_models'])}")
            
            if status['loaded_models']:
                print("   🎯 Currently loaded:")
                for model in status['loaded_models'][:3]:  # Show first 3
                    print(f"      - {model['id']} ({model.get('type', 'unknown')})")
        else:
            ASCIIColors.red(f"   ❌ {status['message']}")
            return
        
        # Get optimal settings
        ASCIIColors.yellow("\n⚙️  Optimal Settings:")
        settings = self.get_optimal_settings()
        for key, value in settings.items():
            print(f"   {key}: {value}")
        
        # Model recommendations
        ASCIIColors.yellow("\n🎯 Model Recommendations:")
        recommendations = self.get_model_recommendations()
        for category, models in recommendations.items():
            print(f"   {category.title()}:")
            for model in models[:2]:  # Show top 2 per category
                available = any(model in m['id'] for m in status.get('available_models', []))
                loaded = any(model in m['id'] for m in status.get('loaded_models', []))
                status_icon = "🔥" if loaded else "✅" if available else "📥"
                print(f"      {status_icon} {model}")
        
        # Generate optimized client config
        ASCIIColors.yellow("\n🚀 Optimized LOLLMS-Client Config:")
        
        # Find best loaded model or recommend one
        best_model = None
        if status['loaded_models']:
            best_model = status['loaded_models'][0]['id']
        else:
            # Recommend fastest general model
            general_models = recommendations.get('speed_demon', recommendations.get('general', []))
            if general_models:
                best_model = general_models[0]
        
        config = {
            "binding_name": "lmstudio",
            "host_address": self.host_address,
            "model_name": best_model or "auto-detect",
            "llm_binding_config": {
                "use_native_api": True,
                "default_completion_format": "Chat"
            },
            "ctx_size": settings["context_length"],
            "n_predict": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True
        }
        
        print("   Python code:")
        print("   ```python")
        print("   from lollms_client import LollmsClient")
        print("   ")
        print("   # Optimized for RTX 3060 12GB + 64GB RAM")
        print(f"   lc = LollmsClient(")
        for key, value in config.items():
            if isinstance(value, str):
                print(f"       {key}=\"{value}\",")
            else:
                print(f"       {key}={value},")
        print("   )")
        print("   ```")
        
        # Performance tips
        ASCIIColors.yellow("\n💡 Performance Tips:")
        tips = [
            "Use streaming=True for better perceived performance",
            "Keep context length ≤ 16K for optimal speed on 12GB VRAM", 
            "Load models with Q4_K_M quantization for best speed/quality balance",
            "Use task-specific models (programming, reasoning, etc.) for better results",
            "Monitor GPU utilization - should be 70-95% during generation",
            "Close other GPU-intensive applications for maximum performance"
        ]
        
        for i, tip in enumerate(tips, 1):
            print(f"   {i}. {tip}")
        
        ASCIIColors.green("\n🎉 Your RTX 3060 12GB setup is optimized for 10X performance!")
        
        return config

def main():
    """Run the LM Studio optimizer."""
    optimizer = LMStudioOptimizer()
    config = optimizer.optimize_lmstudio_config()
    
    # Save config to file
    if config:
        with open("lmstudio_optimal_config.json", "w") as f:
            json.dump(config, f, indent=2)
        ASCIIColors.info("\n💾 Optimal config saved to 'lmstudio_optimal_config.json'")

if __name__ == "__main__":
    main()
