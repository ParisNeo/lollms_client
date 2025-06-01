# lollms_client/ttm_bindings/bark/__init__.py
import io
import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from ascii_colors import trace_exception, ASCIIColors

# --- Package Management and Conditional Imports ---
_bark_deps_installed_with_correct_torch = False
_bark_installation_error = ""
try:
    import pipmaster as pm
    import platform

    preferred_torch_device_for_install = "cpu"
    if platform.system() == "Linux" or platform.system() == "Windows":
        preferred_torch_device_for_install = "cuda"
    elif platform.system() == "Darwin":
        preferred_torch_device_for_install = "mps"

    torch_pkgs = ["torch"]
    bark_core_pkgs = ["transformers", "accelerate", "sentencepiece"]
    other_deps = ["scipy", "numpy"]

    torch_index_url = None
    if preferred_torch_device_for_install == "cuda":
        torch_index_url = "https://download.pytorch.org/whl/cu126"
        ASCIIColors.info(f"Attempting to ensure PyTorch with CUDA support (target index: {torch_index_url}) for Bark binding.")
        pm.ensure_packages(torch_pkgs, index_url=torch_index_url)
        pm.ensure_packages(bark_core_pkgs + other_deps)
    else:
        ASCIIColors.info("Ensuring PyTorch, Bark dependencies, and others using default PyPI index for Bark binding.")
        pm.ensure_packages(torch_pkgs + bark_core_pkgs + other_deps)

    import torch
    from transformers import AutoProcessor, BarkModel, GenerationConfig 
    import scipy.io.wavfile
    import numpy as np

    _bark_deps_installed_with_correct_torch = True
except Exception as e:
    _bark_installation_error = str(e)
    AutoProcessor, BarkModel, GenerationConfig, torch, scipy, np = None, None, None, None, None, None
# --- End Package Management ---

from lollms_client.lollms_ttm_binding import LollmsTTMBinding

BindingName = "BarkTTMBinding"

DEFAULT_BARK_MODELS = [
    "suno/bark",
    "suno/bark-small",
]

BARK_VOICE_PRESETS_EXAMPLES = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
    "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
    "v2/en_speaker_8", "v2/en_speaker_9",
    "v2/de_speaker_0", "v2/es_speaker_0", "v2/fr_speaker_0",
]


class BarkTTMBinding(LollmsTTMBinding):
    def __init__(self,
                 model_name: str = "suno/bark-small",
                 device: Optional[str] = None,
                 default_voice_preset: Optional[str] = "v2/en_speaker_6",
                 enable_better_transformer: bool = True,
                 **kwargs): 

        super().__init__(binding_name="bark")

        if not _bark_deps_installed_with_correct_torch:
            raise ImportError(f"Bark TTM binding dependencies not met. Error: {_bark_installation_error}")

        self.device = device
        if self.device is None:
            if torch.cuda.is_available(): self.device = "cuda"; ASCIIColors.info("CUDA device detected by PyTorch for Bark.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): self.device = "mps"; ASCIIColors.info("MPS device detected for Bark.")
            else: self.device = "cpu"; ASCIIColors.info("No GPU (CUDA/MPS) by PyTorch, using CPU for Bark.")
        elif self.device == "cuda" and not torch.cuda.is_available(): self.device = "cpu"; ASCIIColors.warning("CUDA req, not avail. CPU for Bark.")
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()): self.device = "cpu"; ASCIIColors.warning("MPS req, not avail. CPU for Bark.")

        ASCIIColors.info(f"BarkTTMBinding: Using device '{self.device}'.")

        self.loaded_model_name = None
        self.model: Optional[BarkModel] = None
        self.processor: Optional[AutoProcessor] = None
        self.default_voice_preset = default_voice_preset
        self.enable_better_transformer = enable_better_transformer
        
        self.default_generation_params = {}
        temp_gen_config = GenerationConfig() 
        for key, value in kwargs.items():
            if hasattr(temp_gen_config, key):
                self.default_generation_params[key] = value

        self._load_bark_model(model_name)

    def _load_bark_model(self, model_name_to_load: str):
        if self.model is not None and self.loaded_model_name == model_name_to_load:
            ASCIIColors.info(f"Bark model '{model_name_to_load}' already loaded.")
            return

        ASCIIColors.info(f"Loading Bark model: '{model_name_to_load}' on device '{self.device}'...")
        try:
            dtype_for_bark = torch.float16 if self.device == "cuda" else None 

            self.processor = AutoProcessor.from_pretrained(model_name_to_load)
            self.model = BarkModel.from_pretrained(
                model_name_to_load,
                torch_dtype=dtype_for_bark,
                low_cpu_mem_usage=True if self.device != "cpu" else False
            ).to(self.device)

            if self.enable_better_transformer and self.device == "cuda":
                try:
                    self.model = self.model.to_bettertransformer()
                    ASCIIColors.info("Applied BetterTransformer optimization to Bark model.")
                except Exception as e_bt:
                    ASCIIColors.warning(f"Failed to apply BetterTransformer: {e_bt}. Proceeding without it.")
            
            if "small" not in model_name_to_load and self.device=="cpu":
                ASCIIColors.warning("Using full Bark model on CPU. Generation might be slow.")
            elif self.device != "cpu" and "small" not in model_name_to_load:
                if hasattr(self.model, "enable_model_cpu_offload"):
                    try: self.model.enable_model_cpu_offload(); ASCIIColors.info("Enabled model_cpu_offload for Bark.")
                    except Exception as e: ASCIIColors.warning(f"Could not enable model_cpu_offload: {e}")
                elif hasattr(self.model, "enable_cpu_offload"):
                    try: self.model.enable_cpu_offload(); ASCIIColors.info("Enabled cpu_offload for Bark (older API).")
                    except Exception as e: ASCIIColors.warning(f"Could not enable cpu_offload (older API): {e}")
                else: ASCIIColors.info("CPU offload not explicitly enabled.")

            self.loaded_model_name = model_name_to_load
            ASCIIColors.green(f"Bark model '{model_name_to_load}' loaded successfully.")
        except Exception as e:
            self.model, self.processor, self.loaded_model_name = None, None, None
            ASCIIColors.error(f"Failed to load Bark model '{model_name_to_load}': {e}"); trace_exception(e)
            raise RuntimeError(f"Failed to load Bark model '{model_name_to_load}'") from e

    def generate_music(self,
                       prompt: str,
                       voice_preset: Optional[str] = None,
                       do_sample: Optional[bool] = None, 
                       temperature: Optional[float] = None,
                       **kwargs) -> bytes:
        if self.model is None or self.processor is None:
            raise RuntimeError("Bark model or processor not loaded.")

        effective_voice_preset = voice_preset if voice_preset is not None else self.default_voice_preset
        
        ASCIIColors.info(f"Generating SFX/audio with Bark: '{prompt[:60]}...' (Preset: {effective_voice_preset})")
        try:
            # The processor correctly returns 'input_ids' and 'attention_mask'
            inputs = self.processor(
                text=[prompt], # Processor expects a list of texts
                voice_preset=effective_voice_preset,
                return_tensors="pt",
                # Explicitly ask for padding if tokenizer supports it,
                # though Bark's processor might handle this internally.
                # padding=True, # Let processor decide best padding strategy
                # truncation=True # Ensure inputs fit model context
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Ensure attention_mask is present
            if 'attention_mask' not in inputs:
                ASCIIColors.warning("Processor did not return attention_mask. Creating a default one (all ones). This might lead to suboptimal results if padding was intended.")
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])


            if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
                gen_config = GenerationConfig.from_dict(self.model.generation_config.to_dict())
            else:
                gen_config = GenerationConfig()

            for key, value in self.default_generation_params.items():
                if hasattr(gen_config, key): setattr(gen_config, key, value)

            if do_sample is not None: gen_config.do_sample = do_sample
            
            if temperature is not None:
                if 'semantic_temperature' not in kwargs and hasattr(gen_config, 'semantic_temperature'): gen_config.semantic_temperature = temperature 
                if 'coarse_temperature' not in kwargs and hasattr(gen_config, 'coarse_temperature'): gen_config.coarse_temperature = temperature
                if 'fine_temperature' not in kwargs and hasattr(gen_config, 'fine_temperature'): gen_config.fine_temperature = temperature
            
            for key, value in kwargs.items():
                if hasattr(gen_config, key): setattr(gen_config, key, value)

            # Critical: Set pad_token_id in GenerationConfig.
            # Bark uses specific token IDs for its different codebooks.
            # The processor's tokenizer should have the correct pad_token_id if it's used for text inputs.
            # For Bark, the semantic vocabulary has its own pad_token_id, often same as EOS.
            # Let's try to get it from the model's semantic config or text config.
            pad_token_id_to_set = None
            if hasattr(self.model.config, 'semantic_config') and hasattr(self.model.config.semantic_config, 'pad_token_id'):
                pad_token_id_to_set = self.model.config.semantic_config.pad_token_id
            elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'pad_token_id'):
                 pad_token_id_to_set = self.model.config.text_config.pad_token_id
            elif hasattr(self.processor, 'tokenizer') and self.processor.tokenizer and self.processor.tokenizer.pad_token_id is not None:
                pad_token_id_to_set = self.processor.tokenizer.pad_token_id
            
            if pad_token_id_to_set is not None:
                gen_config.pad_token_id = pad_token_id_to_set
                # Also set EOS token if it's distinct and meaningful for generation stopping
                if hasattr(gen_config, 'eos_token_id') and gen_config.eos_token_id is None:
                    eos_id = None
                    if hasattr(self.model.config, 'semantic_config') and hasattr(self.model.config.semantic_config, 'eos_token_id'):
                         eos_id = self.model.config.semantic_config.eos_token_id
                    if eos_id is not None:
                         gen_config.eos_token_id = eos_id

            else:
                # This state is problematic for Bark if pad_token_id is truly needed and distinct from EOS
                ASCIIColors.warning("Could not determine a specific pad_token_id from Bark's config for GenerationConfig. This might lead to issues.")
                # If eos_token_id is also not set, generation might not stop correctly.
                # Defaulting pad_token_id to eos_token_id if eos_token_id exists.
                if gen_config.eos_token_id is not None:
                    gen_config.pad_token_id = gen_config.eos_token_id
                    ASCIIColors.info(f"Setting pad_token_id to eos_token_id ({gen_config.eos_token_id}) as a fallback.")
                else:
                    # This is a last resort and might not be correct for Bark specifically
                    gen_config.pad_token_id = 0 
                    ASCIIColors.warning("pad_token_id defaulted to 0 as a last resort.")


            ASCIIColors.debug(f"Bark final generation_config: {gen_config.to_json_string()}")

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs['input_ids'], # Explicitly pass input_ids
                    attention_mask=inputs.get('attention_mask'), # Pass attention_mask if available
                    generation_config=gen_config
                )

            if isinstance(output, torch.Tensor): speech_output_tensor = output
            elif isinstance(output, dict) and "audio_features" in output: speech_output_tensor = output["audio_features"]
            elif isinstance(output, dict) and "waveform" in output: speech_output_tensor = output["waveform"] # Bark might return this key
            else: raise TypeError(f"Unexpected output type from BarkModel.generate: {type(output)}. Content: {output}")

            audio_array_np = speech_output_tensor.cpu().numpy().squeeze()
            if audio_array_np.ndim == 0 or audio_array_np.size == 0:
                raise RuntimeError("Bark model returned empty audio data.")

            audio_int16 = (audio_array_np * 32767).astype(np.int16)

            buffer = io.BytesIO()
            sample_rate_to_use = int(self.model.generation_config.sample_rate if hasattr(self.model.generation_config, 'sample_rate') and self.model.generation_config.sample_rate else 24_000)
            scipy.io.wavfile.write(buffer, rate=sample_rate_to_use, data=audio_int16)
            audio_bytes = buffer.getvalue()
            buffer.close()

            ASCIIColors.green("Bark audio generation successful.")
            return audio_bytes
        except Exception as e:
            ASCIIColors.error(f"Bark audio generation failed: {e}"); trace_exception(e)
            if "out of memory" in str(e).lower() and self.device == "cuda":
                 ASCIIColors.yellow("CUDA out of memory. Consider using suno/bark-small or ensure GPU has sufficient VRAM.")
            raise RuntimeError(f"Bark audio generation error: {e}") from e

    def list_models(self, **kwargs) -> List[str]:
        return DEFAULT_BARK_MODELS.copy()

    def list_voice_presets(self) -> List[str]:
        return BARK_VOICE_PRESETS_EXAMPLES.copy()

    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model; self.model = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor; self.processor = None
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            loaded_name = getattr(self, 'loaded_model_name', None)
            msg = f"BarkTTMBinding for model '{loaded_name}' destroyed." if loaded_name else "BarkTTMBinding destroyed."
            ASCIIColors.info(msg)

# --- Main Test Block ---
if __name__ == '__main__':
    if not _bark_deps_installed_with_correct_torch:
        print(f"{ASCIIColors.RED}Bark TTM binding dependencies not met. Skipping tests. Error: {_bark_installation_error}{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- BarkTTMBinding Test ---")
    test_model_id = "suno/bark-small"
    test_output_dir = Path("./test_bark_sfx_output")
    test_output_dir.mkdir(exist_ok=True)
    ttm_binding = None

    try:
        ASCIIColors.cyan(f"\n--- Initializing BarkTTMBinding (model: '{test_model_id}') ---")
        ttm_binding = BarkTTMBinding(model_name=test_model_id)

        ASCIIColors.cyan("\n--- Listing common Bark models ---")
        models = ttm_binding.list_models(); print(f"Common Bark models: {models}")
        ASCIIColors.cyan("\n--- Listing example Bark voice presets ---")
        presets = ttm_binding.list_voice_presets(); print(f"Example presets: {presets[:5]}...")

        sfx_prompts_to_test = [
            ("laser_blast", "A short, sharp laser blast sound effect [SFX]"),
            ("footsteps_gravel", "Footsteps walking on gravel [footsteps]."),
            ("explosion_distant", "A distant explosion [boom] with a slight echo."),
            ("interface_click", "A clean, quick digital interface click sound. [click]"),
            ("creature_roar_short", "[roar] A short, guttural creature roar."),
            ("ambient_wind", "[wind] Gentle wind blowing through trees."),
            ("speech_hello", "Hello, this is a test of Bark's speech capabilities."),
        ]

        for name, prompt in sfx_prompts_to_test:
            ASCIIColors.cyan(f"\n--- Generating SFX/Audio for: '{name}' ---"); print(f"Prompt: {prompt}")
            try:
                call_kwargs = {}
                if "speech" in name:
                    call_kwargs = {"semantic_temperature": 0.6, "coarse_temperature": 0.8, "fine_temperature": 0.5, "do_sample": True}
                elif name == "laser_blast":
                    call_kwargs = {"semantic_temperature": 0.5, "coarse_temperature": 0.6, "fine_temperature": 0.4, "do_sample": True}
                else: # For SFX, sometimes more deterministic sampling helps for consistency
                    call_kwargs = {"do_sample": True, "semantic_temperature": 0.7, "coarse_temperature": 0.7, "fine_temperature": 0.7}


                sfx_bytes = ttm_binding.generate_music(prompt, voice_preset=None, **call_kwargs)
                if sfx_bytes:
                    output_filename = f"sfx_{name}_{test_model_id.split('/')[-1]}.wav"
                    output_path = test_output_dir / output_filename
                    with open(output_path, "wb") as f: f.write(sfx_bytes)
                    ASCIIColors.green(f"SFX for '{name}' saved to: {output_path} ({len(sfx_bytes) / 1024:.2f} KB)")
                else: ASCIIColors.error(f"SFX generation for '{name}' returned empty bytes.")
            except Exception as e_gen: ASCIIColors.error(f"Failed to generate SFX for '{name}': {e_gen}")

    except ImportError as e_imp: ASCIIColors.error(f"Import error: {e_imp}")
    except RuntimeError as e_rt: ASCIIColors.error(f"Runtime error: {e_rt}")
    except Exception as e: ASCIIColors.error(f"Unexpected error: {e}"); trace_exception(e)
    finally:
        if ttm_binding: del ttm_binding
        ASCIIColors.info(f"Test SFX (if any) are in: {test_output_dir.resolve()}")
        print(f"{ASCIIColors.YELLOW}Check the audio files in '{test_output_dir.resolve()}'!{ASCIIColors.RESET}")

    ASCIIColors.yellow("\n--- BarkTTMBinding Test Finished ---")