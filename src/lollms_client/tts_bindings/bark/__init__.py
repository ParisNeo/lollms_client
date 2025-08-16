# lollms_client/tts_bindings/bark/__init__.py
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

    torch_pkgs = ["torch", "torchaudio","xformers"]
    bark_core_pkgs = ["transformers", "accelerate", "sentencepiece"]
    other_deps = ["scipy", "numpy"]

    torch_index_url = None
    if preferred_torch_device_for_install == "cuda":
        torch_index_url = "https://download.pytorch.org/whl/cu126"
        ASCIIColors.info(f"Attempting to ensure PyTorch with CUDA support (target index: {torch_index_url}) for Bark TTS binding.")
        pm.ensure_packages(torch_pkgs, index_url=torch_index_url)
        pm.ensure_packages(bark_core_pkgs + other_deps)
    else:
        ASCIIColors.info("Ensuring PyTorch, Bark dependencies, and others using default PyPI index for Bark TTS binding.")
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

from lollms_client.lollms_tts_binding import LollmsTTSBinding # Changed base class

BindingName = "BarkTTSBinding" # Changed BindingName

# Bark model IDs (can be used as 'model_name' for this binding)
BARK_MODELS = [
    "suno/bark",        # Full model
    "suno/bark-small",  # Smaller, faster model
]

# Bark voice presets, used as the 'voice' argument in generate_audio
BARK_VOICE_PRESETS = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3",
    "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", "v2/en_speaker_7",
    "v2/en_speaker_8", "v2/en_speaker_9",
    "v2/de_speaker_0", "v2/es_speaker_0", "v2/fr_speaker_0", "v2/hi_speaker_0",
    "v2/it_speaker_0", "v2/ja_speaker_0", "v2/ko_speaker_0", "v2/pl_speaker_0",
    "v2/pt_speaker_0", "v2/ru_speaker_0", "v2/tr_speaker_0", "v2/zh_speaker_0",
    # Non-speech sounds (less relevant for pure TTS, but part of Bark's capabilities)
    "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]", "[clears throat]",
    "♪", "...", "[MAN]", "[WOMAN]" # Special tokens
]


class BarkTTSBinding(LollmsTTSBinding): # Changed class name and base class
    def __init__(self,
                 model_name: str = "suno/bark-small", # This is the Bark model ID
                 default_voice: Optional[str] = "v2/en_speaker_6", # This is the default voice_preset
                 device: Optional[str] = None,
                 enable_better_transformer: bool = True,
                 host_address: Optional[str] = None, # Unused for local binding
                 service_key: Optional[str] = None, # Unused for local binding
                 verify_ssl_certificate: bool = True, # Unused for local binding
                 **kwargs):

        super().__init__(binding_name="bark") # Call LollmsTTSBinding's init

        if not _bark_deps_installed_with_correct_torch:
            raise ImportError(f"Bark TTS binding dependencies not met. Error: {_bark_installation_error}")

        self.device = device
        if self.device is None:
            if torch.cuda.is_available(): self.device = "cuda"; ASCIIColors.info("CUDA device detected by PyTorch for Bark TTS.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): self.device = "mps"; ASCIIColors.info("MPS device detected for Bark TTS.")
            else: self.device = "cpu"; ASCIIColors.info("No GPU (CUDA/MPS) by PyTorch, using CPU for Bark TTS.")
        elif self.device == "cuda" and not torch.cuda.is_available(): self.device = "cpu"; ASCIIColors.warning("CUDA req, not avail. CPU for Bark TTS.")
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()): self.device = "cpu"; ASCIIColors.warning("MPS req, not avail. CPU for Bark TTS.")

        ASCIIColors.info(f"BarkTTSBinding: Using device '{self.device}'.")

        self.bark_model_id = model_name # Store the actual Bark model ID separately
        self.loaded_bark_model_id = None
        self.model: Optional[BarkModel] = None
        self.processor: Optional[AutoProcessor] = None
        self.default_voice_preset = default_voice # Renamed for clarity in TTS context
        self.enable_better_transformer = enable_better_transformer
        
        self.default_generation_params = {}
        temp_gen_config = GenerationConfig()
        for key, value in kwargs.items():
            if hasattr(temp_gen_config, key):
                self.default_generation_params[key] = value

        self._load_bark_model(self.bark_model_id)

    def _load_bark_model(self, model_id_to_load: str):
        if self.model is not None and self.loaded_bark_model_id == model_id_to_load:
            ASCIIColors.info(f"Bark model '{model_id_to_load}' already loaded.")
            return

        ASCIIColors.info(f"Loading Bark model for TTS: '{model_id_to_load}' on device '{self.device}'...")
        try:
            dtype_for_bark = torch.float16 if self.device == "cuda" else None

            self.processor = AutoProcessor.from_pretrained(model_id_to_load)
            self.model = BarkModel.from_pretrained(
                model_id_to_load,
                torch_dtype=dtype_for_bark,
                low_cpu_mem_usage=True if self.device != "cpu" else False
            ).to(self.device)

            if self.enable_better_transformer and self.device == "cuda":
                try:
                    self.model = self.model.to_bettertransformer()
                    ASCIIColors.info("Applied BetterTransformer optimization to Bark model.")
                except Exception as e_bt:
                    ASCIIColors.warning(f"Failed to apply BetterTransformer: {e_bt}. Proceeding without it.")
            
            # (CPU offload logic remains the same)
            if "small" not in model_id_to_load and self.device=="cpu":
                ASCIIColors.warning("Using full Bark model on CPU. Generation might be slow.")
            elif self.device != "cpu" and "small" not in model_id_to_load:
                if hasattr(self.model, "enable_model_cpu_offload"):
                    try: self.model.enable_model_cpu_offload(); ASCIIColors.info("Enabled model_cpu_offload for Bark.")
                    except Exception as e: ASCIIColors.warning(f"Could not enable model_cpu_offload: {e}")
                elif hasattr(self.model, "enable_cpu_offload"):
                    try: self.model.enable_cpu_offload(); ASCIIColors.info("Enabled cpu_offload for Bark (older API).")
                    except Exception as e: ASCIIColors.warning(f"Could not enable cpu_offload (older API): {e}")
                else: ASCIIColors.info("CPU offload not explicitly enabled.")

            self.loaded_bark_model_id = model_id_to_load
            ASCIIColors.green(f"Bark model '{model_id_to_load}' for TTS loaded successfully.")
        except Exception as e:
            self.model, self.processor, self.loaded_bark_model_id = None, None, None
            ASCIIColors.error(f"Failed to load Bark model '{model_id_to_load}': {e}"); trace_exception(e)
            raise RuntimeError(f"Failed to load Bark model '{model_id_to_load}'") from e

    def generate_audio(self,
                       text: str,
                       voice: Optional[str] = None, # This will be the Bark voice_preset
                       do_sample: Optional[bool] = True, # Default to True for more natural speech
                       temperature: Optional[float] = 0.7, # General speech temperature
                       **kwargs) -> bytes:
        if self.model is None or self.processor is None:
            raise RuntimeError("Bark model or processor not loaded.")

        effective_voice_preset = voice if voice is not None else self.default_voice_preset
        if effective_voice_preset not in BARK_VOICE_PRESETS and not Path(effective_voice_preset).exists():
            ASCIIColors.warning(f"Voice preset '{effective_voice_preset}' not in known presets. Bark will attempt to use it as is.")
        
        ASCIIColors.info(f"Generating speech with Bark: '{text[:60]}...' (Voice Preset: {effective_voice_preset})")
        try:
            inputs = self.processor(text=[text], voice_preset=effective_voice_preset, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
                gen_config = GenerationConfig.from_dict(self.model.generation_config.to_dict())
            else:
                gen_config = GenerationConfig()

            for key, value in self.default_generation_params.items():
                if hasattr(gen_config, key): setattr(gen_config, key, value)

            # For TTS, do_sample is usually True
            gen_config.do_sample = do_sample if do_sample is not None else True
            
            # Apply general temperature hint for TTS
            if temperature is not None:
                # Bark's main temperatures for speech quality are often coarse and fine.
                # Semantic temperature can also play a role.
                if 'semantic_temperature' not in kwargs and hasattr(gen_config, 'semantic_temperature'):
                    gen_config.semantic_temperature = kwargs.get("semantic_temperature", temperature)
                if 'coarse_temperature' not in kwargs and hasattr(gen_config, 'coarse_temperature'):
                    gen_config.coarse_temperature = kwargs.get("coarse_temperature", temperature)
                if 'fine_temperature' not in kwargs and hasattr(gen_config, 'fine_temperature'):
                    gen_config.fine_temperature = kwargs.get("fine_temperature", temperature * 0.8) # Fine is often lower
            
            for key, value in kwargs.items():
                if hasattr(gen_config, key): setattr(gen_config, key, value)

            pad_token_id_to_set = None
            # (pad_token_id logic remains the same)
            if hasattr(self.model.config, 'semantic_config') and hasattr(self.model.config.semantic_config, 'pad_token_id'):
                pad_token_id_to_set = self.model.config.semantic_config.pad_token_id
            elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'pad_token_id'):
                 pad_token_id_to_set = self.model.config.text_config.pad_token_id
            elif hasattr(self.processor, 'tokenizer') and self.processor.tokenizer and self.processor.tokenizer.pad_token_id is not None:
                pad_token_id_to_set = self.processor.tokenizer.pad_token_id
            
            if pad_token_id_to_set is not None:
                gen_config.pad_token_id = pad_token_id_to_set
                if hasattr(gen_config, 'eos_token_id') and gen_config.eos_token_id is None:
                    eos_id = getattr(getattr(self.model.config, 'semantic_config', None), 'eos_token_id', None)
                    if eos_id is not None: gen_config.eos_token_id = eos_id
            else:
                ASCIIColors.warning("Could not determine pad_token_id for Bark TTS. Using default in GenerationConfig.")
                if gen_config.eos_token_id is not None and gen_config.pad_token_id is None:
                     gen_config.pad_token_id = gen_config.eos_token_id
                elif gen_config.pad_token_id is None:
                     gen_config.pad_token_id = 0

            ASCIIColors.debug(f"Bark TTS final generation_config: {gen_config.to_json_string()}")

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    generation_config=gen_config
                )

            if isinstance(output, torch.Tensor): speech_output_tensor = output
            elif isinstance(output, dict) and ("audio_features" in output or "waveform" in output) : 
                speech_output_tensor = output.get("waveform", output.get("audio_features"))
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

            ASCIIColors.green("Bark TTS audio generation successful.")
            return audio_bytes
        except Exception as e:
            ASCIIColors.error(f"Bark TTS audio generation failed: {e}"); trace_exception(e)
            if "out of memory" in str(e).lower() and self.device == "cuda":
                 ASCIIColors.yellow("CUDA out of memory. Consider using suno/bark-small or ensure GPU has sufficient VRAM.")
            raise RuntimeError(f"Bark TTS audio generation error: {e}") from e

    def list_voices(self, **kwargs) -> List[str]: # Renamed from list_models
        """Lists available Bark voice presets."""
        return BARK_VOICE_PRESETS.copy()
    
    def get_bark_model_ids(self) -> List[str]: # Helper to list actual Bark models
        """Lists available Bark underlying model IDs."""
        return BARK_MODELS.copy()

    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model; self.model = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor; self.processor = None
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            loaded_name = getattr(self, 'loaded_bark_model_id', None) # Use specific attribute
            msg = f"BarkTTSBinding for model '{loaded_name}' destroyed." if loaded_name else "BarkTTSBinding destroyed."
            ASCIIColors.info(msg)

# --- Main Test Block ---
if __name__ == '__main__':
    if not _bark_deps_installed_with_correct_torch:
        print(f"{ASCIIColors.RED}Bark TTS binding dependencies not met. Skipping tests. Error: {_bark_installation_error}{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- BarkTTSBinding Test ---")
    # Use bark_model_id to specify the underlying Bark model
    test_bark_model_id = "suno/bark-small"
    test_output_dir = Path("./test_bark_tts_output")
    test_output_dir.mkdir(exist_ok=True)
    tts_binding = None

    try:
        ASCIIColors.cyan(f"\n--- Initializing BarkTTSBinding (Bark Model: '{test_bark_model_id}') ---")
        tts_binding = BarkTTSBinding(
            model_name=test_bark_model_id, # This is the Bark model ID from HF
            default_voice="v2/en_speaker_3" # This is the default voice_preset
        )

        ASCIIColors.cyan("\n--- Listing available Bark voice presets (voices) ---")
        voices = tts_binding.list_voices(); print(f"Available voice presets (first 10): {voices[:10]}...")
        ASCIIColors.cyan("\n--- Listing available Bark underlying model IDs ---")
        bark_models = tts_binding.get_bark_model_ids(); print(f"Underlying Bark models: {bark_models}")


        texts_to_synthesize = [
            ("hello_world_default_voice", "Hello world, this is a test of the Bark text to speech binding."),
            ("excited_greeting_spk6", "Wow! This is really cool! I can't believe it's working so well.", "v2/en_speaker_6"),
            ("question_spk1", "Can you generate different types of voices?", "v2/en_speaker_1"),
            ("german_example", "Hallo Welt, wie geht es dir heute?", "v2/de_speaker_0"),
            ("laughter_in_text", "This is so funny [laughter] I can't stop laughing.", "v2/en_speaker_0"), # Testing non-speech token
        ]

        for name, text, *voice_arg in texts_to_synthesize:
            voice_to_use = voice_arg[0] if voice_arg else None # Use specified voice or binding's default
            ASCIIColors.cyan(f"\n--- Synthesizing TTS for: '{name}' (Voice: {voice_to_use or tts_binding.default_voice_preset}) ---")
            print(f"Text: {text}")
            try:
                # Example of passing Bark-specific GenerationConfig params for this call
                tts_kwargs = {"semantic_temperature": 0.6, "coarse_temperature": 0.7, "fine_temperature": 0.5}
                if "[laughter]" in text: # Special handling for prompts with non-speech sounds
                    tts_kwargs["semantic_temperature"] = 0.8 # May need higher temp for non-speech
                    tts_kwargs["coarse_temperature"] = 0.8
                
                audio_bytes = tts_binding.generate_audio(text, voice=voice_to_use, **tts_kwargs)
                if audio_bytes:
                    output_filename = f"tts_{name}_{tts_binding.bark_model_id.split('/')[-1]}.wav"
                    output_path = test_output_dir / output_filename
                    with open(output_path, "wb") as f: f.write(audio_bytes)
                    ASCIIColors.green(f"TTS for '{name}' saved to: {output_path} ({len(audio_bytes) / 1024:.2f} KB)")
                else: ASCIIColors.error(f"TTS generation for '{name}' returned empty bytes.")
            except Exception as e_gen: ASCIIColors.error(f"Failed to generate TTS for '{name}': {e_gen}")

    except ImportError as e_imp: ASCIIColors.error(f"Import error: {e_imp}")
    except RuntimeError as e_rt: ASCIIColors.error(f"Runtime error: {e_rt}")
    except Exception as e: ASCIIColors.error(f"Unexpected error: {e}"); trace_exception(e)
    finally:
        if tts_binding: del tts_binding
        ASCIIColors.info(f"Test TTS audio (if any) are in: {test_output_dir.resolve()}")
        print(f"{ASCIIColors.YELLOW}Check the audio files in '{test_output_dir.resolve()}'!{ASCIIColors.RESET}")

    ASCIIColors.yellow("\n--- BarkTTSBinding Test Finished ---")