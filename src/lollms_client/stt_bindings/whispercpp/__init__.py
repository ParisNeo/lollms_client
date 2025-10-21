# lollms_client/stt_bindings/whispercpp/__init__.py
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from ascii_colors import trace_exception, ASCIIColors
# No pipmaster needed here as whisper.cpp is a C++ executable.
# Python dependencies are assumed to be handled by the environment or a higher level.

from lollms_client.lollms_stt_binding import LollmsSTTBinding

BindingName = "WhisperCppSTTBinding"

DEFAULT_WHISPERCPP_EXE_NAMES = ["main", "whisper-cli", "whisper"] # Common names for the executable

class WhisperCppSTTBinding(LollmsSTTBinding):
    def __init__(self,
                 **kwargs): # Catch-all for future compatibility or specific whisper.cpp params

        super().__init__(binding_name="whispercpp")

        # --- Extract values from kwargs with defaults ---
        model_path = kwargs.get("model_path")
        whispercpp_exe_path = kwargs.get("whispercpp_exe_path")
        ffmpeg_path = kwargs.get("ffmpeg_path")
        models_search_path = kwargs.get("models_search_path")
        default_language = kwargs.get("default_language", "auto")
        n_threads = kwargs.get("n_threads", 4)
        extra_whisper_args = kwargs.get("extra_whisper_args", [])  # e.g. ["--no-timestamps"]

        self.default_model_name = "base"

        # --- Validate FFMPEG ---
        self.ffmpeg_exe = None
        if ffmpeg_path:
            resolved_ffmpeg_path = Path(ffmpeg_path)
            if resolved_ffmpeg_path.is_file() and os.access(resolved_ffmpeg_path, os.X_OK):
                self.ffmpeg_exe = str(resolved_ffmpeg_path)
            else:
                raise FileNotFoundError(f"Provided ffmpeg_path '{ffmpeg_path}' not found or not executable.")
        else:
            self.ffmpeg_exe = shutil.which("ffmpeg")

        if not self.ffmpeg_exe:
            ASCIIColors.warning("ffmpeg not found in PATH or explicitly provided. Audio conversion will not be possible for non-WAV files or incompatible WAV files.")
            ASCIIColors.warning("Please install ffmpeg and ensure it's in your system's PATH, or provide ffmpeg_path argument.")
            # Not raising an error here, as user might provide perfectly formatted WAV files.

        # --- Validate Whisper.cpp Executable ---
        self.whispercpp_exe = None
        if whispercpp_exe_path:
            resolved_wcpp_path = Path(whispercpp_exe_path)
            if resolved_wcpp_path.is_file() and os.access(resolved_wcpp_path, os.X_OK):
                self.whispercpp_exe = str(resolved_wcpp_path)
            else:
                raise FileNotFoundError(f"Provided whispercpp_exe_path '{whispercpp_exe_path}' not found or not executable.")
        else:
            for name in DEFAULT_WHISPERCPP_EXE_NAMES:
                found_path = shutil.which(name)
                if found_path:
                    self.whispercpp_exe = found_path
                    ASCIIColors.info(f"Found whisper.cpp executable via PATH: {self.whispercpp_exe}")
                    break

        if not self.whispercpp_exe:
            raise FileNotFoundError(
                f"Whisper.cpp executable (tried: {', '.join(DEFAULT_WHISPERCPP_EXE_NAMES)}) not found in PATH or explicitly provided. "
                "Please build/install whisper.cpp (from https://github.com/ggerganov/whisper.cpp) "
                "and ensure its main executable is in your system's PATH or provide its path via whispercpp_exe_path argument."
            )

        # --- Validate Model Path ---
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            # Try to resolve relative to models_search_path if provided and model_path is not absolute
            if models_search_path and not self.model_path.is_absolute() and Path(models_search_path, self.model_path).is_file():
                self.model_path = Path(models_search_path, self.model_path).resolve()
            else:
                raise FileNotFoundError(f"Whisper GGUF model file not found at '{self.model_path}'. Also checked in models_search_path if applicable.")

        self.models_search_path = Path(models_search_path).resolve() if models_search_path else None
        self.default_language = default_language
        self.n_threads = n_threads
        self.extra_whisper_args = extra_whisper_args

        ASCIIColors.green(f"WhisperCppSTTBinding initialized with model: {self.model_path}")

    def _convert_to_wav(self, input_audio_path: Path, output_wav_path: Path) -> bool:
        if not self.ffmpeg_exe:
            ASCIIColors.error("ffmpeg is required for audio conversion but was not found or configured.")
            return False
        try:
            command = [
                self.ffmpeg_exe,
                "-i", str(input_audio_path),
                "-ar", "16000",          # 16kHz sample rate
                "-ac", "1",              # Mono channel
                "-c:a", "pcm_s16le",     # Signed 16-bit PCM little-endian
                "-y",                    # Overwrite output file if it exists
                str(output_wav_path)
            ]
            ASCIIColors.info(f"Converting audio with ffmpeg: {' '.join(command)}")
            # Using Popen to better handle stderr/stdout if needed for detailed logging
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                ASCIIColors.error(f"ffmpeg conversion failed (exit code {process.returncode}).")
                ASCIIColors.error(f"ffmpeg stdout:\n{stdout}")
                ASCIIColors.error(f"ffmpeg stderr:\n{stderr}")
                return False
            ASCIIColors.info(f"ffmpeg conversion successful: {output_wav_path}")
            return True
        except FileNotFoundError: # Handle case where ffmpeg command itself is not found
            ASCIIColors.error(f"ffmpeg command '{self.ffmpeg_exe}' not found. Ensure ffmpeg is installed and in PATH or ffmpeg_path is correct.")
            return False
        except Exception as e:
            ASCIIColors.error(f"An error occurred during ffmpeg conversion: {e}")
            trace_exception(e)
            return False

    def transcribe_audio(self, audio_path: Union[str, Path], model: Optional[str] = None, **kwargs) -> str:
        input_audio_p = Path(audio_path)
        if not input_audio_p.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_audio_p}")

        current_model_path = self.model_path
        if model: # User specified a different model for this transcription
            potential_model_p = Path(model)
            if potential_model_p.is_absolute() and potential_model_p.is_file():
                current_model_path = potential_model_p
            elif self.models_search_path and (self.models_search_path / model).is_file():
                current_model_path = self.models_search_path / model
            elif Path(model).is_file(): # Relative to current working directory?
                 current_model_path = Path(model)
            else:
                ASCIIColors.warning(f"Specified model '{model}' not found as absolute path, in models_search_path, or current dir. Using default: {self.model_path.name}")
        
        language = kwargs.get("language", self.default_language)
        threads = kwargs.get("n_threads", self.n_threads)
        extra_args_call = kwargs.get("extra_whisper_args", self.extra_whisper_args)

        with tempfile.TemporaryDirectory(prefix="lollms_whispercpp_") as tmpdir:
            tmp_dir_path = Path(tmpdir)
            
            # Always convert to ensure 16kHz mono WAV, unless explicitly told not to by a kwarg (e.g. assume_wav=True)
            force_conversion = not kwargs.get("assume_compatible_wav", False)
            
            if force_conversion or input_audio_p.suffix.lower() != ".wav":
                if not self.ffmpeg_exe:
                    raise RuntimeError("ffmpeg is required for audio pre-processing but is not configured. "
                                       "Please provide a 16kHz mono WAV file or configure ffmpeg.")
                converted_wav_path = tmp_dir_path / (input_audio_p.stem + "_16khz_mono.wav")
                if not self._convert_to_wav(input_audio_p, converted_wav_path):
                    raise Exception(f"Audio conversion to compatible WAV failed for {input_audio_p}.")
                target_audio_file = converted_wav_path
            else: # Input is WAV, assume it's compatible (user's responsibility if assume_compatible_wav=True)
                target_audio_file = input_audio_p
            
            command = [
                self.whispercpp_exe,
                "-m", str(current_model_path),
                "-f", str(target_audio_file),
                "-l", language,
                "-t", str(threads),
                "-otxt" # Output as a .txt file in the same dir as input wav
            ]
            if isinstance(extra_args_call, list):
                command.extend(extra_args_call)
            elif isinstance(extra_args_call, str):
                command.extend(extra_args_call.split())

            ASCIIColors.info(f"Executing Whisper.cpp: {' '.join(command)}")
            try:
                # Run whisper.cpp, making it output its .txt file into our temp directory.
                # To do this, we can copy the target_audio_file into tmp_dir_path if it's not already there,
                # then run whisper.cpp with CWD as tmp_dir_path.
                
                final_target_audio_in_tmp: Path
                if target_audio_file.parent != tmp_dir_path:
                    final_target_audio_in_tmp = tmp_dir_path / target_audio_file.name
                    shutil.copy2(target_audio_file, final_target_audio_in_tmp)
                    # Update command to use the path within tmp_dir_path if we copied it.
                    # The -f argument should be relative to the CWD if CWD is set.
                    command[command.index("-f")+1] = str(final_target_audio_in_tmp.name) 
                else:
                    final_target_audio_in_tmp = target_audio_file
                    command[command.index("-f")+1] = str(final_target_audio_in_tmp.name) 


                process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=str(tmp_dir_path))
                
                output_txt_file = tmp_dir_path / (final_target_audio_in_tmp.name + ".txt")

                if output_txt_file.exists():
                    transcribed_text = output_txt_file.read_text(encoding='utf-8').strip()
                    ASCIIColors.green(f"Whisper.cpp transcription successful for {input_audio_p.name}.")
                    return transcribed_text
                else:
                    ASCIIColors.error(f"Whisper.cpp did not produce the expected output file: {output_txt_file.name} in {tmp_dir_path}")
                    ASCIIColors.info(f"Whisper.cpp stdout:\n{process.stdout}")
                    ASCIIColors.info(f"Whisper.cpp stderr:\n{process.stderr}")
                    raise Exception("Whisper.cpp execution failed to produce output text file.")

            except subprocess.CalledProcessError as e:
                ASCIIColors.error(f"Whisper.cpp execution failed with exit code {e.returncode} for {input_audio_p.name}")
                ASCIIColors.error(f"Command: {' '.join(e.cmd)}")
                ASCIIColors.error(f"Stdout:\n{e.stdout}")
                ASCIIColors.error(f"Stderr:\n{e.stderr}")
                trace_exception(e)
                raise Exception(f"Whisper.cpp execution error: {e.stderr or e.stdout or 'Unknown whisper.cpp error'}") from e
            except Exception as e:
                ASCIIColors.error(f"An error occurred during Whisper.cpp transcription for {input_audio_p.name}: {e}")
                trace_exception(e)
                raise

    def list_models(self, **kwargs) -> List[str]:
        models = []
        # 1. Add the default configured model's name
        if self.model_path and self.model_path.exists():
            # For consistency, list by name. The 'model' arg in transcribe_audio can take this name.
            models.append(self.model_path.name)

        # 2. Scan models_search_path if provided
        if self.models_search_path and self.models_search_path.is_dir():
            for item in self.models_search_path.iterdir():
                if item.is_file() and item.suffix.lower() == ".gguf":
                    # Add name if not already listed (default model might be in search path)
                    if item.name not in models:
                        models.append(item.name)
        
        return sorted(list(set(models))) # Ensure uniqueness and sort


# --- Main Test Block ---
if __name__ == '__main__':
    ASCIIColors.yellow("--- WhisperCppSTTBinding Test ---")

    # --- USER CONFIGURATION REQUIRED FOR TEST ---
    # Find your whisper.cpp build directory and the 'main' or 'whisper-cli' executable.
    # Example: If you built whisper.cpp in /home/user/whisper.cpp, the exe might be /home/user/whisper.cpp/main
    TEST_WHISPERCPP_EXE = None # SET THIS: e.g., "/path/to/whisper.cpp/main" or "whisper-cli" if in PATH
    
    # Download a GGUF model from Hugging Face: https://huggingface.co/ggerganov/whisper.cpp/tree/main
    # Place it somewhere accessible.
    TEST_MODEL_GGUF = "ggml-tiny.en.bin" # SET THIS: e.g., "/path/to/models/ggml-tiny.en.bin"
                                        # If just a name, expects it in CWD or models_search_path.

    # Optional: Path to ffmpeg if not in system PATH
    TEST_FFMPEG_EXE = None # e.g., "/usr/local/bin/ffmpeg"

    # Optional: A directory to put other .gguf models for testing list_models
    TEST_MODELS_SEARCH_DIR = Path("./test_whisper_models_cpp") 
    # --- END USER CONFIGURATION ---


    # Create a dummy audio file for testing (requires scipy and numpy)
    dummy_audio_file = Path("dummy_whispercpp_test.wav")
    if not dummy_audio_file.exists():
        try:
            import numpy as np
            from scipy.io.wavfile import write as write_wav
            samplerate = 44100; duration = 1.5; frequency = 330 # A bit longer, different freq
            t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
            amplitude = np.iinfo(np.int16).max * 0.3
            data = amplitude * np.sin(2. * np.pi * frequency * t)
            # Add some noise
            data += (amplitude * 0.05 * np.random.normal(size=data.shape[0])).astype(data.dtype)
            write_wav(dummy_audio_file, samplerate, data.astype(np.int16))
            ASCIIColors.green(f"Created dummy audio file for testing: {dummy_audio_file}")
        except ImportError:
            ASCIIColors.warning("SciPy/NumPy not installed. Cannot create dummy audio file for testing.")
            ASCIIColors.warning(f"Please manually place a test audio file named '{dummy_audio_file.name}' in the current directory.")
        except Exception as e_dummy:
            ASCIIColors.error(f"Could not create dummy audio file: {e_dummy}")
    
    # Prepare model search directory for list_models test
    if TEST_MODELS_SEARCH_DIR:
        TEST_MODELS_SEARCH_DIR.mkdir(exist_ok=True)
        # Create another dummy GGUF (just an empty file for listing purposes)
        (TEST_MODELS_SEARCH_DIR / "ggml-base.en.bin").touch(exist_ok=True) 


    # Basic check for prerequisites before attempting to initialize
    if TEST_WHISPERCPP_EXE is None and not any(shutil.which(name) for name in DEFAULT_WHISPERCPP_EXE_NAMES):
        ASCIIColors.error(f"Whisper.cpp executable not found in PATH and TEST_WHISPERCPP_EXE not set. Aborting test.")
        exit(1)
    
    if not Path(TEST_MODEL_GGUF).exists() and not (TEST_MODELS_SEARCH_DIR and (TEST_MODELS_SEARCH_DIR / TEST_MODEL_GGUF).exists()):
         # Check if model is just a name and exists in current dir if TEST_MODELS_SEARCH_DIR is not set or model not in it
        if not (Path().cwd()/TEST_MODEL_GGUF).exists():
            ASCIIColors.error(f"Test model GGUF '{TEST_MODEL_GGUF}' not found. Please download/place it or update TEST_MODEL_GGUF path. Aborting test.")
            exit(1)
        else: # Found in CWD
            TEST_MODEL_GGUF = str((Path().cwd()/TEST_MODEL_GGUF).resolve())


    stt_binding = None
    try:
        ASCIIColors.cyan("\n--- Initializing WhisperCppSTTBinding ---")
        stt_binding = WhisperCppSTTBinding(
            model_path=TEST_MODEL_GGUF,
            whispercpp_exe_path=TEST_WHISPERCPP_EXE,
            ffmpeg_path=TEST_FFMPEG_EXE,
            models_search_path=TEST_MODELS_SEARCH_DIR,
            default_language="en",
            n_threads=os.cpu_count() // 2 or 1, # Use half CPU cores or at least 1
        )
        ASCIIColors.green("Binding initialized successfully.")

        ASCIIColors.cyan("\n--- Listing available models ---")
        models = stt_binding.list_models()
        if models:
            print(f"Available models: {models}")
        else:
            ASCIIColors.warning("No models listed. Check paths and models_search_path.")

        if dummy_audio_file.exists():
            ASCIIColors.cyan(f"\n--- Transcribing '{dummy_audio_file.name}' (expected 16kHz mono after conversion) ---")
            transcription = stt_binding.transcribe_audio(str(dummy_audio_file))
            print(f"Transcription: '{transcription}'")

            # Test with a different model if listed and available
            if "ggml-base.en.bin" in models and "ggml-base.en.bin" != Path(TEST_MODEL_GGUF).name :
                if (TEST_MODELS_SEARCH_DIR / "ggml-base.en.bin").exists(): # Ensure it's actually there
                    ASCIIColors.cyan(f"\n--- Transcribing with model 'ggml-base.en.bin' from search path ---")
                    transcription_base = stt_binding.transcribe_audio(str(dummy_audio_file), model="ggml-base.en.bin")
                    print(f"Transcription (ggml-base.en.bin): '{transcription_base}'")
                else:
                    ASCIIColors.warning("Model 'ggml-base.en.bin' listed but not found in search path for test.")
            
            # Test assume_compatible_wav (if user has a 16kHz mono WAV already)
            # Create a specific 16kHz mono wav file for this
            compatible_wav_file = Path("compatible_test.wav")
            try:
                import numpy as np
                from scipy.io.wavfile import write as write_wav
                samplerate = 16000; duration = 1.0; frequency = 550
                t_compat = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
                data_compat = (np.iinfo(np.int16).max * 0.2 * np.sin(2. * np.pi * frequency * t_compat)).astype(np.int16)
                write_wav(compatible_wav_file, samplerate, data_compat)
                ASCIIColors.green(f"Created compatible 16kHz mono WAV: {compatible_wav_file}")

                ASCIIColors.cyan(f"\n--- Transcribing '{compatible_wav_file.name}' with assume_compatible_wav=True ---")
                transcription_compat = stt_binding.transcribe_audio(str(compatible_wav_file), assume_compatible_wav=True)
                print(f"Transcription (compatible WAV): '{transcription_compat}'")

            except ImportError: ASCIIColors.warning("SciPy/NumPy not available, skipping compatible WAV test.")
            except Exception as e_compat: ASCIIColors.error(f"Error in compatible WAV test: {e_compat}")
            finally:
                if compatible_wav_file.exists(): compatible_wav_file.unlink(missing_ok=True)


        else:
            ASCIIColors.warning(f"Dummy audio file '{dummy_audio_file}' not found. Skipping main transcription test.")

    except FileNotFoundError as e:
        ASCIIColors.error(f"Initialization or transcription failed due to FileNotFoundError: {e}")
        ASCIIColors.info("Please ensure whisper.cpp/ffmpeg executables are in PATH or paths are correctly set in the test script, and the GGUF model file exists.")
    except RuntimeError as e:
        ASCIIColors.error(f"Runtime error: {e}")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred: {e}")
        trace_exception(e)
    finally:
        # Clean up dummy files created by this test script
        if "samplerate" in locals() and dummy_audio_file.exists(): # Heuristic: if we created it
             dummy_audio_file.unlink(missing_ok=True)
        if TEST_MODELS_SEARCH_DIR:
            if (TEST_MODELS_SEARCH_DIR / "ggml-base.en.bin").exists():
                 (TEST_MODELS_SEARCH_DIR / "ggml-base.en.bin").unlink(missing_ok=True)
            # Remove dir only if it's empty (or was created by this script and now empty)
            try:
                if not any(TEST_MODELS_SEARCH_DIR.iterdir()):
                    TEST_MODELS_SEARCH_DIR.rmdir()
            except OSError: pass # Ignore if not empty or other issues

    ASCIIColors.yellow("\n--- WhisperCppSTTBinding Test Finished ---")

    def list_models(self) -> List[Dict[str, Any]]:
        return ["base" , "small", "medium", "large"]

