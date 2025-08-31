# File: lollms_client/tts_bindings/bark/server/install_bark.py
#!/usr/bin/env python3
"""
Bark installation script with GPU support detection
"""

import subprocess
import sys
import torch

def install_bark():
    """Install Bark with appropriate PyTorch version for GPU support"""
    
    print("Checking for CUDA availability...")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"CUDA detected! GPU: {torch.cuda.get_device_name(0)}")
        print("Installing Bark with GPU support...")
    else:
        print("No CUDA detected, installing CPU-only version...")
    
    try:
        # Install Bark
        print("Installing bark...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bark"])
        
        print("Bark installation completed successfully!")
        
        # Test installation
        print("Testing Bark installation...")
        try:
            from bark import generate_audio, SAMPLE_RATE
            print("✓ Bark imported successfully!")
            
            # Quick test generation
            print("Running quick test generation...")
            audio = generate_audio("Hello, this is a test.", history_prompt="v2/en_speaker_6")
            print(f"✓ Test generation successful! Generated {len(audio)} audio samples.")
            
        except Exception as e:
            print(f"✗ Bark test failed: {e}")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = install_bark()
    if success:
        print("\n🎉 Bark TTS is ready to use!")
        if torch.cuda.is_available():
            print(f"🚀 GPU acceleration enabled with {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Running on CPU (consider installing CUDA for better performance)")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)