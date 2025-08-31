# File: lollms_client/tts_bindings/xtts/server/setup_voices.py
#!/usr/bin/env python3
"""
Helper script to set up XTTS voices directory with sample speaker files
"""

import os
import urllib.request
from pathlib import Path

def download_sample_voices():
    """Download some sample voice files for XTTS"""
    
    voices_dir = Path(__file__).parent / "voices"
    voices_dir.mkdir(exist_ok=True)
    
    print(f"Setting up voices in: {voices_dir}")
    
    # You can add URLs to sample speaker voice files here
    # For now, let's create instructions for users
    
    readme_content = """
# XTTS Voices Directory

Place your speaker reference WAV files in this directory.

## How to add voices:

1. Record or find WAV files of speakers you want to clone (5-30 seconds recommended)
2. Name them descriptively (e.g., "john.wav", "sarah.wav", "narrator.wav")
3. Place them in this directory
4. The voice name will be the filename without extension

## Requirements for voice files:
- WAV format
- 22050 Hz sample rate (recommended)
- Mono or stereo
- Good quality, clear speech
- 5-30 seconds duration
- Single speaker

## Example usage:
```python
# Use a custom voice file named "john.wav"
audio = tts.generate_audio("Hello world", voice="john")
```

## Getting sample voices:
You can:
1. Record your own voice
2. Use text-to-speech to create reference voices
3. Extract audio clips from videos/podcasts (respect copyright)
4. Use royalty-free voice samples

Note: XTTS works by cloning the voice characteristics from the reference file.
"""
    
    readme_path = voices_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("✓ Created voices directory and README")
    print(f"📁 Add your WAV voice files to: {voices_dir}")
    print("📖 See README.md for detailed instructions")

if __name__ == "__main__":
    download_sample_voices()