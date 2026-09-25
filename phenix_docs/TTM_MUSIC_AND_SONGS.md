# 🎵 Text-to-Music & Song Generation Guide (MiniMax Music 3)

---

## 1. Overview

The **Diffusers TTM (Text-to-Music)** binding enables local, high-fidelity music synthesis and complete song generation directly through `lollms_client`.

Supported models include:
- **`MiniMaxAI/MiniMax-Music3`** (Flagship 5-minute song generation with expressive vocals and structured progression)
- **`stabilityai/stable-audio-open-1.0`** (High-fidelity stereo audio and sound design at 44.1 kHz)
- **`cvssp/audioldm2-music`** (Text-conditioned latent diffusion music model)
- **`facebook/musicgen-small`** / **`facebook/musicgen-medium`** (Lightweight autoregressive models)

---

## 2. Quickstart: Generating Songs in One Line

```python
from lollms_client import LollmsClient, LollmsBindingProfile, LollmsModelProfile

# 1. Initialize client with the Diffusers TTM binding
client = LollmsClient(
    ttm_binding_profiles={
        "local_ttm": LollmsBindingProfile(
            name="local_ttm",
            binding_name="diffusers",
            binding_config={"host": "127.0.0.1", "port": 9637, "auto_start_server": True}
        )
    },
    ttm_model_profiles={
        "minimax_music": LollmsModelProfile(
            name="minimax_music",
            binding_profile_name="local_ttm",
            model_name="MiniMaxAI/MiniMax-Music3",
            is_default=True
        )
    }
)

# 2. Generate a complete song with vocals from lyrics
lyrics = """
[Verse 1]
Midnight shadows on the pavement glow
Walking through the neon in a steady flow
[Chorus]
Electric hearts ignite the night
We are the sound, we are the light
[Outro]
Fading in the morning glow
"""

prompt = "Energetic 80s synthwave pop, driving bassline, bright synthesizers, punchy drums, female lead vocals"

song_wav_bytes = client.generate_song_from_lyrics(
    prompt=prompt,
    lyrics=lyrics,
    duration=60  # Duration in seconds
)

with open("my_synthwave_song.wav", "wb") as f:
    f.write(song_wav_bytes)

print("Song generated and saved to my_synthwave_song.wav!")
```

---

## 3. Generating Instrumental Music

For background soundtracks, lo-fi beats, or ambient sound design:

```python
# Generate instrumental music
music_bytes = client.generate_music(
    prompt="Chill lo-fi study beat, soft electric piano, vinyl crackle, gentle hip-hop drums",
    duration=30
)

with open("lofi_chill.wav", "wb") as f:
    f.write(music_bytes)
```

---

## 4. Model Zoo Reference

Query the built-in model zoo programmatically:

```python
zoo = client.ttm.get_zoo()
for entry in zoo:
    print(f"Name: {entry['name']} | Model ID: {entry['link']} | Size: {entry['size']}")
```

### Pre-configured Models
1. **`MiniMaxAI/MiniMax-Music3`**
   - **Type**: Complete Song Generation (Vocals + Instrumentation)
   - **Max Length**: Up to 5 minutes (300 seconds)
   - **Sample Rate**: 32 kHz / 44.1 kHz Stereo
   - **Recommended VRAM**: 16 GB+ (CUDA bfloat16)
2. **`stabilityai/stable-audio-open-1.0`**
   - **Type**: Instrumental / Sound FX
   - **Max Length**: Up to 47 seconds
   - **Sample Rate**: 44.1 kHz Stereo
   - **Recommended VRAM**: 8 GB+
3. **`cvssp/audioldm2-music`**
   - **Type**: Latent Diffusion Music
   - **Sample Rate**: 16 kHz Mono/Stereo
   - **Recommended VRAM**: 6 GB+

---

## 5. Autonomous Agent Tool Integration

Autonomous agents (`LollmsPersonality` / `Agent`) automatically receive the `tool_generate_song` tool when TTM capabilities are enabled:

```python
from lollms_client.lollms_personality import LollmsPersonality, CapabilityFlags

personality = LollmsPersonality(
    name="ComposerAgent",
    system_prompt="You are an expert composer and lyricist. Write lyrics and generate full songs using tool_generate_song.",
    capabilities=CapabilityFlags(enable_ttm=True),
    lollms_client=client,
    workspace_path="./music_project"
)

# The agent writes the lyrics, selects the musical arrangement, and generates the song into the workspace
response = personality.chat("Write an acoustic folk ballad about space travelers and generate the song.")
```