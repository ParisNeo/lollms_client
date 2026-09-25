# 📚 Phenix Unified API Reference

---

## 1. Client Initialization (`LollmsClient`)

```python
from lollms_client import LollmsClient, LollmsBindingProfile, LollmsModelProfile

client = LollmsClient(
    # Connection Layer
    llm_binding_profiles={
        "local_ollama": LollmsBindingProfile(name="local_ollama", binding_name="ollama", binding_config={"host_address": "http://localhost:11434"})
    },
    ttm_binding_profiles={
        "local_ttm": LollmsBindingProfile(name="local_ttm", binding_name="diffusers", binding_config={"port": 9637})
    },
    stt_binding_profiles={
        "local_stt": LollmsBindingProfile(name="local_stt", binding_name="whisper", binding_config={"port": 9633})
    },

    # Execution Layer
    llm_model_profiles={
        "chat_model": LollmsModelProfile(name="chat_model", binding_profile_name="local_ollama", model_name="llama3.1:8b", is_default=True)
    },
    ttm_model_profiles={
        "song_model": LollmsModelProfile(name="song_model", binding_profile_name="local_ttm", model_name="MiniMaxAI/MiniMax-Music3", is_default=True)
    },
    stt_model_profiles={
        "transcribe_model": LollmsModelProfile(name="transcribe_model", binding_profile_name="local_stt", model_name="base", is_default=True)
    }
)
```

---

## 2. Modality Primitives

### Text Generation (LLM)
```python
# Synchronous text generation
text = client.generate_text("Explain quantum entanglement in simple terms.")

# Chat completions with message history
response = client.generate_from_messages([
    {"role": "system", "content": "You are a senior system architect."},
    {"role": "user", "content": "Outline a clean micro-batching architecture."}
])

# Tool-enabled agentic generation
result = client.generate_with_tools(
    prompt="Calculate sqrt(144) + 10",
    tools=[my_calculator_tool]
)
```

### Music & Full Song Generation (TTM)
```python
# Generate instrumental music
music_bytes = client.generate_music("Epic orchestral cinematic soundtrack", duration=20)

# Generate complete song from lyrics and musical prompt
song_bytes = client.generate_song_from_lyrics(
    prompt="Dreamy indie folk, acoustic guitar, warm female vocals",
    lyrics="[Verse 1]\nStars across the open sky\n[Chorus]\nFly away with me",
    duration=60
)
```

### Speech-to-Text (STT)
```python
# Transcribe audio file using shared Whisper daemon
transcript = client.transcribe_audio("interview.wav", model="base", language="en")
```

### Text-to-Speech (TTS)
```python
# Synthesize speech using Piper or XTTS
speech_bytes = client.generate_audio("Welcome to Project Phenix!", voice="en_US-lessac-medium")
```

### Text-to-Image (TTI)
```python
# Generate image using local Diffusers / SDXL
image_bytes = client.generate_image("A futuristic city floating in the sky, 8k, hyperdetailed")
```

---

## 3. Discussion & Autonomous Agency (`LollmsDiscussion`)

```python
from lollms_client import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_personality import LollmsPersonality

db_manager = LollmsDataManager("sqlite:///project.db")
discussion = LollmsDiscussion.create_new(
    lollms_client=client,
    db_manager=db_manager,
    workspace_path="./my_project"
)

response = discussion.chat(
    user_message="Analyze sales_data.csv and write an automated summary report.",
    enable_artefacts=True,
    enable_data_tools=True,
    max_nb_rounds=15
)

# Access generated files and reports
print(response["ai_message"].content)
for art in response["artefacts"]:
    print(f"Artifact created: {art['title']} (v{art['version']})")
```