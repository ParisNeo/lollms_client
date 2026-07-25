import os
import sys
import base64
import math
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lollms_client.stt_bindings.whisper import WhisperSTTBinding
from ascii_colors import ASCIIColors


def generate_test_audio_file(text: str = "Hello, this is a test."):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sample_rate = 16000.0
    duration = 2.0
    frequency = 440.0
    
    with wave.open(temp_file.name, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        
        num_samples = int(duration * sample_rate)
        for i in range(num_samples):
            sample = int(32767.0 * 0.5 * math.sin(2 * math.pi * frequency * (i / sample_rate)))
            data = struct.pack('<h', sample)
            wav_file.writeframesraw(data)
            
    return temp_file.name


@pytest.fixture
def mock_whisper_binding():
    ASCIIColors.cyan("Setting up mocked Whisper STT Binding environment...")
    
    with patch("lollms_client.stt_bindings.whisper.requests") as mock_requests:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "running"}
        mock_requests.get.return_value = mock_response
        
        binding = WhisperSTTBinding(
            host="localhost",
            port=9655,
            auto_start_server=False,
            wait_for_server=False,
            model_name="tiny",
            venv_path="./venv/stt_whisper_test_venv",
            cache_dir="./data/stt_test_cache"
        )
        binding.server_process = MagicMock()
        binding.server_process.poll.return_value = None
        
        yield binding, mock_requests


def test_whisper_server_running(mock_whisper_binding):
    binding, mock_requests = mock_whisper_binding
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "running"}
    mock_requests.get.return_value = mock_response
    
    assert binding.is_server_running() is True
    
    mock_response.json.return_value = {"status": "stopped"}
    mock_requests.get.return_value = mock_response
    assert binding.is_server_running() is False


def test_transcription_pipeline(mock_whisper_binding):
    binding, mock_requests = mock_whisper_binding
    audio_file_path = generate_test_audio_file()
    
    try:
        ASCIIColors.info(f"Transcribing test audio file: {audio_file_path}")
        
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"text": "Hello, this is a test."}
        mock_requests.post.return_value = mock_post_response
        
        text = binding.transcribe_audio(
            audio_source=audio_file_path,
            model="tiny",
            language="en"
        )
        
        assert isinstance(text, str), "Transcription did not return a string."
        assert text == "Hello, this is a test."
        
        assert mock_requests.post.called, "POST request was not made for transcription."
        _, kwargs = mock_requests.post.call_args
        payload = kwargs.get("json", {})
        
        assert "audio_b64" in payload, "Payload missing base64 audio data."
        assert payload["model_name"] == "tiny"
        assert payload["language"] == "en"
        assert payload["task"] == "transcribe"
        
    finally:
        if os.path.exists(audio_file_path):
            os.unlink(audio_file_path)


def test_shared_state_ps(mock_whisper_binding):
    binding, mock_requests = mock_whisper_binding
    
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = [
        {"model_name": "tiny", "is_loaded": True, "task": "transcribe"}
    ]
    mock_requests.get.return_value = mock_get_response
    
    ps_data = binding.ps()
    assert isinstance(ps_data, list), "/ps did not return a list."
    
    tiny_entry = next((entry for entry in ps_data if entry.get("model_name") == "tiny"), None)
    assert tiny_entry is not None, "Model 'tiny' not found in server registry /ps."
    assert tiny_entry["is_loaded"] is True, "Model 'tiny' should be marked as loaded."
