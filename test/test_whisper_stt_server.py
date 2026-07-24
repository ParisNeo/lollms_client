import os
import sys
import time
import wave
import math
import struct
import subprocess
import tempfile
import pytest
import requests
import socket
from pathlib import Path

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

def kill_process_on_port(port: int):
    """Finds and kills any process listening on the given port."""
    if sys.platform == "win32":
        try:
            # Find PIDs using netstat
            result = subprocess.run(["netstat", "-aon", "-p", "TCP"], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            for line in lines:
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    if pid > 0:
                        ASCIIColors.warning(f"Killing zombie process {pid} on port {port}")
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        except Exception as e:
            ASCIIColors.warning(f"Could not kill process on port {port}: {e}")
    else:
        try:
            # Unix-like systems
            result = subprocess.run(["lsof", "-t", "-i", f":{port}"], capture_output=True, text=True)
            pids = result.stdout.split()
            for pid in pids:
                if pid:
                    ASCIIColors.warning(f"Killing zombie process {pid} on port {port}")
                    subprocess.run(["kill", "-9", pid], capture_output=True)
        except Exception:
            pass

@pytest.fixture(scope="module")
def whisper_binding():
    ASCIIColors.cyan("Setting up Whisper STT Binding test environment...")
    
    port = 9655
    kill_process_on_port(port)
    
    binding = WhisperSTTBinding(
        host="localhost",
        port=port,
        auto_start_server=True,
        wait_for_server=True,
        model_name="tiny",
        venv_path="./venv/stt_whisper_test_venv",
        cache_dir="./data/stt_test_cache"
    )
    
    start_time = time.time()
    while time.time() - start_time < 60:
        if binding.is_server_running():
            break
        if binding.server_process and binding.server_process.poll() is not None:
            pytest.fail("Whisper server process died unexpectedly during setup.")
        time.sleep(1)
    else:
        pytest.fail("Whisper server failed to start within 60 seconds.")
        
    yield binding
    
    ASCIIColors.cyan("Tearing down Whisper STT test environment...")
    if binding.server_process:
        binding.server_process.terminate()
        try:
            binding.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            binding.server_process.kill()
            binding.server_process.wait()


def test_whisper_server_running(whisper_binding):
    assert whisper_binding.is_server_running()
    response = whisper_binding._get_request("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


def test_transcription_pipeline(whisper_binding):
    audio_file_path = generate_test_audio_file()
    
    try:
        ASCIIColors.info(f"Transcribing test audio file: {audio_file_path}")
        text = whisper_binding.transcribe_audio(
            audio_source=audio_file_path,
            model="tiny",
            language="en"
        )
        
        assert isinstance(text, str), "Transcription did not return a string."
        ASCIIColors.green(f"Transcription result: '{text}'")
        
    finally:
        if os.path.exists(audio_file_path):
            os.unlink(audio_file_path)


def test_shared_state_ps(whisper_binding):
    audio_file_path = generate_test_audio_file()
    try:
        whisper_binding.transcribe_audio(audio_source=audio_file_path, model="tiny")
    except Exception:
        pass
    finally:
        if os.path.exists(audio_file_path):
            os.unlink(audio_file_path)

    time.sleep(1)
    
    ps_data = whisper_binding.ps()
    assert isinstance(ps_data, list), "/ps did not return a list."
    
    tiny_entry = next((entry for entry in ps_data if entry.get("model_name") == "tiny"), None)
    assert tiny_entry is not None, "Model 'tiny' not found in server registry /ps."
    assert tiny_entry["is_loaded"] == True, "Model 'tiny' should be marked as loaded."
