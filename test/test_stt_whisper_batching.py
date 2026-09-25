import os
import time
import queue
import secrets
import tempfile
import threading
from pathlib import Path
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
from lollms_client.stt_bindings.whisper import WhisperSTTBinding
from lollms_client.stt_bindings.whisper.server.main import (
    ModelManager,
    TranscriptionJob,
    ServerState,
    verify_auth_token,
)
from fastapi import HTTPException


def test_whisper_binding_initialization_defaults():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        binding = WhisperSTTBinding(
            host="127.0.0.1",
            port=9999,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )
        assert binding.host == "127.0.0.1"
        assert binding.port == 9999
        assert binding.base_url == "http://127.0.0.1:9999"
        assert binding.batch_window == 0.02
        assert binding.max_batch_size == 8
        assert binding.token_file == tmp_path / "cache" / "whisper_server.token"


def test_whisper_auth_token_verification():
    with tempfile.TemporaryDirectory() as tmp_dir:
        token = secrets.token_hex(16)
        state = ServerState(models_cache_dir=Path(tmp_dir), auth_token=token)

        # Patch state in server.main
        with patch("lollms_client.stt_bindings.whisper.server.main.state", state):
            # Valid Bearer header
            verify_auth_token(authorization=f"Bearer {token}", x_server_token=None)

            # Valid X-Server-Token header
            verify_auth_token(authorization=None, x_server_token=token)

            # Missing or invalid token should raise 401
            with pytest.raises(HTTPException) as exc_info:
                verify_auth_token(authorization="Bearer wrong-token", x_server_token=None)
            assert exc_info.value.status_code == 401

            with pytest.raises(HTTPException) as exc_info:
                verify_auth_token(authorization=None, x_server_token=None)
            assert exc_info.value.status_code == 401


def test_whisper_multi_process_probing_and_no_port_drift():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        binding1 = WhisperSTTBinding(
            host="127.0.0.1",
            port=9633,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )
        binding2 = WhisperSTTBinding(
            host="127.0.0.1",
            port=9633,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )

        # Both instances must retain port 9633 without port drifting
        assert binding1.port == 9633
        assert binding2.port == 9633
        assert binding1.base_url == binding2.base_url

        # Simulate binding1 already running
        with patch.object(binding2, "is_server_running", return_value=True):
            # Calling ensure_server_is_running should attach without starting server or modifying port
            with patch.object(binding2, "start_server") as mock_start:
                binding2.ensure_server_is_running(wait=True)
                mock_start.assert_not_called()
                assert binding2.port == 9633


def test_model_manager_micro_batching_loop():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        manager = ModelManager(
            config={"model_name": "base", "device": "cpu"},
            models_cache_dir=cache_dir,
            batch_window=0.03,
            max_batch_size=4,
        )

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed audio output."}
        manager.model = mock_model
        manager.loaded_model_name = "base"

        f1, f2, f3 = Future(), Future(), Future()
        job1 = TranscriptionJob(f1, "base", "dummy1.wav", {"task": "transcribe"})
        job2 = TranscriptionJob(f2, "base", "dummy2.wav", {"task": "transcribe"})
        job3 = TranscriptionJob(f3, "base", "dummy3.wav", {"task": "transcribe"})

        # Mock whisper.load_audio to return 10s audio arrays
        dummy_audio = [0.0] * 160000
        with patch("whisper.load_audio", return_value=dummy_audio):
            # Put 3 concurrent requests into the queue
            manager.queue.put(job1)
            manager.queue.put(job2)
            manager.queue.put(job3)

            # Wait for results
            res1 = f1.result(timeout=5)
            res2 = f2.result(timeout=5)
            res3 = f3.result(timeout=5)

            assert res1 != ""
            assert res2 != ""
            assert res3 != ""

        manager.stop()