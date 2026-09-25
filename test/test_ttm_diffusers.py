import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lollms_client.ttm_bindings.diffusers import DiffusersTTMBinding, DEFAULT_TTM_ZOO
from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile


def test_diffusers_ttm_binding_initialization_defaults():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        binding = DiffusersTTMBinding(
            host="127.0.0.1",
            port=9637,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )
        assert binding.host == "127.0.0.1"
        assert binding.port == 9637
        assert binding.base_url == "http://127.0.0.1:9637"
        assert binding.model_name == "MiniMaxAI/MiniMax-Music3"
        assert binding.token_file == tmp_path / "cache" / "diffusers_ttm.token"


def test_diffusers_ttm_model_zoo_listing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        binding = DiffusersTTMBinding(
            host="127.0.0.1",
            port=9637,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )
        zoo = binding.get_zoo()
        assert len(zoo) >= 4
        names = [item["name"] for item in zoo]
        links = [item["link"] for item in zoo]
        assert "MiniMax Music 3" in names
        assert "MiniMaxAI/MiniMax-Music3" in links
        assert "stabilityai/stable-audio-open-1.0" in links


def test_diffusers_ttm_shared_server_probe_no_port_drift():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        b1 = DiffusersTTMBinding(
            host="127.0.0.1",
            port=9637,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )
        b2 = DiffusersTTMBinding(
            host="127.0.0.1",
            port=9637,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )
        assert b1.port == 9637
        assert b2.port == 9637

        with patch.object(b2, "is_server_running", return_value=True):
            with patch.object(b2, "start_server") as mock_start:
                b2.ensure_server_is_running(wait=True)
                mock_start.assert_not_called()
                assert b2.port == 9637


def test_diffusers_ttm_generation_mock_calls():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        binding = DiffusersTTMBinding(
            host="127.0.0.1",
            port=9637,
            auto_start_server=False,
            cache_dir=str(tmp_path / "cache"),
            venv_path=str(tmp_path / "venv"),
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"RIFF....WAVEfmt "

        with patch.object(binding, "is_server_running", return_value=True):
            with patch.object(binding._session, "post", return_value=mock_resp) as mock_post:
                # 1. Test generate_music
                audio = binding.generate_music("Epic orchestral soundtrack", duration=10)
                assert audio == b"RIFF....WAVEfmt "
                assert "generate_music" in mock_post.call_args[0][0]

                # 2. Test generate_song_from_lyrics
                song_audio = binding.generate_song_from_lyrics(
                    prompt="Acoustic indie pop",
                    lyrics="[Verse]\nWalking in the morning sun\n[Chorus]\nIt has only begun",
                    duration=60
                )
                assert song_audio == b"RIFF....WAVEfmt "
                assert "generate_song" in mock_post.call_args[0][0]


def test_core_client_song_methods():
    client = LollmsClient(
        ttm_binding_profiles={
            "local_ttm": LollmsBindingProfile(
                name="local_ttm",
                binding_name="diffusers",
                binding_config={"host": "127.0.0.1", "port": 9637, "auto_start_server": False}
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

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"WAV_DATA"

    with patch.object(client.ttm, "is_server_running", return_value=True):
        with patch.object(client.ttm._session, "post", return_value=mock_resp):
            music = client.generate_music("chill synthwave")
            assert music == b"WAV_DATA"

            song = client.generate_song_from_lyrics("pop rock", "[Verse]\nHey there")
            assert song == b"WAV_DATA"