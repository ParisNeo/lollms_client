import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lollms_client.tti_bindings.diffusers import DiffusersTTIBinding
from lollms_client.ttm_bindings.diffusers import DiffusersTTMBinding
from lollms_client.ttv_bindings.diffusers import DiffusersTTVBinding
from lollms_client.lollms_core import LollmsClient, LollmsBindingProfile, LollmsModelProfile


def test_diffusers_tti_install_model_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        binding = DiffusersTTIBinding(
            host="127.0.0.1",
            port=9632,
            auto_start_server=False,
            models_path=str(tmp_path / "models"),
            venv_path=str(tmp_path / "venv"),
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "success": True,
            "model_name": "stabilityai__sdxl-turbo",
            "message": "Model 'stabilityai/sdxl-turbo' downloaded successfully"
        }

        with patch.object(binding, "is_server_running", return_value=True):
            with patch.object(binding._session, "post", return_value=mock_resp) as mock_post:
                res = binding.install_model("stabilityai/sdxl-turbo")
                assert res["status"] == "ok"
                assert res["model_name"] == "stabilityai__sdxl-turbo"
                assert "pull_model" in mock_post.call_args[0][0]
                posted_data = mock_post.call_args[1]["json"]
                assert posted_data["model_name"] == "stabilityai/sdxl-turbo"

                # pull_model alias check
                res_alias = binding.pull_model("stabilityai/sdxl-turbo")
                assert res_alias["status"] == "ok"


def test_diffusers_ttm_install_model_command_and_searchability():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        binding = DiffusersTTMBinding(
            host="127.0.0.1",
            port=9637,
            auto_start_server=False,
            cache_dir=str(cache_dir),
            venv_path=str(tmp_path / "venv"),
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": True,
            "message": "Model 'reach-vb/musicgen-synthwave' downloaded successfully."
        }

        with patch.object(binding, "is_server_running", return_value=True):
            with patch.object(binding._session, "post", return_value=mock_resp) as mock_post:
                res = binding.install_model("reach-vb/musicgen-synthwave")
                assert res["status"] is True
                assert "pull_model" in mock_post.call_args[0][0]

        # Simulate downloaded folder in cache_dir
        downloaded_folder = cache_dir / "reach-vb__musicgen-synthwave"
        downloaded_folder.mkdir(parents=True, exist_ok=True)

        # list_models should discover the downloaded model
        with patch.object(binding, "is_server_running", return_value=True):
            with patch.object(binding._session, "get", side_effect=Exception("offline fallback")):
                models = binding.list_models()
                assert "reach-vb/musicgen-synthwave" in models


def test_diffusers_ttv_install_model_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.Popen"):
            binding = DiffusersTTVBinding(
                host="127.0.0.1",
                port=9638,
                models_path=str(models_dir),
            )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": True,
            "message": "Model 'damo-vilab/text-to-video-ms-1.7b' installed successfully."
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            res = binding.install_model("damo-vilab/text-to-video-ms-1.7b")
            assert res["status"] is True
            assert "pull_model" in mock_post.call_args[0][0]

        # Simulate downloaded model folder
        downloaded_dir = models_dir / "ali-vilab__i2vgen-xl"
        downloaded_dir.mkdir(parents=True, exist_ok=True)

        with patch("requests.get", side_effect=Exception("offline fallback")):
            models = binding.list_models()
            assert "ali-vilab/i2vgen-xl" in models
            assert "damo-vilab/text-to-video-ms-1.7b" in models


def test_client_level_install_model_delegation():
    client = LollmsClient(
        tti_binding_profiles={
            "local_diffusers": LollmsBindingProfile(
                name="local_diffusers",
                binding_name="diffusers",
                binding_config={"host": "127.0.0.1", "port": 9632, "auto_start_server": False}
            )
        },
        tti_model_profiles={
            "sdxl": LollmsModelProfile(
                name="sdxl",
                binding_profile_name="local_diffusers",
                model_name="stabilityai/sdxl-turbo",
                is_default=True
            )
        }
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "success": True, "model_name": "sdxl_downloaded"}

    with patch.object(client.tti, "is_server_running", return_value=True):
        with patch.object(client.tti._session, "post", return_value=mock_resp):
            res = client.install_model("stabilityai/sdxl-turbo", modality="tti")
            assert res["status"] == "ok"
            assert res["model_name"] == "sdxl_downloaded"