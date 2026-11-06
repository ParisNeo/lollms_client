"""Unit tests for the centralized path resolution policy in ``lollms_paths``.

Validates that:

* Defaults resolve to ``~/.lollms`` (data) and ``~/.lollms/config.yaml``.
* ``LOLLMS_HOME`` / ``LOLLMS_DATA_FOLDER`` / ``LOLLMS_CONFIG_FILE`` env vars
  override the built-in defaults permanently.
* An explicit app-provided kwarg always wins over env and defaults.
* The base binding adopts the same policy (no more ``<package>/data`` default).
* Directories are materialized on access so bindings can persist immediately.
"""
import os
from pathlib import Path

import pytest

from lollms_client import lollms_paths
from lollms_client.lollms_base_binding import LollmsBaseBinding


class _DummyBinding(LollmsBaseBinding):
    def list_models(self, *args, **kwargs):
        return []


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (lollms_paths.ENV_HOME, lollms_paths.ENV_DATA_FOLDER, lollms_paths.ENV_CONFIG_FILE):
        monkeypatch.delenv(key, raising=False)
    yield


def test_default_home_is_dot_lollms(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert lollms_paths.get_home_dir() == (tmp_path / ".lollms").resolve()


def test_default_data_folder_is_home_data(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = (tmp_path / ".lollms" / "data").resolve()
    assert lollms_paths.get_data_folder() == expected
    assert expected.is_dir()


def test_default_config_file_is_home_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = (tmp_path / ".lollms" / "config.yaml").resolve()
    assert lollms_paths.get_config_file() == expected
    assert expected.parent.is_dir()
    assert not expected.exists()  # File is created lazily on first write


def test_env_data_folder_override(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    custom = tmp_path / "custom_data"
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(custom))
    assert lollms_paths.get_data_folder() == custom.resolve()
    assert custom.is_dir()


def test_env_config_file_override(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    custom = tmp_path / "custom_cfg.yaml"
    monkeypatch.setenv(lollms_paths.ENV_CONFIG_FILE, str(custom))
    assert lollms_paths.get_config_file() == custom.resolve()


def test_env_home_override_affects_data_and_config(monkeypatch, tmp_path):
    alt_home = tmp_path / "alt_home"
    monkeypatch.setenv(lollms_paths.ENV_HOME, str(alt_home))
    assert lollms_paths.get_home_dir() == alt_home.resolve()
    assert lollms_paths.get_data_folder() == (alt_home / "data").resolve()
    assert lollms_paths.get_config_file() == (alt_home / "config.yaml").resolve()


def test_kwarg_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(tmp_path / "env_data"))
    app_choice = tmp_path / "app_data"
    assert lollms_paths.get_data_folder(app_choice) == app_choice.resolve()


def test_base_binding_uses_shared_default_not_package_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    binding = _DummyBinding(binding_name="dummy")
    expected = (tmp_path / ".lollms" / "data").resolve()
    assert binding.get_data_folder() == expected
    assert expected.is_dir()


def test_base_binding_respects_kwarg(monkeypatch, tmp_path):
    custom = tmp_path / "binding_custom"
    binding = _DummyBinding(binding_name="dummy", lollms_data_folder=str(custom))
    assert binding.get_data_folder() == custom.resolve()


def test_base_binding_respects_env(monkeypatch, tmp_path):
    custom = tmp_path / "env_binding_data"
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(custom))
    binding = _DummyBinding(binding_name="dummy")
    assert binding.get_data_folder() == custom.resolve()


def test_whisper_models_dir_defaults_under_data_folder(monkeypatch, tmp_path):
    """Whisper places weights under <lollms_data_folder>/stt/whisper/models."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from lollms_client.stt_bindings.whisper import WhisperSTTBinding

    binding = WhisperSTTBinding(auto_start_server=False, wait_for_server=False)
    expected_root = (tmp_path / ".lollms" / "data" / "stt" / "whisper").resolve()
    assert binding.cache_dir == expected_root
    assert binding.models_dir == (expected_root / "models").resolve()
    assert binding.models_dir.is_dir()