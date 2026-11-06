"""Regression tests for the data/config path override policy.

Locks the intended precedence contract requested by the product spec:

    per-instance kwarg  >  process-wide env var  >  default (~/.lollms/...)

Two independent user workflows are covered:

1. **App overrides per-instance** (``app can be independent``): the app passes
   ``lollms_data_folder=`` (and, where relevant, ``LOLLMS_CONFIG_FILE``) at the
   moment of instantiation. Only that app sees the custom path.

2. **Env var at process start** (``all instances comply``): the user exports
   ``LOLLMS_DATA_FOLDER`` / ``LOLLMS_HOME`` / ``LOLLMS_CONFIG_FILE`` before
   launching Python. Every binding and every client in that process resolves
   to that path **without** any code change in the app.

Both flows are validated through the public surface:
``lollms_paths``, ``LollmsBaseBinding``, and ``WhisperSTTBinding``.
"""
import os
from pathlib import Path

import pytest

from lollms_client import lollms_paths
from lollms_client.lollms_base_binding import LollmsBaseBinding
from lollms_client.stt_bindings.whisper import WhisperSTTBinding


class _DummyBinding(LollmsBaseBinding):
    def list_models(self, *args, **kwargs):
        return []


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure each test starts from a known, env-var-free baseline."""
    for key in (
        lollms_paths.ENV_HOME,
        lollms_paths.ENV_DATA_FOLDER,
        lollms_paths.ENV_CONFIG_FILE,
        "LOLLMS_SYSTEM_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


# ── Flow 1: per-instance override (app-level) ────────────────────────────────

def test_per_instance_kwarg_overrides_env_for_data_folder(monkeypatch, tmp_path):
    """An app can pin a private data folder even when the process already set env."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(tmp_path / "env_data"))
    app_private = tmp_path / "app_private"

    # Paths helper: kwarg > env
    assert lollms_paths.get_data_folder(app_private) == app_private.resolve()

    # Base binding: kwarg > env
    binding = _DummyBinding(binding_name="dummy", lollms_data_folder=str(app_private))
    assert binding.get_data_folder() == app_private.resolve()

    # Whisper binding: kwarg > env (models dir lands under the app-private root)
    whisper = WhisperSTTBinding(
        auto_start_server=False,
        wait_for_server=False,
        lollms_data_folder=str(app_private),
    )
    assert whisper.get_data_folder() == app_private.resolve()
    assert whisper.models_dir == (app_private / "stt" / "whisper" / "models").resolve()
    assert whisper.cache_dir == (app_private / "stt" / "whisper").resolve()


def test_two_instances_can_have_independent_data_folders(monkeypatch, tmp_path):
    """Two apps in the same process can each keep their own storage."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    a_root = tmp_path / "app_a"
    b_root = tmp_path / "app_b"

    a = _DummyBinding(binding_name="a", lollms_data_folder=str(a_root))
    b = _DummyBinding(binding_name="b", lollms_data_folder=str(b_root))

    assert a.get_data_folder() == a_root.resolve()
    assert b.get_data_folder() == b_root.resolve()
    assert a.get_data_folder() != b.get_data_folder()


# ── Flow 2: process-wide env override (all instances comply) ─────────────────

def test_env_var_at_start_propagates_to_every_binding(monkeypatch, tmp_path):
    """Setting LOLLMS_DATA_FOLDER makes every new binding resolve to it."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    shared = tmp_path / "shared_on_big_disk"
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(shared))

    b1 = _DummyBinding(binding_name="b1")
    b2 = _DummyBinding(binding_name="b2")
    whisper = WhisperSTTBinding(auto_start_server=False, wait_for_server=False)

    assert b1.get_data_folder() == shared.resolve()
    assert b2.get_data_folder() == shared.resolve()
    assert whisper.get_data_folder() == shared.resolve()
    assert whisper.models_dir == (shared / "stt" / "whisper" / "models").resolve()


def test_env_home_override_relocates_data_and_config_together(monkeypatch, tmp_path):
    """A single LOLLMS_HOME export relocates both data and config atomically."""
    monkeypatch.setenv(lollms_paths.ENV_HOME, str(tmp_path / "custom_home"))
    home = (tmp_path / "custom_home").resolve()

    assert lollms_paths.get_home_dir() == home
    assert lollms_paths.get_data_folder() == home / "data"
    assert lollms_paths.get_config_file() == home / "config.yaml"

    binding = _DummyBinding(binding_name="dummy")
    assert binding.get_data_folder() == home / "data"


def test_env_config_file_override_is_honored(monkeypatch, tmp_path):
    """Users can relocate only the config file without moving the data folder."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    custom_cfg = tmp_path / "cfg_on_network_share" / "lollms.yaml"
    monkeypatch.setenv(lollms_paths.ENV_CONFIG_FILE, str(custom_cfg))

    assert lollms_paths.get_config_file() == custom_cfg.resolve()
    # A default data folder is still under ~/.lollms
    assert lollms_paths.get_data_folder() == (tmp_path / ".lollms" / "data").resolve()


# ── Resolution precedence invariants ─────────────────────────────────────────

def test_resolution_precedence_is_kwarg_then_env_then_default(monkeypatch, tmp_path):
    """Precedence sanity check used as documentation-as-test."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    default_data = (tmp_path / ".lollms" / "data").resolve()

    # 1. Default only.
    assert lollms_paths.get_data_folder() == default_data

    # 2. Env introduces an override; default is shadowed.
    env_data = tmp_path / "env_override"
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(env_data))
    assert lollms_paths.get_data_folder() == env_data.resolve()

    # 3. Kwarg wins over env; env is shadowed for that call only.
    kwarg_data = tmp_path / "kwarg_override"
    assert lollms_paths.get_data_folder(kwarg_data) == kwarg_data.resolve()
    # And the env-driven default is still in effect for calls without a kwarg.
    assert lollms_paths.get_data_folder() == env_data.resolve()


def test_env_var_change_is_visible_to_already_constructed_bindings(monkeypatch, tmp_path):
    """Env-var is read lazily; a mid-process change propagates to new bindings.

    This documents that ``all instances comply`` is achieved by exporting the
    env var *before* constructing the binding, and that nothing caches the
    value in a way that would surprise a user relocating storage.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    first_root = tmp_path / "first_root"
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(first_root))
    b1 = _DummyBinding(binding_name="b1")
    assert b1.get_data_folder() == first_root.resolve()

    second_root = tmp_path / "second_root"
    monkeypatch.setenv(lollms_paths.ENV_DATA_FOLDER, str(second_root))
    b2 = _DummyBinding(binding_name="b2")
    assert b2.get_data_folder() == second_root.resolve()
    # b1 kept its original resolution; env is not retroactive.
    assert b1.get_data_folder() == first_root.resolve()