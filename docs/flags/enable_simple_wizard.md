# `enable_simple_wizard` — Hardware-Aware Guided Configuration

## Summary
Controls whether `lollms-code` offers the guided **Simple Profile** wizard as
the default first-time-setup experience.

## Behavior
- **Enabled (default)** — On first run (no LLM configured), lollms-code:
  1. Detects hardware (NVIDIA VRAM via `nvidia-smi`, Apple Silicon unified RAM,
     system RAM) and prints a banner with the detected tier.
  2. Presents an explicit choice: **Simple (pre-configured)** vs
     **Expert (manual)**. The Simple option previews the recommended model.
  3. In Simple mode, only lists **trusted** coding GGUFs that fit the detected
     VRAM budget *including KV-cache headroom* — so the default path never
     picks a model that would thrash system RAM.
  4. Downloads the GGUF from Unsloth / Bartowski / official vendor repos,
     installs the llama.cpp runtime, and registers the canonical two-tier
     profile map.
- **Disabled** — Skips the guided choice and jumps directly into the expert
  wizard (`run_wizard_and_save`), preserving pre-Simple-Profile behaviour.

## Model Catalog
The curated catalog is organised by hardware tier and sourced exclusively
from verified Hugging Face repos (Unsloth dynamic quants, Bartowski imatrix,
official vendor releases). Every entry declares `min_vram_gb` (including KV
cache headroom), `file_gb`, `ctx_size`, vision/tool-calling flags, and a
one-line rationale. CPU-only machines get small Qwen2.5-Coder builds.

Selection is guarded: picking an oversized model requires explicit
confirmation, and the safe recommendation remains preselected.

## CLI usage
```bash
# Guided simple flow
python -m lollms_client.lollms_config_cli_env --simple
lollms-code                    # will ask: simple or expert?

# Expert flow (everything)
python -m lollms_client.lollms_config_cli_env --expert
```

## GUI usage
The exact same hardware-aware flow is exposed in the GUI through
`lollms_client.apps.lollms_code.gui.env_config`:

- **`EnvStore.get_simple_setup_plan()`** — hardware scan + curated catalog
  (delegates to `lollms_config_simple.get_simple_setup_plan`).
- **`EnvStore.apply_simple_setup(model_id, enable_ttv=…, install_runtime=…,
  download=…)`** — installs the runtime, downloads the GGUF, registers the
  canonical two-tier profile, and saves via the wizard's own
  `_save_and_validate` (identical file bytes as the CLI produces). Call it
  from a NiceGUI `run.io_bound` worker so heavy I/O never blocks the event
  loop.
- **`build_quick_setup_ui(store, on_complete=…)`** — a ready-to-drop-in
  NiceGUI card offering the "Quick local setup" flow with the hardware tier
  badge, the curated model selector (recommended preselected), an optional
  TTV toggle, and background install. Returns `None` when NiceGUI isn't
  importable, so non-GUI callers can guard trivially.

Both the CLI and GUI surfaces share the same `_LLM_ZOO`, `detect_hardware()`,
`_fits_comfortably()`, and `_register_llm()` implementation — the two paths
cannot drift.

## Related
- `docs/flags/enable_artefacts.md`
- `docs/lcp/00_OVERVIEW.md`
- `src/lollms_client/lollms_config_simple.py` — CLI + headless implementation
- `src/lollms_client/apps/lollms_code/gui/env_config.py` — GUI façade (`EnvStore.apply_simple_setup`, `build_quick_setup_ui`)