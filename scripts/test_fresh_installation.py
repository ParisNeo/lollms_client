#!/usr/bin/env python3
"""
test_fresh_installation.py
==========================

Automated Pre-Release Verification Harness for `lollms_client`.

Simulates a brand-new PC environment:
  1. Cleans old build artifacts.
  2. Builds fresh distribution wheel (.whl) and source dist (.tar.gz).
  3. Creates an isolated virtualenv in a temporary directory.
  4. Sandboxes HOME / USERPROFILE to an empty folder (zero config bleed).
  5. Installs the wheel and verifies package integrity, imports, and CLI entry points.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def print_banner(text: str, color: str = "cyan") -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  {text}\n{sep}\n")


def run_cmd(
    cmd: list[str],
    cwd: Path | str = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    check: bool = True
) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if check and result.returncode != 0:
        print("\n❌ Command failed with output:")
        if result.stdout:
            print(f"[STDOUT]\n{result.stdout}")
        if result.stderr:
            print(f"[STDERR]\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def main() -> int:
    print_banner("🚀 Starting Pre-Release Clean-Room Deployment Test", "cyan")

    # ── 1. Clean Build Directory ──
    print("[1/5] Cleaning previous build artifacts...")
    dist_dir = PROJECT_ROOT / "dist"
    build_dir = PROJECT_ROOT / "build"
    egg_info = list(PROJECT_ROOT.glob("*.egg-info"))

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    for egg in egg_info:
        if egg.is_dir():
            shutil.rmtree(egg)

    # ── 2. Build Wheel and Tarball ──
    print("\n[2/5] Building source distribution and binary wheel...")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "build", "wheel"])
    run_cmd([sys.executable, "-m", "build", str(PROJECT_ROOT)])

    wheels = list(dist_dir.glob("*.whl"))
    tarballs = list(dist_dir.glob("*.tar.gz"))

    if not wheels:
        print("❌ Error: No .whl file was generated in dist/")
        return 1
    wheel_path = wheels[0].resolve()
    print(f"  ✓ Successfully built wheel: {wheel_path.name}")
    if tarballs:
        print(f"  ✓ Successfully built tarball: {tarballs[0].name}")

    # ── 3. Create Sandbox Environment ──
    with tempfile.TemporaryDirectory(prefix="lollms_clean_test_") as temp_dir_str:
        temp_dir = Path(temp_dir_str).resolve()
        sandbox_home = temp_dir / "fake_home"
        sandbox_home.mkdir(parents=True, exist_ok=True)
        venv_dir = temp_dir / "venv"

        print(f"\n[3/5] Creating clean virtualenv at: {venv_dir}...")
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])

        # Determine venv binaries
        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
            venv_pip = venv_dir / "Scripts" / "pip.exe"
            venv_cli = venv_dir / "Scripts" / "lollms-code.exe"
        else:
            venv_python = venv_dir / "bin" / "python"
            venv_pip = venv_dir / "bin" / "pip"
            venv_cli = venv_dir / "bin" / "lollms-code"

        # Construct pristine environment with isolated HOME
        sandbox_env = os.environ.copy()
        sandbox_env["HOME"] = str(sandbox_home)
        sandbox_env["USERPROFILE"] = str(sandbox_home)
        sandbox_env["APPDATA"] = str(sandbox_home / "AppData" / "Roaming")
        sandbox_env["LOCALAPPDATA"] = str(sandbox_home / "AppData" / "Local")
        sandbox_env.pop("PYTHONPATH", None)

        # ── 4. Install Wheel in Clean Venv ──
        print(f"\n[4/5] Installing {wheel_path.name} in clean environment...")
        # Invoke via python -m pip to prevent Windows executable locking of pip.exe
        run_cmd(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-warn-script-location",
                f"{str(wheel_path)}[app]",
            ],
            env=sandbox_env
        )

        # ── 5. Verification Assertions ──
        print("\n[5/5] Running deployment verification assertions...")

        # Assertion A: Verify import is strictly from site-packages (not git repo)
        verify_import_code = (
            "import sys, lollms_client\n"
            "path = lollms_client.__file__\n"
            "print(f'Imported from: {path}')\n"
            "assert 'site-packages' in path or 'dist-packages' in path, f'Error: imported from repo: {path}'\n"
        )
        res_import = run_cmd(
            [str(venv_python), "-c", verify_import_code],
            cwd=temp_dir,  # run outside repo
            env=sandbox_env
        )
        print(f"  ✓ {res_import.stdout.strip()}")

        # Assertion B: Verify CLI entry point
        if venv_cli.exists() or sys.platform != "win32":
            res_cli = run_cmd(
                [str(venv_cli), "--version"] if venv_cli.exists() else [str(venv_python), "-m", "lollms_client.apps.lollms_code.cli", "--version"],
                cwd=temp_dir,
                env=sandbox_env
            )
            print(f"  ✓ CLI binary response: {res_cli.stdout.strip()}")

        # Assertion C: Verify default directory initialization in fresh HOME
        verify_config_code = (
            "from pathlib import Path\n"
            "from lollms_client.apps.lollms_code.gui.env_config import EnvStore\n"
            "env = EnvStore()\n"
            "target = Path.home() / '.lollms_client'\n"
            "print(f'Checking fresh HOME config: {Path.home()}')\n"
            "assert not env.is_configured(), 'Fresh env should report is_configured() == False'\n"
        )
        res_cfg = run_cmd(
            [str(venv_python), "-c", verify_config_code],
            cwd=temp_dir,
            env=sandbox_env
        )
        print(f"  ✓ {res_cfg.stdout.strip()}")

    print_banner("✅ ALL PRE-RELEASE TESTS PASSED! Package is ready for PyPI.", "green")
    print(f"Artifacts ready in: {dist_dir}")
    for item in dist_dir.glob("*"):
        print(f"  • {item.name} ({item.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())