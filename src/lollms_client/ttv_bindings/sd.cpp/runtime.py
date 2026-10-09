"""Locates / installs the stable-diffusion.cpp CLI (sd-cli). No ComfyUI, no Python ML stack."""
import json, os, platform, shutil, stat, subprocess, sys, tarfile, zipfile
from pathlib import Path
from typing import Callable, List, Optional

import requests

Progress = Callable[[str], None]
API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
REPO = "https://github.com/leejet/stable-diffusion.cpp.git"
ACCEL_TOKENS = ("cuda", "vulkan", "rocm", "hip", "metal", "sycl", "opencl", "musa")
EXE = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"


def detect_accel() -> str:
    if sys.platform == "darwin":
        return "metal"
    return "cuda" if shutil.which("nvidia-smi") else "vulkan"


def choose_asset(names: List[str], system: str, machine: str, accel: str) -> Optional[str]:
    sysk = {"windows": ("win",), "linux": ("linux",), "darwin": ("darwin", "macos", "mac")}[system]
    arm = machine.lower() in ("arm64", "aarch64")

    def ok(n: str) -> bool:
        l = n.lower()
        if not (l.endswith(".zip") or l.endswith(".tar.gz")) or "bin" not in l or "cudart" in l:
            return False
        if not any(t in l for t in sysk):
            return False
        if system != "darwin" and (("arm64" in l or "aarch64" in l) != arm):
            return False
        return True

    c = [n for n in names if ok(n)]
    if accel == "cpu":
        pick = [n for n in c if not any(t in n.lower() for t in ACCEL_TOKENS)]
    else:
        pick = [n for n in c if accel in n.lower()]
    if not pick and system == "darwin":
        pick = c
    return sorted(pick, key=len)[0] if pick else None


def run_env(cli: Path) -> dict:
    env = dict(os.environ)
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = f"{cli.parent}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def supports_h3(cli: Path) -> bool:
    """Cheap capability probe: only builds from 2026-08-04 onward know --audio-vae."""
    try:
        r = subprocess.run([str(cli), "--help"], capture_output=True, text=True, timeout=60, env=run_env(cli))
        return "--audio-vae" in (r.stdout + r.stderr)
    except Exception:
        return False


def _find_cli(folder: Path) -> Optional[Path]:
    hits = [p for p in folder.rglob("*") if p.is_file() and p.name.lower() in ("sd-cli", "sd-cli.exe")]
    return sorted(hits, key=lambda p: len(str(p)))[0] if hits else None


def _download(url: str, dst: Path, progress: Progress):
    dst.parent.mkdir(parents=True, exist_ok=True)
    progress(f"downloading {url.rsplit('/', 1)[-1]}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def _extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def build_from_source(root: Path, progress: Progress) -> Path:
    for tool in ("git", "cmake", "nvcc"):
        if not shutil.which(tool):
            raise RuntimeError(f"building the CUDA binary needs '{tool}' in PATH")
    src = root / "src"
    if not (src / "CMakeLists.txt").exists():
        progress("cloning stable-diffusion.cpp")
        subprocess.run(["git", "clone", "--recursive", "--depth", "1", REPO, str(src)], check=True)
    progress("building (CUDA) - several minutes")
    subprocess.run(["cmake", "-S", str(src), "-B", str(src / "build"), "-DSD_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
                    "-DSD_BUILD_EXAMPLES=ON"], check=True)
    subprocess.run(["cmake", "--build", str(src / "build"), "--config", "Release", "-j"], check=True)
    cli = _find_cli(src / "build")
    if not cli:
        raise RuntimeError("build finished but sd-cli not found")
    return cli


def install_runtime(root: Path, accel: str = "auto", progress: Progress = print) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    accel = detect_accel() if accel in (None, "", "auto") else accel
    system = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
    cli = None
    if system == "linux" and accel == "cuda":
        # upstream ships no Linux CUDA prebuilt (as of the docs I found): build, else fall back to Vulkan
        try:
            cli = build_from_source(root, progress)
        except Exception as e:
            progress(f"CUDA build unavailable ({e}); falling back to the Vulkan prebuilt")
            accel = "vulkan"
    if cli is None:
        rel = requests.get(API, timeout=30, headers={"Accept": "application/vnd.github+json"}).json()
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        name = choose_asset(list(assets), system, platform.machine(), accel)
        if not name:
            raise RuntimeError(f"no release asset for {system}/{platform.machine()}/{accel}. Assets: {list(assets)}")
        dest = root / "bin" / rel.get("tag_name", "latest")
        arc = root / "dl" / name
        _download(assets[name], arc, progress)
        _extract(arc, dest)
        if system == "windows" and accel == "cuda":   # CUDA runtime DLLs ship separately on Windows
            rt = next((n for n in assets if "cudart" in n.lower() and "win" in n.lower()), None)
            if rt:
                _download(assets[rt], root / "dl" / rt, progress); _extract(root / "dl" / rt, dest)
        cli = _find_cli(dest)
        if not cli:
            raise RuntimeError(f"sd-cli not found inside {name}")
    if sys.platform != "win32":
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    if not supports_h3(cli):
        raise RuntimeError(f"{cli} does not know --audio-vae: too old for MiniMax-H3 (needs a build from 2026-08-04 or later)")
    (root / "current.json").write_text(json.dumps({"sd_cli": str(cli)}), encoding="utf-8")
    progress(f"sd-cli ready: {cli}")
    return cli


def ensure_sd_cli(root: Path, explicit: Optional[str], auto_install: bool, accel: str, progress: Progress) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"sd_cli_path does not exist: {p}")
        return p
    cur = root / "current.json"
    if cur.exists():
        p = Path(json.loads(cur.read_text(encoding="utf-8"))["sd_cli"])
        if p.exists():
            return p
    if not auto_install:
        raise RuntimeError("sd-cli not installed and auto_install is off (run the install_runtime command)")
    return install_runtime(root, accel, progress)
