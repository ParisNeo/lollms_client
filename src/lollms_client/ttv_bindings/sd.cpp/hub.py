"""Hugging Face download helpers + user model profiles."""
import fnmatch, json, shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

Progress = Callable[[str], None]
SUBFOLDERS = ("diffusion_models", "vae", "text_encoders", "other")


def list_repo_files(repo_id: str, token: Optional[str] = None, revision: Optional[str] = None) -> List[str]:
    from huggingface_hub import list_repo_files as _l
    return _l(repo_id, token=token or None, revision=revision)


def download_file(repo_id: str, filename: str, dest: Path, token: Optional[str] = None,
                  revision: Optional[str] = None, progress: Optional[Progress] = None) -> Path:
    """Downloads one repo file to dest/<basename>. Idempotent."""
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(filename).name
    if target.exists():
        return target
    if progress: progress(f"downloading {repo_id}/{filename}")
    got = hf_hub_download(repo_id, filename, token=token or None, revision=revision, local_dir=str(dest / ".dl"))
    shutil.move(got, target)
    shutil.rmtree(dest / ".dl", ignore_errors=True)
    return target


def find_exact(files: List[str], name: str) -> Optional[str]:
    return next((f for f in files if f == name or Path(f).name == name), None)


def match_pattern(files: List[str], pattern: str) -> List[str]:
    return [f for f in files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)]


class Profiles:
    """User-registered models: {name: {family, files:{role: path}, args:[...], mode}} in profiles.json."""
    ROLE_FLAGS = {"diffusion_model": "--diffusion-model", "vae": "--vae", "audio_vae": "--audio-vae",
                  "llm": "--llm", "llm_vision": "--llm_vision"}

    def __init__(self, models_path: Path):
        self.path = models_path / "profiles.json"

    def load(self) -> Dict[str, dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def register(self, name: str, files: Dict[str, str], args: Optional[List[str]] = None, mode: str = "vid_gen") -> dict:
        bad = [r for r in files if r not in self.ROLE_FLAGS]
        if bad:
            raise ValueError(f"unsupported roles {bad}; valid: {list(self.ROLE_FLAGS)}. Put other flags in args.")
        missing = [p for p in files.values() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"files not found: {missing}")
        data = self.load()
        data[name] = {"files": {k: str(Path(v).resolve()) for k, v in files.items()}, "args": list(args or []), "mode": mode}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data[name]
