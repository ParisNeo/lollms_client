from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ascii_colors import ASCIIColors


class Handbag:
    """
    A self-contained folder carrying ALL of an agent's resources.
    """
    def __init__(self, handbag_path: Union[str, Path]):
        self.path = Path(handbag_path).resolve()
        if not self.path.exists() or not self.path.is_dir():
            raise ValueError(f"Handbag path does not exist or is not a directory: {self.path}")

        self.soul_path = self.path / "SOUL.md"
        self.coworkers_dir = self.path / "coworkers"
        self.tools_dir = self.path / "tools"
        self._skills_dir = self.path / "skills"
        self.assets_dir = self.path / "assets"
        self.memory_dir = self.path / "memory"
        self.rag_dir = self.path / "rag"
        self.workspace_dir = self.path / "workspace"

        self.manifest = self._load_manifest()

        self.tool_files: List[Path] = self._load_tools()
        self.skills_dirs: List[Path] = [self._skills_dir.resolve()] if self._skills_dir.exists() else []
        self.rag_files: List[Path] = self._load_rag_files()
        self.memory_db_path: Optional[str] = f"sqlite:///{self.memory_dir / 'memory.db'}" if self.memory_dir.exists() else None
        self.assets: Dict[str, str] = self._load_assets()

    @property
    def global_memory_db_path(self) -> Optional[str]:
        """Returns the path to the global handbag memory database."""
        return self.memory_db_path

    @property
    def skills_dir(self) -> Path:
        """Returns the canonical skills directory for this handbag."""
        return self._skills_dir

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.path / "handbag.yaml"
        if not manifest_path.exists():
            return {}
        try:
            import yaml
            content = manifest_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_tools(self) -> List[Path]:
        if not self.tools_dir.exists():
            return []
        files = []
        for item in sorted(self.tools_dir.iterdir()):
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                files.append(item.resolve())
            elif item.is_dir():
                for py_file in sorted(item.glob("*.py")):
                    if py_file.stem != "__init__":
                        files.append(py_file.resolve())
        return files

    def _load_rag_files(self) -> List[Path]:
        if not self.rag_dir.exists():
            return []
        files = []
        for f in sorted(self.rag_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append(f.resolve())
        return files

    @staticmethod
    def create_structure(target_path: Union[str, Path], name: str = "My Handbag") -> Path:
        p = Path(target_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "coworkers").mkdir(exist_ok=True)
        (p / "tools").mkdir(exist_ok=True)
        (p / "skills").mkdir(exist_ok=True)
        (p / "rag").mkdir(exist_ok=True)
        (p / "memory").mkdir(exist_ok=True)
        (p / "assets").mkdir(exist_ok=True)
        (p / "workspace").mkdir(exist_ok=True)
        return p

    def _load_assets(self) -> Dict[str, str]:
        if not self.assets_dir.exists():
            return {}
        assets = {}
        for f in self.assets_dir.rglob("*"):
            if f.is_file():
                assets[f.name] = str(f.resolve())
        return assets

    def create_memory_manager(self) -> Optional[Any]:
        if not self.memory_db_path:
            return None
        try:
            from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig
            db_file = self.memory_db_path.replace("sqlite:///", "")
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
            manager = LollmsMemoryManager(
                db_path=self.memory_db_path,
                owner_id=f"handbag_{self.path.name}",
                config=MemoryConfig(working_token_budget=2000),
            )
            return manager
        except Exception:
            return None