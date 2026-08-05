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
        self.skills_dir = self.path / "skills"
        self.assets_dir = self.path / "assets"
        self.memory_dir = self.path / "memory"

        self.manifest = self._load_manifest()
        
        self.tool_files: List[Path] = self._load_tools()
        self.skills_dirs: List[Path] = [self.skills_dir.resolve()] if self.skills_dir.exists() else []
        self.memory_db_path: Optional[str] = f"sqlite:///{self.memory_dir / 'memory.db'}" if self.memory_dir.exists() else None
        self.assets: Dict[str, str] = self._load_assets()

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
                tool_file = item / f"{item.name}.py"
                if tool_file.exists():
                    files.append(tool_file.resolve())
                else:
                    for py_file in sorted(item.glob("*.py")):
                        if py_file.stem != "__init__":
                            files.append(py_file.resolve())
        return files

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