"""
sub_workspace.py — Sub-Workspace & Reference Context Manager for lollms_code.
Enables importing external documentation, specifications, and reference code into
`.lollms_code/sub_workspace/` without cluttering the main repository tree.
Provides peeking, load/unload context control, and LLM prompt generation.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from ascii_colors import ASCIIColors

_EXPLICIT_BINARY_EXTS = {
    ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet",
    ".docx", ".pptx", ".odt", ".pdf", ".epub",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif", ".ico",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".xz", ".bz2", ".zst",
    ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx",
    ".h5", ".hdf5", ".gguf", ".pkl", ".pickle", ".joblib",
    ".npy", ".npz", ".msgpack", ".pb", ".tflite", ".mlmodel",
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".wma", ".aac",
    ".mp4", ".avi", ".mov", ".webm", ".mkv",
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe"
}

_IGNORED_NAMES = {
    "__pycache__", ".git", ".venv", "venv", "node_modules", ".DS_Store"
}


class SubWorkspaceManager:
    """Manages the reference sub-workspace directory inside .lollms_code/sub_workspace."""

    def __init__(self, workspace_path: Union[str, Path]):
        self.workspace_path = Path(workspace_path).resolve()
        self.sandbox_dir = self.workspace_path / ".lollms_code"
        self.sub_ws_dir = self.sandbox_dir / "sub_workspace"
        self.state_file = self.sandbox_dir / "sub_workspace_state.json"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.sub_ws_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {"loaded_files": []}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as e:
            ASCIIColors.warning(f"[SubWorkspace] Failed to read state: {e}")
            return {"loaded_files": []}

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            ASCIIColors.warning(f"[SubWorkspace] Failed to save state: {e}")

    def get_loaded_files(self) -> Set[str]:
        state = self._load_state()
        return set(state.get("loaded_files", []))

    def has_files(self) -> bool:
        if not self.sub_ws_dir.exists():
            return False
        for f in self.sub_ws_dir.rglob("*"):
            if f.is_file() and not self._is_ignored(f):
                return True
        return False

    def _is_ignored(self, p: Path) -> bool:
        for part in p.parts:
            if part in _IGNORED_NAMES or part.startswith("."):
                return True
        return False

    def list_files(self, query: str = "") -> List[Dict[str, Any]]:
        self.ensure_dirs()
        loaded_set = self.get_loaded_files()
        records: List[Dict[str, Any]] = []

        q = query.lower().strip()
        for f in sorted(self.sub_ws_dir.rglob("*")):
            if not f.is_file() or self._is_ignored(f):
                continue
            rel_str = str(f.relative_to(self.sub_ws_dir)).replace("\\", "/")
            if q and q not in rel_str.lower() and q not in f.name.lower():
                continue

            try:
                size = f.stat().st_size
                mtime = f.stat().st_mtime
            except OSError:
                size = 0
                mtime = 0.0

            records.append({
                "rel_path": rel_str,
                "name": f.name,
                "size": size,
                "mtime": mtime,
                "is_loaded": rel_str in loaded_set,
                "is_binary": self.is_binary_file(f),
                "full_path": str(f.resolve()),
            })
        return records

    @staticmethod
    def is_binary_file(p: Path) -> bool:
        if not p.exists() or not p.is_file():
            return False
        if p.suffix.lower() in _EXPLICIT_BINARY_EXTS:
            return True
        try:
            with open(p, "rb") as f:
                chunk = f.read(4096)
                return b"\x00" in chunk
        except Exception:
            return True

    def import_file(self, src: Union[str, Path], dest_rel: Optional[str] = None) -> Path:
        self.ensure_dirs()
        src_path = Path(src).resolve()
        if not src_path.exists() or not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src}")

        rel_dest_clean = dest_rel.replace("\\", "/").lstrip("/") if dest_rel else src_path.name
        dest_path = (self.sub_ws_dir / rel_dest_clean).resolve()

        if not str(dest_path).startswith(str(self.sub_ws_dir.resolve())):
            raise PermissionError(f"Path traversal blocked for destination: '{dest_rel}'")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(dest_path))
        return dest_path

    def import_folder(self, src: Union[str, Path], dest_rel: Optional[str] = None) -> List[Path]:
        self.ensure_dirs()
        src_dir = Path(src).resolve()
        if not src_dir.exists() or not src_dir.is_dir():
            raise NotADirectoryError(f"Source folder not found: {src}")

        target_base = (self.sub_ws_dir / dest_rel.replace("\\", "/").lstrip("/")) if dest_rel else (self.sub_ws_dir / src_dir.name)
        target_base.mkdir(parents=True, exist_ok=True)

        imported: List[Path] = []
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if d not in _IGNORED_NAMES and not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                f_path = Path(root) / fname
                rel = f_path.relative_to(src_dir)
                dest_file = target_base / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f_path), str(dest_file))
                imported.append(dest_file)
        return imported

    def load_file(self, rel_path: str) -> bool:
        clean = rel_path.replace("\\", "/").strip().lstrip("/")
        if clean.startswith("sub_workspace/"):
            clean = clean[len("sub_workspace/"):]

        target = self.sub_ws_dir / clean
        if not target.exists() or not target.is_file():
            return False

        state = self._load_state()
        loaded = set(state.get("loaded_files", []))
        loaded.add(clean)
        state["loaded_files"] = sorted(list(loaded))
        self._save_state(state)
        return True

    def unload_file(self, rel_path: str) -> bool:
        clean = rel_path.replace("\\", "/").strip().lstrip("/")
        if clean.startswith("sub_workspace/"):
            clean = clean[len("sub_workspace/"):]

        state = self._load_state()
        loaded = set(state.get("loaded_files", []))
        if clean in loaded:
            loaded.remove(clean)
            state["loaded_files"] = sorted(list(loaded))
            self._save_state(state)
            return True
        return False

    def load_all(self) -> int:
        files = self.list_files()
        state = self._load_state()
        all_rels = [f["rel_path"] for f in files]
        state["loaded_files"] = sorted(all_rels)
        self._save_state(state)
        return len(all_rels)

    def unload_all(self) -> int:
        state = self._load_state()
        count = len(state.get("loaded_files", []))
        state["loaded_files"] = []
        self._save_state(state)
        return count

    def peek_file(self, rel_path: str, max_chars: int = 40000) -> str:
        clean = rel_path.replace("\\", "/").strip().lstrip("/")
        if clean.startswith("sub_workspace/"):
            clean = clean[len("sub_workspace/"):]

        target = (self.sub_ws_dir / clean).resolve()
        if not target.exists() or not target.is_file():
            return f"[Error: Reference file '{rel_path}' not found on disk.]"

        if self.is_binary_file(target):
            size = target.stat().st_size
            return f"[Non-textual reference file: {clean} ({size:,} bytes). Raw binary content is withheld to protect the context window.]"

        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
            if len(content) > max_chars:
                return content[:max_chars] + f"\n... [truncated, {len(content) - max_chars:,} more chars]"
            return content
        except Exception as e:
            return f"[Error reading reference file '{rel_path}': {e}]"

    def remove_path(self, rel_path: str) -> bool:
        clean = rel_path.replace("\\", "/").strip().lstrip("/")
        if clean.startswith("sub_workspace/"):
            clean = clean[len("sub_workspace/"):]

        target = (self.sub_ws_dir / clean).resolve()
        if not str(target).startswith(str(self.sub_ws_dir.resolve())):
            return False

        if target.is_file():
            target.unlink()
            self.unload_file(clean)
            return True
        elif target.is_dir():
            shutil.rmtree(str(target))
            state = self._load_state()
            loaded = [p for p in state.get("loaded_files", []) if not p.startswith(clean + "/") and p != clean]
            state["loaded_files"] = loaded
            self._save_state(state)
            return True
        return False

    def build_context_block(self, client: Optional[Any] = None) -> str:
        files = self.list_files()
        if not files:
            return ""

        loaded_set = self.get_loaded_files()

        lines = [
            "=== SUB-WORKSPACE (REFERENCE & DOCUMENTATION) ===",
            "The following reference materials are available in `.lollms_code/sub_workspace/`.",
            "These are external documentation, guidelines, or reference code for your task.",
            "Use `<unlock_file>sub_workspace/<path></unlock_file>` to load a file into [C] context.",
            "Use `<lock_file>sub_workspace/<path></lock_file>` to unload a file when finished.",
            "",
            "## Reference Files Index:"
        ]

        for f in files:
            rel = f["rel_path"]
            marker = "[C]" if rel in loaded_set else "[U]"
            size_kb = f["size"] / 1024.0
            bin_tag = " (binary)" if f["is_binary"] else ""
            lines.append(f"- {marker} sub_workspace/{rel} ({size_kb:.1f} KB){bin_tag}")

        loaded_parts = []
        for rel in sorted(loaded_set):
            target = self.sub_ws_dir / rel
            if target.exists() and target.is_file():
                content = self.peek_file(rel, max_chars=30000)
                loaded_parts.append(
                    f"--- Reference File: sub_workspace/{rel} ---\n"
                    f"{content}\n"
                    f"--- End Reference File: sub_workspace/{rel} ---"
                )

        if loaded_parts:
            lines.append("\n## Loaded Reference Contents [C]:")
            lines.extend(loaded_parts)

        lines.append("=== END SUB-WORKSPACE ===")
        return "\n".join(lines)