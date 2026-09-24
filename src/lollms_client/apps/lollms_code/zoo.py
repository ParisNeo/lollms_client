"""
zoo.py — Unified Package Hub & Zoo Manager for lollms_code.
===========================================================

Manages official GitHub Zoo repositories:
  - Tools Zoo:         https://github.com/ParisNeo/lollms_tools_zoo.git
  - Skills Zoo:        https://github.com/ParisNeo/lollms_skills_zoo.git
  - Personalities Zoo: https://github.com/ParisNeo/lollms_personalities_zoo.git

Provides:
  1. Multi-level recursive category and subcategory discovery.
  2. Distinct detection of terminal items vs. categories/subcategories.
  3. Reading of both category-level README.md and item-level README.md/SKILL.md/SOUL.md.
  4. Project Scope (.lollms_code/) and Global Scope (~/.lollms_client/) installation.
  5. Interactive terminal CLI Navigator and GUI support.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors, Menu

try:
    import yaml
except ImportError:
    yaml = None


ZOO_REPOS = {
    "tools": "https://github.com/ParisNeo/lollms_tools_zoo.git",
    "skills": "https://github.com/ParisNeo/lollms_skills_zoo.git",
    "personalities": "https://github.com/ParisNeo/lollms_personalities_zoo.git",
}

ZOO_LABELS = {
    "tools": "🛠️ Tools Zoo",
    "skills": "🧠 Skills Zoo",
    "personalities": "🎭 Personalities Zoo",
}

DEFAULT_ZOO_CACHE_DIR = Path.home() / ".lollms_client" / "zoos"
GLOBAL_APPS_DIR = Path.home() / ".lollms_client" / "lollms_code"

_IGNORED_SCAN_DIRS = {
    ".git", ".github", "__pycache__", ".vscode", ".idea",
    "venv", ".venv", "node_modules", ".DS_Store", "build", "dist"
}


@dataclass
class ZooItem:
    """Represents an individual tool, skill, or persona package."""
    zoo_type: str
    category: str              # Full category path, e.g. "math / algebra"
    top_category: str          # Top-level category, e.g. "math"
    sub_category: str          # Subcategory, e.g. "algebra"
    name: str
    path: Path
    is_installed_project: bool = False
    is_installed_global: bool = False
    description: str = ""
    author: str = ""
    has_readme: bool = False
    readme_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_installed(self) -> bool:
        return self.is_installed_project or self.is_installed_global


@dataclass
class ZooCategory:
    """Represents a category or subcategory node."""
    zoo_type: str
    full_path: str             # e.g. "math / algebra" or "math"
    name: str                  # Display name, e.g. "algebra" or "math"
    top_level: str             # Top-level parent, e.g. "math"
    is_subcategory: bool = False
    path: Optional[Path] = None
    readme_content: str = ""
    items_count: int = 0


class ZooManager:
    """
    Core engine for syncing, browsing, and installing Zoo packages with multi-level category support.
    """

    def __init__(self, workspace_path: Optional[Union[str, Path]] = None):
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
        self.cache_dir = DEFAULT_ZOO_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_repo_dir(self, zoo_type: str) -> Path:
        return self.cache_dir / f"{zoo_type}_zoo"

    def get_project_target_dir(self, zoo_type: str) -> Path:
        sandbox = self.workspace_path / ".lollms_code"
        type_folder_map = {
            "tools": "tools",
            "skills": "skills",
            "personalities": "handbags",
        }
        target = sandbox / type_folder_map.get(zoo_type, zoo_type)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_global_target_dir(self, zoo_type: str) -> Path:
        type_folder_map = {
            "tools": "tools",
            "skills": "skills",
            "personalities": "handbags",
        }
        target = GLOBAL_APPS_DIR / type_folder_map.get(zoo_type, zoo_type)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def is_repo_cloned(self, zoo_type: str) -> bool:
        repo_dir = self.get_repo_dir(zoo_type)
        return repo_dir.exists() and any(repo_dir.iterdir())

    def sync_repo(
        self,
        zoo_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str]:
        """Clones or updates the given zoo repository using Git, falling back to ZIP download."""
        repo_url = ZOO_REPOS.get(zoo_type)
        if not repo_url:
            return False, f"Unknown zoo type: {zoo_type}"

        dest_dir = self.get_repo_dir(zoo_type)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        def _log(msg: str):
            if progress_callback:
                progress_callback(msg)
            else:
                ASCIIColors.info(f"[ZooSync] {msg}")

        # Method 1: Git CLI
        git_executable = shutil.which("git")
        if git_executable:
            if (dest_dir / ".git").exists():
                _log(f"Updating {zoo_type} zoo repository via git pull...")
                try:
                    res = subprocess.run(
                        [git_executable, "pull", "--ff-only"],
                        cwd=str(dest_dir),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        timeout=90,
                    )
                    if res.returncode == 0:
                        _log(f"Successfully updated {zoo_type} zoo.")
                        return True, f"Updated {zoo_type} zoo successfully."
                    else:
                        ASCIIColors.warning(f"[ZooSync] Git pull failed ({res.stderr.strip()}). Re-cloning...")
                except Exception as ex:
                    ASCIIColors.warning(f"[ZooSync] Git pull error: {ex}")

            if not dest_dir.exists() or not (dest_dir / ".git").exists():
                _log(f"Cloning {zoo_type} zoo repository from {repo_url}...")
                if dest_dir.exists():
                    shutil.rmtree(str(dest_dir), ignore_errors=True)
                try:
                    res = subprocess.run(
                        [git_executable, "clone", "--depth", "1", repo_url, str(dest_dir)],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        timeout=180,
                    )
                    if res.returncode == 0:
                        _log(f"Successfully cloned {zoo_type} zoo.")
                        return True, f"Cloned {zoo_type} zoo successfully."
                except Exception as ex:
                    ASCIIColors.warning(f"[ZooSync] Git clone failed: {ex}")

        # Method 2: HTTP ZIP Download Fallback
        _log(f"Fetching {zoo_type} zoo archive via HTTP fallback...")
        base_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        zip_urls = [
            f"https://github.com/ParisNeo/{base_name}/archive/refs/heads/main.zip",
            f"https://github.com/ParisNeo/{base_name}/archive/refs/heads/master.zip",
        ]

        for z_url in zip_urls:
            try:
                _log(f"Downloading archive: {z_url}")
                req = urllib.request.Request(z_url, headers={"User-Agent": "lollms_code-zoo-client"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    zip_data = resp.read()

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_p = Path(temp_dir)
                    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                        zf.extractall(temp_p)

                    extracted_dirs = [d for d in temp_p.iterdir() if d.is_dir()]
                    if extracted_dirs:
                        if dest_dir.exists():
                            shutil.rmtree(str(dest_dir), ignore_errors=True)
                        shutil.copytree(str(extracted_dirs[0]), str(dest_dir), dirs_exist_ok=True)
                        _log(f"Extracted {zoo_type} zoo archive successfully.")
                        return True, f"Downloaded and unpacked {zoo_type} zoo successfully."
            except Exception as dl_err:
                ASCIIColors.warning(f"[ZooSync] Failed downloading {z_url}: {dl_err}")

        return False, f"Failed to sync {zoo_type} zoo from {repo_url}."

    def sync_all(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Tuple[bool, str]]:
        results = {}
        for z_type in ZOO_REPOS:
            results[z_type] = self.sync_repo(z_type, progress_callback=progress_callback)
        return results

    # ── Item & Category Identification Rules ───────────────────────────────

    def _is_item(self, path: Path, zoo_type: str) -> bool:
        """
        Determines if a filesystem path is a terminal item package (tool, skill, or persona),
        as opposed to a category directory containing child items or subcategories.
        """
        if zoo_type == "personalities":
            if not path.is_dir():
                return False
            return (
                (path / "config.yaml").is_file()
                or (path / "config.yml").is_file()
                or (path / "SOUL.md").is_file()
                or (path / "soul.md").is_file()
                or (path / "personality.yaml").is_file()
            )

        if zoo_type == "skills":
            if path.is_dir():
                if (path / "SKILL.md").is_file() or (path / "skill.md").is_file() or (path / "skill.yaml").is_file():
                    return True
                sub_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name not in _IGNORED_SCAN_DIRS]
                if not sub_dirs and any(f.is_file() and f.suffix.lower() == ".md" and f.name.lower() != "readme.md" for f in path.iterdir()):
                    return True
                return False
            if path.is_file():
                return path.suffix.lower() == ".md" and path.name.lower() != "readme.md"
            return False

        if zoo_type == "tools":
            if path.is_file():
                return path.suffix.lower() == ".py" and path.name not in ("__init__.py", "setup.py")
            if not path.is_dir():
                return False
            # Check for standard LCP tool marker files
            if (path / "description.yaml").is_file() or (path / "description.yml").is_file():
                return True
            if (path / f"{path.name}.py").is_file() or (path / "tool.py").is_file() or (path / "main.py").is_file():
                return True

            py_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".py" and f.name != "__init__.py"]
            if py_files:
                sub_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name not in _IGNORED_SCAN_DIRS]
                # If there are no child subdirectories, this directory is a tool package
                if not sub_dirs:
                    return True
                # If subdirectories are NOT tools (e.g. assets, data), it is a tool package
                if not any(self._is_item(sd, "tools") for sd in sub_dirs):
                    return True
            return False

        return False

    def _normalize_category_parts(self, rel_parent: Path, zoo_type: str) -> List[str]:
        """
        Strips redundant top-level wrapper directories (e.g. 'tools/', 'skills/', 'personalities/')
        from category paths so categories are presented cleanly (e.g. 'math / algebra' rather than 'tools / math / algebra').
        """
        parts = [p for p in rel_parent.parts if p not in _IGNORED_SCAN_DIRS]
        if parts and parts[0].lower() in (zoo_type.lower(), f"{zoo_type.lower()}_zoo", "zoos"):
            parts = parts[1:]
        return parts

    # ── Recursive Discovery Engine ──────────────────────────────────────────

    def _discover_all(self, zoo_type: str) -> Tuple[List[ZooItem], List[ZooCategory]]:
        repo_dir = self.get_repo_dir(zoo_type)
        if not repo_dir.exists():
            return [], []

        items: List[ZooItem] = []
        categories_dict: Dict[str, ZooCategory] = {}
        category_dirs_seen: Dict[str, Path] = {}

        project_dir = self.get_project_target_dir(zoo_type)
        global_dir = self.get_global_target_dir(zoo_type)

        def _scan(current_dir: Path):
            if not current_dir.exists() or not current_dir.is_dir():
                return

            try:
                entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except Exception:
                return

            for entry in entries:
                if entry.name.startswith(".") or entry.name in _IGNORED_SCAN_DIRS:
                    continue

                if self._is_item(entry, zoo_type):
                    # Item detected!
                    rel_parent = entry.parent.relative_to(repo_dir)
                    parts = self._normalize_category_parts(rel_parent, zoo_type)

                    full_cat = " / ".join(parts) if parts else "general"
                    top_cat = parts[0] if parts else "general"
                    sub_cat = " / ".join(parts[1:]) if len(parts) > 1 else ""

                    # Register category paths
                    category_dirs_seen[full_cat] = entry.parent
                    if top_cat not in category_dirs_seen and parts:
                        category_dirs_seen[top_cat] = repo_dir / parts[0] if not (repo_dir / zoo_type).exists() else (repo_dir / zoo_type / parts[0])

                    item_name = entry.stem if entry.is_file() else entry.name
                    readme_content = ""
                    description = ""
                    author = "Community"
                    meta_data: Dict[str, Any] = {}

                    doc_candidates = [
                        entry / "README.md" if entry.is_dir() else None,
                        entry / "SKILL.md" if entry.is_dir() else None,
                        entry / "SOUL.md" if entry.is_dir() else None,
                        entry / "description.yaml" if entry.is_dir() else None,
                        entry / "config.yaml" if entry.is_dir() else None,
                        entry if entry.is_file() and entry.suffix.lower() in (".md", ".py") else None,
                    ]

                    for doc in doc_candidates:
                        if doc and doc.exists():
                            try:
                                raw = doc.read_text(encoding="utf-8", errors="ignore")
                                if not readme_content and doc.name.lower().endswith((".md", ".txt", ".yaml", ".yml")):
                                    readme_content = raw

                                if doc.name.endswith(".yaml") and yaml:
                                    y_data = yaml.safe_load(raw)
                                    if isinstance(y_data, dict):
                                        meta_data.update(y_data)
                                        author = y_data.get("author") or author
                                        description = (
                                            y_data.get("description")
                                            or y_data.get("personality_description")
                                            or description
                                        )

                                if raw.startswith("---"):
                                    fm_match = re.match(r"^---\n(.*?)\n---", raw, re.DOTALL)
                                    if fm_match:
                                        for line in fm_match.group(1).splitlines():
                                            if ":" in line:
                                                k, v = line.split(":", 1)
                                                k = k.strip().lower()
                                                v = v.strip().strip("'\"")
                                                meta_data[k] = v
                                                if k == "author":
                                                    author = v
                                                elif k == "description":
                                                    description = v

                                if not description:
                                    for line in raw.splitlines():
                                        line_s = line.strip()
                                        if line_s and not line_s.startswith("#") and not line_s.startswith("---"):
                                            description = line_s[:160]
                                            break
                            except Exception:
                                pass

                    is_inst_proj = (project_dir / entry.name).exists() or (project_dir / f"{item_name}.py").exists()
                    is_inst_glob = (global_dir / entry.name).exists() or (global_dir / f"{item_name}.py").exists()

                    items.append(
                        ZooItem(
                            zoo_type=zoo_type,
                            category=full_cat,
                            top_category=top_cat,
                            sub_category=sub_cat,
                            name=item_name,
                            path=entry,
                            is_installed_project=is_inst_proj,
                            is_installed_global=is_inst_glob,
                            description=description or f"{zoo_type.capitalize()} item in {full_cat}",
                            author=author,
                            has_readme=bool(readme_content),
                            readme_content=readme_content,
                            metadata=meta_data,
                        )
                    )
                elif entry.is_dir():
                    # Recurse down into categories and subcategories
                    _scan(entry)

        _scan(repo_dir)

        # Build clean category and subcategory index
        cat_counts: Dict[str, int] = {}
        for it in items:
            cat_counts[it.category] = cat_counts.get(it.category, 0) + 1
            if it.sub_category:
                cat_counts[it.top_category] = cat_counts.get(it.top_category, 0) + 1

        for cat_str, count in sorted(cat_counts.items()):
            parts = [p.strip() for p in cat_str.split("/") if p.strip()]
            top_level = parts[0] if parts else "general"
            is_sub = len(parts) > 1
            cat_name = parts[-1] if parts else "general"

            cat_dir = category_dirs_seen.get(cat_str)
            readme_text = ""
            if cat_dir and (cat_dir / "README.md").exists():
                try:
                    readme_text = (cat_dir / "README.md").read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass

            categories_dict[cat_str] = ZooCategory(
                zoo_type=zoo_type,
                full_path=cat_str,
                name=cat_name,
                top_level=top_level,
                is_subcategory=is_sub,
                path=cat_dir,
                readme_content=readme_text,
                items_count=count,
            )

        return items, list(categories_dict.values())

    def list_categories(self, zoo_type: str) -> List[ZooCategory]:
        _, categories = self._discover_all(zoo_type)
        return categories

    def list_items(self, zoo_type: str, category: Optional[str] = None) -> List[ZooItem]:
        items, _ = self._discover_all(zoo_type)
        if not category:
            return items

        c_target = category.strip().lower()
        matched: List[ZooItem] = []
        for it in items:
            # Matches exact full category ("math / algebra") or top-level parent ("math")
            if it.category.lower() == c_target or it.top_category.lower() == c_target:
                matched.append(it)
        return matched

    def search(self, query: str, zoo_type: Optional[str] = None) -> List[ZooItem]:
        q = query.lower().strip()
        targets = [zoo_type] if zoo_type else list(ZOO_REPOS.keys())
        results: List[ZooItem] = []
        for z in targets:
            items, _ = self._discover_all(z)
            for item in items:
                if (
                    q in item.name.lower()
                    or q in item.category.lower()
                    or q in item.description.lower()
                    or q in item.readme_content.lower()
                ):
                    results.append(item)
        return results

    # ── Installation & Uninstallation ──────────────────────────────────────

    def install_item(
        self,
        item: ZooItem,
        scope: str = "project",
        overwrite: bool = True,
    ) -> Tuple[bool, str]:
        """Installs a tool, skill, or persona package into the specified scope."""
        target_dir = self.get_project_target_dir(item.zoo_type) if scope == "project" else self.get_global_target_dir(item.zoo_type)
        target_dir.mkdir(parents=True, exist_ok=True)

        dest_path = target_dir / item.path.name

        try:
            if item.path.is_dir():
                if dest_path.exists():
                    if not overwrite:
                        return False, f"Item already exists at {dest_path}"
                    shutil.rmtree(str(dest_path))
                shutil.copytree(str(item.path), str(dest_path), dirs_exist_ok=True)
            else:
                if item.zoo_type == "tools":
                    # For standalone tool .py files, create a dedicated directory to conform to LCP structure
                    dest_tool_dir = target_dir / item.name
                    dest_tool_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(item.path), str(dest_tool_dir / item.path.name))
                    dest_path = dest_tool_dir
                elif item.zoo_type == "skills":
                    dest_skill_dir = target_dir / item.name
                    dest_skill_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(item.path), str(dest_skill_dir / "SKILL.md"))
                    dest_path = dest_skill_dir
                else:
                    shutil.copy2(str(item.path), str(dest_path))

            # Synthesize SOUL.md for personalities if only config.yaml is present
            if item.zoo_type == "personalities":
                handbag_p = dest_path if dest_path.is_dir() else dest_path.parent
                soul_p = handbag_p / "SOUL.md"
                config_yaml_p = handbag_p / "config.yaml"

                if not soul_p.exists() and config_yaml_p.exists() and yaml:
                    try:
                        y_text = config_yaml_p.read_text(encoding="utf-8", errors="ignore")
                        y_cfg = yaml.safe_load(y_text) or {}
                        p_name = y_cfg.get("name") or item.name.replace("_", " ").title()
                        p_author = y_cfg.get("author") or item.author or "Community"
                        p_cat = y_cfg.get("category") or item.top_category or "general"
                        p_desc = y_cfg.get("personality_description") or y_cfg.get("description") or item.description
                        p_cond = y_cfg.get("personality_conditioning") or y_cfg.get("conditioning") or f"You are {p_name}."

                        soul_body = f"""---
name: "{p_name}"
author: "{p_author}"
category: "{p_cat}"
description: "{p_desc}"
---

{p_cond}
"""
                        soul_p.write_text(soul_body, encoding="utf-8")
                        ASCIIColors.info(f"[ZooManager] Synthesized SOUL.md for '{p_name}'.")
                    except Exception as soul_err:
                        ASCIIColors.warning(f"[ZooManager] Could not synthesize SOUL.md: {soul_err}")

                if handbag_p.is_dir():
                    for sub in ("coworkers", "tools", "skills", "memory", "workspace"):
                        (handbag_p / sub).mkdir(exist_ok=True)

            if scope == "project":
                item.is_installed_project = True
            else:
                item.is_installed_global = True

            scope_desc = f"Project ({self.workspace_path.name})" if scope == "project" else "Global User Directory"
            return True, f"Installed '{item.name}' ({item.category}) to {scope_desc} successfully."
        except Exception as ex:
            return False, f"Failed to install '{item.name}': {ex}"

    def uninstall_item(self, item: ZooItem, scope: str = "project") -> Tuple[bool, str]:
        target_dir = self.get_project_target_dir(item.zoo_type) if scope == "project" else self.get_global_target_dir(item.zoo_type)
        dest_path = target_dir / item.path.name
        alt_path = target_dir / item.name

        removed_any = False
        for p in (dest_path, alt_path):
            if p.exists():
                try:
                    if p.is_dir():
                        shutil.rmtree(str(p))
                    else:
                        p.unlink()
                    removed_any = True
                except Exception as ex:
                    return False, f"Failed to delete {p}: {ex}"

        if removed_any:
            if scope == "project":
                item.is_installed_project = False
            else:
                item.is_installed_global = False
            return True, f"Uninstalled '{item.name}' from {scope} successfully."
        return False, f"'{item.name}' was not found in {scope}."

    def list_installed_items(self, zoo_type: str, scope: str = "all") -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        def _scan(base_p: Path, scope_label: str):
            if not base_p.exists():
                return
            for entry in sorted(base_p.iterdir()):
                if entry.name.startswith(".") or entry.name in ("__pycache__", "default_coder"):
                    continue
                results.append({
                    "name": entry.stem if entry.is_file() else entry.name,
                    "zoo_type": zoo_type,
                    "scope": scope_label,
                    "path": str(entry.resolve()),
                    "is_dir": entry.is_dir(),
                })

        if scope in ("all", "project"):
            _scan(self.get_project_target_dir(zoo_type), "project")
        if scope in ("all", "global"):
            _scan(self.get_global_target_dir(zoo_type), "global")

        return results

    def activate_persona_in_session(self, persona_name: str, config: Any, personality: Any, client: Any) -> Tuple[bool, str, Any]:
        """Switches the active session persona to an installed handbag."""
        proj_p = self.get_project_target_dir("personalities") / persona_name
        glob_p = self.get_global_target_dir("personalities") / persona_name

        selected_path = proj_p if proj_p.exists() else (glob_p if glob_p.exists() else None)
        if not selected_path:
            return False, f"Persona handbag '{persona_name}' is not installed.", personality

        try:
            from lollms_client.apps.lollms_code.cli import create_coding_personality
            config.handbag_path = str(selected_path.resolve())
            config.save()
            new_pers = create_coding_personality(config, client)
            return True, f"Active persona switched to '{new_pers.name}' ({selected_path.name}).", new_pers
        except Exception as ex:
            return False, f"Failed to activate persona '{persona_name}': {ex}", personality


# ── Interactive Terminal CLI Navigator ───────────────────────────────────────

def run_cli_zoo_navigator(config: Any, client: Optional[Any] = None, personality: Optional[Any] = None) -> Optional[Any]:
    """Interactive terminal REPL navigator for LoLLMS Zoos."""
    manager = ZooManager(config.workspace_path)
    active_pers = personality

    while True:
        ASCIIColors.rule("[bold cyan]🦁 LoLLMS Zoo Package Hub & Navigator[/bold cyan]")
        ASCIIColors.rich_print("Browse, inspect documentation, and install community Tools, Skills, and Personas.\n")

        main_menu = Menu("Package Hub Main Menu", mode=Menu.MODE_RETURN, exit_text="↩ Back to Agent")
        main_menu.add_choice("🛠️  Browse Tools Zoo", value="tools")
        main_menu.add_choice("🧠 Browse Skills Zoo", value="skills")
        main_menu.add_choice("🎭 Browse Personalities Zoo", value="personalities")
        main_menu.add_choice("🔍 Search All Zoos", value="search")
        main_menu.add_choice("🔄 Synchronize / Update All Zoos", value="sync")
        main_menu.add_choice("📦 List Installed Packages", value="installed")
        main_menu.add_choice("↩ Back to Agent", value="__BACK__")

        choice = main_menu.run()
        if choice in ("__BACK__", None, False) or (isinstance(choice, str) and choice.startswith("↩")):
            break

        if choice == "sync":
            ASCIIColors.info("\nSynchronizing zoo repositories...")
            for z in ZOO_REPOS:
                ok, msg = manager.sync_repo(z)
                if ok:
                    ASCIIColors.success(f"  ✓ {ZOO_LABELS[z]}: {msg}")
                else:
                    ASCIIColors.warning(f"  ⚠ {ZOO_LABELS[z]}: {msg}")
            input("\nPress Enter to continue...")

        elif choice == "installed":
            _render_installed_summary(manager)

        elif choice == "search":
            q = input("\nEnter search keyword: ").strip()
            if q:
                hits = manager.search(q)
                _render_item_results_cli(hits, manager, config, client, active_pers)

        elif choice in ("tools", "skills", "personalities"):
            active_pers = _browse_zoo_cli(choice, manager, config, client, active_pers)

    return active_pers


def _render_installed_summary(manager: ZooManager):
    ASCIIColors.rule("[bold green]📦 Installed Packages Summary[/bold green]")
    for z in ("tools", "skills", "personalities"):
        ASCIIColors.rich_print(f"\n[bold yellow]{ZOO_LABELS[z]}[/bold yellow]")
        items = manager.list_installed_items(z)
        if not items:
            ASCIIColors.rich_print("  [dim](none)[/dim]")
        else:
            for it in items:
                badge = "[green][Project][/green]" if it["scope"] == "project" else "[cyan][Global][/cyan]"
                ASCIIColors.rich_print(f"  • {badge} [bold]{it['name']}[/bold] ({it['path']})")
    input("\nPress Enter to continue...")


def _browse_zoo_cli(
    zoo_type: str, manager: ZooManager, config: Any, client: Any, personality: Any
) -> Any:
    if not manager.is_repo_cloned(zoo_type):
        ASCIIColors.warning(f"\n{ZOO_LABELS[zoo_type]} is not synced locally yet.")
        if input("Clone repository now? [y/n]: ").strip().lower().startswith("y"):
            ok, msg = manager.sync_repo(zoo_type)
            if not ok:
                ASCIIColors.red(f"Sync failed: {msg}")
                input("Press Enter to continue...")
                return personality

    while True:
        categories = manager.list_categories(zoo_type)
        menu = Menu(f"{ZOO_LABELS[zoo_type]} — Categories", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro(f"Select a category or subcategory to explore.")

        for c in categories:
            indent = "    ↳ " if c.is_subcategory else "📁 "
            menu.add_choice(f"{indent}{c.full_path} ({c.items_count} items)", value=c.full_path)

        menu.add_choice("🔄 Update Zoo Repo", value="__SYNC__")
        menu.add_choice("↩ Back", value="__BACK__")

        c_choice = menu.run()
        if c_choice in ("__BACK__", None, False) or (isinstance(c_choice, str) and c_choice.startswith("↩")):
            break
        if c_choice == "__SYNC__":
            manager.sync_repo(zoo_type)
            input("Press Enter to continue...")
            continue

        selected_cat = next((c for c in categories if c.full_path == c_choice), None)
        if selected_cat:
            personality = _browse_category_cli(selected_cat, manager, config, client, personality)

    return personality


def _browse_category_cli(
    cat: ZooCategory, manager: ZooManager, config: Any, client: Any, personality: Any
) -> Any:
    while True:
        items = manager.list_items(cat.zoo_type, category=cat.full_path)
        ASCIIColors.rule(f"[bold cyan]📁 Category: {cat.full_path} ({len(items)} items)[/bold cyan]")

        menu = Menu(f"{cat.full_path} Packages", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro("Select an item to inspect documentation and install.")

        if cat.readme_content:
            menu.add_choice(f"📖 View Category README.md ({cat.name})", value="__CAT_README__")

        for it in items:
            status_tag = ""
            if it.is_installed_project and it.is_installed_global:
                status_tag = " [green]✓ Project+Global[/green]"
            elif it.is_installed_project:
                status_tag = " [green]✓ Project[/green]"
            elif it.is_installed_global:
                status_tag = " [cyan]✓ Global[/cyan]"
            sub_tag = f" [dim]({it.category})[/dim]" if it.category != cat.full_path else ""
            menu.add_choice(f"• {it.name}{sub_tag}{status_tag}", value=it.name)

        menu.add_choice("↩ Back", value="__BACK__")

        choice = menu.run()
        if choice in ("__BACK__", None, False) or (isinstance(choice, str) and choice.startswith("↩")):
            break

        if choice == "__CAT_README__":
            ASCIIColors.panel(
                cat.readme_content,
                title=f"[bold]📖 {cat.full_path} Documentation[/bold]",
                border_style="cyan",
            )
            input("Press Enter to continue...")
            continue

        target_item = next((it for it in items if it.name == choice), None)
        if target_item:
            personality = _inspect_item_cli(target_item, manager, config, client, personality)

    return personality


def _inspect_item_cli(
    item: ZooItem, manager: ZooManager, config: Any, client: Any, personality: Any
) -> Any:
    while True:
        ASCIIColors.rule(f"[bold magenta]Package: {item.name}[/bold magenta]")
        lines = [
            f"[cyan]Name:[/cyan]        {item.name}",
            f"[cyan]Category:[/cyan]    {item.category}",
            f"[cyan]Zoo Type:[/cyan]    {item.zoo_type}",
            f"[cyan]Author:[/cyan]      {item.author}",
            f"[cyan]Description:[/cyan] {item.description}",
            f"[cyan]Installed:[/cyan]   {'Project' if item.is_installed_project else ''} {'Global' if item.is_installed_global else ''}" or "Not Installed",
        ]
        ASCIIColors.rich_print("\n".join(lines))

        if item.readme_content:
            ASCIIColors.panel(
                item.readme_content[:3000] + ("\n... [truncated]" if len(item.readme_content) > 3000 else ""),
                title=f"[bold]📖 {item.name} Documentation[/bold]",
                border_style="blue",
            )

        menu = Menu(f"Action: {item.name}", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.add_choice("📦 Install to Project Workspace (.lollms_code/)", value="inst_proj")
        menu.add_choice("🌐 Install Globally (~/.lollms_client/)", value="inst_glob")
        if item.is_installed_project:
            menu.add_choice("🗑️ Uninstall from Project", value="uninst_proj")
        if item.is_installed_global:
            menu.add_choice("🗑️ Uninstall Globally", value="uninst_glob")
        if item.zoo_type == "personalities" and item.is_installed:
            menu.add_choice("🎭 Activate Persona for Current Session", value="activate_persona")
        menu.add_choice("↩ Back", value="__BACK__")

        act = menu.run()
        if act in ("__BACK__", None, False) or (isinstance(act, str) and act.startswith("↩")):
            break

        if act == "inst_proj":
            ok, msg = manager.install_item(item, scope="project")
            if ok:
                ASCIIColors.success(f"\n{msg}")
            else:
                ASCIIColors.error(f"\n{msg}")
            input("Press Enter to continue...")

        elif act == "inst_glob":
            ok, msg = manager.install_item(item, scope="global")
            if ok:
                ASCIIColors.success(f"\n{msg}")
            else:
                ASCIIColors.error(f"\n{msg}")
            input("Press Enter to continue...")

        elif act == "uninst_proj":
            ok, msg = manager.uninstall_item(item, scope="project")
            if ok:
                ASCIIColors.success(f"\n{msg}")
            else:
                ASCIIColors.error(f"\n{msg}")
            input("Press Enter to continue...")

        elif act == "uninst_glob":
            ok, msg = manager.uninstall_item(item, scope="global")
            if ok:
                ASCIIColors.success(f"\n{msg}")
            else:
                ASCIIColors.error(f"\n{msg}")
            input("Press Enter to continue...")

        elif act == "activate_persona":
            ok, msg, new_p = manager.activate_persona_in_session(item.name, config, personality, client)
            if ok:
                ASCIIColors.success(f"\n{msg}")
                personality = new_p
            else:
                ASCIIColors.error(f"\n{msg}")
            input("Press Enter to continue...")

    return personality


def _render_item_results_cli(
    items: List[ZooItem], manager: ZooManager, config: Any, client: Any, personality: Any
):
    if not items:
        ASCIIColors.yellow("No matching items found.")
        input("Press Enter to continue...")
        return

    menu = Menu(f"Search Results ({len(items)} items)", mode=Menu.MODE_RETURN, exit_text="↩ Back")
    for it in items[:30]:
        menu.add_choice(f"[{it.zoo_type}] {it.category}/{it.name}", value=it.name)
    menu.add_choice("↩ Back", value="__BACK__")

    choice = menu.run()
    if choice and choice != "__BACK__":
        target = next((it for it in items if it.name == choice), None)
        if target:
            _inspect_item_cli(target, manager, config, client, personality)