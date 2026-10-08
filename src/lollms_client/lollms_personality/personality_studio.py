"""
personality_studio.py
Programmatic builder that crafts a Handbag from elements (soul, tools, skills,
coworkers, documents, manifest) and produces a ready-to-run LollmsPersonality.

All operations are idempotent: re-running them replaces the previous artifact.
Document ingestion delegates to Handbag.add_document (the single ingestion path).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional, Union

from ascii_colors import ASCIIColors

from .handbag import Handbag


def _sanitize_dir_name(name: str) -> str:
    cleaned = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in (name or "").strip()
    )
    return cleaned.strip("_").lower() or "unnamed"


def _quote(value: str) -> str:
    return '"' + (value or "").replace('"', "'").replace("\n", " ").strip() + '"'


class PersonalityStudio:
    """
    Crafts a Handbag folder from elements, then builds a LollmsPersonality from it.

    Example:
        studio = PersonalityStudio("./my_handbag")
        studio.set_soul(name="Researcher", system_prompt="...")
        studio.add_tool("./tools/web_search.py")
        studio.add_skill("./skills/bibliography_and_research")
        studio.add_coworker("coder", soul_prompt="You are a coder...")
        studio.add_document("./papers/attention.pdf")
        personality = studio.build(lollms_client=client)
    """

    def __init__(self, handbag_path: Union[str, Path]):
        self.path = Path(handbag_path).resolve()
        Handbag.create_structure(self.path)
        self._handbag = Handbag(self.path)

    @property
    def soul_path(self) -> Path:
        return self.path / "SOUL.md"

    def set_soul(
        self,
        name: str,
        system_prompt: str,
        author: str = "studio",
        version: str = "1.0",
        category: str = "general",
        description: str = "",
        temperature: Optional[float] = None,
    ) -> Path:
        """Writes the SOUL.md (YAML frontmatter + system prompt body)."""
        lines = [
            "---",
            f"name: {_quote(name)}",
            f"author: {_quote(author)}",
            f"version: {_quote(version)}",
            f"category: {_quote(category)}",
        ]
        if description:
            lines.append(f"description: {_quote(description)}")
        if temperature is not None:
            lines.append(f"temperature: {temperature}")
        lines.extend(["---", "", (system_prompt or "").strip()])
        self.soul_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.soul_path

    def add_tool(self, tool_path: Union[str, Path]) -> Path:
        """Copies a .py tool library file (or a tool directory) into the handbag's tools/."""
        source = Path(tool_path)
        if not source.exists():
            raise FileNotFoundError(f"Tool source not found: {source}")
        target_root = self.path / "tools"
        if source.is_dir():
            target = target_root / _sanitize_dir_name(source.name)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        elif source.suffix.lower() == ".py":
            target = target_root / source.name
            shutil.copy2(source, target)
        else:
            raise ValueError("add_tool expects a .py tool library file or a tool directory.")
        ASCIIColors.success(f"[PersonalityStudio] Tool added: {target.relative_to(self.path)}")
        return target

    def add_skill(self, skill_path: Union[str, Path]) -> Path:
        """Copies a SKILL.md file (or a skill directory containing one) into skills/."""
        source = Path(skill_path)
        if not source.exists():
            raise FileNotFoundError(f"Skill source not found: {source}")
        target_root = self.path / "skills"
        if source.is_dir():
            target = target_root / _sanitize_dir_name(source.name)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        elif source.suffix.lower() == ".md":
            target = target_root / _sanitize_dir_name(source.stem)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / "SKILL.md")
        else:
            raise ValueError("add_skill expects a SKILL.md file or a skill directory.")
        ASCIIColors.success(f"[PersonalityStudio] Skill added: {target.relative_to(self.path)}")
        return target

    def add_coworker(
        self,
        name: str,
        soul_prompt: str,
        author: str = "studio",
        category: str = "general",
        description: str = "",
    ) -> Path:
        """Creates a coworker persona (Crew Handbag) under coworkers/<name>/SOUL.md."""
        coworker_dir = self.path / "coworkers" / _sanitize_dir_name(name)
        coworker_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            f"name: {_quote(name)}",
            f"author: {_quote(author)}",
            f"category: {_quote(category)}",
        ]
        if description:
            lines.append(f"description: {_quote(description)}")
        lines.extend(["---", "", (soul_prompt or "").strip()])
        (coworker_dir / "SOUL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ASCIIColors.success(f"[PersonalityStudio] Coworker added: {name}")
        return coworker_dir

    def add_document(
        self,
        file_path: Union[str, Path],
        use_llm: bool = False,
        lollms_client: Optional[Any] = None,
    ) -> Path:
        """Ingests a document (txt/md/pdf/docx/pptx) into the handbag's docs/ library."""
        return self._handbag.add_document(
            file_path, use_llm=use_llm, lollms_client=lollms_client
        )

    def set_manifest(self, **fields) -> Path:
        """Writes handbag.yaml (e.g. skills_mode='loadable'). Re-loads the handbag after."""
        manifest_path = self.path / "handbag.yaml"
        try:
            import yaml
            manifest_path.write_text(
                yaml.safe_dump(fields, allow_unicode=True), encoding="utf-8"
            )
        except ImportError:
            lines = [f"{key}: {value}" for key, value in fields.items()]
            manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._handbag = Handbag(self.path)
        return manifest_path

    def build(
        self,
        lollms_client: Optional[Any] = None,
        extra_tools: Optional[list] = None,
        extra_skills_dirs: Optional[list] = None,
    ) -> "LollmsPersonality":
        """Builds the final LollmsPersonality from the crafted handbag."""
        if not self.soul_path.exists():
            raise ValueError(
                "The handbag has no SOUL.md. Call set_soul() before build()."
            )
        from .lollms_personality import LollmsPersonality
        return LollmsPersonality.from_handbag(
            self.path,
            lollms_client=lollms_client,
            extra_tools=extra_tools,
            extra_skills_dirs=extra_skills_dirs,
        )