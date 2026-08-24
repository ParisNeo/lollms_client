from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .skill import Skill, parse_skill_md


class SkillsManager:
    """
    Manages SKILL.md files from external directories.
    Tiers: "visible" (in sys prompt), "loadable" (listed, tool_load_skill), "searchable" (hidden, tool_search_skills).
    """

    def __init__(
        self,
        skills_dirs: Optional[List[Union[str, Path]]] = None,
        mode: str = "mixed",
        max_visible_skills: int = 10,
        max_visible_tokens: int = 4000
    ):
        self.mode = mode
        self.max_visible_skills = max_visible_skills
        self.max_visible_tokens = max_visible_tokens
        self._skills_dirs: List[Path] = []
        if skills_dirs:
            for d in skills_dirs:
                p = Path(d)
                if p.exists() and p.is_dir():
                    self._skills_dirs.append(p.resolve())
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def _resolve_visibility(self, skill: Skill) -> str:
        # If the skill is text-only (no metadata), always load it unless mode is explicitly searchable
        if not skill.has_metadata:
            return "visible" if self.mode != "searchable" else "searchable"

        if skill.visibility == "visible":
            return "visible"
        if skill.visibility == "searchable":
            return "searchable"
        if self.mode == "visible":
            return "visible"
        if self.mode == "searchable":
            return "searchable"
        return "loadable"

    def reload(self):
        self.skills.clear()
        seen_paths = set()
        for d in self._skills_dirs:
            self._scan_directory(d, seen_paths)

    def _scan_directory(self, directory: Path, seen_paths: set):
        if not directory.exists() or not directory.is_dir():
            return

        direct_skill = directory / "SKILL.md"
        if direct_skill.exists() and direct_skill.resolve() not in seen_paths:
            seen_paths.add(direct_skill.resolve())
            skill = parse_skill_md(direct_skill, default_visibility=self.mode)
            if skill:
                skill.visibility = self._resolve_visibility(skill)
                self.skills[skill.title.lower()] = skill
            return

        for item in sorted(directory.iterdir()):
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists() and skill_file.resolve() not in seen_paths:
                    seen_paths.add(skill_file.resolve())
                    skill = parse_skill_md(skill_file, default_visibility=self.mode)
                    if skill:
                        skill.visibility = self._resolve_visibility(skill)
                        self.skills[skill.title.lower()] = skill
            elif item.is_file() and item.suffix.lower() == ".md" and item.name != "README.md":
                if item.resolve() not in seen_paths:
                    seen_paths.add(item.resolve())
                    skill = parse_skill_md(item, default_visibility=self.mode)
                    if skill:
                        skill.visibility = self._resolve_visibility(skill)
                        self.skills[skill.title.lower()] = skill

    def build_context(self) -> str:
        parts = []

        raw_visible = [s for s in self.skills.values() if s.visibility == "visible"]
        active_visible = []
        overflow_loadable = []

        # Enforce budget: prevent context window saturation if too many skills are visible
        used_chars = 0
        max_chars = self.max_visible_tokens * 4

        for s in raw_visible:
            s_len = len(s.content)
            if len(active_visible) < self.max_visible_skills and (used_chars + s_len <= max_chars or not active_visible):
                active_visible.append(s)
                used_chars += s_len
            else:
                overflow_loadable.append(s)

        if active_visible:
            lines = ["=== ACTIVE SKILLS (Always Visible) ==="]
            for skill in active_visible:
                lines.append(f"\n--- Skill: {skill.title} ---")
                if skill.description:
                    lines.append(f"Description: {skill.description}")
                lines.append(f"\n{skill.content}")
                lines.append(f"--- End Skill: {skill.title} ---")
            lines.append("=== END ACTIVE SKILLS ===")
            parts.append("\n".join(lines))

        loadable = [s for s in self.skills.values() if s.visibility == "loadable"] + overflow_loadable
        if loadable:
            lines = ["=== AVAILABLE SKILLS (Loadable on Demand) ==="]
            lines.append(f"There are {len(loadable)} loadable skills. Use the `tool_load_skill` tool to load the full content of any skill listed below.")
            lines.append("")
            for skill in loadable:
                desc = skill.description or (skill.content.splitlines()[0][:100] if skill.content else "No description")
                cat = f" [{skill.category}]" if skill.category else ""
                lines.append(f"- **{skill.title}**{cat}: {desc}")
            lines.append("=== END AVAILABLE SKILLS ===")
            parts.append("\n".join(lines))

        searchable = [s for s in self.skills.values() if s.visibility == "searchable"]
        if searchable:
            lines = ["=== SEARCHABLE SKILLS ==="]
            lines.append(f"There are {len(searchable)} hidden skills. Use `tool_search_skills` to find them by keyword.")
            lines.append("=== END SEARCHABLE SKILLS ===")
            parts.append("\n".join(lines))

        if not parts:
            return "\n=== SKILLS SYSTEM ===\nThere are currently 0 skills in the library.\n=== END SKILLS SYSTEM ==="

        return "\n\n".join(parts)

    def search_skills(self, query: str) -> List[Skill]:
        query_lower = query.lower()
        results = []
        for skill in self.skills.values():
            score = 0
            if query_lower in skill.title.lower():
                score += 3
            if query_lower in skill.description.lower():
                score += 2
            if any(query_lower in tag.lower() for tag in skill.tags):
                score += 2
            if query_lower in skill.content.lower():
                score += 1
            if score > 0:
                results.append((score, skill))
        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]

    def load_skill(self, title: str) -> Optional[str]:
        skill = self.skills.get(title.lower())
        if skill:
            return f"--- Skill: {skill.title} ---\n{skill.content}\n--- End Skill: {skill.title} ---"
        matches = self.search_skills(title)
        if matches:
            skill = matches[0]
            return f"--- Skill: {skill.title} ---\n{skill.content}\n--- End Skill: {skill.title} ---"
        return None

    def list_skills(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.skills.values()]

    def has_searchable_skills(self) -> bool:
        return any(s.visibility == "searchable" for s in self.skills.values())

    def build_skill_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Conditionally builds tool specifications for skill management based on visibility tiers.
        - `tool_list_skills` is registered ONLY if there is at least 1 skill in the library.
        - `tool_load_skill` is registered if there are 'loadable' or 'searchable' skills.
        - `tool_search_skills` is registered ONLY if there is at least one 'searchable' skill.
        """
        tools: Dict[str, Dict[str, Any]] = {}

        raw_visible_count = sum(1 for s in self.skills.values() if s.visibility == "visible")
        has_loadable = any(s.visibility == "loadable" for s in self.skills.values()) or (raw_visible_count > self.max_visible_skills)
        has_searchable = self.has_searchable_skills()
        total_skills = len(self.skills)

        if total_skills > 0:
            def tool_list_skills() -> dict:
                """
                Lists all available skills in the library, categorized by their visibility tier (visible, loadable, searchable).
                Use this to get an overview of what knowledge is available.
                """
                visible = [s.to_dict() for s in self.skills.values() if s.visibility == "visible"]
                loadable = [s.to_dict() for s in self.skills.values() if s.visibility == "loadable"]
                searchable = [s.to_dict() for s in self.skills.values() if s.visibility == "searchable"]
                
                report = {
                    "visible_skills": visible,
                    "loadable_skills": loadable,
                    "searchable_skills": searchable,
                    "total_count": total_skills
                }
                return {"success": True, "output": report}

            tools["tool_list_skills"] = {
                "name": "tool_list_skills",
                "description": "Lists all available skills in the library, categorized by their visibility tier (visible, loadable, searchable).",
                "parameters": [],
                "callable": tool_list_skills,
            }

        if has_loadable or has_searchable:
            def tool_load_skill(title: str) -> dict:
                """
                Load the full content of a skill by title. Use this to access detailed instructions.

                Args:
                    title (str): The title of the skill to load (case-insensitive).
                """
                content = self.load_skill(title)
                if content:
                    return {"success": True, "output": content}
                return {"success": False, "error": f"Skill '{title}' not found."}

            tools["tool_load_skill"] = {
                "name": "tool_load_skill",
                "description": "Load the full content of a skill by title. Skills contain reusable knowledge, instructions, and best practices.",
                "parameters": [
                    {"name": "title", "type": "str", "description": "The title of the skill to load."}
                ],
                "callable": tool_load_skill,
            }

        if has_searchable:
            def tool_search_skills(query: str) -> dict:
                """
                Search for hidden skills by keyword. Use this to find hidden skills before loading them.

                Args:
                    query (str): The search keyword or phrase.
                """
                matches = self.search_skills(query)
                if not matches:
                    return {"success": True, "output": "No matching skills found."}

                lines = ["Matching skills:"]
                for skill in matches:
                    cat = f" [{skill.category}]" if skill.category else ""
                    lines.append(f"- **{skill.title}**{cat}: {skill.description or 'No description'}")
                return {"success": True, "output": "\n".join(lines)}

            tools["tool_search_skills"] = {
                "name": "tool_search_skills",
                "description": "Search for hidden skills by keyword. Use this to discover skills before loading them with tool_load_skill.",
                "parameters": [
                    {"name": "query", "type": "str", "description": "The search keyword or phrase."}
                ],
                "callable": tool_search_skills,
            }

        return tools