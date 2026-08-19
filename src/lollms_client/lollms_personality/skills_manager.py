from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .skill import Skill, parse_skill_md


class SkillsManager:
    """
    Manages SKILL.md files from external directories.
    Tiers: "visible" (in sys prompt), "loadable" (listed, tool_load_skill), "searchable" (hidden, tool_search_skills).
    """

    def __init__(self, skills_dirs: Optional[List[Union[str, Path]]] = None, mode: str = "mixed"):
        self.mode = mode
        self._skills_dirs: List[Path] = []
        if skills_dirs:
            for d in skills_dirs:
                p = Path(d)
                if p.exists() and p.is_dir():
                    self._skills_dirs.append(p.resolve())
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def _resolve_visibility(self, parsed_visibility: str) -> str:
        if parsed_visibility == "visible":
            return "visible"
        if parsed_visibility == "searchable":
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
                skill.visibility = self._resolve_visibility(skill.visibility)
                self.skills[skill.title.lower()] = skill
            return

        for item in sorted(directory.iterdir()):
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists() and skill_file.resolve() not in seen_paths:
                    seen_paths.add(skill_file.resolve())
                    skill = parse_skill_md(skill_file, default_visibility=self.mode)
                    if skill:
                        skill.visibility = self._resolve_visibility(skill.visibility)
                        self.skills[skill.title.lower()] = skill
            elif item.is_file() and item.suffix.lower() == ".md" and item.name != "README.md":
                if item.resolve() not in seen_paths:
                    seen_paths.add(item.resolve())
                    skill = parse_skill_md(item, default_visibility=self.mode)
                    if skill:
                        skill.visibility = self._resolve_visibility(skill.visibility)
                        self.skills[skill.title.lower()] = skill

    def build_context(self) -> str:
        parts = []
        
        visible = [s for s in self.skills.values() if s.visibility == "visible"]
        if visible:
            lines = ["=== ACTIVE SKILLS (Always Visible) ==="]
            for skill in visible:
                lines.append(f"\n--- Skill: {skill.title} ---")
                if skill.description:
                    lines.append(f"Description: {skill.description}")
                lines.append(f"\n{skill.content}")
                lines.append(f"--- End Skill: {skill.title} ---")
            lines.append("=== END ACTIVE SKILLS ===")
            parts.append("\n".join(lines))

        loadable = [s for s in self.skills.values() if s.visibility == "loadable"]
        if loadable:
            lines = ["=== AVAILABLE SKILLS (Loadable on Demand) ==="]
            lines.append(f"There are {len(loadable)} loadable skills. Use the `tool_load_skill` tool to load the full content of any skill listed below.")
            lines.append("")
            for skill in loadable:
                desc = skill.description or "No description"
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

        has_loadable = any(s.visibility == "loadable" for s in self.skills.values())
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

        if self._skills_dirs:
            def tool_create_skill(name: str, description: str, content: str, category: str = "general", tags: str = "", visibility: str = "loadable") -> dict:
                """
                Creates a new SKILL.md file in the skills directory.

                Args:
                    name (str): The title of the skill (will be used as filename).
                    description (str): A short description of what the skill does.
                    content (str): The full markdown content of the skill.
                    category (str, optional): The category for the skill. Defaults to "general".
                    tags (str, optional): Comma-separated tags. Defaults to "".
                    visibility (str, optional): "visible", "loadable", or "searchable". Defaults to "loadable".
                """
                if not name or not content:
                    return {"success": False, "error": "Name and content are required."}

                safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
                skill_dir = self._skills_dirs[0] / safe_name
                skill_dir.mkdir(parents=True, exist_ok=True)
                
                skill_file = skill_dir / "SKILL.md"
                
                tags_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
                
                yaml_lines = [
                    "---",
                    f"title: {name}",
                    f"description: {description}",
                    f"category: {category}",
                    f"tags: [{', '.join(tags_list)}]",
                    f"visibility: {visibility}",
                    "---"
                ]
                full_content = "\n".join(yaml_lines) + "\n\n" + content
                
                skill_file.write_text(full_content, encoding="utf-8")
                self.reload()
                
                return {"success": True, "output": f"Skill '{name}' created successfully at {skill_file}."}

            tools["tool_create_skill"] = {
                "name": "tool_create_skill",
                "description": "Creates a new SKILL.md file to save a reusable methodology or pattern for future sessions.",
                "parameters": [
                    {"name": "name", "type": "str", "description": "The title of the skill."},
                    {"name": "description", "type": "str", "description": "A short description of what the skill does."},
                    {"name": "content", "type": "str", "description": "The full markdown content of the skill."},
                    {"name": "category", "type": "str", "description": "The category for the skill.", "optional": True},
                    {"name": "tags", "type": "str", "description": "Comma-separated tags.", "optional": True},
                    {"name": "visibility", "type": "str", "description": "Visibility tier: visible, loadable, or searchable.", "optional": True}
                ],
                "callable": tool_create_skill,
            }

            def tool_update_skill(title: str, content: str, description: str = "", category: str = "", tags: str = "", visibility: str = "") -> dict:
                """
                Updates an existing SKILL.md file.

                Args:
                    title (str): The title of the skill to update.
                    content (str): The new full markdown content.
                    description (str, optional): New description. If empty, keeps existing.
                    category (str, optional): New category. If empty, keeps existing.
                    tags (str, optional): New comma-separated tags. If empty, keeps existing.
                    visibility (str, optional): New visibility. If empty, keeps existing.
                """
                skill = self.skills.get(title.lower())
                if not skill:
                    matches = self.search_skills(title)
                    if not matches:
                        return {"success": False, "error": f"Skill '{title}' not found."}
                    skill = matches[0]

                if not skill.file_path or not skill.file_path.exists():
                    return {"success": False, "error": f"Skill file path not found for '{title}'."}

                tags_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else skill.tags
                
                yaml_lines = [
                    "---",
                    f"title: {skill.title}",
                    f"description: {description or skill.description}",
                    f"category: {category or skill.category}",
                    f"tags: [{', '.join(tags_list)}]",
                    f"visibility: {visibility or skill.visibility}",
                    "---"
                ]
                full_content = "\n".join(yaml_lines) + "\n\n" + content
                
                skill.file_path.write_text(full_content, encoding="utf-8")
                self.reload()
                
                return {"success": True, "output": f"Skill '{skill.title}' updated successfully."}

            tools["tool_update_skill"] = {
                "name": "tool_update_skill",
                "description": "Updates an existing SKILL.md file with new content or metadata.",
                "parameters": [
                    {"name": "title", "type": "str", "description": "The title of the skill to update."},
                    {"name": "content", "type": "str", "description": "The new full markdown content."},
                    {"name": "description", "type": "str", "description": "New description.", "optional": True},
                    {"name": "category", "type": "str", "description": "New category.", "optional": True},
                    {"name": "tags", "type": "str", "description": "New comma-separated tags.", "optional": True},
                    {"name": "visibility", "type": "str", "description": "New visibility tier.", "optional": True}
                ],
                "callable": tool_update_skill,
            }

        return tools
    
    