from __future__ import annotations

import re
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

    def _sanitize_title(self, title: str) -> str:
        safe_title = re.sub(r'[^\w\-]', '_', title).strip('_')
        return safe_title or "unnamed_skill"

    def create_skill(
        self,
        title: str,
        content: str,
        description: str = "",
        category: str = "",
        tags: Optional[List[str]] = None,
        visibility: str = "loadable"
    ) -> Optional[Skill]:
        if not self._skills_dirs:
            return None

        target_dir = self._skills_dirs[0]
        safe_title = self._sanitize_title(title)
        
        skill_dir = target_dir / safe_title
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"

        tags_str = ", ".join(tags) if tags else ""
        
        frontmatter = "---\n"
        frontmatter += f"title: \"{title}\"\n"
        if description:
            frontmatter += f"description: \"{description}\"\n"
        if category:
            frontmatter += f"category: \"{category}\"\n"
        if tags_str:
            frontmatter += f"tags: [{tags_str}]\n"
        frontmatter += f"visibility: {visibility}\n"
        frontmatter += "---\n\n"

        skill_path.write_text(frontmatter + content.strip() + "\n", encoding="utf-8")
        
        self.reload()
        return self.skills.get(title.lower())

    def update_skill(
        self,
        title: str,
        content: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Skill]:
        skill = self.skills.get(title.lower())
        if not skill or not skill.file_path:
            matches = self.search_skills(title)
            if matches:
                skill = matches[0]
        
        if not skill or not skill.file_path:
            return None

        if not skill.modifiable:
            return None

        target_dir = skill.file_path.parent
        safe_title = self._sanitize_title(title)
        skill_path = target_dir / "SKILL.md"

        tags_str = ", ".join(tags) if tags else (", ".join(skill.tags) if skill.tags else "")
        
        frontmatter = "---\n"
        frontmatter += f"title: \"{skill.title}\"\n"
        final_desc = description if description is not None else skill.description
        if final_desc:
            frontmatter += f"description: \"{final_desc}\"\n"
        final_cat = category if category is not None else skill.category
        if final_cat:
            frontmatter += f"category: \"{final_cat}\"\n"
        if tags_str:
            frontmatter += f"tags: [{tags_str}]\n"
        frontmatter += f"visibility: {skill.visibility}\n"
        frontmatter += f"modifiable: {'true' if skill.modifiable else 'false'}\n"
        frontmatter += "---\n\n"

        skill_path.write_text(frontmatter + content.strip() + "\n", encoding="utf-8")
        
        self.reload()
        return self.skills.get(skill.title.lower())

    def append_to_skill(
        self,
        title: str,
        content: str
    ) -> Optional[Skill]:
        skill = self.skills.get(title.lower())
        if not skill or not skill.file_path:
            matches = self.search_skills(title)
            if matches:
                skill = matches[0]
        
        if not skill or not skill.file_path:
            return None

        if not skill.modifiable:
            return None

        existing_content = skill.file_path.read_text(encoding="utf-8", errors="ignore")
        
        separator = "\n\n---\n\n"
        new_content = existing_content.rstrip() + separator + content.strip() + "\n"
        skill.file_path.write_text(new_content, encoding="utf-8")
        
        self.reload()
        return self.skills.get(skill.title.lower())

    def remove_skill(self, title: str) -> bool:
        skill = self.skills.get(title.lower())
        if not skill or not skill.file_path:
            matches = self.search_skills(title)
            if matches:
                skill = matches[0]
        
        if not skill or not skill.file_path:
            return False

        if not skill.modifiable:
            return False

        skill_path = skill.file_path
        parent_dir = skill_path.parent

        try:
            skill_path.unlink()
            
            if parent_dir != self._skills_dirs[0] and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
        except Exception:
            return False

        self.reload()
        return True

    def build_context(self) -> str:
        parts = []

        raw_visible = [s for s in self.skills.values() if s.visibility == "visible"]
        active_visible = []
        overflow_loadable = []

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
                if not skill.modifiable:
                    lines.append("\n🚨 **READ-ONLY SKILL (UNMODIFIABLE)** 🚨")
                    lines.append("You are STRICTLY FORBIDDEN from updating, patching, or appending to this skill. Any `<skill>` tag attempting to modify it WILL BE BLOCKED by the system.\n")
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
                mod_str = " (Read-Only)" if not skill.modifiable else ""
                lines.append(f"- **{skill.title}**{cat}{mod_str}: {desc}")
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
            mod_prefix = ""
            if not skill.modifiable:
                mod_prefix = "🚨 **READ-ONLY SKILL (UNMODIFIABLE)** 🚨\nYou are STRICTLY FORBIDDEN from updating, patching, or appending to this skill.\n\n"
            return f"--- Skill: {skill.title} ---\n{mod_prefix}{skill.content}\n--- End Skill: {skill.title} ---"
        matches = self.search_skills(title)
        if matches:
            skill = matches[0]
            mod_prefix = ""
            if not skill.modifiable:
                mod_prefix = "🚨 **READ-ONLY SKILL (UNMODIFIABLE)** 🚨\nYou are STRICTLY FORBIDDEN from updating, patching, or appending to this skill.\n\n"
            return f"--- Skill: {skill.title} ---\n{mod_prefix}{skill.content}\n--- End Skill: {skill.title} ---"
        return None

    def list_skills(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.skills.values()]

    def has_searchable_skills(self) -> bool:
        return any(s.visibility == "searchable" for s in self.skills.values())

    def build_skill_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Conditionally builds tool specifications for skill management based on visibility tiers.
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

        def tool_create_skill(
            title: str,
            content: str,
            description: str = "",
            category: str = "",
            tags: str = "",
            visibility: str = "loadable"
        ) -> dict:
            """
            Create a new persistent skill (SKILL.md) that survives across sessions.
            Use this when you discover a reusable methodology, workaround, or best practice.

            Args:
                title (str): A concise, descriptive title for the skill.
                content (str): The full Markdown content of the skill.
                description (str, optional): A one-sentence summary. Defaults to "".
                category (str, optional): A category for grouping. Defaults to "".
                tags (str, optional): Comma-separated tags for searchability. Defaults to "".
                visibility (str, optional): "visible", "loadable", or "searchable". Defaults to "loadable".
            """
            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            skill = self.create_skill(
                title=title,
                content=content,
                description=description,
                category=category,
                tags=tags_list,
                visibility=visibility
            )
            if skill:
                return {"success": True, "output": f"Skill '{title}' created successfully."}
            return {"success": False, "error": "Failed to create skill."}

        tools["tool_create_skill"] = {
            "name": "tool_create_skill",
            "description": "Create a new persistent skill (SKILL.md) to save reusable knowledge, methodologies, or workarounds.",
            "parameters": [
                {"name": "title", "type": "str", "description": "A concise, descriptive title for the skill."},
                {"name": "content", "type": "str", "description": "The full Markdown content of the skill."},
                {"name": "description", "type": "str", "description": "A one-sentence summary.", "optional": True},
                {"name": "category", "type": "str", "description": "A category for grouping.", "optional": True},
                {"name": "tags", "type": "str", "description": "Comma-separated tags for searchability.", "optional": True},
                {"name": "visibility", "type": "str", "description": "Visibility tier: 'visible', 'loadable', or 'searchable'.", "optional": True}
            ],
            "callable": tool_create_skill,
        }

        def tool_update_skill(
            title: str,
            content: str,
            description: str = "",
            category: str = "",
            tags: str = ""
        ) -> dict:
            """
            Update an existing skill with new content. Overwrites the existing content.

            Args:
                title (str): The exact title of the existing skill to update.
                content (str): The new full Markdown content.
                description (str, optional): New description. If empty, keeps existing.
                category (str, optional): New category. If empty, keeps existing.
                tags (str, optional): New comma-separated tags. If empty, keeps existing.
            """
            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            
            skill = self.skills.get(title.lower())
            if not skill:
                matches = self.search_skills(title)
                if matches:
                    skill = matches[0]
            
            if not skill:
                return {"success": False, "error": f"Skill '{title}' not found."}
                
            if not skill.modifiable:
                return {"success": False, "error": f"Skill '{title}' is marked as READ-ONLY (unmodifiable) and cannot be updated."}
                
            updated_skill = self.update_skill(
                title=title,
                content=content,
                description=description if description else None,
                category=category if category else None,
                tags=tags_list
            )
            if updated_skill:
                return {"success": True, "output": f"Skill '{title}' updated successfully."}
            return {"success": False, "error": "Failed to update skill."}

        tools["tool_update_skill"] = {
            "name": "tool_update_skill",
            "description": "Update an existing skill with new content. Overwrites the existing content. Will fail if the skill is marked as read-only.",
            "parameters": [
                {"name": "title", "type": "str", "description": "The exact title of the existing skill to update."},
                {"name": "content", "type": "str", "description": "The new full Markdown content."},
                {"name": "description", "type": "str", "description": "New description. If empty, keeps existing.", "optional": True},
                {"name": "category", "type": "str", "description": "New category. If empty, keeps existing.", "optional": True},
                {"name": "tags", "type": "str", "description": "New comma-separated tags. If empty, keeps existing.", "optional": True}
            ],
            "callable": tool_update_skill,
        }

        def tool_append_to_skill(title: str, content: str) -> dict:
            """
            Append new content to the end of an existing skill.

            Args:
                title (str): The exact title of the existing skill.
                content (str): The Markdown content to append.
            """
            skill = self.skills.get(title.lower())
            if not skill:
                matches = self.search_skills(title)
                if matches:
                    skill = matches[0]
            
            if not skill:
                return {"success": False, "error": f"Skill '{title}' not found."}
                
            if not skill.modifiable:
                return {"success": False, "error": f"Skill '{title}' is marked as READ-ONLY (unmodifiable) and cannot be appended to."}
                
            updated_skill = self.append_to_skill(title=title, content=content)
            if updated_skill:
                return {"success": True, "output": f"Content appended to skill '{title}' successfully."}
            return {"success": False, "error": "Failed to append to skill."}

        tools["tool_append_to_skill"] = {
            "name": "tool_append_to_skill",
            "description": "Append new content to the end of an existing skill. Will fail if the skill is marked as read-only.",
            "parameters": [
                {"name": "title", "type": "str", "description": "The exact title of the existing skill."},
                {"name": "content", "type": "str", "description": "The Markdown content to append."}
            ],
            "callable": tool_append_to_skill,
        }

        def tool_remove_skill(title: str) -> dict:
            """
            Permanently delete a skill from the library.

            Args:
                title (str): The exact title of the skill to delete.
            """
            skill = self.skills.get(title.lower())
            if not skill:
                matches = self.search_skills(title)
                if matches:
                    skill = matches[0]
            
            if not skill:
                return {"success": False, "error": f"Skill '{title}' not found."}
                
            if not skill.modifiable:
                return {"success": False, "error": f"Skill '{title}' is marked as READ-ONLY (unmodifiable) and cannot be removed."}
                
            success = self.remove_skill(title=title)
            if success:
                return {"success": True, "output": f"Skill '{title}' removed successfully."}
            return {"success": False, "error": "Failed to remove skill."}

        tools["tool_remove_skill"] = {
            "name": "tool_remove_skill",
            "description": "Permanently delete a skill from the library. Will fail if the skill is marked as read-only.",
            "parameters": [
                {"name": "title", "type": "str", "description": "The exact title of the skill to delete."}
            ],
            "callable": tool_remove_skill,
        }

        return tools