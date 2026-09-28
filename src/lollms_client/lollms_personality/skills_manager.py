from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .skill import Skill, parse_skill_md


class SkillsManager:
    """
    Manages SKILL.md files from external directories.

    Visibility tiers:
        - "visible":    full content injected into the system prompt.
        - "loadable":   name + description listed in the prompt; full content
                        retrieved on demand via `tool_load_skill`.
        - "searchable": hidden from the prompt; discovered via
                        `tool_search_skills` then loaded via `tool_load_skill`.
    """

    VALID_VISIBILITIES = ("visible", "loadable", "searchable")

    def __init__(
        self,
        skills_dirs: Optional[List[Union[str, Path]]] = None,
        mode: str = "loadable",
        max_visible_skills: int = 3,
        max_visible_tokens: int = 1200,
        allow_llm_skill_writing: bool = True,
    ):
        self.mode = mode
        self.max_visible_skills = max_visible_skills
        self.max_visible_tokens = max_visible_tokens
        self.allow_llm_skill_writing = allow_llm_skill_writing
        self.active_tool_names: Optional[set] = None
        self.tool_availability_checker: Optional[Callable[[str], bool]] = None
        self.tool_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
        self._skills_dirs: List[Path] = []
        if skills_dirs:
            for d in skills_dirs:
                p = Path(d)
                p.mkdir(parents=True, exist_ok=True)
                self._skills_dirs.append(p.resolve())
        else:
            default_p = Path("./skills").resolve()
            default_p.mkdir(parents=True, exist_ok=True)
            self._skills_dirs.append(default_p)

        self.skills: Dict[str, Skill] = {}
        self.reload()

    # ─────────────────────────────────────────────────────────── tier helpers

    def _resolve_visibility(self, skill: Skill) -> str:
        if self.mode == "loadable":
            return "loadable"
        if self.mode == "searchable":
            return "searchable"
        if self.mode == "visible":
            return "visible"
        if self.mode == "mixed":
            return skill.visibility
        if skill.visibility == "searchable":
            return "searchable"
        return "loadable"

    def get_unique_skills(self) -> List[Skill]:
        """Returns unique Skill instances, eliminating duplicates from dual-key title/slug indexing."""
        seen = set()
        unique = []
        for s in self.skills.values():
            key = str(s.file_path.resolve()) if s.file_path else s.title.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def get_skills_by_visibility(self, tier: str) -> List[Skill]:
        if tier not in self.VALID_VISIBILITIES:
            raise ValueError(
                f"Invalid visibility tier '{tier}'. Must be one of {self.VALID_VISIBILITIES}."
            )
        return [s for s in self.get_unique_skills() if s.visibility == tier]

    def has_loadable_skills(self) -> bool:
        return any(s.visibility == "loadable" for s in self.get_unique_skills())

    def has_searchable_skills(self) -> bool:
        return any(s.visibility == "searchable" for s in self.get_unique_skills())

    def has_visible_skills(self) -> bool:
        return any(s.visibility == "visible" for s in self.get_unique_skills())

    # ─────────────────────────────────────────────────────────── persistence

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
                        # Also index by directory slug name (e.g. "file_organization")
                        self.skills[item.name.lower()] = skill
            elif item.is_file() and item.suffix.lower() == ".md" and item.name != "README.md":
                if item.resolve() not in seen_paths:
                    seen_paths.add(item.resolve())
                    skill = parse_skill_md(item, default_visibility=self.mode)
                    if skill:
                        skill.visibility = self._resolve_visibility(skill)
                        self.skills[skill.title.lower()] = skill
                        self.skills[item.stem.lower()] = skill

    def _sanitize_title(self, title: str) -> str:
        safe_title = re.sub(r'[^\w\-]', '_', title).strip('_')
        return safe_title or "unnamed_skill"

    def _strip_functional_tags(self, content: str) -> str:
        functional_tags = [
            r'<tool>.*?</tool>',
            r'<art(?:ifact|efact)\b[^>]*>.*?</art(?:ifact|efact)>',
            r'<(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder|scratchpad_append|scratchpad_patch|scratchpad_clear|user_profile_update|user_profile_clear|mem_new|mem_update|effort|done|end|processing|tool_result|refactor_history)\b[^>]*/?>',
            r'</(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder|scratchpad_append|scratchpad_patch|scratchpad_clear|user_profile_update|user_profile_clear|mem_new|mem_update|effort|done|end|processing|tool_result|refactor_history)>',
        ]
        cleaned = content
        for pattern in functional_tags:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _validate_visibility(self, visibility: str) -> str:
        normalized = (visibility or "").strip().lower()
        if normalized not in self.VALID_VISIBILITIES:
            raise ValueError(
                f"Invalid visibility '{visibility}'. Must be one of {self.VALID_VISIBILITIES}."
            )
        return normalized

    def _ensure_writable_dir(self) -> Path:
        if not self._skills_dirs:
            default_p = Path("./skills").resolve()
            default_p.mkdir(parents=True, exist_ok=True)
            self._skills_dirs.append(default_p)
        target_dir = self._skills_dirs[0]
        target_dir.mkdir(parents=True, exist_ok=True)
        self._skills_dirs[0] = target_dir.resolve()
        return self._skills_dirs[0]

    def create_skill(
        self,
        title: str,
        content: str,
        description: str = "",
        category: str = "",
        tags: Optional[List[str]] = None,
        visibility: str = "loadable"
    ) -> Optional[Skill]:
        target_dir = self._ensure_writable_dir()
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

        clean_content = self._strip_functional_tags(content)
        skill_path.write_text(frontmatter + clean_content.strip() + "\n", encoding="utf-8")

        self.reload()
        return self.skills.get(title.lower())

    def add_skill(
        self,
        title: str,
        content: str,
        description: str = "",
        category: str = "",
        tags: Optional[List[str]] = None,
        visibility: str = "loadable",
        overwrite: bool = False,
    ) -> Skill:
        """
        Application-grade entry point for hosting apps to register a custom skill.

        Validates the visibility tier, sanitizes the title and content, persists
        the skill as a SKILL.md file inside the manager's primary skills
        directory, and reloads the registry.

        Args:
            title: Human-readable skill name (sanitized for the filesystem).
            content: Markdown body of the skill.
            description: One-line summary shown in loadable lists.
            category: Optional grouping label.
            tags: Optional list of search keywords.
            visibility: One of "visible", "loadable", "searchable".
            overwrite: When False, raises if a skill with the same title exists.

        Returns:
            The persisted Skill object.

        Raises:
            ValueError: On invalid visibility or duplicate title (overwrite=False).
        """
        normalized_visibility = self._validate_visibility(visibility)

        if not title or not title.strip():
            raise ValueError("Skill title must not be empty.")
        title = title.strip()

        existing = self.skills.get(title.lower())
        if existing and not overwrite:
            raise ValueError(
                f"A skill titled '{title}' already exists. Pass overwrite=True to replace it."
            )

        return self.create_skill(
            title=title,
            content=content,
            description=description,
            category=category,
            tags=tags,
            visibility=normalized_visibility,
        )

    @staticmethod
    def _normalize_key(text: str) -> str:
        if not text:
            return ""
        t = text.lower().replace("_", " ").replace("-", " ")
        t = re.sub(r'[^\w\s]', '', t)
        return " ".join(t.split())

    def get_skill(self, title: str) -> Optional[Skill]:
        """Retrieves a skill by exact, slug, or fuzzy normalized title matching."""
        if not title:
            return None
        clean_title = title.strip().lower()
        if clean_title in self.skills:
            return self.skills[clean_title]

        norm_query = self._normalize_key(title)
        query_words = set(norm_query.split())

        best_match = None
        best_overlap = 0

        for s in self.get_unique_skills():
            norm_s_title = self._normalize_key(s.title)
            norm_s_slug = self._normalize_key(s.file_path.parent.name if s.file_path else "")
            norm_s_cat = self._normalize_key(s.category)

            # 1. Exact match on title, slug, or category
            if norm_query in (norm_s_title, norm_s_slug, norm_s_cat):
                return s

            # 2. Tag match (e.g. model calls tool_load_skill("file_organization"))
            if s.tags:
                for t in s.tags:
                    if norm_query == self._normalize_key(t):
                        return s

            # 3. Substring match
            if norm_query in norm_s_title or norm_s_title in norm_query:
                return s
            if norm_s_slug and (norm_query in norm_s_slug or norm_s_slug in norm_query):
                return s
            if norm_s_cat and (norm_query in norm_s_cat or norm_s_cat in norm_query):
                return s

            # 4. Word overlap match
            s_words = set(norm_s_title.split()) | (set(norm_s_slug.split()) if norm_s_slug else set()) | (set(norm_s_cat.split()) if norm_s_cat else set())
            overlap = len(query_words & s_words)
            if overlap > best_overlap and overlap >= max(1, len(query_words) - 1):
                best_overlap = overlap
                best_match = s

        if best_match:
            return best_match

        matches = self.search_skills(title)
        if matches:
            return matches[0]

        return None

    def set_skill_visibility(self, title: str, visibility: str) -> Skill:
        """
        Re-writes a skill's frontmatter with a new visibility tier and reloads.

        Raises:
            ValueError: If the skill is unknown, read-only, or the tier is invalid.
        """
        normalized_visibility = self._validate_visibility(visibility)
        skill = self.get_skill(title)
        if skill is None:
            raise ValueError(f"Skill '{title}' not found.")
        if not skill.modifiable:
            raise ValueError(f"Skill '{title}' is marked read-only and cannot be modified.")
        if skill.file_path is None:
            raise ValueError(f"Skill '{title}' has no backing file.")

        raw = skill.file_path.read_text(encoding="utf-8", errors="ignore")
        updated_raw, substitution_count = re.subn(
            r'(?m)^visibility:\s*\w+\s*$',
            f'visibility: {normalized_visibility}',
            raw,
        )
        if substitution_count == 0:
            raise ValueError(
                f"Skill '{title}' has no visibility field in its frontmatter to update."
            )

        skill.file_path.write_text(updated_raw, encoding="utf-8")
        self.reload()
        refreshed = self.get_skill(skill.title)
        if refreshed is None:
            raise ValueError(f"Skill '{title}' disappeared after visibility update.")
        return refreshed

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
        target_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill.file_path

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

        clean_content = self._strip_functional_tags(content)
        skill_path.write_text(frontmatter + clean_content.strip() + "\n", encoding="utf-8")

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
        clean_content = self._strip_functional_tags(content)
        new_content = existing_content.rstrip() + separator + clean_content.strip() + "\n"
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

    def set_active_tools(self, tool_names: Optional[Any] = None) -> None:
        """Sets the active tool set used to condition skill availability."""
        if isinstance(tool_names, dict):
            self.active_tool_names = {str(k).lower().strip() for k in tool_names.keys()}
        elif isinstance(tool_names, (list, tuple, set)):
            self.active_tool_names = {str(k).lower().strip() for k in tool_names}
        else:
            self.active_tool_names = None

    # ─────────────────────────────────────────────────────────── prompt builders

    def find_relevant_skills(self, query: str, top_k: int = 2) -> List[Skill]:
        """Identifies loadable skills that have high relevance to the user's query."""
        if not query or len(query.strip()) < 3:
            return []
        matches = self.search_skills(query)
        eligible = [
            s for s in matches
            if s.visibility == "loadable" and s.is_available(self.active_tool_names)
        ]
        return eligible[:top_k]

    def build_context(self, active_tool_names: Optional[Any] = None, current_query: Optional[str] = None, round_count: int = 1) -> str:
        if active_tool_names is not None:
            self.set_active_tools(active_tool_names)

        parts = []

        # Filter out skills whose required_tools are neither active nor available to load
        eligible_skills = [
            s for s in self.get_unique_skills()
            if s.is_available(self.active_tool_names, self.tool_availability_checker)
        ]

        raw_visible = [s for s in eligible_skills if s.visibility == "visible"]
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

        loadable = [s for s in eligible_skills if s.visibility == "loadable"] + overflow_loadable
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

        # Proactively highlight relevant skills that match the user's immediate prompt
        if current_query:
            relevant = self.find_relevant_skills(current_query, top_k=2)
            if relevant:
                if round_count <= 1:
                    rec_lines = [
                        "=== 🎯 RECOMMENDED SKILL FOR CURRENT TASK ===",
                        "The user's request matches the following specialized skill(s):"
                    ]
                    for r_skill in relevant:
                        rec_lines.append(
                            f"👉 `{r_skill.title}` — MANDATORY: Call `tool_load_skill(title=\"{r_skill.title}\")` in Round 1 "
                            f"to load its verified methodology before taking action!"
                        )
                    rec_lines.append("=== END RECOMMENDED SKILL ===")
                else:
                    # In Round 2+, switch to active execution directive so model never loops tool_load_skill
                    rec_lines = [
                        "=== 🎯 ACTIVE TASK SKILL (IN PROGRESS) ==="
                    ]
                    for r_skill in relevant:
                        rec_lines.append(
                            f"✅ Skill '{r_skill.title}' methodology is loaded. You have fulfilled the Skill-First mandate.\n"
                            f"DO NOT call `tool_load_skill` again! You MUST now execute the skill protocol (Phase 1: scan and emit artifacts)."
                        )
                    rec_lines.append("=== END ACTIVE TASK SKILL ===")
                parts.append("\n".join(rec_lines))

        searchable = [s for s in eligible_skills if s.visibility == "searchable"]
        if searchable:
            lines = ["=== SEARCHABLE SKILLS ==="]
            lines.append(f"There are {len(searchable)} hidden skills. Use `tool_search_skills` to find them by keyword.")
            lines.append("=== END SEARCHABLE SKILLS ===")
            parts.append("\n".join(lines))

        if not parts:
            return "\n=== SKILLS SYSTEM ===\nThere are currently 0 skills in the library.\n=== END SKILLS SYSTEM ==="

        return "\n\n".join(parts)

    def build_loadable_skills_prompt(self) -> str:
        """
        Compact prompt block listing immediately loadable skills (names only).

        Returns an empty string when there are no loadable skills, so callers
        can append it unconditionally.
        """
        loadable = [s for s in self.get_unique_skills() if s.visibility == "loadable"]
        if not loadable:
            return ""

        lines = ["=== IMMEDIATELY LOADABLE SKILLS ==="]
        lines.append(
            "The following skills are available and can be loaded on demand with the "
            "`tool_load_skill` tool (pass the exact skill title):"
        )
        lines.append("")
        for skill in loadable:
            lines.append(f"- {skill.title}")
        lines.append("=== END IMMEDIATELY LOADABLE SKILLS ===")
        return "\n".join(lines)

    def build_searchable_skills_prompt(self) -> str:
        """
        Compact prompt block announcing the existence of hidden, searchable skills.

        Returns an empty string when there are no searchable skills, so callers
        can append it unconditionally.
        """
        if not self.has_searchable_skills():
            return ""

        lines = ["=== HIDDEN SKILLS (SEARCHABLE) ==="]
        lines.append(
            f"There are {self._searchable_count()} hidden skills in the library. "
            "Use the `tool_search_skills` tool with a keyword to discover them, "
            "then `tool_load_skill` to load the full content."
        )
        lines.append("=== END HIDDEN SKILLS ===")
        return "\n".join(lines)

    def _searchable_count(self) -> int:
        return sum(1 for s in self.get_unique_skills() if s.visibility == "searchable")

    def search_skills(self, query: str) -> List[Skill]:
        query_lower = query.lower()
        results = []
        for skill in self.get_unique_skills():
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
        return [s.to_dict() for s in self.get_unique_skills()]

    # ─────────────────────────────────────────────────────────── tool building

    def build_skill_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Conditionally builds tool specifications for skill management based on visibility tiers.

        Tool activation matrix:
            - `tool_list_skills`   → offered when the library is non-empty.
            - `tool_load_skill`    → offered when ≥1 loadable skill exists.
            - `tool_search_skills` → offered when ≥1 searchable skill exists.
            - CRUD tools           → offered when `allow_llm_skill_writing` is True.
        """
        tools: Dict[str, Dict[str, Any]] = {}

        raw_visible_count = sum(1 for s in self.get_unique_skills() if s.visibility == "visible")
        has_loadable = any(s.visibility == "loadable" for s in self.get_unique_skills()) or (raw_visible_count > self.max_visible_skills)
        has_searchable = self.has_searchable_skills()
        unique_skills_list = self.get_unique_skills()
        total_skills = len(unique_skills_list)

        if total_skills > 0:
            def tool_list_skills() -> dict:
                """
                Lists all available skills in the library, categorized by their visibility tier (visible, loadable, searchable).
                Use this to get an overview of what knowledge is available.
                """
                visible = [s.to_dict() for s in self.get_unique_skills() if s.visibility == "visible"]
                loadable = [s.to_dict() for s in self.get_unique_skills() if s.visibility == "loadable"]
                searchable = [s.to_dict() for s in self.get_unique_skills() if s.visibility == "searchable"]

                report = {
                    "visible_skills": visible,
                    "loadable_skills": loadable,
                    "searchable_skills": searchable,
                    "total_count": len(self.get_unique_skills())
                }
                return {"success": True, "output": report}

            tools["tool_list_skills"] = {
                "name": "tool_list_skills",
                "description": "Lists all available skills in the library, categorized by their visibility tier (visible, loadable, searchable).",
                "parameters": [],
                "callable": tool_list_skills,
            }

        if has_loadable or has_searchable:
            def tool_load_skill(title: str = "", name: str = "", skill_name: str = "", **kwargs) -> dict:
                """
                Load the full content of a skill by title. Use this to access detailed instructions.

                Args:
                    title (str, optional): The title of the skill to load.
                    name (str, optional): Alias for title.
                    skill_name (str, optional): Alias for title.
                """
                target_title = (title or name or skill_name or kwargs.get("skill") or "").strip()
                if not target_title:
                    return {"success": False, "error": "No skill title or name provided."}

                skill = self.get_skill(target_title)
                if not skill:
                    return {"success": False, "error": f"Skill '{target_title}' not found."}

                # ── DYNAMIC TOOL LOADING ON SKILL MOUNT ──
                loaded_tools_now = []
                missing_tools = []

                if skill.required_tools:
                    available_set = set(self.active_tool_names or [])
                    for req in skill.required_tools:
                        clean_req = str(req).strip()
                        if not clean_req:
                            continue
                        if clean_req.lower() in available_set:
                            continue

                        # Attempt to dynamically load the tool
                        loaded_spec = None
                        if self.tool_loader:
                            try:
                                loaded_spec = self.tool_loader(clean_req)
                            except Exception as ex:
                                ASCIIColors.warning(f"[SkillsManager] Tool loader error for '{clean_req}': {ex}")

                        if loaded_spec:
                            loaded_tools_now.append(clean_req)
                            if self.active_tool_names is not None:
                                self.active_tool_names.add(clean_req.lower())
                            available_set.add(clean_req.lower())
                        else:
                            missing_tools.append(clean_req)

                if missing_tools:
                    return {
                        "success": False,
                        "error": f"Skill '{skill.title}' cannot be loaded: required tool(s) ({', '.join(missing_tools)}) are missing or unavailable. The skill requires these tools to operate safely and will not be loaded until they are installed or mounted."
                    }

                skill._loaded_in_session = True
                content = self.load_skill(skill.title)
                if content:
                    tool_notice = ""
                    if loaded_tools_now or skill.required_tools:
                        active_now = loaded_tools_now or skill.required_tools
                        tool_notice = (
                            f"\n\n🛠️ **Required Toolset Automatically Activated**:\n"
                            f"The required tool(s) for this skill ({', '.join(f'`{t}`' for t in active_now)}) "
                            f"have been verified and activated in your session.\n"
                            f"You can now call them directly (e.g. `{active_now[0]}`)."
                        )
                    return {
                        "success": True,
                        "output": content + tool_notice,
                        "loaded_tools": loaded_tools_now
                    }
                return {"success": False, "error": f"Skill '{title}' not found."}

            tools["tool_load_skill"] = {
                "name": "tool_load_skill",
                "description": "Load the full content of a skill by title. Skills contain reusable knowledge, instructions, and best practices.",
                "parameters": [
                    {"name": "title", "type": "str", "description": "The title of the skill to load."}
                ],
                "callable": tool_load_skill,
            }

            def tool_unload_skill(title: str = "", name: str = "", skill_name: str = "", **kwargs) -> dict:
                """
                Unload a previously loaded skill from context back to loadable state to free context tokens.

                Args:
                    title (str, optional): The title of the skill to unload.
                    name (str, optional): Alias for title.
                    skill_name (str, optional): Alias for title.
                """
                target_title = (title or name or skill_name or kwargs.get("skill") or "").strip()
                if not target_title:
                    return {"success": False, "error": "No skill title or name provided."}

                skill = self.get_skill(target_title)
                if not skill:
                    return {"success": False, "error": f"Skill '{target_title}' not found."}

                skill.visibility = "loadable"
                return {"success": True, "output": f"Skill '{skill.title}' has been unloaded from active context to free token space."}

            tools["tool_unload_skill"] = {
                "name": "tool_unload_skill",
                "description": "Unload a skill from active context back to loadable state to free context tokens when it is no longer needed.",
                "parameters": [
                    {"name": "title", "type": "str", "description": "The title of the skill to unload."}
                ],
                "callable": tool_unload_skill,
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

        if not self.allow_llm_skill_writing:
            return tools

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
            try:
                skill = self.add_skill(
                    title=title,
                    content=content,
                    description=description,
                    category=category,
                    tags=tags_list,
                    visibility=visibility,
                    overwrite=True,
                )
                return {"success": True, "output": f"Skill '{skill.title}' created successfully with visibility '{skill.visibility}'."}
            except ValueError as ex:
                return {"success": False, "error": str(ex)}

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