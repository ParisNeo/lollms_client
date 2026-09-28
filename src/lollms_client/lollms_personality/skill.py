from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Skill:
    """Represents a single skill loaded from a SKILL.md file."""
    title: str
    description: str
    category: str
    tags: List[str]
    content: str
    file_path: Optional[Path] = None
    visibility: str = "loadable"
    has_metadata: bool = False
    modifiable: bool = True
    required_tools: List[str] = None

    def __post_init__(self):
        if self.required_tools is None:
            self.required_tools = []

    def is_available(
        self,
        active_tool_names: Optional[Any] = None,
        tool_availability_checker: Optional[Callable[[str], bool]] = None
    ) -> bool:
        """
        Evaluates whether the skill's tool conditions are satisfied.
        A tool requirement is satisfied if the tool is already active OR
        if the tool_availability_checker confirms it is available to be dynamically loaded.
        """
        if not self.required_tools:
            return True
        if active_tool_names is None and tool_availability_checker is None:
            return True

        available_set = set()
        if isinstance(active_tool_names, dict):
            available_set = {t.lower().strip() for t in active_tool_names.keys()}
        elif isinstance(active_tool_names, (list, tuple, set)):
            available_set = {str(t).lower().strip() for t in active_tool_names}

        for req in self.required_tools:
            clean_req = str(req).lower().strip()
            if not clean_req:
                continue
            if clean_req in available_set:
                continue
            if tool_availability_checker and tool_availability_checker(clean_req):
                continue
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "visibility": self.visibility,
            "has_metadata": self.has_metadata,
            "modifiable": self.modifiable,
            "required_tools": self.required_tools,
            "file_path": str(self.file_path) if self.file_path else None,
        }


def parse_skill_md(file_path: Path, default_visibility: str = "loadable") -> Optional[Skill]:
    """Parses a SKILL.md file into a Skill object. Default visibility is strictly loadable."""
    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    title = file_path.parent.name if file_path.stem.upper() == "SKILL" and file_path.parent.name else file_path.stem
    description = ""
    category = ""
    tags: List[str] = []
    body = raw_content
    has_metadata = False
    visibility = "loadable"
    modifiable = True
    required_tools: List[str] = []

    if raw_content.startswith("---"):
        fm_match = re.match(r'^---\n(.*?)\n---\n(.*)', raw_content, re.DOTALL)
        if fm_match:
            has_metadata = True
            fm_text = fm_match.group(1)
            body = fm_match.group(2)

            try:
                import yaml
                parsed_fm = yaml.safe_load(fm_text)
                if isinstance(parsed_fm, dict):
                    title = parsed_fm.get("title") or parsed_fm.get("name") or title
                    description = parsed_fm.get("description", "")
                    category = parsed_fm.get("category", "")
                    raw_tags = parsed_fm.get("tags", [])
                    if isinstance(raw_tags, list):
                        tags = [str(t).strip() for t in raw_tags]
                    elif isinstance(raw_tags, str):
                        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

                    req = parsed_fm.get("required_tools") or parsed_fm.get("tools_required")
                    if not req and isinstance(parsed_fm.get("requirements"), dict):
                        req = parsed_fm["requirements"].get("tools")
                    if isinstance(req, list):
                        required_tools = [str(t).strip() for t in req if str(t).strip()]
                    elif isinstance(req, str):
                        required_tools = [t.strip() for t in req.split(",") if t.strip()]

                    if parsed_fm.get("always_visible") in (True, "true", "yes", 1):
                        visibility = "visible"
                    elif "visibility" in parsed_fm:
                        v_cand = str(parsed_fm["visibility"]).lower().strip()
                        if v_cand in ("visible", "loadable", "searchable"):
                            visibility = v_cand
                    if parsed_fm.get("modifiable") in (False, "false", "no", 0, "off"):
                        modifiable = False
            except Exception:
                for line in fm_text.splitlines():
                    line = line.strip()
                    if line.startswith("title:"):
                        title = line.split(":", 1)[1].strip().strip('"\'')
                    elif line.startswith("name:") and not title:
                        title = line.split(":", 1)[1].strip().strip('"\'')
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"\'')
                    elif line.startswith("category:"):
                        category = line.split(":", 1)[1].strip().strip('"\'')
                    elif line.startswith("required_tools:"):
                        raw_tools = line.split(":", 1)[1].strip().strip("[]")
                        required_tools = [t.strip().strip('"\'') for t in raw_tools.split(",") if t.strip()]
                    elif line.startswith("tags:"):
                        tags_str = line.split(":", 1)[1].strip()
                        if tags_str.startswith("[") and tags_str.endswith("]"):
                            tags_str = tags_str[1:-1]
                        tags = [t.strip().strip('"\'') for t in tags_str.split(",") if t.strip()]
                    elif line.startswith("always_visible:"):
                        val = line.split(":", 1)[1].strip().strip('"\'').lower()
                        if val in ("true", "yes", "1"):
                            visibility = "visible"
                    elif line.startswith("visibility:"):
                        val = line.split(":", 1)[1].strip().strip('"\'').lower()
                        if val in ("visible", "loadable", "searchable"):
                            visibility = val
                    elif line.startswith("modifiable:"):
                        val = line.split(":", 1)[1].strip().strip('"\'').lower()
                        if val in ("false", "no", "0", "off"):
                            modifiable = False
        else:
            visibility = "loadable"
    else:
        has_metadata = False
        visibility = "loadable"
        h1_match = re.match(r'^#\s+(.+)', raw_content)
        if h1_match:
            title = h1_match.group(1).strip()
            rest = raw_content[h1_match.end():].strip()
            desc_match = re.match(r'^([^\n#]+)', rest)
            if desc_match:
                description = desc_match.group(1).strip()

    return Skill(
        title=title,
        description=description,
        category=category,
        tags=tags,
        content=body.strip(),
        file_path=file_path,
        visibility=visibility,
        has_metadata=has_metadata,
        modifiable=modifiable,
        required_tools=required_tools,
    )