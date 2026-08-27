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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "visibility": self.visibility,
            "has_metadata": self.has_metadata,
            "file_path": str(self.file_path) if self.file_path else None,
        }


def parse_skill_md(file_path: Path, default_visibility: str = "loadable") -> Optional[Skill]:
    """Parses a SKILL.md file into a Skill object. Supports YAML frontmatter."""
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

    if raw_content.startswith("---"):
        fm_match = re.match(r'^---\n(.*?)\n---\n(.*)', raw_content, re.DOTALL)
        if fm_match:
            has_metadata = True
            fm_text = fm_match.group(1)
            body = fm_match.group(2)
            visibility = default_visibility
            for line in fm_text.splitlines():
                line = line.strip()
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("category:"):
                    category = line.split(":", 1)[1].strip().strip('"\'')
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
            if visibility == "mixed":
                visibility = "visible"
        else:
            visibility = "visible"
    else:
        has_metadata = False
        visibility = "visible"
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
    )