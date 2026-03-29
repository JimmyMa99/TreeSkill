"""ASO program representation.

AS(skill)O upgrades optimization from a single prompt to a full agent program:

    program = root prompt + skill set + selection policy

The first implementation intentionally keeps the representation simple so it can
be evaluated through Kode by rendering the full program into one AGENTS.md.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from treeskill.schema import Skill
from treeskill.skill import save as save_skill


@dataclass
class ASOSkill:
    """A reusable skill fragment inside an ASO program."""

    name: str
    description: str
    prompt: str
    version: str = "v1.0"
    tags: List[str] = field(default_factory=list)

    def to_skill(self) -> Skill:
        return Skill(
            name=self.name,
            description=self.description,
            version=self.version,
            system_prompt=self.prompt,
        )

    def render_markdown(self) -> str:
        tag_text = ", ".join(self.tags) if self.tags else "general"
        return (
            f"## Skill: {self.name}\n"
            f"Description: {self.description}\n"
            f"Tags: {tag_text}\n\n"
            f"{self.prompt.strip()}\n"
        )


@dataclass
class ASOProgram:
    """A candidate program tracked in the frontier."""

    root_prompt: str
    skills: List[ASOSkill] = field(default_factory=list)
    selection_policy: str = (
        "Use the available skills selectively. First identify the user's need, "
        "then apply only the most relevant skill instructions."
    )
    version: str = "v1.0"
    score: Optional[float] = None
    parent_id: Optional[str] = None
    program_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render_agents_markdown(self) -> str:
        """Render the whole program into an AGENTS.md document for Kode."""
        sections = [
            "# Agent Program",
            "",
            "## Root Prompt",
            self.root_prompt.strip(),
            "",
            "## Skill Selection Policy",
            self.selection_policy.strip(),
            "",
        ]
        if self.skills:
            sections.append("## Available Skills")
            sections.append(
                "Apply these skills when they match the user's problem. "
                "Do not mention the internal skill names unless needed."
            )
            sections.append("")
            for skill in self.skills:
                sections.append(skill.render_markdown())
        else:
            sections.extend(
                [
                    "## Available Skills",
                    "No specialized skills have been added yet.",
                ]
            )
        return "\n".join(sections).strip() + "\n"

    def clone(self) -> "ASOProgram":
        return ASOProgram(
            root_prompt=self.root_prompt,
            skills=[
                ASOSkill(
                    name=skill.name,
                    description=skill.description,
                    prompt=skill.prompt,
                    version=skill.version,
                    tags=list(skill.tags),
                )
                for skill in self.skills
            ],
            selection_policy=self.selection_policy,
            version=self.version,
            score=self.score,
            parent_id=self.parent_id,
            metadata=dict(self.metadata),
        )

    def bump_version(self) -> "ASOProgram":
        clone = self.clone()
        clone.parent_id = self.program_id
        clone.version = _increment_version(self.version)
        return clone

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "parent_id": self.parent_id,
            "version": self.version,
            "score": self.score,
            "selection_policy": self.selection_policy,
            "root_prompt": self.root_prompt,
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "prompt": skill.prompt,
                    "version": skill.version,
                    "tags": skill.tags,
                }
                for skill in self.skills
            ],
            "metadata": self.metadata,
        }

    def save_to_dir(self, output_dir: Path, *, clean: bool = False) -> Path:
        """Persist the program snapshot to disk."""
        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        root_skill = Skill(
            name="root",
            description="ASO program root prompt",
            version=self.version,
            system_prompt=self.root_prompt,
            target=self.selection_policy,
        )
        save_skill(root_skill, output_dir / "root")

        skills_dir = output_dir / "skills"
        skills_dir.mkdir(exist_ok=True)
        for skill in self.skills:
            save_skill(skill.to_skill(), skills_dir / skill.name)

        (output_dir / "AGENTS.md").write_text(
            self.render_agents_markdown(),
            encoding="utf-8",
        )
        (output_dir / "program.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_dir


def _increment_version(version: str) -> str:
    raw = version[1:] if version.startswith("v") else version
    parts = raw.split(".")
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].isdigit():
            parts[idx] = str(int(parts[idx]) + 1)
            return "v" + ".".join(parts)
    return f"{version}.1"
