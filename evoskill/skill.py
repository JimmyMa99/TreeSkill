"""Skill Management — load, save, and compile Skills in Agent Skills format.

Skills follow the Agent Skills standard (https://agentskills.io/specification).
On disk a skill is a **directory** containing:

    my-skill/
    ├── SKILL.md        # YAML frontmatter + Markdown body (= system prompt)
    └── config.yaml     # optional: few-shot examples & model config

``SKILL.md`` format::

    ---
    name: my-skill
    description: What this skill does and when to use it.
    metadata:
      version: "1.0"
      target: "sound more human"
    ---

    You are a professional writing assistant.
    ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from evoskill.schema import Message, Skill

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKILL_FILE = "SKILL.md"
CONFIG_FILE = "config.yaml"
SCRIPT_FILE = "script.py"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


# ---------------------------------------------------------------------------
# SKILL.md parsing helpers
# ---------------------------------------------------------------------------

def _parse_skill_md(text: str) -> Dict[str, Any]:
    """Parse a SKILL.md file into a dict with 'frontmatter' and 'body' keys."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(
            "SKILL.md must start with YAML frontmatter (--- ... ---). "
            "See https://agentskills.io/specification"
        )
    fm_raw, body = m.group(1), m.group(2).strip()
    frontmatter = yaml.safe_load(fm_raw) or {}
    return {"frontmatter": frontmatter, "body": body}


def _frontmatter_to_skill_fields(fm: Dict[str, Any], body: str) -> Dict[str, Any]:
    """Convert parsed frontmatter + body into Skill constructor kwargs."""
    metadata: Dict[str, str] = fm.get("metadata") or {}

    fields: Dict[str, Any] = {
        "name": fm.get("name", "unnamed"),
        "description": fm.get("description", ""),
        "system_prompt": body,
        "version": metadata.get("version", "v1.0"),
        "target": metadata.get("target"),
    }
    return fields


def _skill_to_frontmatter(skill: Skill) -> str:
    """Render the YAML frontmatter block for a Skill."""
    fm: Dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
    }

    metadata: Dict[str, str] = {}
    if skill.version:
        metadata["version"] = skill.version
    if skill.target:
        metadata["target"] = skill.target
    if metadata:
        fm["metadata"] = metadata

    dumped = yaml.dump(
        fm,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip()
    return f"---\n{dumped}\n---"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load(path: Union[str, Path]) -> Skill:
    """Load a Skill from a SKILL.md file or a directory containing one.

    If *path* is a directory, reads ``SKILL.md`` (and optional ``config.yaml``)
    from it.  If *path* points directly to a ``.md`` file, reads that.

    Raises
    ------
    FileNotFoundError
        If SKILL.md cannot be found.
    ValueError
        If the file does not contain valid YAML frontmatter.
    pydantic.ValidationError
        If the parsed data cannot be coerced into a valid ``Skill``.
    """
    path = Path(path)

    if path.is_dir():
        skill_md_path = path / SKILL_FILE
        config_path = path / CONFIG_FILE
    elif path.name.lower().endswith(".md"):
        skill_md_path = path
        config_path = path.parent / CONFIG_FILE
    else:
        raise FileNotFoundError(
            f"Expected a directory or .md file, got: {path}\n"
            f"Skills must follow the Agent Skills standard with SKILL.md."
        )

    if not skill_md_path.is_file():
        raise FileNotFoundError(
            f"SKILL.md not found: {skill_md_path}\n"
            f"Create a SKILL.md with YAML frontmatter. "
            f"See https://agentskills.io/specification"
        )

    # Parse SKILL.md
    raw = skill_md_path.read_text(encoding="utf-8")
    parsed = _parse_skill_md(raw)
    fields = _frontmatter_to_skill_fields(parsed["frontmatter"], parsed["body"])

    # Load optional config.yaml
    if config_path.is_file():
        config_raw = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        ) or {}
        if "few_shot_messages" in config_raw:
            fields["few_shot_messages"] = config_raw.pop("few_shot_messages")
        if "agenda" in config_raw:
            fields["agenda"] = config_raw.pop("agenda")
        if "tools" in config_raw:
            fields["tools"] = config_raw.pop("tools")
        if config_raw:
            fields["config"] = config_raw

    # Load optional script.py
    skill_dir = skill_md_path.parent
    script_path = skill_dir / SCRIPT_FILE
    if script_path.is_file():
        fields["script"] = script_path.read_text(encoding="utf-8")

    return Skill.model_validate(fields)


def save(skill: Skill, path: Union[str, Path]) -> None:
    """Save *skill* as a SKILL.md (+ optional config.yaml) in a directory.

    If *path* is a directory, writes into it.
    If *path* is a file path ending in ``.md``, writes SKILL.md there and
    config.yaml alongside it.
    """
    path = Path(path)

    if path.suffix.lower() == ".md":
        skill_md_path = path
        config_path = path.parent / CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Treat as directory
        path.mkdir(parents=True, exist_ok=True)
        skill_md_path = path / SKILL_FILE
        config_path = path / CONFIG_FILE

    # Write SKILL.md
    frontmatter = _skill_to_frontmatter(skill)
    body = skill.system_prompt or ""
    content = f"{frontmatter}\n\n{body}\n"
    skill_md_path.write_text(content, encoding="utf-8")

    # Write config.yaml (only if there's content)
    config_data: Dict[str, Any] = {}
    if skill.few_shot_messages:
        config_data["few_shot_messages"] = [
            msg.model_dump(mode="json") for msg in skill.few_shot_messages
        ]
    if skill.agenda:
        config_data["agenda"] = [
            entry.model_dump(mode="json") for entry in skill.agenda
        ]
    if skill.tools:
        config_data["tools"] = [
            ref.model_dump(mode="json", exclude_none=True)
            for ref in skill.tools
        ]
    if skill.config:
        config_data.update(skill.config)

    if config_data:
        config_path.write_text(
            yaml.dump(
                config_data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    elif config_path.is_file():
        # Remove stale config.yaml if no config data
        config_path.unlink()

    # Write script.py (only if skill contains a script)
    script_path = skill_md_path.parent / SCRIPT_FILE
    if skill.script:
        script_path.write_text(skill.script, encoding="utf-8")
    elif script_path.is_file():
        # Remove stale script.py if no script content
        script_path.unlink()


# ---------------------------------------------------------------------------
# Message Compilation
# ---------------------------------------------------------------------------

def compile_messages(
    skill: Skill,
    user_input: List[Message],
    *,
    agenda_context: Optional[str] = None,
) -> List[Message]:
    """Build the full message list sent to the LLM.

    Order::

        [SystemMessage(skill.system_prompt + agenda_context)]
        + skill.few_shot_messages
        + user_input

    Parameters
    ----------
    agenda_context : str | None
        Agenda context generated by ``AgendaStore.compile_context()``.
        Appended to the end of the system prompt when non-empty.
    """
    prompt = skill.system_prompt
    if agenda_context:
        prompt = f"{prompt}\n\n{agenda_context}"
    system_msg = Message(role="system", content=prompt)
    return [system_msg] + list(skill.few_shot_messages) + list(user_input)
