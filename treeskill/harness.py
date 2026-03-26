"""Agent Harness — run skills in a real agent loop for evaluation.

Inspired by learn-claude-code's s05_skill_loading.py, this module provides
a minimal agent loop that loads TreeSkill skills and executes tasks with
tool use (bash, file I/O). Used as the "forward pass" environment for APO.

Usage::

    from treeskill.harness import AgentHarness

    harness = AgentHarness(
        model="intern-s1-pro",
        base_url="https://chat.intern-ai.org.cn",
        api_key="sk-...",
        skill_dir=Path("my-skill/"),
    )

    result = harness.run("Build a hello world Python script")
    print(result.output)      # Final text response
    print(result.tool_calls)  # List of tools called
    print(result.success)     # Whether the task completed without error
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Skill Loader (from learn-claude-code s05)
# ---------------------------------------------------------------------------

class SkillLoader:
    """Load SKILL.md files with YAML frontmatter."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def descriptions(self) -> str:
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _run_bash(command: str, workdir: Path) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/", "mkfs"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=str(workdir),
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _read_file(path: str, workdir: Path, limit: int = None) -> str:
    try:
        fp = (workdir / path).resolve()
        if not fp.is_relative_to(workdir.resolve()):
            return "Error: Path escapes workspace"
        lines = fp.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _write_file(path: str, content: str, workdir: Path) -> str:
    try:
        fp = (workdir / path).resolve()
        if not fp.is_relative_to(workdir.resolve()):
            return "Error: Path escapes workspace"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


TOOLS_SCHEMA = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "load_skill",
        "description": "Load specialized knowledge by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to load",
                },
            },
            "required": ["name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HarnessResult:
    """Result of running a task through the agent harness."""
    output: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    success: bool = True
    error: Optional[str] = None
    files_created: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Harness
# ---------------------------------------------------------------------------

class AgentHarness:
    """Minimal agent loop with skill loading and tool use.

    Parameters
    ----------
    model : str
        Anthropic model name.
    base_url : str
        API base URL.
    api_key : str
        API key.
    skill_dir : Path, optional
        Directory containing SKILL.md files to load.
    workdir : Path, optional
        Working directory for file operations. If None, uses a temp dir.
    max_tokens : int
        Max tokens per LLM call.
    extra_tools : list, optional
        Additional tool schemas + handlers.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        skill_dir: Optional[Path] = None,
        workdir: Optional[Path] = None,
        max_tokens: int = 8192,
        extra_tools: Optional[List[dict]] = None,
    ):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.max_tokens = max_tokens
        self.workdir = (workdir or Path(tempfile.mkdtemp(prefix="harness_"))).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Load skills
        self.skill_loader = SkillLoader(skill_dir) if skill_dir else SkillLoader(self.workdir / "skills")

        # Tool handlers
        self._handlers: Dict[str, Callable] = {
            "bash": lambda **kw: _run_bash(kw["command"], self.workdir),
            "read_file": lambda **kw: _read_file(kw["path"], self.workdir, kw.get("limit")),
            "write_file": lambda **kw: _write_file(kw["path"], kw["content"], self.workdir),
            "load_skill": lambda **kw: self.skill_loader.get_content(kw["name"]),
        }

        self._tools = list(TOOLS_SCHEMA)
        if extra_tools:
            for tool in extra_tools:
                self._tools.append(tool["schema"])
                self._handlers[tool["schema"]["name"]] = tool["handler"]

    def _build_system(self, extra_system: str = "") -> str:
        parts = [
            f"You are a coding agent at {self.workdir}.",
            "Use tools to complete tasks. Use load_skill for specialized knowledge.",
            "",
            f"Skills available:\n{self.skill_loader.descriptions()}",
        ]
        if extra_system:
            parts.append(f"\n{extra_system}")
        return "\n".join(parts)

    def run(
        self,
        task: str,
        system_prompt: str = "",
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> HarnessResult:
        """Run a task through the agent loop.

        Parameters
        ----------
        task : str
            The user's task description.
        system_prompt : str, optional
            Additional system prompt to prepend.
        max_iterations : int
            Maximum tool-use iterations.

        Returns
        -------
        HarnessResult
            Contains the final output, tool calls, and success status.
        """
        result = HarnessResult()
        system = self._build_system(system_prompt)
        messages = [{"role": "user", "content": task}]

        for turn in range(max_iterations):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=messages,
                    tools=self._tools,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                result.error = str(e)
                result.success = False
                break

            messages.append({"role": "assistant", "content": response.content})
            result.turns = turn + 1

            # Extract text output
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    result.output = block.text

            if response.stop_reason != "tool_use":
                break

            # Execute tools
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = self._handlers.get(block.name)
                    call_record = {
                        "name": block.name,
                        "input": block.input,
                    }
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                        call_record["output"] = str(output)[:500]
                        call_record["success"] = True
                    except Exception as e:
                        output = f"Error: {e}"
                        call_record["output"] = output
                        call_record["success"] = False

                    result.tool_calls.append(call_record)
                    logger.debug(f"  tool {block.name}: {str(output)[:100]}")

                    # Track created files
                    if block.name == "write_file" and "path" in block.input:
                        result.files_created.append(block.input["path"])

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })

            messages.append({"role": "user", "content": tool_results})

        return result
