"""Built-in local tools for the interactive CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from treeskill.tools import BaseTool, PythonFunctionTool

_MAX_TEXT_CHARS = 12_000
_MAX_FILE_LINES = 400
_MAX_LIST_ENTRIES = 200
_MAX_SEARCH_RESULTS = 100
_MAX_SHELL_TIMEOUT = 60


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _truncate(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} chars]"


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _list_dir(path: str = ".", include_hidden: bool = False, max_entries: int = 100) -> Dict[str, Any]:
    directory = _resolve_path(path)
    if not directory.exists():
        raise FileNotFoundError(f"Path does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    limit = max(1, min(max_entries, _MAX_LIST_ENTRIES))
    entries = []

    for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if not include_hidden and entry.name.startswith("."):
            continue
        info = {
            "name": entry.name,
            "path": str(entry),
            "type": "directory" if entry.is_dir() else "file",
        }
        if entry.is_file():
            try:
                info["size"] = entry.stat().st_size
            except OSError:
                pass
        entries.append(info)
        if len(entries) >= limit:
            break

    return {
        "path": str(directory),
        "entries": entries,
        "returned": len(entries),
    }


def _read_file(path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
    file_path = _resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {file_path}")

    start = max(1, start_line)
    end = max(start, min(end_line, start + _MAX_FILE_LINES - 1))
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    excerpt = lines[start - 1:end]

    return {
        "path": str(file_path),
        "start_line": start,
        "end_line": min(end, len(lines)),
        "total_lines": len(lines),
        "content": "\n".join(excerpt),
    }


def _search_repo(path: str = ".", pattern: str = "", max_results: int = 20) -> Dict[str, Any]:
    if not pattern:
        raise ValueError("pattern is required")

    root = _resolve_path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    limit = max(1, min(max_results, _MAX_SEARCH_RESULTS))
    results = []

    if shutil.which("rg"):
        command = [
            "rg",
            "-n",
            "--hidden",
            "--glob",
            "!.git",
            "--max-count",
            str(limit),
            "--color",
            "never",
            pattern,
            str(root),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.stdout:
            results = completed.stdout.strip().splitlines()
        return {
            "path": str(root),
            "pattern": pattern,
            "results": results[:limit],
            "returned": min(len(results), limit),
        }

    for file_path in sorted(root.rglob("*")):
        if ".git" in file_path.parts or not file_path.is_file():
            continue
        try:
            for line_no, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if pattern in line:
                    results.append(f"{file_path}:{line_no}:{line}")
                    if len(results) >= limit:
                        return {
                            "path": str(root),
                            "pattern": pattern,
                            "results": results,
                            "returned": len(results),
                        }
        except OSError:
            continue

    return {
        "path": str(root),
        "pattern": pattern,
        "results": results,
        "returned": len(results),
    }


def _write_file(path: str, content: str, append: bool = False) -> Dict[str, Any]:
    file_path = _resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    return {
        "path": str(file_path),
        "written_chars": len(content),
        "append": append,
    }


def _run_shell(command: str, cwd: str = ".", timeout_sec: int = 20) -> Dict[str, Any]:
    if not command.strip():
        raise ValueError("command is required")

    working_dir = _resolve_path(cwd)
    timeout = max(1, min(timeout_sec, _MAX_SHELL_TIMEOUT))

    try:
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(working_dir),
            "exit_code": completed.returncode,
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(working_dir),
            "exit_code": None,
            "stdout": _truncate(exc.stdout or ""),
            "stderr": _truncate(exc.stderr or ""),
            "error": f"Command timed out after {timeout} seconds",
        }


def _build_tool(
    *,
    name: str,
    description: str,
    schema: Dict[str, Any],
    func,
) -> BaseTool:
    return PythonFunctionTool(
        _name=name,
        _description=description,
        func=func,
        parameters_schema=schema,
    )


def build_builtin_tools() -> Dict[str, BaseTool]:
    return {
        "list_dir": _build_tool(
            name="list_dir",
            description="List files and directories for a path. Use this before reading a repository.",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to inspect."},
                    "include_hidden": {"type": "boolean", "description": "Include dotfiles and hidden entries."},
                    "max_entries": {"type": "integer", "description": "Maximum number of entries to return."},
                },
            },
            func=_list_dir,
        ),
        "read_file": _build_tool(
            name="read_file",
            description="Read a UTF-8 text file by line range. Prefer this over shell cat for source files.",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                    "start_line": {"type": "integer", "description": "1-based starting line."},
                    "end_line": {"type": "integer", "description": "1-based ending line."},
                },
                "required": ["path"],
            },
            func=_read_file,
        ),
        "search_repo": _build_tool(
            name="search_repo",
            description="Search text inside a directory tree. Useful for finding symbols, commands, or config keys.",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search."},
                    "pattern": {"type": "string", "description": "Literal pattern to search for."},
                    "max_results": {"type": "integer", "description": "Maximum result lines to return."},
                },
                "required": ["pattern"],
            },
            func=_search_repo,
        ),
        "write_file": _build_tool(
            name="write_file",
            description="Create or overwrite a text file. Use only when the user explicitly asks to create or modify files.",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "Text content to write."},
                    "append": {"type": "boolean", "description": "Append instead of overwrite."},
                },
                "required": ["path", "content"],
            },
            func=_write_file,
        ),
        "shell": _build_tool(
            name="shell",
            description=(
                "Run a local shell command. Prefer read-only commands for inspection. "
                "Use only when file tools are insufficient, and avoid destructive commands unless the user explicitly requests them."
            ),
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run with bash -lc."},
                    "cwd": {"type": "string", "description": "Working directory for the command."},
                    "timeout_sec": {"type": "integer", "description": "Timeout in seconds, max 60."},
                },
                "required": ["command"],
            },
            func=_run_shell,
        ),
    }


def format_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return _truncate(result)
    return _truncate(_safe_json(result))


__all__ = ["build_builtin_tools", "format_tool_result"]
