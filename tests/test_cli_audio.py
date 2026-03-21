"""Tests for CLI audio attachment behavior."""

import sys
import types
from pathlib import Path

for missing_module, attrs in {
    "evoskill.script": {
        "ScriptValidator": object,
        "ScriptValidationResult": object,
        "ScriptIssue": object,
        "validate_script": lambda *args, **kwargs: None,
        "validate_script_file": lambda *args, **kwargs: None,
        "load_script": lambda *args, **kwargs: None,
        "save_script": lambda *args, **kwargs: None,
        "load_script_as_tools": lambda *args, **kwargs: {},
    },
    "evoskill.memory": {
        "MEMORY_FILE": "memory.json",
        "MemoryType": object,
        "MemoryEntry": object,
        "MemoryStore": object,
        "MemoryCompiler": object,
    },
    "evoskill.agenda": {
        "AgendaManager": object,
        "compile_agenda_context": lambda *args, **kwargs: "",
        "parse_due": lambda *args, **kwargs: None,
    },
}.items():
    module = types.ModuleType(missing_module)
    for name, value in attrs.items():
        setattr(module, name, value)
    sys.modules.setdefault(missing_module, module)

from evoskill.cli import (
    ChatCLI,
    _build_chat_prompt_session,
    _get_slash_command_suggestions,
)
from evoskill.config import GlobalConfig
from evoskill.schema import Skill


def _make_cli(tmp_path: Path) -> ChatCLI:
    config = GlobalConfig()
    config.storage.trace_path = tmp_path / "traces.jsonl"
    skill = Skill(
        name="test-skill",
        description="test",
        system_prompt="You are a helpful assistant.",
    )
    return ChatCLI(
        config=config,
        skill_obj=skill,
        skill_path=tmp_path / "skill",
    )


def test_cmd_audio_stages_attachment(tmp_path: Path):
    cli = _make_cli(tmp_path)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake wav bytes")

    handled = cli._cmd_audio(str(audio_path))

    assert handled is True
    assert len(cli._pending_media_parts) == 1
    assert cli._pending_media_parts[0].type == "audio_url"
    assert cli._pending_media_parts[0].audio_url.url.startswith(
        "data:audio/wav;base64,"
    )


def test_build_user_message_includes_pending_audio(tmp_path: Path):
    cli = _make_cli(tmp_path)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake wav bytes")

    cli._cmd_audio(str(audio_path))
    message = cli._build_user_message("Please summarize this recording")

    assert message.role == "user"
    assert isinstance(message.content, list)
    assert message.content[0].type == "audio_url"
    assert message.content[1].type == "text"
    assert message.content[1].text == "Please summarize this recording"
    assert cli._pending_media_parts == []


def test_slash_command_suggestions_return_full_command_list():
    suggestions = _get_slash_command_suggestions("/")

    assert "/help" in suggestions
    assert "/audio" in suggestions
    assert "/select" in suggestions
    assert "/quit" in suggestions


def test_slash_command_suggestions_filter_by_prefix():
    suggestions = _get_slash_command_suggestions("/h")

    assert suggestions == ["/help"]


def test_slash_command_suggestions_ignore_non_slash_input():
    assert _get_slash_command_suggestions("hello") == []


def test_build_chat_prompt_session_uses_slash_completer():
    session = _build_chat_prompt_session()

    assert session.completer is not None
