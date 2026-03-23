"""Tests for CLI streaming output behavior."""

import sys
import types
from pathlib import Path

for missing_module, attrs in {
    "treeskill.script": {
        "ScriptValidator": object,
        "ScriptValidationResult": object,
        "ScriptIssue": object,
        "validate_script": lambda *args, **kwargs: None,
        "validate_script_file": lambda *args, **kwargs: None,
        "load_script": lambda *args, **kwargs: None,
        "save_script": lambda *args, **kwargs: None,
        "load_script_as_tools": lambda *args, **kwargs: {},
    },
    "treeskill.memory": {
        "MEMORY_FILE": "memory.json",
        "MemoryType": object,
        "MemoryEntry": object,
        "MemoryStore": object,
        "MemoryCompiler": object,
    },
    "treeskill.agenda": {
        "AgendaManager": object,
        "compile_agenda_context": lambda *args, **kwargs: "",
        "parse_due": lambda *args, **kwargs: None,
    },
}.items():
    module = types.ModuleType(missing_module)
    for name, value in attrs.items():
        setattr(module, name, value)
    sys.modules.setdefault(missing_module, module)

from treeskill import skill as skill_module
import treeskill.cli as cli_module
from treeskill.cli import ChatCLI
from treeskill.config import GlobalConfig
from treeskill.schema import Message, Skill


class _FakePromptSession:
    def __init__(self, prompts):
        self._prompts = iter(prompts)

    def prompt(self, _text):
        value = next(self._prompts)
        if isinstance(value, BaseException):
            raise value
        return value


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


def test_chat_cli_streams_then_rerenders_final_markdown(tmp_path: Path):
    cli = _make_cli(tmp_path)
    cli._prompt_session = _FakePromptSession(["hello", KeyboardInterrupt()])

    streamed = []
    printed = []

    def fake_generate_stream(messages, **kwargs):
        kwargs["on_delta"]("Hel")
        kwargs["on_delta"]("lo")
        return Message(role="assistant", content="Hello")

    cli._llm.generate_stream = fake_generate_stream
    cli._console.print = lambda *args, **kwargs: printed.append((args, kwargs))
    cli._render_streaming_assistant = lambda text: streamed.append(text)
    cli._tool_guidance_text = lambda: "tool guidance"
    original_compile_messages = skill_module.compile_messages
    skill_module.compile_messages = lambda skill, history: list(history)

    try:
        cli.run()
    finally:
        skill_module.compile_messages = original_compile_messages

    assert streamed == ["Hel", "Hello"]
    assert len(cli._history) == 2
    assert cli._history[0].content == "hello"
    assert cli._history[1].content == "Hello"
    traces = cli._storage.load_all()
    assert len(traces) == 1
    assert traces[0].prediction.content == "Hello"


def test_chat_cli_keeps_single_assistant_panel_after_streaming(tmp_path: Path):
    cli = _make_cli(tmp_path)
    cli._prompt_session = _FakePromptSession(["hello", KeyboardInterrupt()])

    printed = []
    live_updates = []

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            live_updates.append(renderable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, renderable):
            live_updates.append(renderable)

    def fake_generate_stream(messages, **kwargs):
        kwargs["on_delta"]("Hel")
        kwargs["on_delta"]("lo")
        return Message(role="assistant", content="Hello")

    original_live = cli_module.Live
    original_compile_messages = skill_module.compile_messages
    cli_module.Live = _FakeLive
    skill_module.compile_messages = lambda skill, history: list(history)
    cli._llm.generate_stream = fake_generate_stream
    cli._console.print = lambda *args, **kwargs: printed.append((args, kwargs))

    try:
        cli.run()
    finally:
        cli_module.Live = original_live
        skill_module.compile_messages = original_compile_messages

    assert len(printed) == 2
    assert len(live_updates) >= 3
    assert live_updates[-1].renderable.markup == "Hello"
