"""Tests for trace storage deduplication and feedback updates."""

import json
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
from treeskill.cli import ChatCLI
from treeskill.config import GlobalConfig
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.storage import TraceStorage


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


class _FakePromptSession:
    def __init__(self, prompts):
        self._prompts = iter(prompts)

    def prompt(self, _text):
        value = next(self._prompts)
        if isinstance(value, BaseException):
            raise value
        return value


def test_trace_session_id_round_trips_through_json():
    trace = Trace(
        id="trace-1",
        session_id="session-1",
        inputs=[Message(role="user", content="hello")],
        prediction=Message(role="assistant", content="world"),
    )

    loaded = Trace.model_validate_json(trace.model_dump_json())

    assert loaded.session_id == "session-1"
    assert loaded.id == "trace-1"


def test_load_all_handles_legacy_traces_without_session_id(tmp_path: Path):
    config = GlobalConfig()
    config.storage.trace_path = tmp_path / "traces.jsonl"
    storage = TraceStorage(config.storage)
    legacy_trace = {
        "id": "trace-legacy",
        "timestamp": "2026-03-23T00:00:00Z",
        "inputs": [{"role": "user", "content": "hello"}],
        "prediction": {"role": "assistant", "content": "world"},
        "feedback": None,
        "node_path": None,
    }
    config.storage.trace_path.write_text(
        json.dumps(legacy_trace) + "\n",
        encoding="utf-8",
    )

    traces = storage.load_all()

    assert len(traces) == 1
    assert traces[0].id == "trace-legacy"
    assert traces[0].session_id is None


def test_load_all_prefers_latest_trace_version_for_same_id(tmp_path: Path):
    config = GlobalConfig()
    config.storage.trace_path = tmp_path / "traces.jsonl"
    storage = TraceStorage(config.storage)

    trace = Trace(
        id="trace-1",
        inputs=[Message(role="user", content="hello")],
        prediction=Message(role="assistant", content="first reply"),
    )
    storage.append(trace)

    updated_trace = trace.model_copy(
        update={
            "feedback": Feedback(score=0.1, critique="too vague"),
            "prediction": Message(role="assistant", content="revised reply"),
        }
    )
    storage.append(updated_trace)

    traces = storage.load_all()

    assert len(traces) == 1
    assert traces[0].id == "trace-1"
    assert traces[0].prediction.content == "revised reply"
    assert traces[0].feedback is not None
    assert traces[0].feedback.critique == "too vague"


def test_chat_cli_reuses_one_session_id_for_multiple_traces(tmp_path: Path):
    cli = _make_cli(tmp_path)
    cli._prompt_session = _FakePromptSession(["hello", "hello again", KeyboardInterrupt()])

    original_compile_messages = skill_module.compile_messages
    skill_module.compile_messages = lambda skill, history: list(history)

    def fake_generate_stream(messages, **kwargs):
        return Message(role="assistant", content=f"reply-{len(messages)}")

    cli._llm.generate_stream = fake_generate_stream

    try:
        cli.run()
    finally:
        skill_module.compile_messages = original_compile_messages

    traces = cli._storage.load_all()

    assert len(traces) == 2
    assert {trace.session_id for trace in traces} == {cli._session_id}
    assert traces[0].id != traces[1].id


def test_cmd_bad_updates_last_trace_without_duplicate_records(tmp_path: Path):
    cli = _make_cli(tmp_path)
    trace = Trace(
        id="trace-1",
        session_id=cli._session_id,
        inputs=[Message(role="user", content="hello")],
        prediction=Message(role="assistant", content="world"),
    )
    cli._storage.append(trace)
    cli._last_trace = trace

    handled = cli._cmd_bad("missed the point")
    traces = cli._storage.load_all()

    assert handled is True
    assert len(traces) == 1
    assert traces[0].feedback is not None
    assert traces[0].feedback.critique == "missed the point"


def test_cmd_rewrite_updates_last_trace_without_duplicate_records(tmp_path: Path):
    cli = _make_cli(tmp_path)
    trace = Trace(
        id="trace-1",
        session_id=cli._session_id,
        inputs=[Message(role="user", content="hello")],
        prediction=Message(role="assistant", content="world"),
    )
    cli._storage.append(trace)
    cli._last_trace = trace

    handled = cli._cmd_rewrite("better answer")
    traces = cli._storage.load_all()

    assert handled is True
    assert len(traces) == 1
    assert traces[0].feedback is not None
    assert traces[0].feedback.critique == "Rewrite provided"
    assert traces[0].feedback.correction == "better answer"
    assert traces[0].session_id == cli._session_id
