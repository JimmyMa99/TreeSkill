"""Tests for trace storage deduplication and feedback updates."""

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

from evoskill.cli import ChatCLI
from evoskill.config import GlobalConfig
from evoskill.schema import Feedback, Message, Skill, Trace
from evoskill.storage import TraceStorage


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


def test_cmd_bad_updates_last_trace_without_duplicate_records(tmp_path: Path):
    cli = _make_cli(tmp_path)
    trace = Trace(
        id="trace-1",
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
