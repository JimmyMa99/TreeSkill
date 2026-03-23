"""Tests for run-level session_id behavior across trace producers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from treeskill.annotate import AnnotateCLI
from treeskill.config import GlobalConfig
from treeskill.evaluator import Evaluator
from treeskill.schema import Message, Skill
from treeskill.storage import TraceStorage


class _FakeLLM:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.batch_calls = 0

    def generate(self, messages, model=None):
        self.generate_calls += 1
        return Message(
            role="assistant",
            content=f"annotated-{self.generate_calls}",
        )

    def generate_batch(self, batches, model=None):
        self.batch_calls += 1
        if self.batch_calls % 2 == 1:
            return [
                Message(role="assistant", content=f"prediction-{idx}")
                for idx, _batch in enumerate(batches, start=1)
            ]

        return [
            Message(
                role="assistant",
                content='{"score": 0.9, "critique": "looks good"}',
            )
            for _batch in batches
        ]


class _FakeDataset:
    def __init__(self, samples):
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)


def _make_skill() -> Skill:
    return Skill(
        name="test-skill",
        description="test",
        system_prompt="You are a helpful assistant.",
    )


def _make_sample(user_text: str, answer_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        input_messages=[Message(role="user", content=user_text)],
        ground_truth=Message(role="assistant", content=answer_text),
    )


def test_annotate_run_reuses_one_session_id_for_all_traces(
    tmp_path: Path,
    monkeypatch,
):
    config = GlobalConfig()
    config.storage.trace_path = tmp_path / "annotate-traces.jsonl"
    dataset = _FakeDataset([
        _make_sample("hello", "world"),
        _make_sample("bye", "ciao"),
    ])
    storage = TraceStorage(config.storage)
    annotator = AnnotateCLI(
        config=config,
        llm=_FakeLLM(),
        skill=_make_skill(),
        dataset=dataset,
        storage=storage,
        auto=False,
    )

    responses = iter(["nice", "clear"])
    monkeypatch.setattr(
        "treeskill.annotate.Prompt.ask",
        lambda *args, **kwargs: next(responses),
    )

    traces = annotator.run()

    assert len(traces) == 2
    assert {trace.session_id for trace in traces} == {annotator._session_id}
    assert len({trace.id for trace in traces}) == 2


def test_evaluator_run_reuses_one_session_id_for_all_traces(
    tmp_path: Path,
):
    config = GlobalConfig()
    config.storage.trace_path = tmp_path / "evaluator-traces.jsonl"
    dataset = _FakeDataset([
        _make_sample("hello", "world"),
        _make_sample("bye", "ciao"),
    ])
    evaluator = Evaluator(config, _FakeLLM())

    first_run = evaluator.evaluate(_make_skill(), dataset)
    first_session_id = first_run[0].session_id
    second_run = evaluator.evaluate(_make_skill(), dataset)

    assert len(first_run) == 2
    assert len(second_run) == 2
    assert {trace.session_id for trace in first_run} == {first_session_id}
    assert {trace.session_id for trace in second_run} == {
        second_run[0].session_id
    }
    assert first_session_id != second_run[0].session_id
    assert len({trace.id for trace in first_run}) == 2


def test_synthetic_memory_traces_remain_sessionless(monkeypatch, tmp_path: Path):
    import treeskill.schema as schema_module

    monkeypatch.delitem(sys.modules, "treeskill.memory", raising=False)

    tresskill_pkg = types.ModuleType("tresskill")
    tresskill_pkg.__path__ = []
    tresskill_pkg.schema = schema_module
    monkeypatch.setitem(sys.modules, "tresskill", tresskill_pkg)
    monkeypatch.setitem(sys.modules, "tresskill.schema", schema_module)

    memory_module = importlib.import_module("treeskill.memory")
    store = memory_module.MemoryStore(tmp_path)
    store.add_correction(
        "please be shorter",
        before="This is the long version.",
        after="Short version.",
    )

    traces = memory_module.MemoryCompiler(store).compile_traces()

    assert len(traces) == 1
    assert traces[0].session_id is None
    assert traces[0].prediction.content == "This is the long version."
