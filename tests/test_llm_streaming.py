"""Unit tests for streaming generation in ``LLMClient``."""

from types import SimpleNamespace

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.schema import Message


class _FakeChunk:
    def __init__(self, text: str | None = None, tool_calls=None):
        delta = SimpleNamespace(content=text, tool_calls=tool_calls or [])
        choice = SimpleNamespace(delta=delta)
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._streams.pop(0))


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


def test_generate_stream_yields_text_deltas_and_final_message():
    config = GlobalConfig()
    client = LLMClient(config)
    fake = _FakeClient([[_FakeChunk("Hel"), _FakeChunk("lo")]])
    client._client = fake

    yielded = []
    final = client.generate_stream(
        [Message(role="user", content="hello")],
        on_delta=yielded.append,
    )

    assert yielded == ["Hel", "lo"]
    assert final.role == "assistant"
    assert final.content == "Hello"
    assert fake.chat.completions.calls[0]["stream"] is True


def test_generate_stream_ignores_empty_deltas():
    config = GlobalConfig()
    client = LLMClient(config)
    client._client = _FakeClient(
        [[_FakeChunk(None), _FakeChunk(""), _FakeChunk("Hi"), _FakeChunk(" there")]]
    )

    yielded = []
    final = client.generate_stream(
        [Message(role="user", content="hello")],
        on_delta=yielded.append,
    )

    assert yielded == ["Hi", " there"]
    assert final.content == "Hi there"


def test_generate_stream_handles_tool_calls():
    config = GlobalConfig()
    client = LLMClient(config)
    client._client = _FakeClient(
        [
            [
                _FakeChunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="demo", arguments='{"value":'),
                        )
                    ]
                ),
                _FakeChunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id=None,
                            function=SimpleNamespace(name=None, arguments=' 1}'),
                        )
                    ]
                ),
            ],
            [_FakeChunk("done")],
        ]
    )

    events = []

    class _DemoTool:
        def to_schema(self):
            return {
                "name": "demo",
                "description": "demo tool",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            }

        def execute(self, **kwargs):
            return {"ok": kwargs["value"]}

    final = client.generate_stream(
        [Message(role="user", content="hello")],
        tools={"demo": _DemoTool()},
        on_tool_event=lambda event, payload: events.append((event, payload["name"])),
    )

    assert final.content == "done"
    assert events == [("start", "demo"), ("finish", "demo")]
