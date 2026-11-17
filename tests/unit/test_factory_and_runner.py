from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot_qa.adapters import AdapterError, AdapterFactory, BaseChatbotAdapter
from chatbot_qa.models import BotReply, Channel
from chatbot_qa.runner import ConversationError, ConversationRunner


class ScriptedAdapter(BaseChatbotAdapter):
    """Echoes messages; fails on the message 'boom'. Records lifecycle calls."""

    channel = Channel.UI
    events: list[str] = []

    def start_session(self) -> None:
        self.events.append("start")

    def send(self, message: str) -> BotReply:
        if message == "boom":
            raise AdapterError(self.channel, "widget did not answer", url="http://chatbot.test/")
        return BotReply(text=f"echo: {message}", latency_ms=1.0, channel=self.channel)

    def close(self) -> None:
        self.events.append("close")

    def capture_failure_artifacts(self, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        evidence = directory / "screenshot.png"
        evidence.write_bytes(b"png")
        return [evidence]


@pytest.fixture()
def factory(settings) -> AdapterFactory:
    ScriptedAdapter.events = []
    factory = AdapterFactory(settings)
    factory.register(Channel.UI, ScriptedAdapter)
    return factory


def test_factory_resolves_channels_and_defaults_to_settings(settings, factory):
    assert isinstance(factory.create("ui"), ScriptedAdapter)
    assert isinstance(factory.create(Channel.UI), ScriptedAdapter)
    assert factory.create().channel is settings.channel is Channel.API
    with pytest.raises(ValueError):
        factory.create("fax")


def test_runner_records_every_turn_and_always_closes_the_session(settings, factory):
    transcript = ConversationRunner(factory, settings).run("conv-1", ["hello", "bye"], channel="ui")
    assert [t.index for t in transcript.turns] == [1, 2]
    assert transcript.as_text() == "User: hello\nAssistant: echo: hello\nUser: bye\nAssistant: echo: bye"
    assert transcript.escalated is None  # this channel exposes no escalation signal
    assert ScriptedAdapter.events == ["start", "close"]


def test_channel_failure_produces_a_failure_trace_with_evidence(settings, factory):
    runner = ConversationRunner(factory, settings)
    with pytest.raises(ConversationError) as raised:
        runner.run("conv/with spaces", ["hello", "boom", "never sent"], channel="ui")

    trace_path = raised.value.trace_path
    assert trace_path.is_file() and settings.artifacts_dir in trace_path.parents
    trace = json.loads(trace_path.read_text())
    assert trace["failed_on_turn"] == 2 and trace["failed_message"] == "boom"
    assert trace["error"]["type"] == "AdapterError"
    assert trace["error"]["context"] == {"url": "http://chatbot.test/"}
    assert [t["user"] for t in trace["transcript"]["turns"]] == ["hello"]  # what happened before the failure
    assert trace["artifacts"] and Path(trace["artifacts"][0]).is_file()
    assert ScriptedAdapter.events == ["start", "close"]
