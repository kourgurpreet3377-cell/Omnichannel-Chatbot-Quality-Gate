"""RestApiAdapter against an in-memory transport wired to the mock chat service."""

from __future__ import annotations

import json

import httpx
import pytest

from chatbot_qa.adapters import AdapterError
from chatbot_qa.adapters.rest_api import _MISSING, RestApiAdapter, dig
from chatbot_qa.config import Settings
from mock_chatbot.service import ChatRequest, ChatService


def service_transport(service: ChatService, fail_first: int = 0, status: int = 503) -> httpx.MockTransport:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return httpx.Response(status, text="upstream unavailable")
        payload = ChatRequest(**json.loads(request.content))
        return httpx.Response(200, json=service.chat(payload).model_dump())

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_multi_turn_session_keeps_context_and_tracks_latency(settings):
    with RestApiAdapter(settings, transport=service_transport(ChatService())) as adapter:
        adapter.send("My order number is NT-48213.")
        reply = adapter.send("Can you remind me of my order number?")
    assert "NT-48213" in reply.text
    assert reply.latency_ms >= 0 and reply.escalated is False
    assert reply.raw["turn"] == 2


def test_each_adapter_instance_is_a_separate_session(settings):
    service = ChatService()
    with RestApiAdapter(settings, transport=service_transport(service)) as first:
        first.send("My order number is NT-48213.")
    with RestApiAdapter(settings, transport=service_transport(service)) as second:
        assert "NT-48213" not in second.send("Can you remind me of my order number?").text


def test_reply_carries_retrieval_context_and_escalation(settings):
    with RestApiAdapter(settings, transport=service_transport(ChatService())) as adapter:
        answer = adapter.send("When is my bill due?")
        handoff = adapter.send("I want to speak to a human agent.")
    assert answer.retrieval_context and "Billing" in answer.retrieval_context[0]
    assert handoff.escalated is True


def test_transient_503_is_retried(settings):
    transport = service_transport(ChatService(), fail_first=2)
    with RestApiAdapter(settings, transport=transport) as adapter:
        assert adapter.send("Hello").text.startswith("Hello")
    assert transport.calls["n"] == 3


def test_persistent_503_raises_adapter_error_with_context(settings):
    transport = service_transport(ChatService(), fail_first=99)
    with RestApiAdapter(settings, transport=transport) as adapter, pytest.raises(AdapterError) as raised:
        adapter.send("Hello")
    assert raised.value.context["status"] == 503
    assert transport.calls["n"] == settings.retries


def test_client_error_is_not_retried(settings):
    transport = service_transport(ChatService(), fail_first=99, status=401)
    with RestApiAdapter(settings, transport=transport) as adapter, pytest.raises(AdapterError, match="HTTP 401"):
        adapter.send("Hello")
    assert transport.calls["n"] == 1


def test_custom_response_contract_is_configurable(settings):
    """A different bot's API shape is handled by configuration alone."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert set(body) == {"conversationId", "text"}
        return httpx.Response(200, json={"data": {"messages": [{"text": " Hi there "}], "handoff": True}})

    custom = Settings(
        base_url="http://other.test",
        api_session_field="conversationId",
        api_message_field="text",
        api_reply_field="data.messages.0.text",
        api_escalated_field="data.handoff",
        api_context_field="data.sources",
    )
    with RestApiAdapter(custom, transport=httpx.MockTransport(handler)) as adapter:
        reply = adapter.send("Hello")
    assert reply.text == "Hi there" and reply.escalated is True and reply.retrieval_context == []


def test_missing_reply_field_is_a_contract_error(settings):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"answer": "wrong field"}))
    with RestApiAdapter(settings, transport=transport) as adapter, pytest.raises(AdapterError, match="no reply text"):
        adapter.send("Hello")


def test_send_before_start_session_fails_clearly(settings):
    with pytest.raises(AdapterError, match="before start_session"):
        RestApiAdapter(settings).send("Hello")


def test_dig():
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert dig(data, "a.b.1.c") == 2
    assert dig(data, "a.b.-1.c") == 2
    assert dig(data, "a.x") is _MISSING
    assert dig(data, "a.b.7.c") is _MISSING
