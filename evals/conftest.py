"""Fixtures for the LLM-judged stage of the release gate.

Each golden conversation is played once per session through the configured
channel (CHATBOT_CHANNEL=api|ui); every metric then scores the same transcript.
When a test fails, the transcript and the failure reason are written to
``artifacts/failures/`` so the cause can be read without re-running anything.
"""

from __future__ import annotations

import os

import httpx
import pytest
from eval_support import SETTINGS, judge_key_present

from chatbot_qa.adapters import AdapterFactory
from chatbot_qa.config import Settings
from chatbot_qa.logging_config import configure_logging
from chatbot_qa.models import Transcript
from chatbot_qa.runner import ConversationRunner


def _stop(message: str) -> None:
    """In CI a missing precondition fails the gate; locally it skips with the reason."""
    if os.getenv("CI"):
        pytest.fail(message)  # a skipped quality gate that shows green is worse than no gate
    pytest.skip(message)


@pytest.fixture(scope="session")
def settings() -> Settings:
    configure_logging("WARNING", json_logs=False)
    return SETTINGS


@pytest.fixture(scope="session")
def runner(settings: Settings) -> ConversationRunner:
    if not judge_key_present():
        _stop("LLM evals need a judge API key: OPENAI_API_KEY, or ANTHROPIC_API_KEY with EVAL_JUDGE_PROVIDER=anthropic")
    url = settings.url(settings.health_path)
    try:
        httpx.get(url, timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        _stop(f"Chatbot is not reachable at {url}: {exc!r}")
    return ConversationRunner(AdapterFactory(settings), settings)


@pytest.fixture(scope="session")
def _transcripts() -> dict[str, Transcript]:
    return {}


@pytest.fixture()
def converse(request, runner: ConversationRunner, _transcripts: dict[str, Transcript]):
    """``converse(conversation_id, messages) -> Transcript``, cached for the session."""
    used: list[Transcript] = []

    def _converse(conversation_id: str, messages: list[str]) -> Transcript:
        if conversation_id not in _transcripts:
            _transcripts[conversation_id] = runner.run(conversation_id, messages)
        used.append(_transcripts[conversation_id])
        return _transcripts[conversation_id]

    yield _converse

    report = getattr(request.node, "rep_call", None)  # set by the hook in the root conftest.py
    if report is not None and report.failed:
        for transcript in used:
            runner.traces.write(
                request.node.name,
                transcript,
                details={"test": request.node.nodeid, "failure": str(report.longrepr)[:6000]},
            )
