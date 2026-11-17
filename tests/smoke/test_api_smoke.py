"""Stage 1 of the release gate: deterministic API checks.

Milliseconds per test, zero judge tokens. If any of these fail, the build stops
before the LLM evaluation stage starts.
"""

from __future__ import annotations

import pytest
from smoke_support import DATASET

from chatbot_qa.checks import (
    escalation_violations,
    forbidden_content_violations,
    percentile,
    turn_expectation_violations,
)

pytestmark = [pytest.mark.smoke, pytest.mark.api]
LATENCIES: list[float] = []


def run(runner, conversation_id, messages):
    transcript = runner.run(conversation_id, messages, channel="api")
    LATENCIES.extend(turn.reply.latency_ms for turn in transcript.turns)
    return transcript


def test_bot_answers_with_text_and_latency(runner):
    reply = run(runner, "smoke-contract", ["Hello"]).last
    assert reply.text.strip()
    assert reply.latency_ms > 0


def test_sessions_are_isolated(runner):
    run(runner, "smoke-isolation-a", ["My order number is NT-90417."])
    other = run(runner, "smoke-isolation-b", ["Can you remind me of my order number?"])
    assert "NT-90417" not in other.last.text, "one customer's data leaked into another session"


@pytest.mark.parametrize("golden", DATASET.conversations, ids=lambda g: g.id)
def test_conversation_goldens_meet_turn_expectations(golden, runner):
    transcript = run(runner, golden.id, golden.user_messages)
    violations = turn_expectation_violations(golden, transcript) + escalation_violations(golden, transcript)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("golden", DATASET.rag, ids=lambda g: g.id)
def test_rag_goldens_return_knowledge_context(golden, runner):
    reply = run(runner, golden.id, [golden.input]).last
    if not reply.retrieval_context:
        pytest.skip("this API does not expose retrieval context; evals fall back to the golden's expected context")
    assert reply.retrieval_context == golden.expected_context, "the bot answered from the wrong knowledge chunk"


@pytest.mark.parametrize("golden", DATASET.guardrails, ids=lambda g: g.id)
def test_guardrail_goldens_leak_nothing(golden, runner, settings):
    transcript = run(runner, golden.id, golden.turns)
    violations = forbidden_content_violations(golden, transcript, settings.canaries)
    assert not violations, "\n".join(violations)


def test_p95_latency_is_within_budget(settings):
    """Runs last: uses the latency of every API call made by this module."""
    assert LATENCIES, "no latency samples were collected"
    p95 = percentile(LATENCIES, 95)
    assert p95 <= settings.api_latency_budget_ms, (
        f"API p95 latency {p95:.0f} ms over {len(LATENCIES)} turns exceeds the "
        f"{settings.api_latency_budget_ms:.0f} ms budget"
    )
