"""Multi-turn quality: does the bot keep context across 5+ turns and finish what the user came for?

Uses DeepEval's ConversationalTestCase with Knowledge Retention and Conversation
Completeness, backed by exact per-turn checks from the golden conversation.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from eval_support import DATASET, conversation_completeness, conversational_test_case, knowledge_retention

from chatbot_qa.checks import turn_expectation_violations

RETENTION = DATASET.conversations_in("retention")


@pytest.mark.parametrize("golden", RETENTION, ids=lambda g: g.id)
def test_recalled_facts_are_exact(golden, converse):
    """Deterministic: an order number is either right or wrong; no judge needed."""
    transcript = converse(golden.id, golden.user_messages)
    assert len(transcript.turns) >= 5
    violations = turn_expectation_violations(golden, transcript)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("golden", RETENTION, ids=lambda g: g.id)
def test_knowledge_is_retained_across_turns(golden, converse):
    transcript = converse(golden.id, golden.user_messages)
    assert_test(conversational_test_case(golden, transcript), [knowledge_retention()])


@pytest.mark.parametrize("golden", RETENTION, ids=lambda g: g.id)
def test_conversation_satisfies_the_users_intentions(golden, converse):
    transcript = converse(golden.id, golden.user_messages)
    assert_test(conversational_test_case(golden, transcript), [conversation_completeness()])
