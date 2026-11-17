"""RAG quality: is each answer grounded in the knowledge base, on topic, and built on the right chunks?

Release-blocking: Faithfulness >= 0.85 and Answer Relevancy >= 0.85.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from eval_support import DATASET, answer_relevancy, contextual_precision, faithfulness, rag_test_case


@pytest.mark.parametrize("golden", DATASET.rag, ids=lambda g: g.id)
def test_answer_is_faithful_to_the_knowledge_base(golden, converse):
    transcript = converse(golden.id, [golden.input])
    assert_test(rag_test_case(golden, transcript), [faithfulness()])


@pytest.mark.parametrize("golden", DATASET.rag, ids=lambda g: g.id)
def test_answer_is_relevant_to_the_question(golden, converse):
    transcript = converse(golden.id, [golden.input])
    assert_test(rag_test_case(golden, transcript), [answer_relevancy()])


@pytest.mark.parametrize("golden", DATASET.rag, ids=lambda g: g.id)
def test_relevant_knowledge_is_ranked_first(golden, converse):
    transcript = converse(golden.id, [golden.input])
    assert_test(rag_test_case(golden, transcript), [contextual_precision()])
