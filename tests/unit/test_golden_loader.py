from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from chatbot_qa.golden import GoldenDatasetError, load_dataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def golden_copy(tmp_path: Path) -> Path:
    target = tmp_path / "golden"
    shutil.copytree(ROOT / "datasets" / "golden", target)
    return target


def edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_bundled_dataset_is_valid_and_covers_every_suite(dataset):
    assert len(dataset.rag) >= 8
    assert len(dataset.conversations_in("retention")) >= 2
    assert all(len(c.turns) >= 5 for c in dataset.conversations_in("retention"))
    outcomes = {c.should_escalate for c in dataset.conversations_in("escalation")}
    assert outcomes == {True, False}, "escalation needs positive and negative cases"
    assert {g.category for g in dataset.guardrails} >= {"prompt_injection", "toxicity", "unanswerable"}


def test_extra_rag_files_are_merged(golden_copy):
    extra = [{"id": "RAG-900", "input": "Is an eSIM free?", "expected_output": "Yes.", "expected_context": ["x"]}]
    (golden_copy / "rag_reviewed_synthetic.json").write_text(json.dumps(extra))
    assert "RAG-900" in {g.id for g in load_dataset(golden_copy).rag}


def test_duplicate_ids_are_rejected(golden_copy):
    edit(golden_copy / "guardrails.json", lambda data: data[0].update(id="RAG-001"))
    with pytest.raises(GoldenDatasetError, match="Duplicate golden ids"):
        load_dataset(golden_copy)


def test_short_retention_conversation_is_rejected(golden_copy):
    edit(golden_copy / "conversations.json", lambda data: data[0].update(turns=data[0]["turns"][:3]))
    with pytest.raises(GoldenDatasetError, match="at least 5 turns"):
        load_dataset(golden_copy)


def test_unknown_field_and_missing_file_are_rejected(golden_copy):
    edit(golden_copy / "rag.json", lambda data: data[0].update(expected_answer="typo"))
    with pytest.raises(GoldenDatasetError, match="rag.json"):
        load_dataset(golden_copy)
    with pytest.raises(GoldenDatasetError, match="not found"):
        load_dataset(golden_copy / "nope")
