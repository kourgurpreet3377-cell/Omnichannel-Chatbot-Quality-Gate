"""Loads and validates the golden dataset.

Layout of the dataset directory:

    rag.json             single-turn questions with expected answer + knowledge chunks
    rag_*.json           optional extra files, e.g. reviewed synthetic goldens
    conversations.json   multi-turn scenarios (retention, escalation)
    guardrails.json      adversarial and out-of-scope probes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from chatbot_qa.models import ConversationGolden, GoldenDataset, GuardrailGolden, RagGolden, StrictModel

T = TypeVar("T", bound=StrictModel)


class GoldenDatasetError(ValueError):
    pass


def _load_list(path: Path, model: type[T]) -> list[T]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TypeAdapter(list[model]).validate_python(raw)
    except (OSError, ValueError) as exc:
        raise GoldenDatasetError(f"{path}: {exc}") from exc


def load_dataset(directory: Path) -> GoldenDataset:
    directory = Path(directory)
    if not directory.is_dir():
        raise GoldenDatasetError(f"Golden dataset directory not found: {directory}")

    rag: list[RagGolden] = []
    for path in [directory / "rag.json", *sorted(directory.glob("rag_*.json"))]:
        rag += _load_list(path, RagGolden)
    dataset = GoldenDataset(
        rag=rag,
        conversations=_load_list(directory / "conversations.json", ConversationGolden),
        guardrails=_load_list(directory / "guardrails.json", GuardrailGolden),
    )

    ids = [g.id for g in [*dataset.rag, *dataset.conversations, *dataset.guardrails]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise GoldenDatasetError(f"Duplicate golden ids: {duplicates}")
    return dataset
