"""Hermetic fixtures: no network, no browser, no API keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot_qa.config import Settings
from chatbot_qa.golden import load_dataset
from chatbot_qa.models import GoldenDataset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        base_url="http://chatbot.test",
        golden_dir=ROOT / "datasets" / "golden",
        artifacts_dir=tmp_path / "artifacts",
        retries=3,
        retry_backoff_s=0,
        log_json=False,
    )


@pytest.fixture(scope="session")
def dataset() -> GoldenDataset:
    return load_dataset(ROOT / "datasets" / "golden")
