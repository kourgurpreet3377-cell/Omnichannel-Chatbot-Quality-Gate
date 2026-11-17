"""Fixtures for the deterministic smoke gate. Needs a running chatbot; needs no LLM."""

from __future__ import annotations

import httpx
import pytest

from smoke_support import DATASET, SETTINGS

from chatbot_qa.adapters import AdapterFactory
from chatbot_qa.config import Settings
from chatbot_qa.logging_config import configure_logging
from chatbot_qa.models import GoldenDataset
from chatbot_qa.runner import ConversationRunner


@pytest.fixture(scope="session")
def settings() -> Settings:
    configure_logging(SETTINGS.log_level, json_logs=False)
    return SETTINGS


@pytest.fixture(scope="session")
def dataset() -> GoldenDataset:
    return DATASET


@pytest.fixture(scope="session", autouse=True)
def target_is_reachable(settings: Settings) -> None:
    """Fail the whole stage in one clear line if the chatbot is not up."""
    url = settings.url(settings.health_path)
    try:
        httpx.get(url, timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.exit(f"Chatbot is not reachable at {url}: {exc!r}. Start it with `make serve`.", returncode=2)


@pytest.fixture(scope="session")
def runner(settings: Settings) -> ConversationRunner:
    return ConversationRunner(AdapterFactory(settings), settings)
