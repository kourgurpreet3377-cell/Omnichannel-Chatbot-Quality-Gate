"""Settings and golden data loaded once at import time, so tests can be parametrised from them."""

from __future__ import annotations

from chatbot_qa.config import Settings
from chatbot_qa.golden import load_dataset

SETTINGS = Settings()
DATASET = load_dataset(SETTINGS.golden_dir)
