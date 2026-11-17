"""Retry with exponential backoff for transient failures only."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from chatbot_qa.logging_config import get_logger

log = get_logger(__name__)
T = TypeVar("T")


def call_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_s: float,
    retry_on: tuple[type[BaseException], ...],
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``operation``; on an exception in ``retry_on`` wait and try again.

    Waits ``backoff_s``, then double, and so on. Any other exception, and the
    last failure, propagate unchanged so the real error is what gets reported.
    """
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retry_on as exc:
            if attempt == attempts:
                log.error("retry_exhausted", label=label, attempts=attempts, error=repr(exc))
                raise
            delay = backoff_s * 2 ** (attempt - 1)
            log.warning("retrying", label=label, attempt=attempt, next_delay_s=delay, error=repr(exc))
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
