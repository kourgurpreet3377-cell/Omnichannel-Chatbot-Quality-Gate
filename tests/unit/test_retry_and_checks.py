from __future__ import annotations

import pytest

from chatbot_qa.checks import percentile
from chatbot_qa.retry import call_with_retry


class Flaky:
    def __init__(self, failures: int, error: Exception) -> None:
        self.failures, self.error, self.calls = failures, error, 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def test_retries_transient_errors_with_exponential_backoff():
    sleeps: list[float] = []
    operation = Flaky(2, ConnectionError("reset"))
    result = call_with_retry(
        operation, attempts=4, backoff_s=0.5, retry_on=(ConnectionError,), label="t", sleep=sleeps.append
    )
    assert result == "ok" and operation.calls == 3
    assert sleeps == [0.5, 1.0]


def test_gives_up_after_the_last_attempt_and_raises_the_real_error():
    operation = Flaky(5, ConnectionError("still down"))
    with pytest.raises(ConnectionError, match="still down"):
        call_with_retry(operation, attempts=3, backoff_s=0, retry_on=(ConnectionError,), label="t", sleep=lambda _: None)
    assert operation.calls == 3


def test_non_retryable_errors_fail_immediately():
    operation = Flaky(5, ValueError("bad contract"))
    with pytest.raises(ValueError):
        call_with_retry(operation, attempts=3, backoff_s=0, retry_on=(ConnectionError,), label="t", sleep=lambda _: None)
    assert operation.calls == 1


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 95) == 95.0
    assert percentile(values, 50) == 50.0
    assert percentile([120.0], 95) == 120.0
    with pytest.raises(ValueError):
        percentile([], 95)
