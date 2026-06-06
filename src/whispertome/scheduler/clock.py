"""Clock abstraction so the scheduler is testable with a controllable time source."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta


class Clock:
    """Real wall clock (UTC-aware)."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def now_iso(self) -> str:
        return self.now().replace(microsecond=0).isoformat()


class FakeClock(Clock):
    """Deterministic clock for tests."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        self._now = start

    def now(self) -> datetime:
        return self._now

    def set(self, value: datetime) -> None:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        self._now = value

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)
