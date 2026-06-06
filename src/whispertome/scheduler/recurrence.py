"""Canonical recurrence rules and next-slot computation.

Rules (authored by the LLM at creation, validated at save):
  - daily@HH:MM
  - weekly@<dow>@HH:MM         (dow = mon|tue|wed|thu|fri|sat|sun)
  - every@<N>@minutes|hours

daily/weekly are computed on the LOCAL wall clock then converted to UTC (so they stay on
the same local HH:MM across DST); `every` is a fixed UTC interval.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

_DOW = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class RecurrenceError(ValueError):
    pass


def _parse_hhmm(text: str) -> tuple[int, int]:
    hh, mm = text.split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise RecurrenceError(f"invalid time in recurrence: {text}")
    return h, m


def validate_recurrence(rule: str) -> None:
    """Raise RecurrenceError if the rule is not a supported canonical form."""
    next_recurrence(rule, datetime.now(UTC))


def next_recurrence(rule: str, after_utc: datetime) -> datetime:
    """Return the next firing time strictly after `after_utc`, as a UTC-aware datetime."""
    if after_utc.tzinfo is None:
        after_utc = after_utc.replace(tzinfo=UTC)
    parts = (rule or "").strip().lower().split("@")
    kind = parts[0]

    if kind == "every":
        if len(parts) != 3:
            raise RecurrenceError("every rule must be 'every@N@minutes|hours'")
        n = int(parts[1])
        if n <= 0:
            raise RecurrenceError("every interval must be positive")
        unit = parts[2]
        if unit.startswith("min"):
            delta = timedelta(minutes=n)
        elif unit.startswith("hour"):
            delta = timedelta(hours=n)
        else:
            raise RecurrenceError("every unit must be minutes or hours")
        return (after_utc + delta).astimezone(UTC)

    after_local = after_utc.astimezone()  # system local tz
    if kind == "daily":
        if len(parts) != 2:
            raise RecurrenceError("daily rule must be 'daily@HH:MM'")
        h, m = _parse_hhmm(parts[1])
        cand = after_local.replace(hour=h, minute=m, second=0, microsecond=0)
        while cand <= after_local:
            cand += timedelta(days=1)
        return cand.astimezone(UTC)

    if kind == "weekly":
        if len(parts) != 3 or parts[1][:3] not in _DOW:
            raise RecurrenceError("weekly rule must be 'weekly@<dow>@HH:MM'")
        target = _DOW.index(parts[1][:3])
        h, m = _parse_hhmm(parts[2])
        cand = after_local.replace(hour=h, minute=m, second=0, microsecond=0)
        while cand <= after_local or cand.weekday() != target:
            cand += timedelta(days=1)
        return cand.astimezone(UTC)

    raise RecurrenceError(f"unsupported recurrence rule: {rule!r}")
