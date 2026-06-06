from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from whispertome.organizer.store import OrganizerStore
from whispertome.scheduler.clock import FakeClock
from whispertome.scheduler.executor import ActionExecutor
from whispertome.scheduler.model import parse_action, validate_tool_names
from whispertome.scheduler.recurrence import next_recurrence
from whispertome.scheduler.scheduler import SchedulerThread


class FakeEvent:
    def __init__(self, output):
        self.output = output


class FakeRegistry:
    def __init__(self, results=None, delay=0.0):
        self.calls: list[tuple] = []
        self._results = results or {}
        self._delay = delay

    def execute(self, name, args):
        self.calls.append((name, args))
        if self._delay:
            time.sleep(self._delay)
        return FakeEvent(self._results.get(name, {"ok": True}))


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def _build(tmp_path, *, registry=None, clock=None, step_timeout_s=20.0):
    store = OrganizerStore(tmp_path / "org")
    spoken: list[str] = []
    reg = registry or FakeRegistry()
    clock = clock or FakeClock(datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC))
    executor = ActionExecutor(tool_registry=reg, speak=spoken.append, step_timeout_s=step_timeout_s)
    sched = SchedulerThread(store=store, executor=executor, clock=clock)
    return store, sched, spoken, reg, clock


def test_one_shot_fires_and_completes(tmp_path):
    store, sched, spoken, _reg, _clock = _build(tmp_path)
    store.add_scheduled_event(
        title="stretch", trigger="one_shot", next_fire_at="2026-06-06T11:59:00+00:00",
        action_json='{"version":1,"steps":[{"type":"speak","text":"stretch"}]}',
    )
    sched._fire_due()
    assert spoken == ["stretch"]
    assert store.list_scheduled_events(include_terminal=True)[0]["status"] == "done"


def test_no_double_fire(tmp_path):
    store, sched, spoken, _reg, _clock = _build(tmp_path)
    store.add_scheduled_event(
        title="once", trigger="one_shot", next_fire_at="2026-06-06T11:00:00+00:00",
        action_json='{"version":1,"steps":[{"type":"speak","text":"hi"}]}',
    )
    sched._fire_due()
    sched._fire_due()  # second pass must not re-fire
    assert spoken == ["hi"]


def test_recurring_advances_and_collapses_backlog(tmp_path):
    clock = FakeClock(datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC))
    store, sched, _spoken, reg, _ = _build(tmp_path, clock=clock)
    store.add_scheduled_event(
        title="ping", trigger="recurring", recurrence_rule="every@1@minutes",
        next_fire_at="2026-06-06T11:00:00+00:00",  # an hour overdue
        action_json='{"version":1,"steps":[{"type":"tool","name":"noop","args":{}}]}',
    )
    sched._fire_due()
    # fired once (no backlog burst), rescheduled strictly after now
    assert len(reg.calls) == 1
    row = store.list_scheduled_events()[0]
    assert row["status"] == "pending"
    assert datetime.fromisoformat(row["next_fire_at"]) > clock.now()


def test_catch_up_fires_past_due(tmp_path):
    store, sched, spoken, _reg, _clock = _build(tmp_path)
    store.add_scheduled_event(
        title="old", trigger="one_shot", next_fire_at="2020-01-01T00:00:00+00:00",
        action_json='{"version":1,"steps":[{"type":"speak","text":"late"}]}',
    )
    sched._fire_due()
    assert spoken == ["late"]


def test_crash_recovery_marks_needs_review_without_replay(tmp_path):
    store, sched, _spoken, reg, _clock = _build(tmp_path)
    e = store.add_scheduled_event(
        title="job", trigger="one_shot", next_fire_at="2020-01-01T00:00:00+00:00",
        action_json='{"version":1,"steps":[{"type":"tool","name":"danger","args":{}}]}',
    )
    # simulate a crash mid-fire: claimed -> left 'firing'
    store.claim_event_for_fire(
        event_id=e["id"], expected_next_fire_at=e["next_fire_at"], lease_token="t", lease_pid=1
    )
    recovered: list = []
    sched._on_recover = recovered.extend
    sched.start()
    sched.stop()
    assert reg.calls == []  # tool steps NOT replayed
    statuses = {x["title"]: x["status"] for x in store.list_scheduled_events(include_terminal=True)}
    assert statuses["job"] == "needs_review"
    assert [o["title"] for o in recovered] == ["job"]


def test_tool_failure_one_shot_marks_failed(tmp_path):
    reg = FakeRegistry(results={"boom": {"ok": False, "error": "nope"}})
    store, sched, _spoken, _reg, _clock = _build(tmp_path, registry=reg)
    store.add_scheduled_event(
        title="x", trigger="one_shot", next_fire_at="2020-01-01T00:00:00+00:00",
        action_json='{"version":1,"steps":[{"type":"tool","name":"boom","args":{}}]}',
    )
    sched._fire_due()
    assert store.list_scheduled_events(include_terminal=True)[0]["status"] == "failed"


def test_recurring_survives_tool_failure(tmp_path):
    reg = FakeRegistry(results={"boom": {"ok": False, "error": "nope"}})
    store, sched, _spoken, _reg, clock = _build(tmp_path, registry=reg)
    store.add_scheduled_event(
        title="r", trigger="recurring", recurrence_rule="every@5@minutes",
        next_fire_at="2020-01-01T00:00:00+00:00",
        action_json='{"version":1,"steps":[{"type":"tool","name":"boom","args":{}}]}',
    )
    sched._fire_due()
    row = store.list_scheduled_events()[0]
    assert row["status"] == "pending"  # keeps recurring despite failure
    assert row["last_error"]


def test_step_timeout_is_bounded(tmp_path):
    reg = FakeRegistry(delay=3.0)
    store, sched, _spoken, _reg, _clock = _build(tmp_path, registry=reg, step_timeout_s=1.0)
    store.add_scheduled_event(
        title="slow", trigger="one_shot", next_fire_at="2020-01-01T00:00:00+00:00",
        action_json='{"version":1,"steps":[{"type":"tool","name":"hang","args":{}},{"type":"speak","text":"after"}]}',
    )
    start = time.perf_counter()
    sched._fire_due()
    # the speak step still runs after the bounded tool timeout
    assert time.perf_counter() - start < 2.5
    assert store.list_scheduled_events(include_terminal=True)[0]["status"] == "failed"


def test_template_substitution_and_missing_key(tmp_path):
    reg = FakeRegistry(results={"count_tool": {"ok": True, "count": 7}})
    store, sched, spoken, _reg, _clock = _build(tmp_path, registry=reg)
    store.add_scheduled_event(
        title="t", trigger="one_shot", next_fire_at="2020-01-01T00:00:00+00:00",
        action_json=(
            '{"version":1,"steps":['
            '{"type":"tool","name":"count_tool","args":{}},'
            '{"type":"speak","text":"have {step0.count} and {step0.missing}x"}]}'
        ),
    )
    sched._fire_due()
    assert spoken == ["have 7 and x"]


def test_save_time_validation_rejects_unknown_tool():
    action = parse_action([{"type": "tool", "name": "ghost", "args": {}}])
    assert validate_tool_names(action, {"real"}) == ["ghost"]


def test_recurrence_next_slot():
    after = datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC)
    nxt = next_recurrence("every@30@minutes", after)
    assert nxt == after + timedelta(minutes=30)
    daily = next_recurrence("daily@09:00", after)
    assert daily > after  # next 09:00 local strictly after now
