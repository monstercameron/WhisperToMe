"""In-process scheduler thread: fires due events, catches up missed ones on launch,
and recovers crash orphans. Executes a recorded action deterministically (no LLM) under
a shared turn_lock so it never overlaps a live conversation turn or audio playback.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from uuid import uuid4

from whispertome.scheduler.clock import Clock
from whispertome.scheduler.executor import ActionExecutor
from whispertome.scheduler.model import action_from_json
from whispertome.scheduler.recurrence import next_recurrence

LOGGER = logging.getLogger(__name__)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


class SchedulerThread:
    def __init__(
        self,
        *,
        store,
        executor: ActionExecutor,
        clock: Clock | None = None,
        turn_lock: AbstractContextManager | None = None,
        on_fire: Callable[[dict], None] | None = None,
        on_recover: Callable[[list[dict]], None] | None = None,
        on_view: Callable[[str | None, float | None], None] | None = None,
        check_interval_s: float = 30.0,
        max_sleep_s: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store
        self._executor = executor
        self._clock = clock or Clock()
        self._turn_lock = turn_lock or nullcontext()
        self._on_fire = on_fire or (lambda _e: None)
        self._on_recover = on_recover or (lambda _o: None)
        self._on_view = on_view or (lambda _t, _e: None)
        self._check_interval_s = max(1.0, check_interval_s)
        self._max_sleep_s = max(1.0, max_sleep_s)
        self._log = logger or LOGGER
        self._pid = os.getpid()
        self._cv = threading.Condition()
        self._stop = False
        self._pending_wake = False
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        orphans = self._store.recover_orphaned_firing()
        if orphans:
            self._log.warning("scheduler_recovered_orphans count=%d", len(orphans))
            try:
                self._on_recover(orphans)
            except Exception:  # noqa: BLE001
                pass
        self._stop = False
        self._refresh_view()
        self._thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def wake(self) -> None:
        """Signal that events changed (added/cancelled) so the loop recomputes its sleep."""
        with self._cv:
            self._pending_wake = True
            self._cv.notify_all()

    # -- loop ---------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                self._pending_wake = False  # consume before working; wake() during work re-loops
            self._fire_due()
            self._refresh_view()
            timeout = self._sleep_seconds()
            with self._cv:
                if self._stop:
                    return
                if self._pending_wake:  # a new/cancelled event arrived during the fire pass
                    continue
                self._cv.wait(timeout=timeout)

    def _fire_due(self) -> None:
        try:
            due = self._store.get_due_events(self._clock.now_iso())
        except Exception as exc:  # noqa: BLE001 — never let a DB hiccup kill the thread
            self._log.warning("scheduler_due_query_failed error=%s", exc)
            return
        for event in due:
            with self._cv:
                if self._stop:
                    return
            self._fire(event)

    def _refresh_view(self) -> None:
        """Push the next-up event (title + epoch) to the UI for the 'next up' display."""
        try:
            event = self._store.next_pending_event()
        except Exception:  # noqa: BLE001
            return
        if not event:
            self._on_view(None, None)
            return
        try:
            epoch = datetime.fromisoformat(event["next_fire_at"]).timestamp()
        except (ValueError, TypeError):
            epoch = None
        try:
            self._on_view(event.get("title"), epoch)
        except Exception:  # noqa: BLE001
            pass

    def _sleep_seconds(self) -> float:
        try:
            nxt = self._store.peek_next_fire_at()
        except Exception:  # noqa: BLE001
            nxt = None
        if not nxt:
            return self._check_interval_s
        try:
            delta = (datetime.fromisoformat(nxt) - self._clock.now()).total_seconds()
        except ValueError:
            return self._check_interval_s
        return max(0.0, min(delta, self._max_sleep_s, self._check_interval_s))

    # -- firing -------------------------------------------------------------

    def _fire(self, event: dict) -> None:
        token = uuid4().hex
        claimed = self._store.claim_event_for_fire(
            event_id=event["id"],
            expected_next_fire_at=event["next_fire_at"],
            lease_token=token,
            lease_pid=self._pid,
        )
        if claimed is None:
            return  # lost the race / already advanced — never double-fire

        title = claimed.get("title", "")
        self._log.info("scheduler_fire id=%s title=%r", claimed["id"], title)
        try:
            action = action_from_json(claimed["action_json"])
        except Exception as exc:  # noqa: BLE001 — malformed stored action
            self._store.mark_event_failed(
                claimed["id"], error=f"bad action: {exc}", fired_at=self._clock.now_iso()
            )
            return

        ok = False
        error: str | None = None
        try:
            with self._turn_lock:
                try:
                    self._on_fire(claimed)
                except Exception:  # noqa: BLE001
                    pass
                result = self._executor.run(action)
            ok = result.ok
            if not ok:
                error = "; ".join(s.detail for s in result.steps if not s.ok)[:300]
        except Exception as exc:  # noqa: BLE001 — executor shouldn't raise, but be safe
            error = str(exc)

        fired_at = self._clock.now_iso()
        if claimed.get("trigger") == "recurring" and claimed.get("recurrence_rule"):
            try:
                nxt = _iso(next_recurrence(claimed["recurrence_rule"], self._clock.now()))
                self._store.reschedule_recurring(
                    claimed["id"], next_fire_at=nxt, fired_at=fired_at, last_error=error
                )
                return
            except Exception as exc:  # noqa: BLE001 — bad rule -> stop recurring
                error = f"recurrence failed: {exc}"
        if ok:
            self._store.complete_one_shot(claimed["id"], fired_at=fired_at)
        else:
            self._store.mark_event_failed(
                claimed["id"], error=error or "action failed", fired_at=fired_at
            )
