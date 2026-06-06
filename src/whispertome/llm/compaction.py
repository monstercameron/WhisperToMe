"""Automatic conversation compaction with two triggers feeding one compaction action:

1. Idle: after `idle_timeout_s` with no interaction, compact (default 1 hour).
2. Usage (the "Codex" strategy): after a turn whose *non-cached* context tokens cross a
   percentage of the model's context window, compact proactively. Cached prompt tokens are
   excluded ("context % less the context cache") since they don't grow effective context.

Compaction summarizes the chat and replaces history with the summary (reset fallback). It runs
under the shared turn_lock so it never overlaps a live turn or scheduled-event playback. The
idle trigger runs on a daemon thread; the usage trigger runs synchronously after a turn.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from time import monotonic

from whispertome.llm.base import LlmResponse
from whispertome.llm.conversation_tools import ConversationController

LOGGER = logging.getLogger(__name__)


class ConversationCompactionManager:
    def __init__(
        self,
        *,
        controller: ConversationController,
        turn_lock: AbstractContextManager | None = None,
        idle_enabled: bool = True,
        idle_timeout_s: float = 3600.0,
        usage_enabled: bool = True,
        context_window_tokens: int = 0,
        threshold_pct: float = 0.75,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._controller = controller
        self._turn_lock = turn_lock or nullcontext()
        self._idle_enabled = idle_enabled
        self._idle_timeout_s = max(1.0, idle_timeout_s)
        self._usage_enabled = usage_enabled
        self._context_window = max(0, context_window_tokens)
        self._threshold_pct = min(0.99, max(0.1, threshold_pct))
        self._log = logger or LOGGER
        self._clock = clock
        self._cv = threading.Condition()
        self._last_activity = clock()
        self._turns_since_compact = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    # -- activity / triggers ------------------------------------------------

    def note_turn(self, response: LlmResponse | None) -> None:
        """Record a completed turn: reset the idle timer and run the usage trigger."""
        with self._cv:
            self._last_activity = self._clock()
            self._turns_since_compact += 1
            self._cv.notify_all()
        if self._usage_enabled and self._context_window > 0 and response is not None:
            effective = max(0, (response.input_tokens or 0) - (response.cached_tokens or 0))
            pct = effective / self._context_window
            if pct >= self._threshold_pct:
                self._log.info(
                    "conversation_compact_usage pct=%.2f effective_tokens=%d window=%d",
                    pct, effective, self._context_window,
                )
                self._compact("usage")

    # -- lifecycle (idle trigger) -------------------------------------------

    def start(self) -> None:
        if self._thread is not None or not self._idle_enabled:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="compaction", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                elapsed = self._clock() - self._last_activity
                remaining = self._idle_timeout_s - elapsed
                due = remaining <= 0 and self._turns_since_compact > 0
                if not due:
                    # wait until the timer should elapse (or re-check periodically when idle/empty)
                    self._cv.wait(timeout=remaining if remaining > 0 else 60.0)
                    continue
            self._compact("idle")

    # -- compaction ---------------------------------------------------------

    def _compact(self, reason: str) -> None:
        with self._cv:
            if self._turns_since_compact <= 0:
                return
        try:
            with self._turn_lock:
                result = self._controller.auto_compact()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("conversation_compact_failed reason=%s error=%s", reason, exc)
            return
        with self._cv:
            self._turns_since_compact = 0
            self._last_activity = self._clock()
        self._log.info(
            "conversation_compacted reason=%s action=%s", reason, result.get("action")
        )
