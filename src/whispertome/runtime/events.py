"""Async runtime event bus for live speech events.

Producers (STT, later VAD/TTS) `publish()` events without blocking; a background
dispatch thread fans them out to listeners. This decouples inference from reaction:
the STT thread keeps transcribing while the UI (or any observer) reacts off-thread.

Mirrors the model-load event API (`models/loading.py`) but for *runtime* events that
fire repeatedly during a session, not just at load.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

EventSource = Literal["stt", "tts", "vad"]
EventKind = Literal["start", "progress", "final", "error", "level"]


@dataclass(frozen=True)
class RuntimeEvent:
    source: EventSource
    kind: EventKind
    value: float = 0.0      # generic magnitude, e.g. 0..1 activity/level/progress
    text: str = ""          # transcript / message payload
    detail: str = ""        # human note
    provider: str | None = None
    elapsed_ms: float | None = None


EventListener = Callable[[RuntimeEvent], None]

_STOP = object()  # sentinel enqueued by stop() to release the blocking dispatcher


class EventBus:
    """Thread-safe, non-blocking publish; background dispatch to listeners.

    `publish()` never blocks the producer (it enqueues). A daemon dispatch thread
    drains the queue and calls each listener, isolating listener errors. If the bus
    has not been `start()`ed, `publish()` dispatches inline so events are never lost.
    """

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._queue: "queue.Queue[object]" = queue.Queue()  # RuntimeEvent | _STOP
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def subscribe(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._queue = queue.Queue()  # drop any stale _STOP sentinel from a prior stop()
        self._thread = threading.Thread(target=self._dispatch_loop, name="event-bus", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            self._queue.put(_STOP)  # unblock the dispatcher immediately
            thread.join(timeout=1.0)

    def publish(self, event: RuntimeEvent) -> None:
        if self._thread is None:
            self._fan_out(event)   # no dispatcher running -> inline, never drop
        else:
            self._queue.put(event)

    def _dispatch_loop(self) -> None:
        # Block on the queue (no idle polling) so an idle session spends zero CPU here —
        # the dispatcher only wakes when an event actually arrives. A _STOP sentinel from
        # stop() releases the block at shutdown.
        while not self._stop.is_set():
            event = self._queue.get()
            if event is _STOP:
                break
            self._fan_out(event)

    def _fan_out(self, event: RuntimeEvent) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # pragma: no cover - observers must not break producers
                pass


class _NullBus(EventBus):
    def publish(self, event: RuntimeEvent) -> None:  # noqa: D401
        return None


NULL_BUS: EventBus = _NullBus()


def attach_speech_reactions(bus: EventBus, ui) -> None:
    """Subscribe a listener that maps STT events onto a voice UI (duck-typed:
    needs `status`, `activity`, `user_text`). Drives the activity waveform and status
    line dynamically as speech is recognized — reacting off the STT thread."""

    def react(event: RuntimeEvent) -> None:
        if event.source != "stt":
            return
        if event.kind == "start":
            ui.status("transcribing")
            ui.activity(0.65)
        elif event.kind in ("progress", "level"):
            ui.activity(max(0.0, min(1.0, event.value)))
        elif event.kind == "final":
            if event.text:
                ui.user_text(event.text)
            ui.status("heard")
            ui.activity(0.0)
        elif event.kind == "error":
            ui.status("stt error", event.detail)
            ui.activity(0.0)

    bus.subscribe(react)
