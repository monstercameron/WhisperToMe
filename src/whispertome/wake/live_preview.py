from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from whispertome.audio.types import AudioBuffer, AudioChunk
from whispertome.config import AppConfig
from whispertome.stt.base import SpeechToTextModel
from whispertome.wake.router import WakeCommandRouter, WakeRouterEvent
from whispertome.wake.sliding_window import SlidingWakeDetector

StatusCallback = Callable[[str, str], None]
LineCallback = Callable[[str], None]
WakeDetectedCallback = Callable[["InterimWakePreviewResult"], None]


@dataclass(frozen=True)
class InterimWakePreviewResult:
    event: WakeRouterEvent
    transcript: str
    wall_ms: float
    stt_latency_ms: float


class InterimWakePreview:
    """Runs rolling STT during active speech to update wake UI before speech pause."""

    def __init__(
        self,
        *,
        config: AppConfig,
        stt_model: SpeechToTextModel,
        stt_lock: Any | None = None,
        min_audio_ms: int = 900,
        interval_ms: int = 700,
        max_audio_ms: int = 4500,
        logger: logging.Logger | None = None,
        status_callback: StatusCallback | None = None,
        line_callback: LineCallback | None = None,
        on_wake_detected: WakeDetectedCallback | None = None,
    ) -> None:
        self._config = config
        self._stt_model = stt_model
        self._stt_lock = stt_lock
        self._min_audio_ms = max(config.audio.block_ms, min_audio_ms)
        self._interval_ms = max(config.audio.block_ms, interval_ms)
        self._max_audio_ms = max(self._min_audio_ms, max_audio_ms)
        self._logger = logger or logging.getLogger(__name__)
        self._status_callback = status_callback
        self._line_callback = line_callback
        self._on_wake_detected = on_wake_detected
        self._lock = threading.Lock()
        self._chunks: list[AudioChunk] = []
        self._utterance_id = 0
        self._last_started_ms = -1_000_000
        self._worker_running = False
        self._detected = False
        self._result: InterimWakePreviewResult | None = None

    @property
    def result(self) -> InterimWakePreviewResult | None:
        return self._result

    def start(self, chunks: tuple[AudioChunk, ...]) -> None:
        with self._lock:
            self._utterance_id += 1
            self._chunks = list(chunks)
            self._last_started_ms = -1_000_000
            self._worker_running = False
            self._detected = False
            self._result = None
            utterance_id = self._utterance_id
        self._maybe_schedule(utterance_id)

    def add_chunk(self, chunk: AudioChunk) -> None:
        with self._lock:
            if not self._chunks:
                return
            self._chunks.append(chunk)
            utterance_id = self._utterance_id
        self._maybe_schedule(utterance_id)

    def finish(self) -> None:
        with self._lock:
            self._utterance_id += 1
            self._chunks = []

    def _maybe_schedule(self, utterance_id: int) -> None:
        with self._lock:
            if self._detected or self._worker_running or utterance_id != self._utterance_id:
                return
            duration_ms = _duration_ms(self._chunks)
            if duration_ms < self._min_audio_ms:
                return
            now_ms = int(perf_counter() * 1000)
            if now_ms - self._last_started_ms < self._interval_ms:
                return
            snapshot = _tail_buffer(self._chunks, max_audio_ms=self._max_audio_ms)
            self._last_started_ms = now_ms
            self._worker_running = True

        thread = threading.Thread(
            target=self._run_snapshot,
            args=(utterance_id, snapshot),
            daemon=True,
        )
        thread.start()

    def _run_snapshot(self, utterance_id: int, audio: AudioBuffer) -> None:
        started = perf_counter()
        try:
            if self._stt_lock is None:
                transcript = self._stt_model.transcribe(audio)
            else:
                with self._stt_lock:
                    transcript = self._stt_model.transcribe(audio)
            wall_ms = (perf_counter() - started) * 1000.0
            text = transcript.text.strip()
            self._logger.info(
                (
                    "wake_interim_transcript utterance=%d duration_ms=%d wall_ms=%.1f "
                    "latency_ms=%.1f provider=%s text=%r"
                ),
                utterance_id,
                audio.duration_ms,
                wall_ms,
                transcript.latency_ms,
                transcript.provider,
                text,
            )
            if not text:
                return
            router = WakeCommandRouter(SlidingWakeDetector(self._config.wake))
            event = router.process_utterance(text)
            if event.kind == "idle":
                return
            result = InterimWakePreviewResult(
                event=event,
                transcript=text,
                wall_ms=wall_ms,
                stt_latency_ms=transcript.latency_ms,
            )
            with self._lock:
                if utterance_id != self._utterance_id or self._detected:
                    return
                self._detected = True
                self._result = result
            self._logger.info(
                (
                    "wake_interim_detected utterance=%d kind=%s transcript=%r "
                    "command=%r phrase=%r"
                ),
                utterance_id,
                event.kind,
                text,
                event.command,
                event.match.phrase if event.match is not None else None,
            )
            if self._status_callback is not None:
                detail = event.command or (event.match.phrase if event.match is not None else text)
                self._status_callback("wake detected", f"interim: {detail}")
            if self._line_callback is not None:
                self._line_callback(f"interim wake: {text}")
            if self._on_wake_detected is not None:
                try:
                    self._on_wake_detected(result)
                except Exception:
                    self._logger.exception("wake_interim_callback_failed")
        finally:
            with self._lock:
                self._worker_running = False
            self._maybe_schedule(utterance_id)


def _duration_ms(chunks: list[AudioChunk]) -> int:
    if not chunks:
        return 0
    return sum(chunk.duration_ms for chunk in chunks)


def _tail_buffer(chunks: list[AudioChunk], *, max_audio_ms: int) -> AudioBuffer:
    if not chunks:
        raise ValueError("Cannot build interim buffer from no chunks")
    selected: list[AudioChunk] = []
    total_ms = 0
    for chunk in reversed(chunks):
        selected.append(chunk)
        total_ms += chunk.duration_ms
        if total_ms >= max_audio_ms:
            break
    selected.reverse()
    sample_rate = selected[0].sample_rate
    return AudioBuffer(
        samples=np.concatenate([chunk.samples for chunk in selected]).astype(
            np.float32,
            copy=False,
        ),
        sample_rate=sample_rate,
    )
