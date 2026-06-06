from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from whispertome.audio.types import AudioChunk
from whispertome.audio.vad import EnergyVad, UtteranceSegmenter
from whispertome.config import AppConfig
from whispertome.stt.base import SpeechToTextModel
from whispertome.wake.router import WakeCommandRouter, WakeRouterEvent
from whispertome.wake.sliding_window import SlidingWakeDetector


StatusCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class PlaybackInterruption:
    event: WakeRouterEvent
    transcript: str
    stt_latency_ms: float
    wall_ms: float


class PlaybackWakeMonitor:
    """Consumes the live microphone stream during playback and interrupts on wake."""

    def __init__(
        self,
        *,
        config: AppConfig,
        chunks: Iterator[AudioChunk],
        stt_model: SpeechToTextModel,
        min_speech_ms: int,
        vad_threshold: float,
        logger: logging.Logger | None = None,
        status_callback: StatusCallback | None = None,
    ) -> None:
        self.interrupt_event = threading.Event()
        self._stop_event = threading.Event()
        self._config = config
        self._chunks = chunks
        self._stt_model = stt_model
        self._min_speech_ms = min_speech_ms
        self._vad_threshold = vad_threshold
        self._logger = logger or logging.getLogger(__name__)
        self._status_callback = status_callback
        self._thread: threading.Thread | None = None
        self._result: PlaybackInterruption | None = None
        self._error: BaseException | None = None

    @property
    def result(self) -> PlaybackInterruption | None:
        return self._result

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._logger.warning("playback_wake_monitor_stop_timeout=true")
            else:
                self._thread = None

    def _run(self) -> None:
        try:
            router = WakeCommandRouter(SlidingWakeDetector(self._config.wake))
            segmenter = UtteranceSegmenter(
                self._config.audio,
                EnergyVad(self._vad_threshold),
            )
            for utterance in segmenter.utterances(self._chunks_until_stopped()):
                if self._stop_event.is_set():
                    break
                active_ms = _active_ms(
                    utterance_samples=utterance.samples,
                    sample_rate=utterance.sample_rate,
                    block_ms=self._config.audio.block_ms,
                    vad_threshold=self._vad_threshold,
                )
                if active_ms < self._min_speech_ms:
                    continue
                self._emit_status("interruption stt", "checking wake")
                started = perf_counter()
                transcript = self._stt_model.transcribe(utterance)
                wall_ms = (perf_counter() - started) * 1000.0
                text = transcript.text.strip()
                self._logger.info(
                    (
                        "playback_wake_transcript duration_ms=%d active_ms=%d "
                        "wall_ms=%.1f latency_ms=%.1f provider=%s text=%r"
                    ),
                    utterance.duration_ms,
                    active_ms,
                    wall_ms,
                    transcript.latency_ms,
                    transcript.provider,
                    text,
                )
                if not text:
                    continue
                event = router.process_utterance(text)
                if event.kind == "idle":
                    continue
                self._result = PlaybackInterruption(
                    event=event,
                    transcript=text,
                    stt_latency_ms=transcript.latency_ms,
                    wall_ms=wall_ms,
                )
                self._logger.info(
                    (
                        "playback_wake_interrupt kind=%s transcript=%r command=%r "
                        "phrase=%r"
                    ),
                    event.kind,
                    text,
                    event.command,
                    event.match.phrase if event.match is not None else None,
                )
                self._emit_status("interruption detected", text)
                self.interrupt_event.set()
                self._stop_event.set()
                break
        except BaseException as exc:  # pragma: no cover - defensive background boundary
            self._error = exc
            self._logger.exception("playback_wake_monitor_failed")

    def _chunks_until_stopped(self) -> Iterator[AudioChunk]:
        while not self._stop_event.is_set():
            try:
                yield next(self._chunks)
            except StopIteration:
                return

    def _emit_status(self, text: str, detail: str) -> None:
        if self._status_callback is not None:
            self._status_callback(text, detail)


def _active_ms(
    *,
    utterance_samples,
    sample_rate: int,
    block_ms: int,
    vad_threshold: float,
) -> int:
    import numpy as np

    samples = np.asarray(utterance_samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        return 0

    frame_count = max(1, int(sample_rate * block_ms / 1000))
    active_frames = 0
    for start in range(0, samples.size, frame_count):
        chunk = samples[start : start + frame_count]
        if chunk.size and float(np.sqrt(np.mean(np.square(chunk)))) >= vad_threshold:
            active_frames += 1
    return active_frames * block_ms
