from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread
from time import perf_counter

from whispertome.audio.playback import SpeakerOutput
from whispertome.audio.types import SynthesizedSpeech
from whispertome.errors import WhisperToMeError
from whispertome.tts.base import SpeechSynthesisResult, TextToSpeechModel


_SENTINEL = object()


@dataclass(frozen=True)
class StreamingSpeechChunk:
    index: int
    text: str
    speech: SynthesizedSpeech
    latency_ms: float
    provider: str


@dataclass(frozen=True)
class StreamingSpeechResult:
    chunks: tuple[StreamingSpeechChunk, ...]
    total_ms: float
    synthesis_ms: float
    playback_ms: float
    first_text_ms: float | None
    first_audio_ms: float | None
    first_playback_ms: float | None
    interrupted: bool

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def total_audio_ms(self) -> int:
        return sum(chunk.speech.duration_ms for chunk in self.chunks)

    @property
    def provider(self) -> str:
        return self.chunks[0].provider if self.chunks else ""

    @property
    def sample_rate(self) -> int | None:
        return self.chunks[0].speech.sample_rate if self.chunks else None


class StreamingSpeechPlayer:
    """Synthesizes text chunks and plays completed audio chunks in order."""

    def __init__(
        self,
        *,
        tts_model: TextToSpeechModel,
        speaker: SpeakerOutput | None,
        interrupt_event: Event | None = None,
        logger: logging.Logger | None = None,
        status_callback=None,
    ) -> None:
        self._tts_model = tts_model
        self._speaker = speaker
        self._interrupt_event = interrupt_event
        self._logger = logger or logging.getLogger(__name__)
        self._status_callback = status_callback
        self._text_queue: Queue[str | object] = Queue()
        self._audio_queue: Queue[StreamingSpeechChunk | object] = Queue()
        self._chunks: list[StreamingSpeechChunk] = []
        self._lock = Lock()
        self._error: BaseException | None = None
        self._started = perf_counter()
        self._first_text_ms: float | None = None
        self._first_audio_ms: float | None = None
        self._first_playback_ms: float | None = None
        self._synthesis_ms = 0.0
        self._playback_ms = 0.0
        self._interrupted = False
        self._closed = False
        self._tts_thread = Thread(target=self._run_tts, daemon=True)
        self._playback_thread = Thread(target=self._run_playback, daemon=True)
        self._tts_thread.start()
        self._playback_thread.start()

    def enqueue_text(self, text: str) -> None:
        text = text.strip()
        if not text or self._closed:
            return
        if self._first_text_ms is None:
            self._first_text_ms = self._elapsed_ms()
        self._text_queue.put(text)

    def finish(self) -> StreamingSpeechResult:
        self._closed = True
        self._text_queue.put(_SENTINEL)
        self._tts_thread.join()
        self._playback_thread.join()
        if self._error is not None:
            raise WhisperToMeError(f"Streaming TTS failed: {self._error}") from self._error
        return StreamingSpeechResult(
            chunks=tuple(self._chunks),
            total_ms=self._elapsed_ms(),
            synthesis_ms=self._synthesis_ms,
            playback_ms=self._playback_ms,
            first_text_ms=self._first_text_ms,
            first_audio_ms=self._first_audio_ms,
            first_playback_ms=self._first_playback_ms,
            interrupted=self._interrupted,
        )

    def cancel(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._interrupted = True
        self._text_queue.put(_SENTINEL)
        self._audio_queue.put(_SENTINEL)
        self._tts_thread.join(timeout=1.0)
        self._playback_thread.join(timeout=1.0)

    def _run_tts(self) -> None:
        index = 0
        try:
            while True:
                text = self._text_queue.get()
                if text is _SENTINEL:
                    break
                if self._interrupt_event is not None and self._interrupt_event.is_set():
                    self._interrupted = True
                    continue
                index += 1
                self._set_status("running streaming tts", f"chunk {index}")
                started = perf_counter()
                result = self._tts_model.synthesize(str(text))
                wall_ms = (perf_counter() - started) * 1000.0
                self._synthesis_ms += wall_ms
                if self._first_audio_ms is None:
                    self._first_audio_ms = self._elapsed_ms()
                chunk = StreamingSpeechChunk(
                    index=index,
                    text=str(text),
                    speech=result.speech,
                    latency_ms=result.latency_ms,
                    provider=result.provider,
                )
                with self._lock:
                    self._chunks.append(chunk)
                self._audio_queue.put(chunk)
        except BaseException as exc:  # pragma: no cover - exercised via caller tests
            self._error = exc
            self._logger.exception("streaming_tts_failed")
        finally:
            self._audio_queue.put(_SENTINEL)

    def _run_playback(self) -> None:
        try:
            while True:
                item = self._audio_queue.get()
                if item is _SENTINEL:
                    break
                assert isinstance(item, StreamingSpeechChunk)
                if self._speaker is None:
                    continue
                if self._interrupt_event is not None and self._interrupt_event.is_set():
                    self._interrupted = True
                    break
                self._set_status("playing streaming speech", f"chunk {item.index}")
                if self._first_playback_ms is None:
                    self._first_playback_ms = self._elapsed_ms()
                started = perf_counter()
                playback = self._speaker.play_interruptible(
                    item.speech,
                    interrupt_event=self._interrupt_event,
                )
                self._playback_ms += (perf_counter() - started) * 1000.0
                if playback.interrupted:
                    self._interrupted = True
                    break
        except BaseException as exc:  # pragma: no cover - device dependent
            self._error = exc
            self._logger.exception("streaming_playback_failed")

    def _set_status(self, text: str, detail: str = "") -> None:
        if self._status_callback is not None:
            self._status_callback(text, detail)

    def _elapsed_ms(self) -> float:
        return (perf_counter() - self._started) * 1000.0


def concatenate_speech(chunks: tuple[StreamingSpeechChunk, ...]) -> SynthesizedSpeech | None:
    if not chunks:
        return None

    import numpy as np

    sample_rate = chunks[0].speech.sample_rate
    if any(chunk.speech.sample_rate != sample_rate for chunk in chunks):
        return None
    return SynthesizedSpeech(
        samples=np.concatenate([chunk.speech.samples for chunk in chunks]),
        sample_rate=sample_rate,
    )
