from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import perf_counter, sleep

import numpy as np

from whispertome.audio.types import SynthesizedSpeech
from whispertome.errors import AudioError


@dataclass(frozen=True)
class PlaybackResult:
    elapsed_ms: float
    interrupted: bool


class SpeakerOutput:
    def play(self, speech: SynthesizedSpeech) -> None:
        self.play_interruptible(speech)

    def play_interruptible(
        self,
        speech: SynthesizedSpeech,
        *,
        interrupt_event: Event | None = None,
        poll_ms: int = 25,
    ) -> PlaybackResult:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("sounddevice is required for speaker playback") from exc

        try:
            samples = np.asarray(speech.samples, dtype=np.float32)
            started = perf_counter()
            sd.play(samples, samplerate=speech.sample_rate, blocking=False)
            while (perf_counter() - started) * 1000.0 < speech.duration_ms:
                if interrupt_event is not None and interrupt_event.is_set():
                    sd.stop()
                    return PlaybackResult(
                        elapsed_ms=(perf_counter() - started) * 1000.0,
                        interrupted=True,
                    )
                sleep(max(1, poll_ms) / 1000.0)
            sd.wait()
            return PlaybackResult(
                elapsed_ms=(perf_counter() - started) * 1000.0,
                interrupted=False,
            )
        except Exception as exc:  # pragma: no cover - device dependent
            raise AudioError(f"Speaker playback failed: {exc}") from exc
