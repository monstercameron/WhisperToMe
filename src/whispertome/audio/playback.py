from __future__ import annotations

import numpy as np

from whispertome.audio.types import SynthesizedSpeech
from whispertome.errors import AudioError


class SpeakerOutput:
    def play(self, speech: SynthesizedSpeech) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("sounddevice is required for speaker playback") from exc

        try:
            samples = np.asarray(speech.samples, dtype=np.float32)
            sd.play(samples, samplerate=speech.sample_rate, blocking=True)
        except Exception as exc:  # pragma: no cover - device dependent
            raise AudioError(f"Speaker playback failed: {exc}") from exc

