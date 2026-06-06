from __future__ import annotations

import sys
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from whispertome.audio.playback import SpeakerOutput
from whispertome.audio.types import SynthesizedSpeech


class FakeSoundDevice:
    def __init__(self) -> None:
        self.play_calls = []
        self.stop_calls = 0
        self.wait_calls = 0

    def play(self, samples, *, samplerate: int, blocking: bool) -> None:
        self.play_calls.append(
            SimpleNamespace(samples=samples, samplerate=samplerate, blocking=blocking)
        )

    def stop(self) -> None:
        self.stop_calls += 1

    def wait(self) -> None:
        self.wait_calls += 1


class PlaybackTests(unittest.TestCase):
    def test_interruptible_playback_stops_when_event_is_set(self) -> None:
        fake_sd = FakeSoundDevice()
        interrupt = Event()
        interrupt.set()
        speech = SynthesizedSpeech(
            samples=np.ones(24_000, dtype=np.float32),
            sample_rate=24_000,
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = SpeakerOutput().play_interruptible(
                speech,
                interrupt_event=interrupt,
                poll_ms=1,
            )

        self.assertTrue(result.interrupted)
        self.assertEqual(fake_sd.stop_calls, 1)
        self.assertEqual(fake_sd.wait_calls, 0)
        self.assertEqual(fake_sd.play_calls[0].blocking, False)

    def test_interruptible_playback_waits_when_not_interrupted(self) -> None:
        fake_sd = FakeSoundDevice()
        speech = SynthesizedSpeech(samples=np.zeros(0, dtype=np.float32), sample_rate=24_000)

        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = SpeakerOutput().play_interruptible(speech, poll_ms=1)

        self.assertFalse(result.interrupted)
        self.assertEqual(fake_sd.stop_calls, 0)
        self.assertEqual(fake_sd.wait_calls, 1)


if __name__ == "__main__":
    unittest.main()
