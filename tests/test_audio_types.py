from __future__ import annotations

import unittest

import numpy as np

from whispertome.audio.types import SynthesizedSpeech


class AudioTypesTests(unittest.TestCase):
    def test_synthesized_speech_reports_duration_ms(self) -> None:
        speech = SynthesizedSpeech(
            samples=np.zeros(12_000, dtype=np.float32),
            sample_rate=24_000,
        )

        self.assertEqual(speech.duration_ms, 500)


if __name__ == "__main__":
    unittest.main()
