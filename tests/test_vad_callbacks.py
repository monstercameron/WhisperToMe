from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from whispertome.audio.types import AudioChunk
from whispertome.audio.vad import EnergyVad, UtteranceSegmenter
from whispertome.config import load_config


class VadCallbackTests(unittest.TestCase):
    def test_segmenter_calls_speech_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp), require_openai_key=False)
            config = replace(
                config,
                audio=replace(
                    config.audio,
                    block_ms=10,
                    speech_start_ms=10,
                    speech_end_ms=20,
                    pre_roll_ms=10,
                ),
            )
            chunks = [
                AudioChunk(np.ones(160, dtype=np.float32) * 0.1, 16_000, 0),
                AudioChunk(np.ones(160, dtype=np.float32) * 0.1, 16_000, 10),
                AudioChunk(np.zeros(160, dtype=np.float32), 16_000, 20),
                AudioChunk(np.zeros(160, dtype=np.float32), 16_000, 30),
            ]
            starts: list[int] = []
            active_chunks: list[int] = []
            ends: list[bool] = []

            utterances = list(
                UtteranceSegmenter(config.audio, EnergyVad(0.01)).utterances(
                    chunks,
                    on_speech_start=lambda started: starts.append(len(started)),
                    on_speech_chunk=lambda chunk: active_chunks.append(chunk.timestamp_ms),
                    on_speech_end=lambda: ends.append(True),
                )
            )

        self.assertEqual(len(utterances), 1)
        self.assertEqual(starts, [1])
        self.assertEqual(active_chunks, [10, 20, 30])
        self.assertEqual(ends, [True])


if __name__ == "__main__":
    unittest.main()
