from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from whispertome.audio.types import AudioChunk
from whispertome.config import load_config, with_wake_phrases
from whispertome.stt.base import SpeechToTextModel, Transcript
from whispertome.wake.live_preview import InterimWakePreview


class FakePreviewStt(SpeechToTextModel):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def load(self) -> None:
        return

    def transcribe(self, audio) -> Transcript:
        self.calls += 1
        return Transcript(
            text=self.text,
            language="en",
            latency_ms=1.0,
            model="fake",
            provider="fake",
        )


class LiveWakePreviewTests(unittest.TestCase):
    def test_interim_preview_detects_wake_before_utterance_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = with_wake_phrases(
                load_config(Path(tmp), require_openai_key=False),
                ["computer"],
            )
            config = replace(
                config,
                audio=replace(config.audio, block_ms=10),
            )
            statuses: list[tuple[str, str]] = []
            lines: list[str] = []
            detections = []
            preview = InterimWakePreview(
                config=config,
                stt_model=FakePreviewStt("Computer, keep listening"),
                min_audio_ms=20,
                interval_ms=10,
                max_audio_ms=100,
                status_callback=lambda text, detail: statuses.append((text, detail)),
                line_callback=lines.append,
                on_wake_detected=detections.append,
            )
            chunks = tuple(
                AudioChunk(np.ones(160, dtype=np.float32) * 0.1, 16_000, index * 10)
                for index in range(3)
            )

            preview.start(chunks[:1])
            preview.add_chunk(chunks[1])
            preview.add_chunk(chunks[2])
            for _ in range(100):
                if preview.result is not None:
                    break
                time.sleep(0.01)

        self.assertIsNotNone(preview.result)
        assert preview.result is not None
        self.assertEqual(preview.result.event.kind, "command_ready")
        self.assertEqual(preview.result.event.command, "keep listening")
        self.assertIn(("wake detected", "interim: keep listening"), statuses)
        self.assertTrue(any(line.startswith("interim wake:") for line in lines))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].event.command, "keep listening")


if __name__ == "__main__":
    unittest.main()
