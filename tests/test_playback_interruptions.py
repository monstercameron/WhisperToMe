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
from whispertome.wake.interruptions import PlaybackWakeMonitor


class FakeStt(SpeechToTextModel):
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


class PlaybackWakeMonitorTests(unittest.TestCase):
    def test_monitor_sets_interrupt_event_when_wake_phrase_is_heard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = with_wake_phrases(
                load_config(Path(tmp), require_openai_key=False),
                ["computer"],
            )
            config = replace(
                config,
                audio=replace(
                    config.audio,
                    block_ms=10,
                    speech_start_ms=10,
                    speech_end_ms=20,
                    pre_roll_ms=10,
                    vad_rms_threshold=0.01,
                ),
            )
            chunks = iter(
                [
                    AudioChunk(np.ones(160, dtype=np.float32) * 0.10, 16_000, 0),
                    AudioChunk(np.ones(160, dtype=np.float32) * 0.10, 16_000, 10),
                    AudioChunk(np.ones(160, dtype=np.float32) * 0.10, 16_000, 20),
                    AudioChunk(np.zeros(160, dtype=np.float32), 16_000, 30),
                    AudioChunk(np.zeros(160, dtype=np.float32), 16_000, 40),
                ]
            )
            stt = FakeStt("Computer stop talking")
            detected = []
            speech_detected = []
            monitor = PlaybackWakeMonitor(
                config=config,
                chunks=chunks,
                stt_model=stt,
                min_speech_ms=0,
                vad_threshold=0.01,
                on_speech_detected=speech_detected.append,
                on_wake_detected=detected.append,
            )

            monitor.start()
            for _ in range(100):
                if monitor.interrupt_event.is_set():
                    break
                time.sleep(0.01)
            monitor.stop()

        self.assertTrue(monitor.interrupt_event.is_set())
        self.assertIsNotNone(monitor.result)
        assert monitor.result is not None
        self.assertEqual(monitor.result.transcript, "Computer stop talking")
        self.assertEqual(monitor.result.event.kind, "command_ready")
        self.assertEqual(monitor.result.event.command, "stop talking")
        self.assertEqual(stt.calls, 1)
        self.assertEqual(len(speech_detected), 1)
        self.assertGreaterEqual(len(speech_detected[0]), 1)
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].transcript, "Computer stop talking")


if __name__ == "__main__":
    unittest.main()
