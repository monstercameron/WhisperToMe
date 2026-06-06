from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from whispertome.audio.types import SynthesizedSpeech
from whispertome.tts.base import SpeechSynthesisResult, TextToSpeechModel
from whispertome.tts.streaming import StreamingSpeechPlayer, concatenate_speech


class FakeStreamingTts(TextToSpeechModel):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def load(self) -> None:
        return

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        self.texts.append(text)
        return SpeechSynthesisResult(
            speech=SynthesizedSpeech(
                samples=np.ones(len(text), dtype=np.float32),
                sample_rate=24_000,
            ),
            latency_ms=2.0,
            model="fake",
            provider="fake-provider",
        )


class FakeStreamingSpeaker:
    def __init__(self) -> None:
        self.played: list[SynthesizedSpeech] = []

    def play_interruptible(self, speech, *, interrupt_event=None):
        self.played.append(speech)
        return SimpleNamespace(interrupted=False, elapsed_ms=0.0)


class StreamingTtsTests(unittest.TestCase):
    def test_synthesizes_queued_text_chunks(self) -> None:
        tts = FakeStreamingTts()
        player = StreamingSpeechPlayer(tts_model=tts, speaker=None)

        player.enqueue_text("Hello.")
        player.enqueue_text("Next.")
        result = player.finish()

        self.assertEqual(tts.texts, ["Hello.", "Next."])
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.provider, "fake-provider")
        self.assertFalse(result.interrupted)

    def test_plays_audio_chunks_in_order(self) -> None:
        tts = FakeStreamingTts()
        speaker = FakeStreamingSpeaker()
        player = StreamingSpeechPlayer(tts_model=tts, speaker=speaker)  # type: ignore[arg-type]

        player.enqueue_text("One.")
        player.enqueue_text("Two.")
        result = player.finish()

        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(len(speaker.played), 2)
        self.assertEqual([len(item.samples) for item in speaker.played], [4, 4])

    def test_concatenates_compatible_chunks(self) -> None:
        tts = FakeStreamingTts()
        player = StreamingSpeechPlayer(tts_model=tts, speaker=None)

        player.enqueue_text("Hi.")
        player.enqueue_text("There.")
        result = player.finish()
        speech = concatenate_speech(result.chunks)

        assert speech is not None
        self.assertEqual(speech.sample_rate, 24_000)
        self.assertEqual(len(speech.samples), len("Hi.") + len("There."))


if __name__ == "__main__":
    unittest.main()
