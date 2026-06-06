from __future__ import annotations

import unittest

from whispertome.config import WakeConfig
from whispertome.wake.sliding_window import SlidingWakeDetector, normalize_text


class WakeWindowTests(unittest.TestCase):
    def test_normalizes_text(self) -> None:
        self.assertEqual(normalize_text("Whisper, to me!"), "whisper to me")

    def test_detects_exact_phrase(self) -> None:
        detector = SlidingWakeDetector(
            WakeConfig(
                phrases=("whisper to me",),
                fuzzy_threshold=0.9,
                window_words=8,
                cooldown_ms=0,
            )
        )

        match = detector.push_transcript("could you whisper to me")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.phrase, "whisper to me")
        self.assertEqual(match.score, 1.0)

    def test_respects_window_size(self) -> None:
        detector = SlidingWakeDetector(
            WakeConfig(
                phrases=("whisper to me",),
                fuzzy_threshold=0.99,
                window_words=3,
                cooldown_ms=0,
            )
        )

        detector.push_transcript("whisper")
        detector.push_transcript("lots of unrelated words")
        match = detector.push_transcript("to me")

        self.assertIsNone(match)

    def test_detects_phrase_split_across_transcripts(self) -> None:
        detector = SlidingWakeDetector(
            WakeConfig(
                phrases=("whisper to me",),
                fuzzy_threshold=0.9,
                window_words=8,
                cooldown_ms=0,
            )
        )

        self.assertIsNone(detector.push_transcript("whisper"))
        match = detector.push_transcript("to me")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.phrase, "whisper to me")
        self.assertEqual(match.transcript_end_char, len("to me"))

    def test_does_not_retrigger_from_stale_window_only(self) -> None:
        detector = SlidingWakeDetector(
            WakeConfig(
                phrases=("whisper to me",),
                fuzzy_threshold=0.9,
                window_words=8,
                cooldown_ms=0,
            )
        )

        self.assertIsNotNone(detector.push_transcript("whisper to me"))
        match = detector.push_transcript("unrelated words")

        self.assertIsNone(match)

    def test_detects_wake_at_start_of_long_current_transcript(self) -> None:
        detector = SlidingWakeDetector(
            WakeConfig(
                phrases=("computer",),
                fuzzy_threshold=0.9,
                window_words=3,
                cooldown_ms=0,
            )
        )

        match = detector.push_transcript(
            "Computer, what should I have for breakfast today?"
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.phrase, "computer")
        self.assertEqual(match.transcript_end_char, len("Computer"))

    def test_fuzzy_match_reports_current_transcript_span(self) -> None:
        detector = SlidingWakeDetector(
            WakeConfig(
                phrases=("whisper to me",),
                fuzzy_threshold=0.86,
                window_words=8,
                cooldown_ms=0,
            )
        )

        match = detector.push_transcript("whispered to me open settings")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.phrase, "whisper to me")
        self.assertEqual(match.matched_text, "whispered to me")
        self.assertEqual(match.transcript_end_char, len("whispered to me"))


if __name__ == "__main__":
    unittest.main()
