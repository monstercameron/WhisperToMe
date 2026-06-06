from __future__ import annotations

import unittest

from whispertome.config import WakeConfig
from whispertome.wake.router import WakeCommandRouter
from whispertome.wake.sliding_window import SlidingWakeDetector


def make_router(*phrases: str) -> WakeCommandRouter:
    return WakeCommandRouter(
        SlidingWakeDetector(
            WakeConfig(
                phrases=phrases,
                fuzzy_threshold=0.9,
                window_words=12,
                cooldown_ms=0,
            )
        )
    )


class WakeRouterTests(unittest.TestCase):
    def test_wake_and_command_in_same_pause_delimited_utterance(self) -> None:
        router = make_router("computer")

        event = router.process_utterance("computer tell me the time")

        self.assertEqual(event.kind, "command_ready")
        self.assertEqual(event.command, "tell me the time")

    def test_wake_only_arms_next_pause_delimited_utterance(self) -> None:
        router = make_router("whisper to me")

        wake_event = router.process_utterance("whisper to me")
        command_event = router.process_utterance("summarize my calendar")

        self.assertEqual(wake_event.kind, "wake_detected")
        self.assertEqual(command_event.kind, "command_ready")
        self.assertEqual(command_event.command, "summarize my calendar")

    def test_arbitrary_multi_word_wake_phrase(self) -> None:
        router = make_router("hello dashboard")

        event = router.process_utterance("hello dashboard open diagnostics")

        self.assertEqual(event.kind, "command_ready")
        self.assertEqual(event.command, "open diagnostics")


if __name__ == "__main__":
    unittest.main()

