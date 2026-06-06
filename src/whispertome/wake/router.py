from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from whispertome.wake.sliding_window import SlidingWakeDetector, WakeMatch, normalize_text


WakeEventKind = Literal["idle", "wake_detected", "command_ready"]


@dataclass(frozen=True)
class WakeRouterEvent:
    kind: WakeEventKind
    transcript: str
    command: str | None = None
    match: WakeMatch | None = None


class WakeCommandRouter:
    """Turns speech-pause-delimited transcripts into assistant commands."""

    def __init__(self, detector: SlidingWakeDetector) -> None:
        self._detector = detector
        self._awaiting_command = False

    def process_utterance(self, transcript: str) -> WakeRouterEvent:
        transcript = transcript.strip()
        if not transcript:
            return WakeRouterEvent(kind="idle", transcript=transcript)

        if self._awaiting_command:
            self._awaiting_command = False
            return WakeRouterEvent(
                kind="command_ready",
                transcript=transcript,
                command=transcript,
            )

        match = self._detector.push_transcript(transcript)
        if match is None:
            return WakeRouterEvent(kind="idle", transcript=transcript)

        command = self._extract_after_wake(transcript, match)
        if command:
            return WakeRouterEvent(
                kind="command_ready",
                transcript=transcript,
                command=command,
                match=match,
            )

        self._awaiting_command = True
        return WakeRouterEvent(kind="wake_detected", transcript=transcript, match=match)

    @staticmethod
    def _extract_after_wake(transcript: str, match: WakeMatch) -> str:
        if match.transcript_end_char is not None:
            return transcript[match.transcript_end_char :].lstrip(" \t\r\n,.;:!?-")

        normalized = normalize_text(transcript)
        idx = normalized.find(match.phrase)
        if idx < 0:
            return ""
        return normalized[idx + len(match.phrase) :].strip()
