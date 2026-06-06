from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from time import monotonic

from whispertome.config import WakeConfig


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class WakeMatch:
    phrase: str
    score: float
    transcript_window: str


class SlidingWakeDetector:
    def __init__(self, config: WakeConfig) -> None:
        self._phrases = tuple(normalize_text(phrase) for phrase in config.phrases)
        self._threshold = config.fuzzy_threshold
        self._tokens: deque[str] = deque(maxlen=config.window_words)
        self._cooldown_s = config.cooldown_ms / 1000.0
        self._last_match = 0.0

    def push_transcript(self, transcript: str) -> WakeMatch | None:
        for token in normalize_text(transcript).split():
            self._tokens.append(token)
        return self.check()

    def check(self) -> WakeMatch | None:
        now = monotonic()
        if now - self._last_match < self._cooldown_s:
            return None

        window = " ".join(self._tokens)
        if not window:
            return None

        best_phrase = ""
        best_score = 0.0
        for phrase in self._phrases:
            if phrase in window:
                self._last_match = now
                return WakeMatch(phrase=phrase, score=1.0, transcript_window=window)
            score = SequenceMatcher(None, phrase, window[-max(len(phrase) * 2, 1) :]).ratio()
            if score > best_score:
                best_phrase = phrase
                best_score = score

        if best_score >= self._threshold:
            self._last_match = now
            return WakeMatch(phrase=best_phrase, score=best_score, transcript_window=window)
        return None

