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
class WakeToken:
    text: str
    source_id: int
    start_char: int
    end_char: int
    sequence: int


@dataclass(frozen=True)
class WakeMatch:
    phrase: str
    score: float
    transcript_window: str
    transcript_start_char: int | None = None
    transcript_end_char: int | None = None
    matched_text: str = ""


@dataclass(frozen=True)
class _Candidate:
    phrase: str
    score: float
    tokens: tuple[WakeToken, ...]


class SlidingWakeDetector:
    def __init__(self, config: WakeConfig) -> None:
        self._phrases = tuple(
            phrase
            for phrase in (normalize_text(phrase) for phrase in config.phrases)
            if phrase
        )
        self._phrase_tokens = {
            phrase: tuple(phrase.split())
            for phrase in self._phrases
        }
        self._threshold = config.fuzzy_threshold
        self._tokens: deque[WakeToken] = deque(maxlen=config.window_words)
        self._cooldown_s = config.cooldown_ms / 1000.0
        self._last_match = 0.0
        self._source_id = 0
        self._sequence = 0

    def push_transcript(self, transcript: str) -> WakeMatch | None:
        self._source_id += 1
        source_id = self._source_id
        current_tokens: list[WakeToken] = []
        for match in _TOKEN_RE.finditer(transcript.lower()):
            self._sequence += 1
            current_tokens.append(
                WakeToken(
                    text=match.group(0),
                    source_id=source_id,
                    start_char=match.start(),
                    end_char=match.end(),
                    sequence=self._sequence,
                )
            )
        tokens_to_check = tuple(self._tokens) + tuple(current_tokens)
        self._tokens.extend(current_tokens)
        return self._check_tokens(tokens_to_check, current_source_id=source_id)

    def check(self, *, current_source_id: int | None = None) -> WakeMatch | None:
        return self._check_tokens(tuple(self._tokens), current_source_id=current_source_id)

    def _check_tokens(
        self,
        tokens: tuple[WakeToken, ...],
        *,
        current_source_id: int | None = None,
    ) -> WakeMatch | None:
        now = monotonic()
        if now - self._last_match < self._cooldown_s:
            return None

        window = " ".join(token.text for token in tokens)
        if not window:
            return None

        candidate = self._best_candidate(tokens, current_source_id=current_source_id)
        if candidate is None:
            return None

        if candidate.score >= self._threshold:
            self._last_match = now
            return self._to_match(candidate, current_source_id, window)
        return None

    def _best_candidate(
        self,
        tokens: tuple[WakeToken, ...],
        *,
        current_source_id: int | None,
    ) -> _Candidate | None:
        best: _Candidate | None = None
        for phrase, phrase_tokens in self._phrase_tokens.items():
            if not phrase_tokens:
                continue
            phrase_len = len(phrase_tokens)
            for candidate_len in _candidate_lengths(phrase_len):
                if candidate_len > len(tokens):
                    continue
                for start in range(0, len(tokens) - candidate_len + 1):
                    candidate_tokens = tokens[start : start + candidate_len]
                    if (
                        current_source_id is not None
                        and not any(token.source_id == current_source_id for token in candidate_tokens)
                    ):
                        continue
                    candidate_text = " ".join(token.text for token in candidate_tokens)
                    score = (
                        1.0
                        if tuple(token.text for token in candidate_tokens) == phrase_tokens
                        else SequenceMatcher(None, phrase, candidate_text).ratio()
                    )
                    candidate = _Candidate(
                        phrase=phrase,
                        score=score,
                        tokens=candidate_tokens,
                    )
                    if _is_better_candidate(candidate, best):
                        best = candidate
        return best

    @staticmethod
    def _to_match(
        candidate: _Candidate,
        current_source_id: int | None,
        transcript_window: str,
    ) -> WakeMatch:
        start_char = None
        end_char = None
        if current_source_id is not None:
            first = candidate.tokens[0]
            last = candidate.tokens[-1]
            if first.source_id == current_source_id:
                start_char = first.start_char
            if last.source_id == current_source_id:
                end_char = last.end_char
        return WakeMatch(
            phrase=candidate.phrase,
            score=candidate.score,
            transcript_window=transcript_window,
            transcript_start_char=start_char,
            transcript_end_char=end_char,
            matched_text=" ".join(token.text for token in candidate.tokens),
        )


def _candidate_lengths(phrase_len: int) -> tuple[int, ...]:
    lengths = {phrase_len}
    if phrase_len > 1:
        lengths.add(phrase_len - 1)
    lengths.add(phrase_len + 1)
    return tuple(sorted(lengths))


def _is_better_candidate(candidate: _Candidate, best: _Candidate | None) -> bool:
    if best is None:
        return True
    if candidate.score != best.score:
        return candidate.score > best.score
    return candidate.tokens[-1].sequence > best.tokens[-1].sequence
