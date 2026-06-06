from __future__ import annotations

import re

from whispertome.text.markdown import parse_spoken_markdown


_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")
_WHITESPACE_RE = re.compile(r"\s+")


class SpokenTextChunker:
    """Build TTS-safe sentence chunks from a streaming markdown response."""

    def __init__(self, *, max_chunk_chars: int = 220) -> None:
        self._max_chunk_chars = max(40, max_chunk_chars)
        self._raw_text = ""
        self._safe_prefix_len = 0
        self._pending_spoken = ""
        self._emitted_spoken = ""

    @property
    def raw_text(self) -> str:
        return self._raw_text

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._raw_text += delta
        safe_spoken = _spoken_prefix_outside_fences(self._raw_text)
        if len(safe_spoken) > self._safe_prefix_len:
            self._pending_spoken += safe_spoken[self._safe_prefix_len :]
            self._safe_prefix_len = len(safe_spoken)
        return self._drain_ready_chunks(final=False)

    def finish(self) -> list[str]:
        final_spoken = parse_spoken_markdown(self._raw_text).speech_text
        emitted = _normalize(self._emitted_spoken)
        final = _normalize(final_spoken)
        if emitted and final.startswith(emitted):
            self._pending_spoken = final[len(emitted) :].strip()
        elif not emitted:
            self._pending_spoken = final
        else:
            self._pending_spoken = ""
        return self._drain_ready_chunks(final=True)

    def _drain_ready_chunks(self, *, final: bool) -> list[str]:
        chunks: list[str] = []
        while True:
            boundary = _ready_boundary(
                self._pending_spoken,
                final=final,
                max_chunk_chars=self._max_chunk_chars,
            )
            if boundary is None:
                break
            chunk = _normalize(self._pending_spoken[:boundary])
            self._pending_spoken = self._pending_spoken[boundary:].lstrip()
            if chunk:
                chunks.append(chunk)
                self._emitted_spoken = _normalize(
                    f"{self._emitted_spoken} {chunk}"
                )
        return chunks


def _ready_boundary(text: str, *, final: bool, max_chunk_chars: int) -> int | None:
    if not text.strip():
        return None

    sentence_end = 0
    for match in _SENTENCE_END_RE.finditer(text):
        sentence_end = match.end()
    if sentence_end:
        return sentence_end

    if len(text) >= max_chunk_chars:
        split_at = text.rfind(" ", 0, max_chunk_chars)
        if split_at >= 40:
            return split_at

    if final:
        return len(text)
    return None


def _spoken_prefix_outside_fences(text: str) -> str:
    output: list[str] = []
    cursor = 0
    inside_fence = False

    while cursor < len(text):
        fence_at = text.find("```", cursor)
        if fence_at == -1:
            if not inside_fence:
                output.append(_without_partial_fence(text[cursor:]))
            break
        if not inside_fence:
            output.append(text[cursor:fence_at])
            inside_fence = True
        else:
            inside_fence = False
        cursor = fence_at + 3

    return "".join(output)


def _without_partial_fence(text: str) -> str:
    trailing = len(text) - len(text.rstrip("`"))
    if trailing and trailing < 3:
        return text[:-trailing]
    return text


def _normalize(text: str) -> str:
    text = text.replace("`", "")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()
