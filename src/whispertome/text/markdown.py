from __future__ import annotations

import re
from dataclasses import dataclass


_FENCE_RE = re.compile(
    r"(?P<fence>`{3,})(?P<info>[^\n`]*)\n(?P<body>.*?)(?P=fence)",
    re.DOTALL,
)
_UNTERMINATED_FENCE_RE = re.compile(
    r"(?P<fence>`{3,})(?P<info>[^\n`]*)\n(?P<body>.*)\Z",
    re.DOTALL,
)
_CODE_START_RE = re.compile(
    r"^(?:"
    r"package\s+\w+|"
    r"import\s+[\"(]|"
    r"func\s+\w+\s*\(|"
    r"def\s+\w+\s*\(|"
    r"class\s+\w+|"
    r"from\s+[\w.]+\s+import\s+|"
    r"#include\b|"
    r"using\s+namespace\b|"
    r"public\s+(?:class|static)\b|"
    r"function\s+\w+\s*\(|"
    r"(?:const|let|var)\s+[\w$]+\s*=|"
    r"select\s+.+\s+from\s+|"
    r"create\s+table\b|"
    r"param\s*\(|"
    r"write-host\b|"
    r"<!doctype\s+html\b|"
    r"<html\b|"
    r"<\?php|"
    r"#!/"
    r")",
    re.IGNORECASE,
)
_INLINE_CODE_RE = re.compile(r"\bpackage\s+main\b", re.IGNORECASE)


@dataclass(frozen=True)
class FencedCodeBlock:
    language: str
    code: str

    @property
    def title(self) -> str:
        return self.language or "code"


@dataclass(frozen=True)
class SpokenMarkdown:
    raw_text: str
    speech_text: str
    display_text: str
    code_blocks: tuple[FencedCodeBlock, ...]

    @property
    def primary_code_block(self) -> FencedCodeBlock | None:
        for block in self.code_blocks:
            if block.language == "script":
                return block
        return self.code_blocks[0] if self.code_blocks else None


@dataclass(frozen=True)
class _UnfencedCodeBlock:
    start: int
    language: str
    code: str


def parse_spoken_markdown(text: str) -> SpokenMarkdown:
    unfenced_block = _extract_unfenced_code(text) if "```" not in text else None
    if unfenced_block is not None:
        before = text[: unfenced_block.start]
        block = FencedCodeBlock(
            language=unfenced_block.language,
            code=unfenced_block.code,
        )
        return _build_response_with_blocks(
            raw_text=text,
            speech_parts=[before, _spoken_placeholder(block.language)],
            display_parts=[
                before,
                f"[{_display_label(block.language)} shown in the TUI]",
            ],
            blocks=[block],
        )

    blocks: list[FencedCodeBlock] = []
    speech_parts: list[str] = []
    display_parts: list[str] = []
    cursor = 0

    for match in _FENCE_RE.finditer(text):
        before = text[cursor : match.start()]
        speech_parts.append(before)
        display_parts.append(before)

        language = _language_from_info(match.group("info"))
        code = match.group("body").strip("\r\n")
        blocks.append(FencedCodeBlock(language=language, code=code))
        speech_parts.append(_spoken_placeholder(language))
        display_parts.append(f"[{_display_label(language)} shown in the TUI]")
        cursor = match.end()

    tail = text[cursor:]
    unterminated = _UNTERMINATED_FENCE_RE.search(tail)
    if unterminated is not None:
        before = tail[: unterminated.start()]
        speech_parts.append(before)
        display_parts.append(before)

        language = _language_from_info(unterminated.group("info"))
        code = unterminated.group("body").strip("\r\n")
        blocks.append(FencedCodeBlock(language=language, code=code))
        speech_parts.append(_spoken_placeholder(language))
        display_parts.append(f"[{_display_label(language)} shown in the TUI]")
    else:
        speech_parts.append(tail)
        display_parts.append(tail)

    return _build_response_with_blocks(
        raw_text=text,
        speech_parts=speech_parts,
        display_parts=display_parts,
        blocks=blocks,
    )


def _build_response_with_blocks(
    *,
    raw_text: str,
    speech_parts: list[str],
    display_parts: list[str],
    blocks: list[FencedCodeBlock],
) -> SpokenMarkdown:
    speech_text = _normalize_spoken_text(" ".join(speech_parts))
    if blocks and not speech_text:
        speech_text = _spoken_placeholder(blocks[0].language)

    return SpokenMarkdown(
        raw_text=raw_text,
        speech_text=speech_text,
        display_text=_normalize_display_text(" ".join(display_parts)),
        code_blocks=tuple(blocks),
    )


def _extract_unfenced_code(text: str) -> _UnfencedCodeBlock | None:
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and _CODE_START_RE.search(stripped):
            code = text[offset:].strip()
            if _is_plausible_code_body(code):
                return _UnfencedCodeBlock(
                    start=offset,
                    language=_detect_language(text[:offset], code),
                    code=code,
                )
        offset += len(line)

    inline = _INLINE_CODE_RE.search(text)
    if inline is None:
        return None
    code = text[inline.start() :].strip()
    if not _is_plausible_code_body(code):
        return None
    return _UnfencedCodeBlock(
        start=inline.start(),
        language=_detect_language(text[: inline.start()], code),
        code=code,
    )


def _is_plausible_code_body(code: str) -> bool:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    if not lines:
        return False
    if lines[0].lower().startswith("package main"):
        return True

    signal_count = 0
    for line in lines:
        if _CODE_START_RE.search(line):
            signal_count += 2
        if re.search(r"(:=|==|<=|>=|&&|\|\||[{}();])", line):
            signal_count += 1
        if re.search(
            r"\b(?:return|switch|case|for|if|else|println|console\.log)\b",
            line,
            re.IGNORECASE,
        ):
            signal_count += 1
    return signal_count >= 3 or (len(lines) >= 3 and signal_count >= 2)


def _detect_language(before: str, code: str) -> str:
    code_lower = code.lower()
    context_lower = before.lower()
    if (
        "package main" in code_lower
        or "fmt.println" in code_lower
        or re.search(r"\bfunc\s+\w+\s*\(", code)
    ):
        return "go"
    if re.search(r"\b(def|class)\s+\w+", code) or "print(" in code_lower:
        return "python"
    if "write-host" in code_lower or code_lower.startswith("param("):
        return "powershell"
    if "<html" in code_lower or "<!doctype html" in code_lower:
        return "html"
    if re.search(r"\b(select|create table)\b", code_lower):
        return "sql"
    if re.search(r"\b(function|const|let|var)\b", code):
        return "javascript"
    if re.search(r"\b(go|golang)\b", context_lower):
        return "go"
    return "code"


def _language_from_info(info: str) -> str:
    first = info.strip().split(maxsplit=1)[0] if info.strip() else ""
    return first.lower()


def _spoken_placeholder(language: str) -> str:
    label = _spoken_label(language)
    return f"I put the {label} on screen."


def _spoken_label(language: str) -> str:
    if language == "script":
        return "script"
    if language:
        return f"{language} code"
    return "code"


def _display_label(language: str) -> str:
    return language.upper() if language else "CODE"


def _normalize_spoken_text(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _normalize_display_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()
