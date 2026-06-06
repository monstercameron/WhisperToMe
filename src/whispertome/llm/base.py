from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from whispertome.agent.tools import AgentToolEvent


@dataclass(frozen=True)
class LlmResponse:
    text: str
    latency_ms: float
    model: str
    response_id: str | None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None


class LlmResponder(Protocol):
    def generate(
        self,
        user_text: str,
        *,
        on_tool_event: Callable[[AgentToolEvent], None] | None = None,
    ) -> LlmResponse: ...

    def generate_stream(
        self,
        user_text: str,
        *,
        on_delta: Callable[[str], None],
        on_tool_event: Callable[[AgentToolEvent], None] | None = None,
    ) -> LlmResponse: ...

    def reset_conversation(self) -> None: ...

    def compact_conversation(self, summary: str) -> None: ...


def build_dictation_user_text(transcript: str) -> str:
    local_time = datetime.now().astimezone().replace(microsecond=0).isoformat()
    return (
        "Speech-to-text transcript from the user. "
        "Interpret it as spoken dictation or a spoken command.\n\n"
        "Voice/TUI output contract: keep spoken prose brief. If your "
        "reply includes code, scripts, templates, commands, JSON, YAML, "
        "XML, or exact "
        "copyable text, put that content in a fenced markdown block after a "
        "short spoken lead-in so the app can display it instead of reading it "
        "aloud. Use at most one fenced block unless the transcript explicitly "
        "asks for multiple files, examples, or blocks.\n\n"
        f"Local date/time: {local_time}\n\n"
        "Transcript:\n"
        f"{transcript}"
    )
