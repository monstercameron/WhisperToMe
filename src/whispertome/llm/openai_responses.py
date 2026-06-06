from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from whispertome.config import OpenAIConfig
from whispertome.errors import ConfigError, WhisperToMeError


@dataclass(frozen=True)
class LlmResponse:
    text: str
    latency_ms: float
    model: str
    response_id: str | None


class OpenAIResponder:
    """Responses API client for dictation-shaped voice turns."""

    def __init__(self, config: OpenAIConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client
        self._last_response_id: str | None = None

    def generate(self, user_text: str) -> LlmResponse:
        transcript = user_text.strip()
        if not transcript:
            raise ValueError("user_text cannot be empty")
        client = self._ensure_client()

        started = perf_counter()
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "instructions": self._config.system_prompt,
            "input": self._build_input(transcript),
            "store": self._config.stateful,
        }
        if self._config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._config.max_output_tokens
        if self._config.stateful and self._last_response_id is not None:
            kwargs["previous_response_id"] = self._last_response_id

        try:
            response = client.responses.create(**kwargs)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"OpenAI API request failed: {exc}") from exc

        elapsed_ms = (perf_counter() - started) * 1000.0
        text = _extract_output_text(response)
        response_id = getattr(response, "id", None)
        if self._config.stateful and response_id:
            self._last_response_id = str(response_id)

        return LlmResponse(
            text=text,
            latency_ms=elapsed_ms,
            model=self._config.model,
            response_id=str(response_id) if response_id else None,
        )

    def reset_conversation(self) -> None:
        self._last_response_id = None

    @staticmethod
    def _build_input(transcript: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Speech-to-text transcript from the user. "
                            "Interpret it as spoken dictation or a spoken command:\n"
                            f"{transcript}"
                        ),
                    }
                ],
            }
        ]

    def _ensure_client(self):  # type: ignore[no-untyped-def]
        if self._config.api_key is None:
            raise ConfigError("OPENAI_API_KEY is required before making API calls")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigError("The openai package is required for API calls") from exc
            self._client = OpenAI(api_key=self._config.api_key)
        return self._client


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    text_parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                text_parts.append(str(text))

    text = "".join(text_parts).strip()
    if not text:
        raise WhisperToMeError("OpenAI response did not include output text")
    return text
