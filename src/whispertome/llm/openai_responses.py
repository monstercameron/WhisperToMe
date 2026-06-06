from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from whispertome.config import OpenAIConfig
from whispertome.errors import ConfigError, WhisperToMeError


@dataclass(frozen=True)
class LlmResponse:
    text: str
    latency_ms: float
    model: str


class OpenAIResponder:
    def __init__(self, config: OpenAIConfig) -> None:
        self._config = config
        self._client = None

    def generate(self, user_text: str) -> LlmResponse:
        if not user_text.strip():
            raise ValueError("user_text cannot be empty")
        client = self._ensure_client()

        started = perf_counter()
        kwargs = {
            "model": self._config.model,
            "instructions": self._config.system_prompt,
            "input": user_text,
        }
        if self._config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._config.max_output_tokens

        try:
            response = client.responses.create(**kwargs)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"OpenAI API request failed: {exc}") from exc

        elapsed_ms = (perf_counter() - started) * 1000.0
        text = getattr(response, "output_text", None)
        if not text:
            raise WhisperToMeError("OpenAI response did not include output_text")
        return LlmResponse(text=str(text), latency_ms=elapsed_ms, model=self._config.model)

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

