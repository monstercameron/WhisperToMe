from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any

from whispertome.agent.tools import (
    AgentToolEvent,
    AgentToolRegistry,
    parse_tool_arguments,
    tool_output_json,
)
from whispertome.config import OpenAIConfig
from whispertome.errors import ConfigError, WhisperToMeError


@dataclass(frozen=True)
class LlmResponse:
    text: str
    latency_ms: float
    model: str
    response_id: str | None


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str
    call_id: str


class OpenAIResponder:
    """Responses API client for dictation-shaped voice turns."""

    def __init__(
        self,
        config: OpenAIConfig,
        *,
        client: Any | None = None,
        tool_registry: AgentToolRegistry | None = None,
        system_context_provider: Callable[[], str | None] | None = None,
        max_tool_iterations: int = 4,
    ) -> None:
        self._config = config
        self._client = client
        self._tool_registry = tool_registry
        self._system_context_provider = system_context_provider
        self._max_tool_iterations = max_tool_iterations
        self._last_response_id: str | None = None

    def generate(
        self,
        user_text: str,
        *,
        on_tool_event: Callable[[AgentToolEvent], None] | None = None,
    ) -> LlmResponse:
        transcript = user_text.strip()
        if not transcript:
            raise ValueError("user_text cannot be empty")
        client = self._ensure_client()

        started = perf_counter()
        if self._tool_registry is not None:
            return self._generate_with_tools(
                client,
                transcript,
                started=started,
                on_tool_event=on_tool_event,
            )

        kwargs = self._build_request_kwargs(transcript)
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

    def generate_stream(
        self,
        user_text: str,
        *,
        on_delta: Callable[[str], None],
        on_tool_event: Callable[[AgentToolEvent], None] | None = None,
    ) -> LlmResponse:
        transcript = user_text.strip()
        if not transcript:
            raise ValueError("user_text cannot be empty")
        client = self._ensure_client()

        started = perf_counter()
        if self._tool_registry is not None:
            return self._generate_stream_with_tools(
                client,
                transcript,
                started=started,
                on_delta=on_delta,
                on_tool_event=on_tool_event,
            )

        kwargs = self._build_request_kwargs(transcript)
        kwargs["stream"] = True
        text_parts: list[str] = []
        completed_response: Any | None = None
        stream: Any | None = None
        try:
            stream = client.responses.create(**kwargs)
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = str(getattr(event, "delta", ""))
                    if delta:
                        text_parts.append(delta)
                        on_delta(delta)
                elif event_type == "response.completed":
                    completed_response = getattr(event, "response", None)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"OpenAI streaming API request failed: {exc}") from exc
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        elapsed_ms = (perf_counter() - started) * 1000.0
        text = "".join(text_parts).strip()
        if not text and completed_response is not None:
            text = _extract_output_text(completed_response)
        if not text:
            raise WhisperToMeError("OpenAI streaming response did not include output text")

        response_id = getattr(completed_response, "id", None)
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

    def _build_request_kwargs(self, transcript: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "instructions": self._instructions(),
            "input": self._build_input(transcript),
            "store": self._config.stateful,
        }
        if self._config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._config.max_output_tokens
        if self._config.stateful and self._last_response_id is not None:
            kwargs["previous_response_id"] = self._last_response_id
        return kwargs

    def _build_tool_followup_kwargs(
        self,
        *,
        input_items: list[dict[str, Any]],
        previous_response_id: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "instructions": self._instructions(),
            "input": input_items,
            "store": self._config.stateful,
        }
        if self._config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._config.max_output_tokens
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id
        return kwargs

    def _add_tool_kwargs(self, kwargs: dict[str, Any]) -> None:
        if self._tool_registry is None:
            return
        kwargs["tools"] = self._tool_registry.tool_specs()
        kwargs["tool_choice"] = "auto"
        kwargs["parallel_tool_calls"] = False
        kwargs["max_tool_calls"] = 4

    def _instructions(self) -> str:
        if self._system_context_provider is None:
            return self._config.system_prompt
        context = self._system_context_provider()
        if not context:
            return self._config.system_prompt
        stripped = context.strip()
        if not stripped:
            return self._config.system_prompt
        return f"{self._config.system_prompt}\n\n{stripped}"

    def _generate_with_tools(
        self,
        client: Any,
        transcript: str,
        *,
        started: float,
        on_tool_event: Callable[[AgentToolEvent], None] | None,
    ) -> LlmResponse:
        kwargs = self._build_request_kwargs(transcript)
        self._add_tool_kwargs(kwargs)
        response: Any | None = None

        try:
            for _ in range(self._max_tool_iterations):
                response = client.responses.create(**kwargs)
                calls = _extract_function_calls(response)
                if not calls:
                    return self._build_llm_response(response, started)
                kwargs = self._build_tool_followup_kwargs(
                    input_items=self._execute_tool_calls(calls, on_tool_event),
                    previous_response_id=_response_id(response),
                )
                self._add_tool_kwargs(kwargs)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"OpenAI API tool loop failed: {exc}") from exc

        raise WhisperToMeError("OpenAI tool loop exceeded max_tool_iterations")

    def _generate_stream_with_tools(
        self,
        client: Any,
        transcript: str,
        *,
        started: float,
        on_delta: Callable[[str], None],
        on_tool_event: Callable[[AgentToolEvent], None] | None,
    ) -> LlmResponse:
        kwargs = self._build_request_kwargs(transcript)
        self._add_tool_kwargs(kwargs)
        text_parts: list[str] = []
        completed_response: Any | None = None

        try:
            for _ in range(self._max_tool_iterations):
                kwargs["stream"] = True
                calls, completed_response = self._consume_agent_stream(
                    client,
                    kwargs,
                    text_parts=text_parts,
                    on_delta=on_delta,
                )
                if not calls:
                    break
                kwargs = self._build_tool_followup_kwargs(
                    input_items=self._execute_tool_calls(calls, on_tool_event),
                    previous_response_id=_response_id(completed_response),
                )
                self._add_tool_kwargs(kwargs)
            else:
                raise WhisperToMeError("OpenAI streaming tool loop exceeded max_tool_iterations")
        except WhisperToMeError:
            raise
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"OpenAI streaming API tool loop failed: {exc}") from exc

        elapsed_ms = (perf_counter() - started) * 1000.0
        text = "".join(text_parts).strip()
        if not text and completed_response is not None:
            text = _extract_output_text(completed_response)
        if not text:
            raise WhisperToMeError("OpenAI streaming response did not include output text")

        response_id = _response_id(completed_response)
        if self._config.stateful and response_id:
            self._last_response_id = response_id

        return LlmResponse(
            text=text,
            latency_ms=elapsed_ms,
            model=self._config.model,
            response_id=response_id,
        )

    def _consume_agent_stream(
        self,
        client: Any,
        kwargs: dict[str, Any],
        *,
        text_parts: list[str],
        on_delta: Callable[[str], None],
    ) -> tuple[list[FunctionCall], Any | None]:
        calls: list[FunctionCall] = []
        completed_response: Any | None = None
        stream: Any | None = None
        try:
            stream = client.responses.create(**kwargs)
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = str(getattr(event, "delta", ""))
                    if delta:
                        text_parts.append(delta)
                        on_delta(delta)
                elif event_type == "response.output_item.done":
                    call = _function_call_from_item(getattr(event, "item", None))
                    if call is not None:
                        calls.append(call)
                elif event_type == "response.completed":
                    completed_response = getattr(event, "response", None)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        return calls, completed_response

    def _execute_tool_calls(
        self,
        calls: list[FunctionCall],
        on_tool_event: Callable[[AgentToolEvent], None] | None,
    ) -> list[dict[str, Any]]:
        if self._tool_registry is None:
            raise WhisperToMeError("No tool registry is configured")

        outputs: list[dict[str, Any]] = []
        for call in calls:
            try:
                arguments = parse_tool_arguments(call.arguments)
            except Exception as exc:
                event = AgentToolEvent(
                    name=call.name,
                    arguments={},
                    output={"ok": False, "error": f"Invalid tool arguments: {exc}"},
                    latency_ms=0.0,
                )
            else:
                event = self._tool_registry.execute(call.name, arguments)
            if on_tool_event is not None:
                on_tool_event(event)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": tool_output_json(event.output),
                }
            )
        return outputs

    def _build_llm_response(self, response: Any, started: float) -> LlmResponse:
        elapsed_ms = (perf_counter() - started) * 1000.0
        text = _extract_output_text(response)
        response_id = _response_id(response)
        if self._config.stateful and response_id:
            self._last_response_id = response_id
        return LlmResponse(
            text=text,
            latency_ms=elapsed_ms,
            model=self._config.model,
            response_id=response_id,
        )

    @staticmethod
    def _build_input(transcript: str) -> list[dict[str, Any]]:
        local_time = datetime.now().astimezone().replace(microsecond=0).isoformat()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
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


def _response_id(response: Any | None) -> str | None:
    if response is None:
        return None
    value = getattr(response, "id", None)
    return str(value) if value else None


def _extract_function_calls(response: Any) -> list[FunctionCall]:
    return [
        call
        for item in getattr(response, "output", []) or []
        if (call := _function_call_from_item(item)) is not None
    ]


def _function_call_from_item(item: Any | None) -> FunctionCall | None:
    if item is None or getattr(item, "type", "") != "function_call":
        return None
    name = getattr(item, "name", "")
    call_id = getattr(item, "call_id", "")
    if not name or not call_id:
        return None
    return FunctionCall(
        name=str(name),
        arguments=str(getattr(item, "arguments", "")),
        call_id=str(call_id),
    )


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
