from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from whispertome.agent.tools import (
    OPENAI_INPUT_IMAGES_KEY,
    AgentToolEvent,
    AgentToolRegistry,
    parse_tool_arguments,
    tool_output_json,
)
from whispertome.config import CerebrasConfig
from whispertome.errors import ConfigError, WhisperToMeError
from whispertome.llm.base import LlmResponse, build_dictation_user_text

ChatMessage = dict[str, Any]


@dataclass(frozen=True)
class ChatToolCall:
    name: str
    arguments: str
    call_id: str


class CerebrasResponder:
    """OpenAI-compatible Chat Completions client for Cerebras inference."""

    def __init__(
        self,
        config: CerebrasConfig,
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
        self._history: list[ChatMessage] = []

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

        user_message = self._user_message(transcript)
        messages = self._build_messages(user_message)
        kwargs = self._build_request_kwargs(messages)
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"Cerebras API request failed: {exc}") from exc

        text = _extract_chat_text(response)
        self._remember_turn(user_message, text)
        return LlmResponse(
            text=text,
            latency_ms=(perf_counter() - started) * 1000.0,
            model=self._config.model,
            response_id=_response_id(response),
        )

    def generate_stream(
        self,
        user_text: str,
        *,
        on_delta: Callable[[str], None],
        on_tool_event: Callable[[AgentToolEvent], None] | None = None,
    ) -> LlmResponse:
        if self._tool_registry is not None:
            response = self.generate(user_text, on_tool_event=on_tool_event)
            if response.text:
                on_delta(response.text)
            return response

        transcript = user_text.strip()
        if not transcript:
            raise ValueError("user_text cannot be empty")
        client = self._ensure_client()
        started = perf_counter()
        user_message = self._user_message(transcript)
        messages = self._build_messages(user_message)
        kwargs = self._build_request_kwargs(messages)
        kwargs["stream"] = True

        text_parts: list[str] = []
        response_id: str | None = None
        stream: Any | None = None
        try:
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                if response_id is None:
                    response_id = _response_id(chunk)
                delta = _extract_stream_delta(chunk)
                if delta:
                    text_parts.append(delta)
                    on_delta(delta)
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"Cerebras streaming API request failed: {exc}") from exc
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        text = "".join(text_parts).strip()
        if not text:
            raise WhisperToMeError("Cerebras streaming response did not include output text")
        self._remember_turn(user_message, text)
        return LlmResponse(
            text=text,
            latency_ms=(perf_counter() - started) * 1000.0,
            model=self._config.model,
            response_id=response_id,
        )

    def reset_conversation(self) -> None:
        self._history.clear()

    def _generate_with_tools(
        self,
        client: Any,
        transcript: str,
        *,
        started: float,
        on_tool_event: Callable[[AgentToolEvent], None] | None,
    ) -> LlmResponse:
        if self._tool_registry is None:
            raise WhisperToMeError("No tool registry is configured")

        user_message = self._user_message(transcript)
        messages = self._build_messages(user_message)
        response: Any | None = None

        try:
            for _ in range(self._max_tool_iterations):
                kwargs = self._build_request_kwargs(messages)
                kwargs["tools"] = _chat_tool_specs(self._tool_registry)
                kwargs["tool_choice"] = "auto"
                kwargs["parallel_tool_calls"] = False
                response = client.chat.completions.create(**kwargs)
                assistant_message = _extract_message(response)
                calls = _extract_tool_calls(assistant_message)
                if not calls:
                    text = _extract_chat_text(response)
                    self._remember_turn(user_message, text)
                    return LlmResponse(
                        text=text,
                        latency_ms=(perf_counter() - started) * 1000.0,
                        model=self._config.model,
                        response_id=_response_id(response),
                    )
                messages.append(_assistant_message_dict(assistant_message))
                messages.extend(self._execute_tool_calls(calls, on_tool_event))
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise WhisperToMeError(f"Cerebras API tool loop failed: {exc}") from exc

        raise WhisperToMeError("Cerebras tool loop exceeded max_tool_iterations")

    def _execute_tool_calls(
        self,
        calls: list[ChatToolCall],
        on_tool_event: Callable[[AgentToolEvent], None] | None,
    ) -> list[ChatMessage]:
        if self._tool_registry is None:
            raise WhisperToMeError("No tool registry is configured")

        messages: list[ChatMessage] = []
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

            public_output = _public_tool_output(event.output)
            public_event = (
                event
                if public_output is event.output
                else AgentToolEvent(
                    name=event.name,
                    arguments=event.arguments,
                    output=public_output,
                    latency_ms=event.latency_ms,
                )
            )
            if on_tool_event is not None:
                on_tool_event(public_event)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": tool_output_json(public_output),
                }
            )
        return messages

    def _build_request_kwargs(self, messages: list[ChatMessage]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }
        if self._config.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = self._config.max_output_tokens
        if self._config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._config.reasoning_effort
        return kwargs

    def _build_messages(self, user_message: ChatMessage) -> list[ChatMessage]:
        messages: list[ChatMessage] = [{"role": "system", "content": self._instructions()}]
        if self._config.stateful:
            messages.extend(dict(message) for message in self._history)
        messages.append(user_message)
        return messages

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

    @staticmethod
    def _user_message(transcript: str) -> ChatMessage:
        return {"role": "user", "content": build_dictation_user_text(transcript)}

    def _remember_turn(self, user_message: ChatMessage, assistant_text: str) -> None:
        if not self._config.stateful:
            return
        self._history.append(dict(user_message))
        self._history.append({"role": "assistant", "content": assistant_text})

    def _ensure_client(self):  # type: ignore[no-untyped-def]
        if self._config.api_key is None:
            raise ConfigError("CEREBRAS_API_KEY is required before making API calls")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigError("The openai package is required for Cerebras API calls") from exc
            self._client = OpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
                max_retries=self._config.max_retries,
            )
        return self._client


def _chat_tool_specs(tool_registry: AgentToolRegistry) -> list[dict[str, Any]]:
    chat_specs: list[dict[str, Any]] = []
    for spec in tool_registry.tool_specs():
        function = {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
        }
        if "strict" in spec:
            function["strict"] = spec["strict"]
        chat_specs.append({"type": "function", "function": function})
    return chat_specs


def _extract_message(response: Any) -> Any:
    choices = getattr(response, "choices", []) or []
    if not choices:
        raise WhisperToMeError("Cerebras response did not include choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise WhisperToMeError("Cerebras response did not include a message")
    return message


def _extract_chat_text(response: Any) -> str:
    text = getattr(_extract_message(response), "content", None)
    if text:
        return str(text).strip()
    raise WhisperToMeError("Cerebras response did not include output text")


def _extract_stream_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", []) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    return str(content) if content else ""


def _extract_tool_calls(message: Any) -> list[ChatToolCall]:
    calls: list[ChatToolCall] = []
    for call in getattr(message, "tool_calls", []) or []:
        function = getattr(call, "function", None)
        name = getattr(function, "name", "") if function is not None else ""
        call_id = getattr(call, "id", "")
        if not name or not call_id:
            continue
        calls.append(
            ChatToolCall(
                name=str(name),
                arguments=str(getattr(function, "arguments", "") or ""),
                call_id=str(call_id),
            )
        )
    return calls


def _assistant_message_dict(message: Any) -> ChatMessage:
    model_dump = getattr(message, "model_dump", None)
    if model_dump is not None:
        return model_dump(exclude_none=True)

    data: ChatMessage = {
        "role": getattr(message, "role", "assistant") or "assistant",
        "content": getattr(message, "content", None),
    }
    tool_calls = []
    for call in getattr(message, "tool_calls", []) or []:
        function = getattr(call, "function", None)
        tool_calls.append(
            {
                "id": getattr(call, "id", ""),
                "type": getattr(call, "type", "function") or "function",
                "function": {
                    "name": getattr(function, "name", "") if function is not None else "",
                    "arguments": (
                        getattr(function, "arguments", "") if function is not None else ""
                    ),
                },
            }
        )
    if tool_calls:
        data["tool_calls"] = tool_calls
    return data


def _public_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    if OPENAI_INPUT_IMAGES_KEY not in output:
        return output
    public = dict(output)
    public.pop(OPENAI_INPUT_IMAGES_KEY, None)
    public["image_attached_to_model"] = False
    public["image_omitted_reason"] = "cerebras_chat_text_only"
    return public


def _response_id(response: Any | None) -> str | None:
    if response is None:
        return None
    value = getattr(response, "id", None)
    return str(value) if value else None
