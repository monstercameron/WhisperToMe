from __future__ import annotations

import unittest
from types import SimpleNamespace

from whispertome.agent.tools import (
    OPENAI_INPUT_IMAGES_KEY,
    AgentTool,
    AgentToolRegistry,
    object_schema,
    string_schema,
)
from whispertome.config import CerebrasConfig
from whispertome.errors import WhisperToMeError
from whispertome.llm.cerebras_chat import CerebrasResponder


class FakeStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeStream(
                [
                    SimpleNamespace(
                        id="chat_stream",
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))],
                    ),
                    SimpleNamespace(
                        id="chat_stream",
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="lo."))],
                    ),
                ]
            )
        return SimpleNamespace(
            id=f"chat_{len(self.calls)}",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="Done.",
                        tool_calls=None,
                    )
                )
            ],
        )


class ToolLoopChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                id="chat_tool",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="notes_add",
                                        arguments='{"content":"call Sam"}',
                                    ),
                                )
                            ],
                        )
                    )
                ],
            )
        return SimpleNamespace(
            id="chat_final",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="Saved that note.",
                        tool_calls=None,
                    )
                )
            ],
        )


class FailingToolLoopChatCompletions(ToolLoopChatCompletions):
    def create(self, **kwargs):
        if len(self.calls) == 1:
            self.calls.append(kwargs)
            raise RuntimeError("429 Too Many Requests")
        return super().create(**kwargs)


class CaptureToolLoopChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                id="chat_tool",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_capture",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="desktop_capture",
                                        arguments="{}",
                                    ),
                                )
                            ],
                        )
                    )
                ],
            )
        return SimpleNamespace(
            id="chat_final",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="I saved a screenshot, but cannot inspect it here.",
                        tool_calls=None,
                    )
                )
            ],
        )


class FakeClient:
    def __init__(self, completions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def config(*, stateful: bool = True) -> CerebrasConfig:
    return CerebrasConfig(
        api_key="test",
        base_url="https://api.cerebras.ai/v1",
        model="gpt-oss-120b",
        system_prompt="System prompt",
        max_output_tokens=50,
        stateful=stateful,
        reasoning_effort="low",
        timeout_seconds=20.0,
        max_retries=0,
    )


def tool_registry() -> AgentToolRegistry:
    return AgentToolRegistry(
        [
            AgentTool(
                name="notes_add",
                description="Save a note.",
                parameters=object_schema(
                    {"content": string_schema("Note content.")},
                    required=["content"],
                ),
                handler=lambda args: {"ok": True, "saved": args["content"]},
            )
        ]
    )


def capture_registry() -> AgentToolRegistry:
    return AgentToolRegistry(
        [
            AgentTool(
                name="desktop_capture",
                description="Capture the desktop.",
                parameters=object_schema({}),
                handler=lambda _args: {
                    "ok": True,
                    "path": "C:/project/artifacts/captures/desktop.png",
                    OPENAI_INPUT_IMAGES_KEY: [
                        {
                            "image_url": "data:image/png;base64,abc",
                            "detail": "low",
                        }
                    ],
                },
            )
        ]
    )


class CerebrasResponderTests(unittest.TestCase):
    def test_sends_chat_completion_shape(self) -> None:
        completions = FakeChatCompletions()

        response = CerebrasResponder(
            config(),
            client=FakeClient(completions),
        ).generate("rite this down")

        self.assertEqual(response.text, "Done.")
        call = completions.calls[0]
        self.assertEqual(call["model"], "gpt-oss-120b")
        self.assertEqual(call["max_completion_tokens"], 50)
        self.assertEqual(call["reasoning_effort"], "low")
        self.assertEqual(call["messages"][0], {"role": "system", "content": "System prompt"})
        self.assertIn("Speech-to-text transcript", call["messages"][1]["content"])
        self.assertIn("rite this down", call["messages"][1]["content"])

    def test_stateful_responder_keeps_local_chat_history(self) -> None:
        completions = FakeChatCompletions()
        responder = CerebrasResponder(config(), client=FakeClient(completions))

        responder.generate("first")
        responder.generate("second")

        second_messages = completions.calls[1]["messages"]
        self.assertEqual(second_messages[1]["role"], "user")
        self.assertIn("first", second_messages[1]["content"])
        self.assertEqual(second_messages[2], {"role": "assistant", "content": "Done."})
        self.assertIn("second", second_messages[3]["content"])

    def test_streaming_responder_emits_deltas(self) -> None:
        completions = FakeChatCompletions()
        deltas: list[str] = []

        response = CerebrasResponder(
            config(),
            client=FakeClient(completions),
        ).generate_stream("hello", on_delta=deltas.append)

        self.assertEqual(deltas, ["Hel", "lo."])
        self.assertEqual(response.text, "Hello.")
        self.assertEqual(response.response_id, "chat_stream")
        self.assertTrue(completions.calls[0]["stream"])

    def test_agent_tool_loop_sends_chat_function_outputs(self) -> None:
        completions = ToolLoopChatCompletions()
        events = []

        response = CerebrasResponder(
            config(),
            client=FakeClient(completions),
            tool_registry=tool_registry(),
        ).generate("take a note", on_tool_event=events.append)

        self.assertEqual(response.text, "Saved that note.")
        self.assertEqual(response.response_id, "chat_final")
        first_call = completions.calls[0]
        self.assertEqual(first_call["tool_choice"], "auto")
        self.assertEqual(first_call["tools"][0]["function"]["name"], "notes_add")
        second_messages = completions.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_1")
        self.assertEqual(events[0].output["saved"], "call Sam")

    def test_agent_tool_loop_wraps_followup_provider_failure(self) -> None:
        completions = FailingToolLoopChatCompletions()
        events = []

        with self.assertRaisesRegex(WhisperToMeError, "Cerebras API tool loop failed"):
            CerebrasResponder(
                config(),
                client=FakeClient(completions),
                tool_registry=tool_registry(),
            ).generate("take a note", on_tool_event=events.append)

        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(events[0].output["saved"], "call Sam")

    def test_desktop_capture_image_is_not_claimed_attached_for_cerebras(self) -> None:
        completions = CaptureToolLoopChatCompletions()
        events = []

        CerebrasResponder(
            config(),
            client=FakeClient(completions),
            tool_registry=capture_registry(),
        ).generate("what am I looking at", on_tool_event=events.append)

        tool_output = events[0].output
        self.assertNotIn(OPENAI_INPUT_IMAGES_KEY, tool_output)
        self.assertFalse(tool_output["image_attached_to_model"])
        self.assertEqual(tool_output["image_omitted_reason"], "cerebras_chat_text_only")


if __name__ == "__main__":
    unittest.main()
