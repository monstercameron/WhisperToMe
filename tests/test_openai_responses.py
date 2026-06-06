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
from whispertome.config import OpenAIConfig
from whispertome.llm.openai_responses import OpenAIResponder


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeStream(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="Hel"),
                    SimpleNamespace(type="response.output_text.delta", delta="lo."),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(id=f"resp_{len(self.calls)}"),
                    ),
                ]
            )
        return SimpleNamespace(id=f"resp_{len(self.calls)}", output_text="Done.")


class FakeStream:
    def __init__(self, events) -> None:
        self._events = events
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class ToolLoopResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                id="resp_tool",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="notes_add",
                        arguments='{"content":"call Sam"}',
                        call_id="call_1",
                    )
                ],
            )
        return SimpleNamespace(id="resp_final", output_text="Saved that note.")


class CaptureToolLoopResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                id="resp_tool",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="desktop_capture",
                        arguments="{}",
                        call_id="call_capture",
                    )
                ],
            )
        return SimpleNamespace(id="resp_final", output_text="I can see your desktop.")


class StreamingToolLoopResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return FakeStream(
                [
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(
                            type="function_call",
                            name="notes_add",
                            arguments='{"content":"call Sam"}',
                            call_id="call_1",
                        ),
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(id="resp_tool"),
                    ),
                ]
            )
        return FakeStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="Saved"),
                SimpleNamespace(type="response.output_text.delta", delta="."),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(id="resp_final"),
                ),
            ]
        )


class PreferenceToolLoopResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                id="resp_tool",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="preferences_save",
                        arguments=(
                            '{"key":"preferred_name",'
                            '"preference":"Call the user Marcus."}'
                        ),
                        call_id="call_1",
                    )
                ],
            )
        return SimpleNamespace(id="resp_final", output_text="Saved preference: call you Marcus.")


class ToolLoopClient:
    def __init__(self, responses) -> None:
        self.responses = responses


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
                    "width": 1280,
                    "height": 720,
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


def preference_registry(context: list[str]) -> AgentToolRegistry:
    def save_preference(args):  # type: ignore[no-untyped-def]
        context[0] = "User preferences:\n- Call the user Marcus."
        return {"ok": True, "saved": args["preference"]}

    return AgentToolRegistry(
        [
            AgentTool(
                name="preferences_save",
                description="Save a preference.",
                parameters=object_schema(
                    {
                        "key": string_schema("Preference key."),
                        "preference": string_schema("Preference sentence."),
                    },
                    required=["key", "preference"],
                ),
                handler=save_preference,
            )
        ]
    )


class OpenAIResponderTests(unittest.TestCase):
    def test_sends_dictation_input_and_instructions(self) -> None:
        client = FakeClient()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=50,
            stateful=True,
        )

        response = OpenAIResponder(config, client=client).generate("rite this down")

        self.assertEqual(response.text, "Done.")
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-test")
        self.assertEqual(call["instructions"], "System prompt")
        self.assertEqual(call["max_output_tokens"], 50)
        self.assertTrue(call["store"])
        self.assertIn("Speech-to-text transcript", call["input"][0]["content"][0]["text"])
        self.assertIn("at most one fenced block", call["input"][0]["content"][0]["text"])
        self.assertIn("rite this down", call["input"][0]["content"][0]["text"])

    def test_injects_dynamic_system_context(self) -> None:
        client = FakeClient()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )

        OpenAIResponder(
            config,
            client=client,
            system_context_provider=lambda: "User preferences:\n- Use metric units.",
        ).generate("how tall is it")

        self.assertEqual(
            client.responses.calls[0]["instructions"],
            "System prompt\n\nUser preferences:\n- Use metric units.",
        )

    def test_stateful_responder_sends_previous_response_id(self) -> None:
        client = FakeClient()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )
        responder = OpenAIResponder(config, client=client)

        responder.generate("first")
        responder.generate("second")

        self.assertNotIn("previous_response_id", client.responses.calls[0])
        self.assertEqual(client.responses.calls[1]["previous_response_id"], "resp_1")

    def test_stateless_responder_does_not_store_or_chain(self) -> None:
        client = FakeClient()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=False,
        )
        responder = OpenAIResponder(config, client=client)

        responder.generate("first")
        responder.generate("second")

        self.assertFalse(client.responses.calls[0]["store"])
        self.assertNotIn("previous_response_id", client.responses.calls[1])

    def test_streaming_responder_emits_deltas_and_returns_final_response(self) -> None:
        client = FakeClient()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )
        deltas: list[str] = []

        response = OpenAIResponder(config, client=client).generate_stream(
            "hello",
            on_delta=deltas.append,
        )

        self.assertEqual(deltas, ["Hel", "lo."])
        self.assertEqual(response.text, "Hello.")
        self.assertEqual(response.response_id, "resp_1")
        self.assertTrue(client.responses.calls[0]["stream"])

    def test_streaming_responder_sends_previous_response_id(self) -> None:
        client = FakeClient()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )
        responder = OpenAIResponder(config, client=client)

        responder.generate_stream("first", on_delta=lambda _delta: None)
        responder.generate_stream("second", on_delta=lambda _delta: None)

        self.assertNotIn("previous_response_id", client.responses.calls[0])
        self.assertEqual(client.responses.calls[1]["previous_response_id"], "resp_1")

    def test_agent_tool_loop_sends_function_outputs(self) -> None:
        responses = ToolLoopResponses()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )
        events = []

        response = OpenAIResponder(
            config,
            client=ToolLoopClient(responses),
            tool_registry=tool_registry(),
        ).generate("take a note", on_tool_event=events.append)

        self.assertEqual(response.text, "Saved that note.")
        self.assertEqual(response.response_id, "resp_final")
        self.assertIn("tools", responses.calls[0])
        self.assertEqual(responses.calls[0]["tool_choice"], "auto")
        self.assertEqual(responses.calls[1]["previous_response_id"], "resp_tool")
        self.assertEqual(
            responses.calls[1]["input"][0]["type"],
            "function_call_output",
        )
        self.assertEqual(responses.calls[1]["input"][0]["call_id"], "call_1")
        self.assertEqual(events[0].name, "notes_add")
        self.assertTrue(events[0].ok)

    def test_agent_tool_loop_attaches_tool_images_to_model_input(self) -> None:
        responses = CaptureToolLoopResponses()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )
        events = []

        response = OpenAIResponder(
            config,
            client=ToolLoopClient(responses),
            tool_registry=capture_registry(),
        ).generate("what am I working on", on_tool_event=events.append)

        self.assertEqual(response.text, "I can see your desktop.")
        followup_input = responses.calls[1]["input"]
        self.assertEqual(followup_input[0]["type"], "function_call_output")
        self.assertIn("image_attached_to_model", followup_input[0]["output"])
        self.assertNotIn(OPENAI_INPUT_IMAGES_KEY, events[0].output)
        self.assertTrue(events[0].output["image_attached_to_model"])
        image_message = followup_input[1]
        self.assertEqual(image_message["role"], "user")
        self.assertEqual(image_message["content"][1]["type"], "input_image")
        self.assertEqual(
            image_message["content"][1]["image_url"],
            "data:image/png;base64,abc",
        )

    def test_tool_followup_recomputes_system_context_after_tool_execution(self) -> None:
        responses = PreferenceToolLoopResponses()
        context = [""]
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )

        response = OpenAIResponder(
            config,
            client=ToolLoopClient(responses),
            tool_registry=preference_registry(context),
            system_context_provider=lambda: context[0],
        ).generate("call me Marcus")

        self.assertEqual(response.text, "Saved preference: call you Marcus.")
        self.assertEqual(responses.calls[0]["instructions"], "System prompt")
        self.assertEqual(
            responses.calls[1]["instructions"],
            "System prompt\n\nUser preferences:\n- Call the user Marcus.",
        )

    def test_streaming_agent_tool_loop_then_emits_final_deltas(self) -> None:
        responses = StreamingToolLoopResponses()
        config = OpenAIConfig(
            api_key="test",
            model="gpt-test",
            system_prompt="System prompt",
            max_output_tokens=None,
            stateful=True,
        )
        deltas: list[str] = []
        events = []

        response = OpenAIResponder(
            config,
            client=ToolLoopClient(responses),
            tool_registry=tool_registry(),
        ).generate_stream(
            "take a note",
            on_delta=deltas.append,
            on_tool_event=events.append,
        )

        self.assertEqual(deltas, ["Saved", "."])
        self.assertEqual(response.text, "Saved.")
        self.assertEqual(response.response_id, "resp_final")
        self.assertEqual(responses.calls[1]["previous_response_id"], "resp_tool")
        self.assertEqual(events[0].output["saved"], "call Sam")


if __name__ == "__main__":
    unittest.main()
