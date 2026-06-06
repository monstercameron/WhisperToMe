from __future__ import annotations

import unittest
from types import SimpleNamespace

from whispertome.config import OpenAIConfig
from whispertome.llm.openai_responses import OpenAIResponder


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id=f"resp_{len(self.calls)}", output_text="Done.")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


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
        self.assertIn("rite this down", call["input"][0]["content"][0]["text"])

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


if __name__ == "__main__":
    unittest.main()
