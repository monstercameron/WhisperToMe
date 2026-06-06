from __future__ import annotations

import unittest

from whispertome.agent.tools import AgentToolRegistry
from whispertome.llm.base import LlmResponse
from whispertome.llm.conversation_tools import (
    ConversationController,
    build_conversation_tools,
)


class FakeResponder:
    def __init__(self) -> None:
        self.reset_count = 0
        self.compacted_summary: str | None = None

    def generate(self, user_text, *, on_tool_event=None):  # type: ignore[no-untyped-def]
        return LlmResponse(text="ok", latency_ms=0.0, model="fake", response_id=None)

    def generate_stream(  # type: ignore[no-untyped-def]
        self,
        user_text,
        *,
        on_delta,
        on_tool_event=None,
    ):
        on_delta("ok")
        return LlmResponse(text="ok", latency_ms=0.0, model="fake", response_id=None)

    def reset_conversation(self) -> None:
        self.reset_count += 1
        self.compacted_summary = None

    def compact_conversation(self, summary: str) -> None:
        self.compacted_summary = summary


class ConversationToolsTests(unittest.TestCase):
    def test_builds_expected_tool_names(self) -> None:
        registry = AgentToolRegistry(build_conversation_tools(ConversationController()))

        names = {tool["name"] for tool in registry.tool_specs()}

        self.assertIn("conversation_compact", names)
        self.assertIn("conversation_start_new", names)

    def test_compact_and_new_chat_call_bound_responder(self) -> None:
        controller = ConversationController()
        responder = FakeResponder()
        controller.bind(responder)  # type: ignore[arg-type]
        registry = AgentToolRegistry(build_conversation_tools(controller))

        compact_event = registry.execute(
            "conversation_compact",
            {"summary": "User wants a clean desktop TUI."},
        )
        new_chat_event = registry.execute("conversation_start_new", {})

        self.assertTrue(compact_event.ok)
        self.assertTrue(new_chat_event.ok)
        self.assertEqual(responder.reset_count, 1)
        self.assertIsNone(responder.compacted_summary)

    def test_unbound_controller_fails_at_tool_boundary(self) -> None:
        registry = AgentToolRegistry(build_conversation_tools(ConversationController()))

        event = registry.execute("conversation_start_new", {})

        self.assertFalse(event.ok)
        self.assertIn("not bound", event.output["error"])


if __name__ == "__main__":
    unittest.main()
