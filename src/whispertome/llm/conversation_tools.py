from __future__ import annotations

from typing import Any

from whispertome.agent.tools import AgentTool, nullable_string_schema, object_schema, string_schema
from whispertome.llm.base import LlmResponder


class ConversationController:
    """Late-bound controller for provider-neutral conversation state tools."""

    def __init__(self) -> None:
        self._responder: LlmResponder | None = None

    def bind(self, responder: LlmResponder) -> None:
        self._responder = responder

    def start_new_chat(self, *, reason: str | None = None) -> dict[str, Any]:
        responder = self._require_responder()
        responder.reset_conversation()
        output: dict[str, Any] = {
            "ok": True,
            "action": "new_chat_started",
            "message": "The next user turn will start without prior chat history.",
        }
        if reason:
            output["reason"] = reason
        return output

    def compact_context(self, *, summary: str, reason: str | None = None) -> dict[str, Any]:
        responder = self._require_responder()
        responder.compact_conversation(summary)
        output: dict[str, Any] = {
            "ok": True,
            "action": "context_compacted",
            "summary_chars": len(summary),
            "message": (
                "The next user turn will use the compacted summary instead of "
                "full chat history."
            ),
        }
        if reason:
            output["reason"] = reason
        return output

    def auto_compact(self) -> dict[str, Any]:
        """Autonomous compaction (no user turn): ask the model for a carry-forward summary,
        then replace history with it. Falls back to a clean reset if summarizing fails."""
        responder = self._require_responder()
        summary = ""
        try:
            response = responder.generate(_AUTO_COMPACT_INSTRUCTION)
            summary = (response.text or "").strip()
        except Exception:  # noqa: BLE001 — never let auto-compaction crash the loop
            summary = ""
        if summary:
            responder.compact_conversation(summary)
            return {"ok": True, "action": "auto_compacted", "summary_chars": len(summary)}
        responder.reset_conversation()
        return {"ok": True, "action": "auto_reset"}

    def _require_responder(self) -> LlmResponder:
        if self._responder is None:
            raise RuntimeError("Conversation controller is not bound to an LLM responder")
        return self._responder


_AUTO_COMPACT_INSTRUCTION = (
    "Summarize our conversation so far into a brief carry-forward note: the user's goals, "
    "key facts, decisions, and any open tasks or current intent. Reply with only the summary, "
    "no preamble."
)


def build_conversation_tools(controller: ConversationController) -> list[AgentTool]:
    return [
        AgentTool(
            name="conversation_compact",
            description=(
                "Replace the active chat history with a compact summary. Use when the "
                "user asks to compact context, summarize the current chat for carryover, "
                "or when the conversation is getting long but important state should be kept."
            ),
            parameters=object_schema(
                {
                    "summary": string_schema(
                        "Concise carry-forward summary of useful context, decisions, "
                        "open tasks, and current intent."
                    ),
                    "reason": nullable_string_schema("Optional short reason for compacting."),
                },
                required=["summary"],
            ),
            handler=lambda args: controller.compact_context(
                summary=str(args["summary"]),
                reason=_optional_text(args.get("reason")),
            ),
        ),
        AgentTool(
            name="conversation_start_new",
            description=(
                "Start a fresh chat by clearing active conversational history. Use when "
                "the user asks for a new chat, fresh conversation, reset context, or to forget "
                "the current chat thread. This does not delete saved notes, preferences, tasks, "
                "reminders, or other organizer records."
            ),
            parameters=object_schema(
                {
                    "reason": nullable_string_schema("Optional short reason for starting fresh."),
                },
            ),
            handler=lambda args: controller.start_new_chat(
                reason=_optional_text(args.get("reason")),
            ),
        ),
    ]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
