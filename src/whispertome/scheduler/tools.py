"""Agent tools for creating/listing/cancelling scheduled events.

The LLM authors the structured action (steps) at creation time; the scheduler replays it
deterministically. Tool names referenced by steps are validated against the live registry
here so a fire never hits an unknown tool.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from whispertome.agent.tools import AgentTool, object_schema, string_schema
from whispertome.scheduler.clock import Clock
from whispertome.scheduler.model import action_to_json, parse_action, validate_tool_names
from whispertome.scheduler.recurrence import next_recurrence, validate_recurrence

NameProvider = Callable[[], set[str]]


def _to_utc_iso(when: str, clock: Clock) -> str:
    dt = datetime.fromisoformat(when.strip())
    if dt.tzinfo is None:
        dt = dt.astimezone()  # naive -> assume system local tz
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def _preauthorize_steps(steps: object) -> object:
    """Scheduling an event IS the user's authorization, and fired events run unattended,
    so pre-authorize tool steps that otherwise need interactive confirmation. For
    powershell_run this sets allow_mutation=True (truly destructive commands stay hard-
    blocked by the tool regardless)."""
    if not isinstance(steps, list):
        return steps
    for step in steps:
        if isinstance(step, dict) and step.get("type") == "tool" and step.get("name") == "powershell_run":
            args = step.get("args")
            if isinstance(args, dict):
                args["allow_mutation"] = True
    return steps


def build_scheduler_tools(
    *,
    store,
    clock: Clock,
    name_provider: NameProvider,
    on_change: Callable[[], None],
) -> list[AgentTool]:
    return [
        _schedule_event(store, clock, name_provider, on_change),
        _list_scheduled_events(store),
        _cancel_scheduled_event(store, on_change),
    ]


def _schedule_event(
    store, clock: Clock, name_provider: NameProvider, on_change: Callable[[], None]
) -> AgentTool:
    def handler(args: dict) -> dict:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"ok": False, "error": "title is required"}

        # Resolve the action steps (a 'say' shortcut builds a single speak step).
        steps = args.get("action_steps")
        if not steps and args.get("say"):
            steps = [{"type": "speak", "text": str(args["say"])}]
        if not steps:
            return {"ok": False, "error": "provide action_steps or 'say' text"}
        steps = _preauthorize_steps(steps)
        try:
            action = parse_action(steps)
        except ValueError as exc:
            return {"ok": False, "error": f"invalid action: {exc}"}
        unknown = validate_tool_names(action, set(name_provider()))
        if unknown:
            return {"ok": False, "error": f"unknown tool(s) in steps: {', '.join(unknown)}"}

        recurrence = (args.get("recurrence") or "").strip() or None
        when = (args.get("when") or "").strip() or None
        try:
            if recurrence:
                validate_recurrence(recurrence)
                trigger = "recurring"
                if when:
                    next_fire_at = _to_utc_iso(when, clock)
                else:
                    next_fire_at = (
                        next_recurrence(recurrence, clock.now())
                        .replace(microsecond=0)
                        .isoformat()
                    )
            else:
                if not when:
                    return {"ok": False, "error": "one-time events need a 'when' time"}
                trigger = "one_shot"
                next_fire_at = _to_utc_iso(when, clock)
        except ValueError as exc:
            return {"ok": False, "error": f"invalid time/recurrence: {exc}"}

        event = store.add_scheduled_event(
            title=title,
            action_json=action_to_json(action),
            trigger=trigger,
            next_fire_at=next_fire_at,
            recurrence_rule=recurrence,
        )
        on_change()
        return {
            "ok": True,
            "event_id": event["id"],
            "title": event["title"],
            "trigger": trigger,
            "next_fire_at": next_fire_at,
            "recurrence": recurrence,
        }

    return AgentTool(
        name="schedule_event",
        description=(
            "Schedule something to happen later: a spoken reminder or a workflow that runs "
            "automatically at the time. Use for 'remind me', 'in an hour', 'every morning', "
            "'at 3pm do X'. Provide a 'when' ISO datetime for one-time events, and/or a "
            "'recurrence' rule for repeating ones. Author the work as 'action_steps': an "
            "ordered list of steps, each either {\"type\":\"speak\",\"text\":...} or "
            "{\"type\":\"tool\",\"name\":<an existing tool name>,\"args\":{...}}. For a plain "
            "reminder just pass 'say'. Speak text may reference an earlier step's result like "
            "'{step0.count}'. Recurrence rules: daily@HH:MM, weekly@<mon|tue|...>@HH:MM, "
            "every@<N>@minutes|hours. Use 24h local times and ISO datetimes."
        ),
        parameters=object_schema(
            {
                "title": string_schema("Short title for the event."),
                "when": string_schema("ISO datetime of the (first) firing, local time ok."),
                "recurrence": string_schema(
                    "Optional recurrence rule: daily@HH:MM, weekly@<dow>@HH:MM, every@N@minutes|hours."
                ),
                "say": string_schema("Shortcut: text to speak (creates one speak step)."),
                "action_steps": {
                    "type": "array",
                    "description": "Ordered steps: speak {type,text} or tool {type,name,args}.",
                    "items": {"type": "object", "additionalProperties": True},
                },
            },
            required=["title"],
        ),
        handler=handler,
    )


def _list_scheduled_events(store) -> AgentTool:
    def handler(_args: dict) -> dict:
        events = store.list_scheduled_events()
        return {
            "ok": True,
            "events": [
                {
                    "id": e["id"],
                    "title": e["title"],
                    "trigger": e["trigger"],
                    "next_fire_at": e["next_fire_at"],
                    "recurrence": e.get("recurrence_rule"),
                    "status": e["status"],
                }
                for e in events
            ],
        }

    return AgentTool(
        name="list_scheduled_events",
        description="List upcoming scheduled events/reminders and when they next fire.",
        parameters=object_schema({}),
        handler=handler,
    )


def _cancel_scheduled_event(store, on_change: Callable[[], None]) -> AgentTool:
    def handler(args: dict) -> dict:
        event_id = (args.get("event_id") or "").strip() or None
        title = (args.get("title") or "").strip() or None
        if not event_id and not title:
            return {"ok": False, "error": "provide event_id or title"}
        try:
            event = store.cancel_scheduled_event(event_id=event_id, title_query=title)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        on_change()
        return {"ok": True, "cancelled": event["id"], "title": event["title"]}

    return AgentTool(
        name="cancel_scheduled_event",
        description="Cancel a scheduled event/reminder by id or by matching its title.",
        parameters=object_schema(
            {
                "event_id": string_schema("The event id from list_scheduled_events."),
                "title": string_schema("Or a title to match if the id is unknown."),
            }
        ),
        handler=handler,
    )
