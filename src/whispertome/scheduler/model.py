"""Structured action model for scheduled events.

An action is an ordered list of steps replayed deterministically at fire time (no LLM):
  - {"type": "speak", "text": "..."}            -> speak fixed/templated text
  - {"type": "tool",  "name": "...", "args": {}} -> run an agent tool with fixed args

Speak text may reference earlier step outputs with a literal key-path template, e.g.
"You have {step0.count} reminders." (no eval; missing keys render empty).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

ACTION_VERSION = 1
STEP_TYPES = ("speak", "tool")


@dataclass(frozen=True)
class Step:
    type: str
    text: str | None = None
    name: str | None = None
    args: dict | None = None


@dataclass(frozen=True)
class Action:
    version: int
    steps: tuple[Step, ...]


class ActionParseError(ValueError):
    """Raised when an action payload is malformed."""


def parse_action(raw: object) -> Action:
    """Parse a dict / JSON string / bare step-list into an Action. Strict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"action is not valid JSON: {exc}") from exc

    if isinstance(raw, list):
        steps_raw = raw
        version = ACTION_VERSION
    elif isinstance(raw, dict):
        steps_raw = raw.get("steps")
        version = int(raw.get("version", ACTION_VERSION))
    else:
        raise ActionParseError("action must be an object or a list of steps")

    if not isinstance(steps_raw, list) or not steps_raw:
        raise ActionParseError("action must contain a non-empty 'steps' list")

    steps: list[Step] = []
    for index, item in enumerate(steps_raw):
        if not isinstance(item, dict):
            raise ActionParseError(f"step {index} must be an object")
        step_type = item.get("type")
        if step_type not in STEP_TYPES:
            raise ActionParseError(f"step {index} type must be one of {STEP_TYPES}")
        if step_type == "speak":
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ActionParseError(f"speak step {index} requires non-empty 'text'")
            steps.append(Step(type="speak", text=text))
        else:  # tool
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ActionParseError(f"tool step {index} requires a 'name'")
            args = item.get("args", {})
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ActionParseError(f"tool step {index} 'args' must be an object")
            steps.append(Step(type="tool", name=name, args=args))

    return Action(version=version, steps=tuple(steps))


def action_to_json(action: Action) -> str:
    payload = {
        "version": action.version,
        "steps": [
            {"type": s.type, "text": s.text}
            if s.type == "speak"
            else {"type": "tool", "name": s.name, "args": s.args or {}}
            for s in action.steps
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def action_from_json(text: str) -> Action:
    return parse_action(text)


def tool_step_names(action: Action) -> list[str]:
    return [s.name for s in action.steps if s.type == "tool" and s.name]


def validate_tool_names(action: Action, known: set[str]) -> list[str]:
    """Return the tool names referenced by the action that are NOT in `known`."""
    return sorted({n for n in tool_step_names(action) if n not in known})
