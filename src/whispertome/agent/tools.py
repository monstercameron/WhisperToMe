from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

JsonObject = dict[str, Any]
ToolHandler = Callable[[JsonObject], JsonObject]
OPENAI_INPUT_IMAGES_KEY = "_openai_input_images"


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: JsonObject
    handler: ToolHandler

    def to_openai_tool(self) -> JsonObject:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": False,
        }


@dataclass(frozen=True)
class AgentToolEvent:
    name: str
    arguments: JsonObject
    output: JsonObject
    latency_ms: float

    @property
    def ok(self) -> bool:
        return bool(self.output.get("ok", False))


class AgentToolRegistry:
    def __init__(self, tools: list[AgentTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def tool_specs(self) -> list[JsonObject]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, arguments: JsonObject) -> AgentToolEvent:
        started = perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            output = {"ok": False, "error": f"Unknown tool: {name}"}
        else:
            try:
                output = tool.handler(arguments)
            except Exception as exc:  # pragma: no cover - defensive tool boundary
                output = {"ok": False, "error": str(exc)}
        return AgentToolEvent(
            name=name,
            arguments=arguments,
            output=output,
            latency_ms=(perf_counter() - started) * 1000.0,
        )


def parse_tool_arguments(arguments: str) -> JsonObject:
    if not arguments.strip():
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return parsed


def tool_output_json(output: JsonObject) -> str:
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def string_schema(description: str) -> JsonObject:
    return {"type": "string", "description": description}


def nullable_string_schema(description: str) -> JsonObject:
    return {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": description,
    }

def string_array_schema(description: str) -> JsonObject:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
    }


def integer_schema(description: str, *, minimum: int | None = None) -> JsonObject:
    schema: JsonObject = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def boolean_schema(description: str) -> JsonObject:
    return {"type": "boolean", "description": description}


def object_schema(
    properties: JsonObject,
    *,
    required: list[str] | None = None,
) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
