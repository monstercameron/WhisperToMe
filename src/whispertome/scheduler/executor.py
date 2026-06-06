"""Deterministic action executor: replays a recorded step sequence at fire time.

No LLM is involved. Tool steps run through the same AgentToolRegistry the live loop uses;
speak steps go through an injected callback. Never raises — a failed/timed-out step is
recorded and execution continues so later steps (e.g. a spoken confirmation) still run.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from whispertome.scheduler.model import Action

LOGGER = logging.getLogger(__name__)

# Literal key-path template like {step0.count} or {step1.result.name}. No eval.
_TEMPLATE = re.compile(r"\{step(\d+)((?:\.[A-Za-z0-9_]+)+)\}")


@dataclass(frozen=True)
class StepResult:
    index: int
    type: str
    ok: bool
    detail: str
    output: dict[str, Any] | None = None


@dataclass
class ExecResult:
    ok: bool
    steps: list[StepResult] = field(default_factory=list)


class ActionExecutor:
    def __init__(
        self,
        *,
        tool_registry: Any,
        speak: Callable[[str], None],
        step_timeout_s: float = 20.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = tool_registry
        self._speak = speak
        self._step_timeout_s = max(1.0, step_timeout_s)
        self._log = logger or LOGGER

    def run(self, action: Action) -> ExecResult:
        outputs: list[dict[str, Any]] = []
        results: list[StepResult] = []
        for index, step in enumerate(action.steps):
            if step.type == "speak":
                result = self._run_speak(index, step.text or "", outputs)
            else:
                result = self._run_tool(index, step.name or "", step.args or {})
            results.append(result)
            outputs.append(result.output or {})
        return ExecResult(ok=all(r.ok for r in results), steps=results)

    # -- step kinds ---------------------------------------------------------

    def _run_speak(self, index: int, text: str, outputs: list[dict]) -> StepResult:
        rendered = self._render(text, outputs)
        try:
            self._speak(rendered)
        except Exception as exc:  # noqa: BLE001 — executor must never raise
            self._log.warning("scheduler_step_speak_failed index=%d error=%s", index, exc)
            return StepResult(index, "speak", False, f"speak failed: {exc}", {"text": rendered})
        return StepResult(index, "speak", True, "spoken", {"text": rendered})

    def _run_tool(self, index: int, name: str, args: dict) -> StepResult:
        # The watchdog bounds how long a hung tool can hold the caller (and turn_lock); it
        # cannot kill the worker, so a timed-out tool may still complete its single operation
        # in the background. That is acceptable: scheduler tool steps are individual organizer/
        # system operations and never emit audio (speak steps run inline under turn_lock), so a
        # late-completing tool cannot overlap conversation audio.
        holder: dict[str, Any] = {}

        def call() -> None:
            try:
                event = self._registry.execute(name, args)
                holder["output"] = dict(getattr(event, "output", {}) or {})
            except Exception as exc:  # noqa: BLE001 — registry.execute shouldn't raise, but be safe
                holder["error"] = str(exc)

        worker = threading.Thread(target=call, name=f"sched-step-{index}", daemon=True)
        worker.start()
        worker.join(self._step_timeout_s)
        if worker.is_alive():
            self._log.warning("scheduler_step_tool_timeout index=%d name=%s", index, name)
            return StepResult(index, "tool", False, f"{name} timed out", {"ok": False, "timed_out": True})
        if "error" in holder:
            return StepResult(index, "tool", False, holder["error"], {"ok": False, "error": holder["error"]})
        output = holder.get("output", {})
        ok = bool(output.get("ok", False))
        detail = output.get("error") if not ok else f"{name} ok"
        return StepResult(index, "tool", ok, str(detail), output)

    # -- templating ---------------------------------------------------------

    def _render(self, text: str, outputs: list[dict]) -> str:
        def replace(match: re.Match) -> str:
            step_index = int(match.group(1))
            path = match.group(2).lstrip(".").split(".")
            if step_index >= len(outputs):
                return ""
            value: Any = outputs[step_index]
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return ""
            return "" if value is None else str(value)

        return _TEMPLATE.sub(replace, text)
