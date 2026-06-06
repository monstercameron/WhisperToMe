from __future__ import annotations

from typing import Any

from whispertome.agent.tools import (
    AgentTool,
    boolean_schema,
    integer_schema,
    object_schema,
)
from whispertome.system.windows import (
    BrightnessController,
    VolumeController,
    WindowsBrightnessController,
    WindowsVolumeController,
    clamp_percent,
)


def build_system_control_tools(
    *,
    volume_controller: VolumeController | None = None,
    brightness_controller: BrightnessController | None = None,
) -> list[AgentTool]:
    volume = volume_controller or WindowsVolumeController()
    brightness = brightness_controller or WindowsBrightnessController()
    return [
        _system_volume_get(volume),
        _system_volume_set(volume),
        _system_volume_change(volume),
        _system_volume_mute(volume),
        _screen_brightness_get(brightness),
        _screen_brightness_set(brightness),
        _screen_brightness_change(brightness),
    ]


def _system_volume_get(controller: VolumeController) -> AgentTool:
    return AgentTool(
        name="system_volume_get",
        description="Get the current Windows master speaker volume and mute state.",
        parameters=object_schema({}),
        handler=lambda _args: {
            "ok": True,
            "volume_percent": controller.get_volume_percent(),
            "muted": controller.get_muted(),
        },
    )


def _system_volume_set(controller: VolumeController) -> AgentTool:
    return AgentTool(
        name="system_volume_set",
        description=(
            "Set Windows master speaker volume on direct user request. "
            "The percent is clamped from 0 to 100."
        ),
        parameters=object_schema(
            {
                "percent": integer_schema("Target volume percent from 0 to 100.", minimum=0),
            },
            required=["percent"],
        ),
        handler=lambda args: {
            "ok": True,
            "volume_percent": controller.set_volume_percent(clamp_percent(args["percent"])),
            "muted": controller.get_muted(),
        },
    )


def _system_volume_change(controller: VolumeController) -> AgentTool:
    return AgentTool(
        name="system_volume_change",
        description=(
            "Change Windows master speaker volume by a relative delta on direct user request. "
            "Use -10 for vague 'turn it down' and +10 for vague 'turn it up'."
        ),
        parameters=object_schema(
            {
                "delta_percent": integer_schema(
                    "Relative percent change, such as -10 or 10.",
                ),
            },
            required=["delta_percent"],
        ),
        handler=lambda args: _change_volume(controller, args.get("delta_percent")),
    )


def _system_volume_mute(controller: VolumeController) -> AgentTool:
    return AgentTool(
        name="system_volume_mute",
        description="Mute or unmute Windows master speaker audio on direct user request.",
        parameters=object_schema(
            {
                "muted": boolean_schema("True to mute, false to unmute."),
            },
            required=["muted"],
        ),
        handler=lambda args: {
            "ok": True,
            "muted": controller.set_muted(bool(args["muted"])),
            "volume_percent": controller.get_volume_percent(),
        },
    )


def _screen_brightness_get(controller: BrightnessController) -> AgentTool:
    return AgentTool(
        name="screen_brightness_get",
        description=(
            "Get built-in Windows display brightness. External monitors may not support this."
        ),
        parameters=object_schema({}),
        handler=lambda _args: _brightness_result(controller.get_brightness_levels()),
    )


def _screen_brightness_set(controller: BrightnessController) -> AgentTool:
    return AgentTool(
        name="screen_brightness_set",
        description=(
            "Set built-in Windows display brightness on direct user request. "
            "The percent is clamped from 0 to 100."
        ),
        parameters=object_schema(
            {
                "percent": integer_schema("Target brightness percent from 0 to 100.", minimum=0),
            },
            required=["percent"],
        ),
        handler=lambda args: _brightness_result(
            controller.set_brightness_percent(clamp_percent(args["percent"]))
        ),
    )


def _screen_brightness_change(controller: BrightnessController) -> AgentTool:
    return AgentTool(
        name="screen_brightness_change",
        description=(
            "Change built-in Windows display brightness by a relative delta on direct user "
            "request. Use -10 for vague 'dim it' and +10 for vague 'brighten it'."
        ),
        parameters=object_schema(
            {
                "delta_percent": integer_schema(
                    "Relative percent change, such as -10 or 10.",
                ),
            },
            required=["delta_percent"],
        ),
        handler=lambda args: _change_brightness(controller, args.get("delta_percent")),
    )


def _change_volume(controller: VolumeController, delta: Any) -> dict[str, Any]:
    current = controller.get_volume_percent()
    updated = controller.set_volume_percent(clamp_percent(current + _int_value(delta)))
    return {"ok": True, "volume_percent": updated, "muted": controller.get_muted()}


def _change_brightness(controller: BrightnessController, delta: Any) -> dict[str, Any]:
    current_levels = controller.get_brightness_levels()
    current = round(sum(current_levels) / len(current_levels))
    return _brightness_result(
        controller.set_brightness_percent(clamp_percent(current + _int_value(delta)))
    )


def _brightness_result(levels: list[int]) -> dict[str, Any]:
    if not levels:
        return {"ok": False, "error": "No brightness-capable display found"}
    return {
        "ok": True,
        "brightness_percent": round(sum(levels) / len(levels)),
        "display_brightness": levels,
        "display_count": len(levels),
    }


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
