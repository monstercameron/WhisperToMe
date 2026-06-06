from __future__ import annotations

from pathlib import Path
from typing import Any

from whispertome.agent.tools import (
    OPENAI_INPUT_IMAGES_KEY,
    AgentTool,
    boolean_schema,
    integer_schema,
    object_schema,
    string_schema,
)
from whispertome.system.windows import (
    AgentWindowController,
    BrightnessController,
    DesktopCaptureController,
    DesktopCaptureResult,
    PowerShellController,
    PowerShellRunResult,
    VolumeController,
    WindowActionResult,
    WindowsAgentWindowController,
    WindowsBrightnessController,
    WindowsDesktopCaptureController,
    WindowsPowerShellController,
    WindowsVolumeController,
    WindowsWindowSwitcher,
    WindowSwitcher,
    clamp_percent,
)


def build_system_control_tools(
    *,
    project_root: Path | None = None,
    volume_controller: VolumeController | None = None,
    brightness_controller: BrightnessController | None = None,
    window_controller: AgentWindowController | None = None,
    capture_controller: DesktopCaptureController | None = None,
    powershell_controller: PowerShellController | None = None,
    window_switcher: WindowSwitcher | None = None,
) -> list[AgentTool]:
    volume = volume_controller or WindowsVolumeController()
    brightness = brightness_controller or WindowsBrightnessController()
    window = window_controller or WindowsAgentWindowController()
    capture = capture_controller or WindowsDesktopCaptureController(project_root=project_root)
    powershell = powershell_controller or WindowsPowerShellController()
    switcher = window_switcher or WindowsWindowSwitcher()
    return [
        _system_volume_get(volume),
        _system_volume_set(volume),
        _system_volume_change(volume),
        _system_volume_mute(volume),
        _screen_brightness_get(brightness),
        _screen_brightness_set(brightness),
        _screen_brightness_change(brightness),
        _agent_window_minimize(window),
        _list_windows(switcher),
        _focus_window(switcher),
        _desktop_capture(capture),
        _powershell_run(powershell),
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


def _agent_window_minimize(controller: AgentWindowController) -> AgentTool:
    return AgentTool(
        name="agent_window_minimize",
        description=(
            "Hide the WhisperToMe desktop assistant window to the system tray when the user "
            "asks to hide, minimize, dismiss, go away, get the assistant/app/window out of "
            "the way, or says they are done talking for now. This affects only the "
            "assistant's own desktop window."
        ),
        parameters=object_schema({}),
        handler=lambda _args: _window_action_result(controller.minimize_agent_window()),
    )


def _list_windows(switcher: WindowSwitcher) -> AgentTool:
    def handler(_args: dict) -> dict:
        windows = switcher.list_windows()
        return {
            "ok": True,
            "windows": [
                {"hwnd": w.hwnd, "app": w.app, "title": w.title} for w in windows
            ],
        }

    return AgentTool(
        name="list_windows",
        description=(
            "List the currently open application windows (their app name and title). Call "
            "this when the user asks to switch to, go to, bring up, focus, or show another "
            "app or window. Then pick the window whose app/title best matches what the user "
            "said and call focus_window with its hwnd. If nothing matches, tell the user."
        ),
        parameters=object_schema({}),
        handler=handler,
    )


def _focus_window(switcher: WindowSwitcher) -> AgentTool:
    def handler(args: dict) -> dict:
        hwnd = args.get("hwnd")
        if not isinstance(hwnd, int):
            return {"ok": False, "error": "hwnd (integer from list_windows) is required"}
        return _window_action_result(switcher.focus_window(hwnd))

    return AgentTool(
        name="focus_window",
        description=(
            "Bring a specific open window to the foreground (switch to it). Use the hwnd "
            "from a list_windows result for the window that best matches the user's request."
        ),
        parameters=object_schema(
            {"hwnd": integer_schema("The window handle (hwnd) from list_windows.")},
            required=["hwnd"],
        ),
        handler=handler,
    )


def _desktop_capture(controller: DesktopCaptureController) -> AgentTool:
    return AgentTool(
        name="desktop_capture",
        description=(
            "Capture the current Windows desktop and attach it as an image for visual "
            "model context when the user asks what is on screen, asks for help with "
            "what they are working on, or refers to something visible on the desktop. "
            "By default, temporarily excludes the WhisperToMe window from the capture."
        ),
        parameters=object_schema(
            {
                "include_agent_window": boolean_schema(
                    "True only if the user explicitly wants the WhisperToMe window included."
                ),
                "max_width": integer_schema(
                    "Maximum preview image width sent to the model, from 512 to 1920.",
                    minimum=512,
                ),
            },
        ),
        handler=lambda args: _desktop_capture_result(
            controller.capture_desktop(
                include_agent_window=bool(args.get("include_agent_window", False)),
                max_width=_positive_int(args.get("max_width"), default=1280),
            )
        ),
    )


def _powershell_run(controller: PowerShellController) -> AgentTool:
    return AgentTool(
        name="powershell_run",
        description=(
            "Run a short, bounded Windows PowerShell command for explicit user requests "
            "to inspect local time, OS, hardware, process, service, environment, or "
            "Windows configuration not covered by narrower tools. Prefer read-only "
            "Get/Test commands. Mutating commands require allow_mutation=true after "
            "explicit user confirmation; high-risk commands are blocked."
        ),
        parameters=object_schema(
            {
                "command": string_schema("PowerShell command to run."),
                "timeout_seconds": integer_schema(
                    "Command timeout in seconds, clamped from 1 to 15. Default 5.",
                    minimum=1,
                ),
                "max_output_chars": integer_schema(
                    "Maximum stdout/stderr characters returned, clamped from 200 to 12000. "
                    "Default 4000.",
                    minimum=200,
                ),
                "allow_mutation": boolean_schema(
                    "True only when the user explicitly confirmed a state-changing command."
                ),
            },
            required=["command"],
        ),
        handler=lambda args: _powershell_run_result(
            controller.run(
                str(args["command"]),
                timeout_seconds=_positive_int(args.get("timeout_seconds"), default=5),
                max_output_chars=_positive_int(args.get("max_output_chars"), default=4000),
                allow_mutation=bool(args.get("allow_mutation", False)),
            )
        ),
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


def _window_action_result(result: WindowActionResult) -> dict[str, Any]:
    output: dict[str, Any] = {
        "ok": result.ok,
        "process_id": result.process_id,
        "window_title": result.window_title,
        "hwnd": result.hwnd,
    }
    if result.reason is not None:
        output["reason"] = result.reason
    return output


def _desktop_capture_result(result: DesktopCaptureResult) -> dict[str, Any]:
    output: dict[str, Any] = {
        "ok": result.ok,
        "path": result.path,
        "width": result.width,
        "height": result.height,
        "preview_width": result.preview_width,
        "preview_height": result.preview_height,
        "excluded_agent_window": result.excluded_agent_window,
    }
    if result.reason is not None:
        output["reason"] = result.reason
    if result.image_url:
        output[OPENAI_INPUT_IMAGES_KEY] = [
            {
                "image_url": result.image_url,
                "detail": "low",
                "label": "desktop capture",
            }
        ]
    return output


def _powershell_run_result(result: PowerShellRunResult) -> dict[str, Any]:
    output: dict[str, Any] = {
        "ok": result.ok,
        "command": result.command,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_ms": round(result.duration_ms, 1),
        "timed_out": result.timed_out,
        "blocked": result.blocked,
        "truncated": result.truncated,
    }
    if result.reason is not None:
        output["reason"] = result.reason
    return output


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
