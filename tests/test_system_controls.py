from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from whispertome.agent.tools import AgentToolRegistry
from whispertome.system.tools import build_system_control_tools
from whispertome.system.windows import (
    DESKTOP_WINDOW_COMMAND_FILE_ENV,
    DesktopCaptureResult,
    PowerShellRunResult,
    SystemVolumeDucker,
    WindowActionResult,
    WindowsAgentWindowController,
    WindowsBackgroundAudioDucker,
    WindowsPowerShellController,
)


class FakeVolumeController:
    def __init__(self, volume: int = 50, muted: bool = False) -> None:
        self.volume = volume
        self.muted = muted
        self.set_calls: list[int] = []

    def get_volume_percent(self) -> int:
        return self.volume

    def set_volume_percent(self, percent: int) -> int:
        self.volume = percent
        self.set_calls.append(percent)
        return self.volume

    def get_muted(self) -> bool:
        return self.muted

    def set_muted(self, muted: bool) -> bool:
        self.muted = muted
        return self.muted


class FakeBrightnessController:
    def __init__(self, levels: list[int] | None = None) -> None:
        self.levels = levels or [60]

    def get_brightness_levels(self) -> list[int]:
        return self.levels

    def set_brightness_percent(self, percent: int) -> list[int]:
        self.levels = [percent for _ in self.levels]
        return self.levels


class FakeWindowController:
    def __init__(self, *, result: WindowActionResult | None = None) -> None:
        self.calls = 0
        self.result = result or WindowActionResult(
            ok=True,
            process_id=123,
            window_title="WhisperToMe",
            hwnd=456,
        )

    def minimize_agent_window(self) -> WindowActionResult:
        self.calls += 1
        return self.result


class FakeCaptureController:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def capture_desktop(
        self,
        *,
        include_agent_window: bool = False,
        max_width: int = 1280,
    ) -> DesktopCaptureResult:
        self.calls.append(
            {
                "include_agent_window": include_agent_window,
                "max_width": max_width,
            }
        )
        return DesktopCaptureResult(
            ok=True,
            path="C:/project/artifacts/captures/desktop.png",
            width=1920,
            height=1080,
            preview_width=max_width,
            preview_height=576,
            image_url="data:image/png;base64,abc",
            excluded_agent_window=not include_agent_window,
        )


class FakePowerShellController:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        command: str,
        *,
        timeout_seconds: int = 5,
        max_output_chars: int = 4000,
        allow_mutation: bool = False,
    ) -> PowerShellRunResult:
        self.calls.append(
            {
                "command": command,
                "timeout_seconds": timeout_seconds,
                "max_output_chars": max_output_chars,
                "allow_mutation": allow_mutation,
            }
        )
        return PowerShellRunResult(
            ok=True,
            command=command,
            exit_code=0,
            stdout="Saturday, June 6, 2026",
            stderr="",
            duration_ms=12.3,
        )


class FakeAudioSession:
    def __init__(
        self,
        *,
        pid: int | None,
        process_name: str,
        volume: float,
        display_name: str = "",
    ) -> None:
        self.pid = pid
        self.process_name = process_name
        self.display_name = display_name
        self.volume = volume
        self.set_calls: list[float] = []

    def get_volume_scalar(self) -> float:
        return self.volume

    def set_volume_scalar(self, scalar: float) -> float:
        self.volume = scalar
        self.set_calls.append(scalar)
        return self.volume


class FakeAudioSessionController:
    def __init__(self, sessions: list[FakeAudioSession]) -> None:
        self.sessions = sessions

    def get_sessions(self) -> list[FakeAudioSession]:
        return self.sessions


class SystemControlToolTests(unittest.TestCase):
    def test_volume_and_brightness_tools_clamp_and_change_values(self) -> None:
        volume = FakeVolumeController(volume=50)
        brightness = FakeBrightnessController(levels=[40, 60])
        window = FakeWindowController()
        capture = FakeCaptureController()
        powershell = FakePowerShellController()
        registry = AgentToolRegistry(
            build_system_control_tools(
                volume_controller=volume,
                brightness_controller=brightness,
                window_controller=window,
                capture_controller=capture,
                powershell_controller=powershell,
            )
        )

        self.assertEqual(
            registry.execute("system_volume_change", {"delta_percent": -15}).output[
                "volume_percent"
            ],
            35,
        )
        self.assertEqual(
            registry.execute("system_volume_set", {"percent": 150}).output["volume_percent"],
            100,
        )
        self.assertTrue(
            registry.execute("system_volume_mute", {"muted": True}).output["muted"]
        )
        self.assertEqual(
            registry.execute("screen_brightness_change", {"delta_percent": 15}).output[
                "brightness_percent"
            ],
            65,
        )
        self.assertEqual(
            registry.execute("screen_brightness_set", {"percent": -20}).output[
                "display_brightness"
            ],
            [0, 0],
        )
        minimize = registry.execute("agent_window_minimize", {}).output
        self.assertTrue(minimize["ok"])
        self.assertEqual(minimize["window_title"], "WhisperToMe")
        self.assertEqual(window.calls, 1)
        screenshot = registry.execute("desktop_capture", {"max_width": 1024}).output
        self.assertTrue(screenshot["ok"])
        self.assertEqual(screenshot["path"], "C:/project/artifacts/captures/desktop.png")
        self.assertEqual(screenshot["preview_width"], 1024)
        self.assertIn("_openai_input_images", screenshot)
        self.assertEqual(capture.calls[0]["max_width"], 1024)
        shell = registry.execute(
            "powershell_run",
            {"command": "Get-Date", "timeout_seconds": 2, "max_output_chars": 500},
        ).output
        self.assertTrue(shell["ok"])
        self.assertEqual(shell["stdout"], "Saturday, June 6, 2026")
        self.assertEqual(powershell.calls[0]["command"], "Get-Date")
        self.assertEqual(powershell.calls[0]["timeout_seconds"], 2)
        self.assertEqual(powershell.calls[0]["max_output_chars"], 500)

    def test_powershell_controller_blocks_high_risk_commands(self) -> None:
        output = WindowsPowerShellController().run("Remove-Item C:\\important -Recurse")

        self.assertFalse(output.ok)
        self.assertTrue(output.blocked)
        self.assertIn("blocked", output.reason or "")

    def test_agent_window_minimize_reports_unavailable_window(self) -> None:
        window = FakeWindowController(
            result=WindowActionResult(
                ok=False,
                process_id=123,
                window_title="WhisperToMe",
                hwnd=None,
                reason="No window",
            )
        )
        registry = AgentToolRegistry(build_system_control_tools(window_controller=window))

        output = registry.execute("agent_window_minimize", {}).output

        self.assertFalse(output["ok"])
        self.assertEqual(output["reason"], "No window")

    def test_agent_window_minimize_writes_desktop_host_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_file = Path(tmp) / "window.command"
            old_command_file = os.environ.get(DESKTOP_WINDOW_COMMAND_FILE_ENV)
            os.environ[DESKTOP_WINDOW_COMMAND_FILE_ENV] = str(command_file)
            try:
                controller = WindowsAgentWindowController(
                    target_pid=123,
                    title="WhisperToMe",
                )
                result = controller.minimize_agent_window()
            finally:
                if old_command_file is None:
                    os.environ.pop(DESKTOP_WINDOW_COMMAND_FILE_ENV, None)
                else:
                    os.environ[DESKTOP_WINDOW_COMMAND_FILE_ENV] = old_command_file

            self.assertTrue(result.ok)
            self.assertIsNone(result.hwnd)
            self.assertEqual(command_file.read_text(encoding="utf-8"), "hide_to_tray\n")

    def test_volume_ducker_lowers_then_restores_previous_volume(self) -> None:
        volume = FakeVolumeController(volume=80)
        ducker = SystemVolumeDucker(controller=volume, target_percent=25)

        ducked = ducker.duck()
        restored = ducker.restore()

        self.assertTrue(ducked.ok)
        self.assertEqual(ducked.original_percent, 80)
        self.assertEqual(ducked.ducked_percent, 25)
        self.assertTrue(restored.ok)
        self.assertEqual(volume.volume, 80)
        self.assertEqual(volume.set_calls, [25, 80])

    def test_volume_ducker_never_raises_low_volume(self) -> None:
        volume = FakeVolumeController(volume=10)
        ducker = SystemVolumeDucker(controller=volume, target_percent=25)

        ducked = ducker.duck()
        restored = ducker.restore()

        self.assertTrue(ducked.ok)
        self.assertEqual(ducked.ducked_percent, 10)
        self.assertTrue(restored.ok)
        self.assertEqual(volume.set_calls, [10])

    def test_background_audio_ducker_lowers_other_sessions_only(self) -> None:
        assistant = FakeAudioSession(pid=123, process_name="python.exe", volume=1.0)
        browser = FakeAudioSession(pid=456, process_name="chrome.exe", volume=1.0)
        quiet = FakeAudioSession(pid=789, process_name="steam.exe", volume=0.10)
        ducker = WindowsBackgroundAudioDucker(
            controller=FakeAudioSessionController([assistant, browser, quiet]),
            target_percent=25,
            excluded_pids={123},
        )

        ducked = ducker.duck()
        second_duck = ducker.duck()
        restored = ducker.restore()

        self.assertTrue(ducked.ok)
        self.assertEqual(ducked.controlled_count, 2)
        self.assertEqual(ducked.lowered_count, 1)
        self.assertEqual(second_duck.reason, "already_ducked")
        self.assertEqual(browser.volume, 1.0)
        self.assertEqual(quiet.volume, 0.10)
        self.assertEqual(assistant.set_calls, [])
        self.assertEqual(browser.set_calls, [0.25, 1.0])
        self.assertEqual(quiet.set_calls, [0.10])
        self.assertTrue(restored.ok)
        self.assertEqual(restored.restored_count, 2)


if __name__ == "__main__":
    unittest.main()
