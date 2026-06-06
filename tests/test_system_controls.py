from __future__ import annotations

import unittest

from whispertome.agent.tools import AgentToolRegistry
from whispertome.system.tools import build_system_control_tools
from whispertome.system.windows import SystemVolumeDucker, WindowsBackgroundAudioDucker


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
        registry = AgentToolRegistry(
            build_system_control_tools(
                volume_controller=volume,
                brightness_controller=brightness,
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
