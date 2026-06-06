from __future__ import annotations

import unittest

from whispertome.agent.tools import AgentToolRegistry
from whispertome.system.tools import build_system_control_tools
from whispertome.system.windows import WindowActionResult, WindowInfo


class _FakeSwitcher:
    def __init__(self) -> None:
        self.focused: list[int] = []
        self._windows = [
            WindowInfo(hwnd=101, process_id=10, title="Inbox - Outlook", app="outlook"),
            WindowInfo(hwnd=202, process_id=20, title="YouTube - Google Chrome", app="chrome"),
        ]

    def list_windows(self) -> list[WindowInfo]:
        return self._windows

    def focus_window(self, hwnd: int) -> WindowActionResult:
        self.focused.append(hwnd)
        info = next((w for w in self._windows if w.hwnd == hwnd), None)
        return WindowActionResult(
            ok=info is not None,
            process_id=info.process_id if info else None,
            window_title=info.title if info else None,
            hwnd=hwnd,
            reason=None if info else "not found",
        )


def _registry(switcher):
    return AgentToolRegistry(build_system_control_tools(window_switcher=switcher))


class WindowSwitchToolTests(unittest.TestCase):
    def test_list_windows_returns_app_and_title(self) -> None:
        reg = _registry(_FakeSwitcher())
        out = reg.execute("list_windows", {}).output
        self.assertTrue(out["ok"])
        apps = {w["app"] for w in out["windows"]}
        self.assertEqual(apps, {"outlook", "chrome"})
        self.assertTrue(all({"hwnd", "app", "title"} <= set(w) for w in out["windows"]))

    def test_focus_window_by_hwnd(self) -> None:
        fake = _FakeSwitcher()
        reg = _registry(fake)
        out = reg.execute("focus_window", {"hwnd": 202}).output
        self.assertTrue(out["ok"])
        self.assertEqual(fake.focused, [202])
        self.assertIn("Chrome", out["window_title"])

    def test_focus_window_requires_int_hwnd(self) -> None:
        fake = _FakeSwitcher()
        reg = _registry(fake)
        out = reg.execute("focus_window", {}).output
        self.assertFalse(out["ok"])
        self.assertEqual(fake.focused, [])  # nothing focused on bad args

    def test_both_tools_registered(self) -> None:
        names = {spec["name"] for spec in _registry(_FakeSwitcher()).tool_specs()}
        self.assertIn("list_windows", names)
        self.assertIn("focus_window", names)


if __name__ == "__main__":
    unittest.main()
