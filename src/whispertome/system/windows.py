from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from whispertome.errors import WhisperToMeError

DESKTOP_HOST_PID_ENV = "WHISPERTOME_DESKTOP_HOST_PID"
DESKTOP_WINDOW_COMMAND_FILE_ENV = "WHISPERTOME_DESKTOP_WINDOW_COMMAND_FILE"
DESKTOP_WINDOW_TITLE_ENV = "WHISPERTOME_DESKTOP_WINDOW_TITLE"
DEFAULT_DESKTOP_WINDOW_TITLE = "WhisperToMe"


class SystemControlError(WhisperToMeError):
    """Raised when a narrow Windows system control cannot be completed."""


class VolumeController(Protocol):
    def get_volume_percent(self) -> int: ...

    def set_volume_percent(self, percent: int) -> int: ...

    def get_muted(self) -> bool: ...

    def set_muted(self, muted: bool) -> bool: ...


class BrightnessController(Protocol):
    def get_brightness_levels(self) -> list[int]: ...

    def set_brightness_percent(self, percent: int) -> list[int]: ...


class AgentWindowController(Protocol):
    def minimize_agent_window(self) -> WindowActionResult: ...


class AudioSessionVolume(Protocol):
    pid: int | None
    process_name: str
    display_name: str

    def get_volume_scalar(self) -> float: ...

    def set_volume_scalar(self, scalar: float) -> float: ...


class AudioSessionController(Protocol):
    def get_sessions(self) -> list[AudioSessionVolume]: ...


@dataclass(frozen=True)
class WindowActionResult:
    ok: bool
    process_id: int | None
    window_title: str | None
    hwnd: int | None
    reason: str | None = None


class WindowsVolumeController:
    """Controls the Windows default speaker endpoint through Core Audio."""

    def get_volume_percent(self) -> int:
        endpoint = self._endpoint()
        return round(endpoint.GetMasterVolumeLevelScalar() * 100)

    def set_volume_percent(self, percent: int) -> int:
        endpoint = self._endpoint()
        clamped = clamp_percent(percent)
        endpoint.SetMasterVolumeLevelScalar(clamped / 100.0, None)
        return self.get_volume_percent()

    def get_muted(self) -> bool:
        return bool(self._endpoint().GetMute())

    def set_muted(self, muted: bool) -> bool:
        endpoint = self._endpoint()
        endpoint.SetMute(1 if muted else 0, None)
        return self.get_muted()

    @staticmethod
    def _endpoint():  # type: ignore[no-untyped-def]
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        except ImportError as exc:
            raise SystemControlError(
                "pycaw and comtypes are required for Windows volume control"
            ) from exc

        speakers = AudioUtilities.GetSpeakers()
        if speakers is None:
            raise SystemControlError("No default Windows speaker endpoint was found")
        endpoint = getattr(speakers, "EndpointVolume", None)
        if endpoint is not None:
            return endpoint
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))


@dataclass
class WindowsAudioSessionVolume:
    pid: int | None
    process_name: str
    display_name: str
    volume: Any

    def get_volume_scalar(self) -> float:
        return float(self.volume.GetMasterVolume())

    def set_volume_scalar(self, scalar: float) -> float:
        clamped = clamp_scalar(scalar)
        self.volume.SetMasterVolume(clamped, None)
        return float(self.volume.GetMasterVolume())


class WindowsAudioSessionController:
    """Controls per-application Windows audio sessions through Core Audio."""

    def get_sessions(self) -> list[AudioSessionVolume]:
        try:
            from pycaw.pycaw import AudioUtilities
        except ImportError as exc:
            raise SystemControlError("pycaw is required for Windows audio ducking") from exc

        sessions: list[AudioSessionVolume] = []
        for session in AudioUtilities.GetAllSessions():
            volume = getattr(session, "SimpleAudioVolume", None)
            if volume is None:
                continue
            process = getattr(session, "Process", None)
            pid = getattr(process, "pid", None)
            process_name = ""
            if process is not None:
                try:
                    process_name = process.name()
                except Exception:
                    process_name = ""
            sessions.append(
                WindowsAudioSessionVolume(
                    pid=pid,
                    process_name=process_name,
                    display_name=getattr(session, "DisplayName", "") or "",
                    volume=volume,
                )
            )
        return sessions


class WindowsBrightnessController:
    """Controls built-in display brightness through Windows WMI/CIM classes."""

    def get_brightness_levels(self) -> list[int]:
        output = _run_powershell(
            "$ErrorActionPreference='Stop'; "
            "$items=@(Get-CimInstance -Namespace 'root/WMI' "
            "-ClassName WmiMonitorBrightness | "
            "ForEach-Object { [int]$_.CurrentBrightness }); "
            "if ($items.Count -eq 0) { throw 'No brightness-capable display found' }; "
            "$items -join ','"
        )
        levels = [_parse_percent(value) for value in output.replace("\n", ",").split(",")]
        if not levels:
            raise SystemControlError("No brightness-capable display found")
        return levels

    def set_brightness_percent(self, percent: int) -> list[int]:
        clamped = clamp_percent(percent)
        _run_powershell(
            "$ErrorActionPreference='Stop'; "
            "$methods=@(Get-CimInstance -Namespace 'root/WMI' "
            "-ClassName WmiMonitorBrightnessMethods); "
            "if ($methods.Count -eq 0) { throw 'No brightness-capable display found' }; "
            f"$brightness={clamped}; "
            "foreach ($method in $methods) { "
            "Invoke-CimMethod -InputObject $method -MethodName WmiSetBrightness "
            "-Arguments @{Timeout=1; Brightness=$brightness} | Out-Null "
            "}; "
            "$methods.Count"
        )
        return self.get_brightness_levels()


class WindowsAgentWindowController:
    """Controls the owned WhisperToMe desktop host window."""

    def __init__(
        self,
        *,
        target_pid: int | None = None,
        title: str | None = None,
    ) -> None:
        self._target_pid = target_pid if target_pid is not None else _env_pid()
        self._title = title or os.environ.get(
            DESKTOP_WINDOW_TITLE_ENV,
            DEFAULT_DESKTOP_WINDOW_TITLE,
        )
        self._command_file = os.environ.get(DESKTOP_WINDOW_COMMAND_FILE_ENV)

    def minimize_agent_window(self) -> WindowActionResult:
        host_result = self._request_host_hide_to_tray()
        if host_result is not None:
            return host_result

        if sys.platform != "win32":
            return WindowActionResult(
                ok=False,
                process_id=self._target_pid,
                window_title=self._title,
                hwnd=None,
                reason="Agent window control is only available on Windows",
            )

        window = _find_top_level_window(pid=self._target_pid, title=self._title)
        if window is None:
            target = (
                f"pid {self._target_pid}"
                if self._target_pid is not None
                else f"title {self._title!r}"
            )
            return WindowActionResult(
                ok=False,
                process_id=self._target_pid,
                window_title=self._title,
                hwnd=None,
                reason=f"No WhisperToMe desktop window found for {target}",
            )

        hwnd, pid, title = window
        try:
            import ctypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            sw_hide = 0
            user32.ShowWindow(hwnd, sw_hide)
            hidden = not bool(user32.IsWindowVisible(hwnd))
        except Exception as exc:
            return WindowActionResult(
                ok=False,
                process_id=pid,
                window_title=title,
                hwnd=hwnd,
                reason=str(exc),
            )

        return WindowActionResult(
            ok=hidden,
            process_id=pid,
            window_title=title,
            hwnd=hwnd,
            reason=None if hidden else "Windows did not report the window as hidden",
        )

    def _request_host_hide_to_tray(self) -> WindowActionResult | None:
        if not self._command_file:
            return None
        try:
            command_file = Path(self._command_file).expanduser()
            command_file.parent.mkdir(parents=True, exist_ok=True)
            command_file.write_text("hide_to_tray\n", encoding="utf-8")
        except Exception as exc:
            return WindowActionResult(
                ok=False,
                process_id=self._target_pid,
                window_title=self._title,
                hwnd=None,
                reason=f"Could not request the desktop host to hide to tray: {exc}",
            )
        return WindowActionResult(
            ok=True,
            process_id=self._target_pid,
            window_title=self._title,
            hwnd=None,
        )


@dataclass(frozen=True)
class DuckResult:
    ok: bool
    original_percent: int | None
    ducked_percent: int | None
    reason: str | None = None


@dataclass(frozen=True)
class AudioSessionDuckResult:
    ok: bool
    controlled_count: int
    lowered_count: int
    restored_count: int
    target_percent: int | None
    reason: str | None = None


@dataclass
class _SessionSnapshot:
    session: AudioSessionVolume
    original_scalar: float
    ducked_scalar: float
    pid: int | None
    process_name: str
    display_name: str


class WindowsBackgroundAudioDucker:
    """Temporarily lowers other app sessions while leaving this assistant process alone."""

    def __init__(
        self,
        *,
        controller: AudioSessionController | None = None,
        target_percent: int = 25,
        enabled: bool = True,
        excluded_pids: set[int] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._controller = controller or WindowsAudioSessionController()
        self._target_percent = clamp_percent(target_percent)
        self._enabled = enabled
        self._excluded_pids = set(excluded_pids or {os.getpid()})
        self._logger = logger or logging.getLogger(__name__)
        self._lock = Lock()
        self._snapshots: list[_SessionSnapshot] = []

    def duck(self) -> AudioSessionDuckResult:
        if not self._enabled:
            return AudioSessionDuckResult(
                ok=False,
                controlled_count=0,
                lowered_count=0,
                restored_count=0,
                target_percent=None,
                reason="disabled",
            )

        with self._lock:
            if self._snapshots:
                return AudioSessionDuckResult(
                    ok=True,
                    controlled_count=len(self._snapshots),
                    lowered_count=sum(
                        1
                        for snapshot in self._snapshots
                        if snapshot.ducked_scalar < snapshot.original_scalar
                    ),
                    restored_count=0,
                    target_percent=self._target_percent,
                    reason="already_ducked",
                )
            try:
                target_scalar = self._target_percent / 100.0
                snapshots: list[_SessionSnapshot] = []
                lowered_count = 0
                skipped_count = 0
                for session in self._controller.get_sessions():
                    if session.pid in self._excluded_pids:
                        skipped_count += 1
                        continue
                    original = clamp_scalar(session.get_volume_scalar())
                    ducked = min(original, target_scalar)
                    if ducked < original:
                        session.set_volume_scalar(ducked)
                        lowered_count += 1
                    snapshots.append(
                        _SessionSnapshot(
                            session=session,
                            original_scalar=original,
                            ducked_scalar=ducked,
                            pid=session.pid,
                            process_name=session.process_name,
                            display_name=session.display_name,
                        )
                    )
                self._snapshots = snapshots
                self._logger.info(
                    (
                        "windows_audio_sessions_ducked controlled=%d lowered=%d "
                        "skipped=%d target_percent=%d sessions=%s"
                    ),
                    len(snapshots),
                    lowered_count,
                    skipped_count,
                    self._target_percent,
                    _format_session_names(snapshots),
                )
                return AudioSessionDuckResult(
                    ok=True,
                    controlled_count=len(snapshots),
                    lowered_count=lowered_count,
                    restored_count=0,
                    target_percent=self._target_percent,
                )
            except Exception as exc:  # pragma: no cover - Windows endpoint dependent
                self._logger.warning("windows_audio_session_duck_failed error=%s", exc)
                return AudioSessionDuckResult(
                    ok=False,
                    controlled_count=0,
                    lowered_count=0,
                    restored_count=0,
                    target_percent=self._target_percent,
                    reason=str(exc),
                )

    def restore(self) -> AudioSessionDuckResult:
        with self._lock:
            snapshots = self._snapshots
            self._snapshots = []
        if not snapshots:
            return AudioSessionDuckResult(
                ok=False,
                controlled_count=0,
                lowered_count=0,
                restored_count=0,
                target_percent=self._target_percent,
                reason="not_ducked",
            )

        restored_count = 0
        failed_count = 0
        for snapshot in snapshots:
            try:
                snapshot.session.set_volume_scalar(snapshot.original_scalar)
                restored_count += 1
            except Exception as exc:  # pragma: no cover - session may disappear
                failed_count += 1
                self._logger.warning(
                    (
                        "windows_audio_session_restore_failed pid=%s process=%s "
                        "display=%s error=%s"
                    ),
                    snapshot.pid,
                    snapshot.process_name,
                    snapshot.display_name,
                    exc,
                )
        self._logger.info(
            (
                "windows_audio_sessions_restored controlled=%d restored=%d "
                "failed=%d sessions=%s"
            ),
            len(snapshots),
            restored_count,
            failed_count,
            _format_session_names(snapshots),
        )
        return AudioSessionDuckResult(
            ok=failed_count == 0,
            controlled_count=len(snapshots),
            lowered_count=sum(
                1
                for snapshot in snapshots
                if snapshot.ducked_scalar < snapshot.original_scalar
            ),
            restored_count=restored_count,
            target_percent=self._target_percent,
            reason=None if failed_count == 0 else f"{failed_count} restore failures",
        )


class SystemVolumeDucker:
    """Temporarily lowers Windows master volume and restores the exact prior value."""

    def __init__(
        self,
        *,
        controller: VolumeController | None = None,
        target_percent: int = 25,
        enabled: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self._controller = controller or WindowsVolumeController()
        self._target_percent = clamp_percent(target_percent)
        self._enabled = enabled
        self._logger = logger or logging.getLogger(__name__)
        self._lock = Lock()
        self._original_percent: int | None = None
        self._ducked_percent: int | None = None

    def duck(self) -> DuckResult:
        if not self._enabled:
            return DuckResult(
                ok=False,
                original_percent=None,
                ducked_percent=None,
                reason="disabled",
            )
        with self._lock:
            if self._original_percent is not None:
                return DuckResult(
                    ok=True,
                    original_percent=self._original_percent,
                    ducked_percent=self._ducked_percent,
                )
            try:
                current = self._controller.get_volume_percent()
                target = min(current, self._target_percent)
                self._original_percent = current
                self._ducked_percent = target
                if target < current:
                    self._controller.set_volume_percent(target)
                self._logger.info(
                    "system_volume_ducked original_percent=%d ducked_percent=%d",
                    current,
                    target,
                )
                return DuckResult(ok=True, original_percent=current, ducked_percent=target)
            except Exception as exc:  # pragma: no cover - Windows endpoint dependent
                self._logger.warning("system_volume_duck_failed error=%s", exc)
                return DuckResult(
                    ok=False,
                    original_percent=None,
                    ducked_percent=None,
                    reason=str(exc),
                )

    def restore(self) -> DuckResult:
        with self._lock:
            original = self._original_percent
            ducked = self._ducked_percent
            self._original_percent = None
            self._ducked_percent = None
        if original is None:
            return DuckResult(
                ok=False,
                original_percent=None,
                ducked_percent=None,
                reason="not_ducked",
            )
        try:
            self._controller.set_volume_percent(original)
            self._logger.info(
                "system_volume_restored original_percent=%d ducked_percent=%s",
                original,
                ducked,
            )
            return DuckResult(ok=True, original_percent=original, ducked_percent=ducked)
        except Exception as exc:  # pragma: no cover - Windows endpoint dependent
            self._logger.warning(
                "system_volume_restore_failed original_percent=%d error=%s",
                original,
                exc,
            )
            return DuckResult(
                ok=False,
                original_percent=original,
                ducked_percent=ducked,
                reason=str(exc),
            )


def clamp_percent(value: int | float | str) -> int:
    try:
        parsed = round(float(value))
    except (TypeError, ValueError) as exc:
        raise SystemControlError("percent must be a number from 0 to 100") from exc
    return min(100, max(0, int(parsed)))


def clamp_scalar(value: int | float | str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemControlError("volume scalar must be a number from 0.0 to 1.0") from exc
    return min(1.0, max(0.0, parsed))


def _parse_percent(value: str) -> int:
    value = value.strip()
    if not value:
        raise SystemControlError("empty percent value")
    return clamp_percent(value)


def _format_session_names(snapshots: list[_SessionSnapshot]) -> str:
    names = []
    for snapshot in snapshots:
        name = snapshot.process_name or snapshot.display_name or "<unknown>"
        if snapshot.pid is not None:
            name = f"{name}:{snapshot.pid}"
        names.append(name)
    return ",".join(names)


def _run_powershell(command: str) -> str:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "PowerShell command failed"
        raise SystemControlError(error)
    return completed.stdout.strip()


def _env_pid() -> int | None:
    raw = os.environ.get(DESKTOP_HOST_PID_ENV)
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _find_top_level_window(
    *,
    pid: int | None,
    title: str,
) -> tuple[int, int | None, str] | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    matches: list[tuple[int, int | None, str]] = []

    enum_windows_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    @enum_windows_proc
    def enum_window(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        window_title = buffer.value
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if pid is not None:
            if int(process_id.value) == pid:
                matches.append((int(hwnd), int(process_id.value), window_title))
            return True
        if window_title == title:
            matches.append((int(hwnd), int(process_id.value), window_title))
        return True

    user32.EnumWindows(enum_window, 0)
    if not matches:
        return None
    if pid is not None:
        for match in matches:
            if match[2] == title:
                return match
    return matches[0]
