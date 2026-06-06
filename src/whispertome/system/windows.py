from __future__ import annotations

import base64
import io
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter, sleep
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


class WindowSwitcher(Protocol):
    def list_windows(self) -> list[WindowInfo]: ...

    def focus_window(self, hwnd: int) -> WindowActionResult: ...


class DesktopCaptureController(Protocol):
    def capture_desktop(
        self,
        *,
        include_agent_window: bool = False,
        max_width: int = 1280,
    ) -> DesktopCaptureResult: ...


class PowerShellController(Protocol):
    def run(
        self,
        command: str,
        *,
        timeout_seconds: int = 5,
        max_output_chars: int = 4000,
        allow_mutation: bool = False,
    ) -> PowerShellRunResult: ...


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


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    process_id: int | None
    title: str
    app: str | None  # process exe stem, e.g. "chrome", "msedge"


@dataclass(frozen=True)
class DesktopCaptureResult:
    ok: bool
    path: str | None
    width: int | None
    height: int | None
    preview_width: int | None
    preview_height: int | None
    image_url: str | None = None
    excluded_agent_window: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class PowerShellRunResult:
    ok: bool
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    blocked: bool = False
    truncated: bool = False
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


class WindowsWindowSwitcher:
    """Lists open top-level windows and focuses one by handle.

    The model picks the window whose title/app best matches the user's request, then calls
    focus_window(hwnd). Focusing uses the AttachThreadInput workaround for the Windows
    foreground lock. The assistant's own window is excluded from the list.
    """

    def __init__(self, *, exclude_pid: int | None = None, exclude_title: str | None = None) -> None:
        self._exclude_pid = exclude_pid if exclude_pid is not None else _env_pid()
        self._exclude_title = exclude_title or os.environ.get(
            DESKTOP_WINDOW_TITLE_ENV, DEFAULT_DESKTOP_WINDOW_TITLE
        )

    def list_windows(self) -> list[WindowInfo]:
        if sys.platform != "win32":
            return []
        windows: list[WindowInfo] = []
        seen: set[int] = set()
        for hwnd, pid, title in _iter_visible_windows():
            if not title.strip():
                continue
            if self._exclude_pid is not None and pid == self._exclude_pid:
                continue
            if self._exclude_title and title == self._exclude_title:
                continue
            if hwnd in seen:
                continue
            seen.add(hwnd)
            app = _process_exe_stem(pid) if pid else None
            windows.append(WindowInfo(hwnd=hwnd, process_id=pid, title=title, app=app))
        return windows

    def focus_window(self, hwnd: int) -> WindowActionResult:
        if sys.platform != "win32":
            return WindowActionResult(
                False, None, None, hwnd, reason="Window switching is only available on Windows"
            )
        # Resolve title/pid for a useful result and to confirm the handle is still live.
        info = next((w for w in self.list_windows() if w.hwnd == hwnd), None)
        try:
            focused = _focus_window(hwnd)
        except Exception as exc:  # noqa: BLE001
            return WindowActionResult(
                False, info.process_id if info else None, info.title if info else None, hwnd,
                reason=str(exc),
            )
        return WindowActionResult(
            ok=focused,
            process_id=info.process_id if info else None,
            window_title=info.title if info else None,
            hwnd=hwnd,
            reason=None if focused else "Windows did not bring the window to the foreground",
        )


class WindowsDesktopCaptureController:
    """Captures the visible Windows desktop for multimodal model context."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        target_pid: int | None = None,
        title: str | None = None,
    ) -> None:
        self._project_root = project_root or Path.cwd()
        self._target_pid = target_pid if target_pid is not None else _env_pid()
        self._title = title or os.environ.get(
            DESKTOP_WINDOW_TITLE_ENV,
            DEFAULT_DESKTOP_WINDOW_TITLE,
        )

    def capture_desktop(
        self,
        *,
        include_agent_window: bool = False,
        max_width: int = 1280,
    ) -> DesktopCaptureResult:
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            return DesktopCaptureResult(
                ok=False,
                path=None,
                width=None,
                height=None,
                preview_width=None,
                preview_height=None,
                reason=f"Pillow is required for desktop capture: {exc}",
            )

        hwnd: int | None = None
        was_visible = False
        if not include_agent_window and sys.platform == "win32":
            hwnd, was_visible = _hide_agent_window_for_capture(
                pid=self._target_pid,
                title=self._title,
            )

        try:
            image = ImageGrab.grab(all_screens=True)
            output_path = self._capture_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG")
            preview = image.copy()
            preview.thumbnail((_max_capture_width(max_width), _max_capture_width(max_width)))
            buffer = io.BytesIO()
            preview.save(buffer, format="PNG", optimize=True)
            image_url = "data:image/png;base64," + base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")
            width, height = image.size
            preview_width, preview_height = preview.size
            return DesktopCaptureResult(
                ok=True,
                path=str(output_path),
                width=width,
                height=height,
                preview_width=preview_width,
                preview_height=preview_height,
                image_url=image_url,
                excluded_agent_window=bool(hwnd and was_visible),
            )
        except Exception as exc:
            return DesktopCaptureResult(
                ok=False,
                path=None,
                width=None,
                height=None,
                preview_width=None,
                preview_height=None,
                reason=str(exc),
            )
        finally:
            if hwnd is not None and was_visible:
                _restore_agent_window_after_capture(hwnd)

    def _capture_path(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return self._project_root / "artifacts" / "captures" / f"desktop-{stamp}.png"


class WindowsPowerShellController:
    """Runs short, bounded PowerShell commands for explicit local system requests."""

    def run(
        self,
        command: str,
        *,
        timeout_seconds: int = 5,
        max_output_chars: int = 4000,
        allow_mutation: bool = False,
    ) -> PowerShellRunResult:
        command = command.strip()
        timeout_seconds = min(15, max(1, int(timeout_seconds)))
        max_output_chars = min(12000, max(200, int(max_output_chars)))
        started = perf_counter()

        reason = _blocked_powershell_reason(command, allow_mutation=allow_mutation)
        if reason is not None:
            return PowerShellRunResult(
                ok=False,
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=(perf_counter() - started) * 1000.0,
                blocked=True,
                reason=reason,
            )

        try:
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, stdout_truncated = _truncate_text(exc.stdout or "", max_output_chars)
            stderr, stderr_truncated = _truncate_text(exc.stderr or "", max_output_chars)
            return PowerShellRunResult(
                ok=False,
                command=command,
                exit_code=None,
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                duration_ms=(perf_counter() - started) * 1000.0,
                timed_out=True,
                truncated=stdout_truncated or stderr_truncated,
                reason=f"PowerShell timed out after {timeout_seconds} seconds",
            )
        except FileNotFoundError:
            return PowerShellRunResult(
                ok=False,
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=(perf_counter() - started) * 1000.0,
                reason="powershell.exe was not found",
            )

        stdout, stdout_truncated = _truncate_text(completed.stdout, max_output_chars)
        stderr, stderr_truncated = _truncate_text(completed.stderr, max_output_chars)
        return PowerShellRunResult(
            ok=completed.returncode == 0,
            command=command,
            exit_code=completed.returncode,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            duration_ms=(perf_counter() - started) * 1000.0,
            truncated=stdout_truncated or stderr_truncated,
            reason=None if completed.returncode == 0 else "PowerShell command failed",
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


_POWERSHELL_ALWAYS_BLOCKED = re.compile(
    r"(?i)(?:"
    r"\b(?:Remove-Item|rm|del|erase|rmdir|rd|Format-Volume|Clear-Disk|"
    r"Initialize-Disk|diskpart|Restart-Computer|Stop-Computer|shutdown|"
    r"Invoke-Expression|iex|Set-ExecutionPolicy)\b"
    r"|(?:^|\s)reg(?:\.exe)?\s+(?:add|delete|import|restore)\b"
    r")"
)

_POWERSHELL_MUTATION_BLOCKED = re.compile(
    r"(?i)(?:"
    r"\b(?:New-Item|Set-Item|Set-ItemProperty|Set-Content|Add-Content|"
    r"Out-File|Move-Item|Copy-Item|Rename-Item|Start-Process|Stop-Process|"
    r"Set-[A-Za-z]+|Clear-[A-Za-z]+|Export-[A-Za-z]+|Import-[A-Za-z]+|"
    r"Enable-[A-Za-z]+|Disable-[A-Za-z]+|Install-[A-Za-z]+|Uninstall-[A-Za-z]+)\b"
    r"|>{1,2}"
    r")"
)


def _blocked_powershell_reason(command: str, *, allow_mutation: bool) -> str | None:
    if not command:
        return "PowerShell command cannot be empty"
    if len(command) > 2000:
        return "PowerShell command is too long"
    if _POWERSHELL_ALWAYS_BLOCKED.search(command):
        return "Command matched a blocked high-risk PowerShell pattern"
    if not allow_mutation and _POWERSHELL_MUTATION_BLOCKED.search(command):
        return "Command appears to mutate state; ask for explicit confirmation first"
    return None


def _truncate_text(text: str | bytes, max_chars: int) -> tuple[str, bool]:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text, False
    return f"{text[: max_chars - 18]}\n...[truncated]...", True


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


def _iter_visible_windows() -> list[tuple[int, int | None, str]]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    found: list[tuple[int, int | None, str]] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @proc
    def enum_window(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        found.append((int(hwnd), int(process_id.value) or None, buffer.value))
        return True

    user32.EnumWindows(enum_window, 0)
    return found


def _process_exe_stem(pid: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(260)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return Path(buffer.value).stem.lower()
    finally:
        kernel32.CloseHandle(handle)


def _focus_window(hwnd: int) -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.SetActiveWindow.argtypes = [wintypes.HWND]
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    sw_restore = 9
    target = wintypes.HWND(hwnd)
    foreground = user32.GetForegroundWindow()
    this_thread = kernel32.GetCurrentThreadId()
    target_thread = user32.GetWindowThreadProcessId(target, None)
    fore_thread = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0

    # Attaching input queues lets SetForegroundWindow bypass the foreground lock.
    attach_target = bool(target_thread) and target_thread != this_thread
    attach_fore = bool(fore_thread) and fore_thread not in (this_thread, target_thread)
    if attach_target:
        user32.AttachThreadInput(this_thread, target_thread, True)
    if attach_fore:
        user32.AttachThreadInput(this_thread, fore_thread, True)
    try:
        if user32.IsIconic(target):
            user32.ShowWindow(target, sw_restore)
        user32.SetForegroundWindow(target)
        user32.BringWindowToTop(target)
        user32.SetActiveWindow(target)
    finally:
        if attach_target:
            user32.AttachThreadInput(this_thread, target_thread, False)
        if attach_fore:
            user32.AttachThreadInput(this_thread, fore_thread, False)

    now = user32.GetForegroundWindow()
    return now is not None and int(now) == int(hwnd)


def _max_capture_width(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1280
    return max(512, min(parsed, 1920))


def _hide_agent_window_for_capture(
    *,
    pid: int | None,
    title: str,
) -> tuple[int | None, bool]:
    window = _find_top_level_window(pid=pid, title=title)
    if window is None:
        return None, False
    hwnd = window[0]
    try:
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        was_visible = bool(user32.IsWindowVisible(hwnd))
        if was_visible:
            sw_hide = 0
            user32.ShowWindow(hwnd, sw_hide)
            sleep(0.15)
        return hwnd, was_visible
    except Exception:
        return None, False


def _restore_agent_window_after_capture(hwnd: int) -> None:
    try:
        import ctypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        sw_show = 5
        user32.ShowWindow(hwnd, sw_show)
    except Exception:
        return
