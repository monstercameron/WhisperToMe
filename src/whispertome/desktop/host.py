from __future__ import annotations

import codecs
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

from whispertome.desktop.tray import DesktopTrayIcon
from whispertome.errors import WhisperToMeError
from whispertome.system.windows import (
    DESKTOP_HOST_PID_ENV,
    DESKTOP_WINDOW_COMMAND_FILE_ENV,
    DESKTOP_WINDOW_TITLE_ENV,
)
from whispertome.ui.terminal import TUI_HEIGHT_ENV, TUI_WIDTH_ENV

ANSI_RE = re.compile(r"\x1b\[([0-9?;]*)([A-Za-z])")
FULL_FRAME_CLEAR = "\x1b[H\x1b[2J"
LOGGER = logging.getLogger(__name__)
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_FONT_SIZE = 10
WAKE_SHOW_DEBOUNCE_S = 1.5
GEOMETRY_TRACKING_DELAY_MS = 600
GEOMETRY_PROGRAMMATIC_S = 0.6
COLOR_CODES = {
    "30": "fg_black",
    "31": "fg_red",
    "32": "fg_green",
    "33": "fg_yellow",
    "34": "fg_blue",
    "35": "fg_magenta",
    "36": "fg_cyan",
    "37": "fg_white",
}


@dataclass(frozen=True)
class VoiceLoopLaunchOptions:
    project_root: Path
    wake_phrases: tuple[str, ...] = ()
    max_commands: int = 0
    allow_non_npu: bool = False
    save_audio: bool = False
    warmup: bool = True
    min_speech_ms: int = 250
    vad_threshold: float | None = None
    speech_end_ms: int | None = None
    play: bool = True
    stream_tts: bool = True
    tui_lines: int = 10
    stop_file: Path | None = None
    wake_event_file: Path | None = None


@dataclass(frozen=True)
class DesktopHostConfig:
    command: list[str]
    cwd: Path
    title: str = "WhisperToMe"
    width: int = DEFAULT_WINDOW_WIDTH
    height: int = DEFAULT_WINDOW_HEIGHT
    font_size: int = DEFAULT_FONT_SIZE
    stop_file: Path | None = None
    wake_event_file: Path | None = None
    window_command_file: Path | None = None
    window_state_file: Path | None = None
    graceful_shutdown_timeout_s: float = 5.0
    slow_render_log_ms: float = 50.0


@dataclass(frozen=True)
class WindowPlacement:
    x: int
    y: int
    width: int
    height: int
    manual: bool = False


def build_voice_loop_command(
    python_executable: str,
    options: VoiceLoopLaunchOptions,
) -> list[str]:
    command = [python_executable]
    # A frozen one-file build IS the entry point, so `-m whispertome` is invalid there;
    # only prepend the module flag when running under a real Python interpreter.
    if not getattr(sys, "frozen", False):
        command += ["-m", "whispertome"]
    command += [
        "--project-root",
        str(options.project_root),
        "run",
        "--turns",
        str(options.max_commands),
        "--min-speech-ms",
        str(options.min_speech_ms),
        "--tui",
        "--tui-lines",
        str(options.tui_lines),
    ]
    for phrase in options.wake_phrases:
        command.extend(["--wake", phrase])
    if options.allow_non_npu:
        command.append("--allow-non-npu")
    if options.save_audio:
        command.append("--save-audio")
    if not options.warmup:
        command.append("--no-warmup")
    if options.vad_threshold is not None:
        command.extend(["--vad-threshold", str(options.vad_threshold)])
    if options.speech_end_ms is not None:
        command.extend(["--speech-end-ms", str(options.speech_end_ms)])
    if not options.play:
        command.append("--no-play")
    if not options.stream_tts:
        command.append("--no-stream-tts")
    if options.stop_file is not None:
        command.extend(["--stop-file", str(options.stop_file)])
    if options.wake_event_file is not None:
        command.extend(["--wake-event-file", str(options.wake_event_file)])
    return command


class DesktopTerminalHost:
    """Owns a desktop window and renders the existing terminal TUI inside it."""

    def __init__(self, config: DesktopHostConfig) -> None:
        self._config = config
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._terminal: object | None = None
        self._root: object | None = None
        self._status_var: object | None = None
        self._stop_button: object | None = None
        self._tray_icon: object | None = None
        self._terminal_font: object | None = None
        self._current_tags: tuple[str, ...] = ()
        self._stop_requested = False
        self._close_after_stop = False
        self._shutdown_deadline: float | None = None
        self._last_wake_event_mtime_ns: int | None = None
        self._last_window_command_mtime_ns: int | None = None
        self._last_show_monotonic = 0.0
        self._tracking_window_position = False
        self._programmatic_geometry_until = 0.0
        self._window_placement: WindowPlacement | None = None

    def run(self) -> int:
        try:
            import tkinter as tk
            from tkinter import font as tkfont
        except ImportError as exc:  # pragma: no cover - platform install dependent
            raise WhisperToMeError("tkinter is required for the desktop host") from exc

        root = tk.Tk()
        self._root = root
        root.withdraw()
        root.title(self._config.title)
        root.geometry(f"{self._config.width}x{self._config.height}")
        root.minsize(self._config.width, self._config.height)
        root.maxsize(self._config.width, self._config.height)
        root.resizable(False, False)
        root.configure(bg="#05070b")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        status_var = tk.StringVar(value="starting voice loop")
        self._status_var = status_var

        header = tk.Frame(root, bg="#080d14", padx=10, pady=6)
        header.pack(side=tk.TOP, fill=tk.X)
        title = tk.Label(
            header,
            text="WHISPER TO ME",
            fg="#7df9ff",
            bg="#080d14",
            font=("Segoe UI Semibold", 11),
        )
        title.pack(side=tk.LEFT)
        status = tk.Label(
            header,
            textvariable=status_var,
            fg="#95a3b8",
            bg="#080d14",
            font=("Segoe UI", 9),
        )
        status.pack(side=tk.LEFT, padx=(18, 0))
        stop_button = tk.Button(
            header,
            text="Stop",
            command=self._request_stop,
            bg="#141b27",
            fg="#e6edf7",
            activebackground="#253247",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=2,
        )
        stop_button.pack(side=tk.RIGHT)
        self._stop_button = stop_button

        terminal_font = tkfont.Font(family="Cascadia Mono", size=self._config.font_size)
        self._terminal_font = terminal_font
        terminal = tk.Text(
            root,
            bg="#05070b",
            fg="#d7faff",
            insertbackground="#d7faff",
            font=terminal_font,
            wrap=tk.NONE,
            borderwidth=0,
            highlightthickness=0,
            padx=4,
            pady=0,
        )
        terminal.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        terminal.configure(state=tk.DISABLED)
        self._terminal = terminal
        self._configure_terminal_tags(terminal)
        self._restore_or_center_window(reason="startup")
        root.bind("<Configure>", self._on_root_configure)
        root.bind("<Unmap>", self._on_root_unmap, add="+")
        root.after(GEOMETRY_TRACKING_DELAY_MS, self._enable_position_tracking)
        root.deiconify()

        self._tray_icon = DesktopTrayIcon(
            title=self._config.title,
            on_show=lambda: root.after(0, lambda: self._show_window(force=True)),  # type: ignore[attr-defined]
            on_stop_listening=lambda: root.after(0, self._request_stop),  # type: ignore[attr-defined]
            on_close=lambda: root.after(  # type: ignore[attr-defined]
                0,
                lambda: self._request_stop(close_after_stop=True),
            ),
        )
        self._tray_icon.start()  # type: ignore[attr-defined]
        root.after(80, self._start_process)  # type: ignore[attr-defined]
        root.after(16, self._drain_output)
        root.after(120, self._poll_wake_event)
        root.after(120, self._poll_window_command)
        root.mainloop()
        return self._process.returncode if self._process is not None else 0

    def _restore_or_center_window(self, *, reason: str) -> None:
        root = self._root
        if root is None:
            return
        screen_width, screen_height = self._screen_size()
        saved = _load_window_placement(
            self._config.window_state_file,
            expected_width=self._config.width,
            expected_height=self._config.height,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        placement = saved or _center_placement(
            width=self._config.width,
            height=self._config.height,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self._apply_window_placement(placement, reason=reason)

    def _enable_position_tracking(self) -> None:
        self._tracking_window_position = True

    def _on_root_configure(self, event: object) -> None:
        root = self._root
        if root is None or getattr(event, "widget", None) is not root:
            return
        if not self._tracking_window_position:
            return
        if perf_counter() < self._programmatic_geometry_until:
            return
        if not _root_window_visible(root):
            return
        placement = self._read_root_placement(manual=True)
        if placement is None:
            return
        if placement == self._window_placement:
            return
        self._window_placement = placement
        _save_window_placement(self._config.window_state_file, placement)
        LOGGER.info(
            "desktop_window_position_saved x=%d y=%d width=%d height=%d",
            placement.x,
            placement.y,
            placement.width,
            placement.height,
        )

    def _on_root_unmap(self, event: object) -> None:
        root = self._root
        if root is None or getattr(event, "widget", None) is not root:
            return
        try:
            state = str(root.state())  # type: ignore[attr-defined]
        except Exception:
            return
        if state == "iconic":
            root.after(0, self._hide_iconified_window_to_tray)  # type: ignore[attr-defined]

    def _hide_iconified_window_to_tray(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            state = str(root.state())  # type: ignore[attr-defined]
        except Exception as exc:
            LOGGER.warning("desktop_minimize_state_check_failed error=%s", exc)
            return
        if state != "iconic":
            return
        self._hide_to_tray()
        LOGGER.info("desktop_window_hidden reason=minimize_button")

    def _read_root_placement(self, *, manual: bool) -> WindowPlacement | None:
        root = self._root
        if root is None:
            return None
        try:
            root.update_idletasks()  # type: ignore[attr-defined]
            return WindowPlacement(
                x=int(root.winfo_x()),  # type: ignore[attr-defined]
                y=int(root.winfo_y()),  # type: ignore[attr-defined]
                width=int(root.winfo_width()),  # type: ignore[attr-defined]
                height=int(root.winfo_height()),  # type: ignore[attr-defined]
                manual=manual,
            )
        except Exception:
            return None

    def _screen_size(self) -> tuple[int, int]:
        root = self._root
        if root is None:
            return self._config.width, self._config.height
        try:
            root.update_idletasks()  # type: ignore[attr-defined]
            return (
                int(root.winfo_screenwidth()),  # type: ignore[attr-defined]
                int(root.winfo_screenheight()),  # type: ignore[attr-defined]
            )
        except Exception:
            return self._config.width, self._config.height

    def _apply_window_placement(self, placement: WindowPlacement, *, reason: str) -> None:
        root = self._root
        if root is None:
            return
        self._programmatic_geometry_until = perf_counter() + GEOMETRY_PROGRAMMATIC_S
        self._window_placement = placement
        root.geometry(_format_geometry(placement))  # type: ignore[attr-defined]
        LOGGER.info(
            "desktop_window_position_applied reason=%s x=%d y=%d width=%d height=%d manual=%s",
            reason,
            placement.x,
            placement.y,
            placement.width,
            placement.height,
            placement.manual,
        )

    def _start_process(self) -> None:
        if self._process is not None:
            return
        root = self._root
        if root is not None:
            root.update_idletasks()  # type: ignore[attr-defined]
        creationflags = 0
        if sys.platform == "win32":
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        fallback_columns, fallback_rows = _estimate_tui_viewport(
            self._config.width,
            self._config.height,
            self._config.font_size,
        )
        tui_columns, tui_rows = _measure_tui_viewport(
            self._terminal,
            self._terminal_font,
            fallback_columns=fallback_columns,
            fallback_rows=fallback_rows,
        )
        env = os.environ.copy()
        env[TUI_WIDTH_ENV] = str(tui_columns)
        env[TUI_HEIGHT_ENV] = str(tui_rows)
        env["PYTHONIOENCODING"] = "utf-8:replace"
        env[DESKTOP_HOST_PID_ENV] = str(os.getpid())
        env[DESKTOP_WINDOW_TITLE_ENV] = self._config.title
        if self._config.window_command_file is not None:
            env[DESKTOP_WINDOW_COMMAND_FILE_ENV] = str(self._config.window_command_file)
        LOGGER.info(
            "desktop_tui_viewport width_px=%d height_px=%d font_size=%d cols=%d rows=%d",
            self._config.width,
            self._config.height,
            self._config.font_size,
            tui_columns,
            tui_rows,
        )
        self._process = subprocess.Popen(
            self._config.command,
            cwd=str(self._config.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            creationflags=creationflags,
            env=env,
        )
        self._set_status(f"running pid {self._process.pid}")
        self._reader_thread = threading.Thread(
            target=self._read_output,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._reader_thread.start()

    def _read_output(self, stream: BinaryIO | None) -> None:
        if stream is None:
            self._output_queue.put(None)
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = os.read(stream.fileno(), 4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self._output_queue.put(text)
            remaining = decoder.decode(b"", final=True)
            if remaining:
                self._output_queue.put(remaining)
        finally:
            self._output_queue.put(None)

    def _drain_output(self) -> None:
        chunks: list[str] = []
        stopped = False
        while True:
            try:
                item = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                stopped = True
                continue
            chunks.append(item)
        if chunks:
            self._write_terminal(_latest_terminal_frame("".join(chunks)))
        if stopped:
            self._set_status("voice loop stopped")
            if self._stop_requested:
                self._finish_stop()
        if self._root is not None:
            self._root.after(16, self._drain_output)  # type: ignore[attr-defined]

    def _write_terminal(self, text: str) -> None:
        terminal = self._terminal
        if terminal is None:
            return
        started = perf_counter()
        full_frame = FULL_FRAME_CLEAR in text
        terminal.configure(state="normal")  # type: ignore[attr-defined]
        index = 0
        for match in ANSI_RE.finditer(text):
            if match.start() > index:
                self._insert_text(text[index : match.start()])
            self._handle_ansi(match.group(1), match.group(2))
            index = match.end()
        if index < len(text):
            self._insert_text(text[index:])
        terminal.configure(state="disabled")  # type: ignore[attr-defined]
        if full_frame:
            terminal.yview_moveto(0.0)  # type: ignore[attr-defined]
        else:
            terminal.see("end")  # type: ignore[attr-defined]
        render_ms = (perf_counter() - started) * 1000.0
        if render_ms >= self._config.slow_render_log_ms:
            LOGGER.info("desktop_tui_slow_render latency_ms=%.1f chars=%d", render_ms, len(text))

    def _insert_text(self, text: str) -> None:
        if not text or self._terminal is None:
            return
        self._terminal.insert("end", text, self._current_tags)  # type: ignore[attr-defined]

    def _handle_ansi(self, params: str, command: str) -> None:
        if command == "J" and "2" in params:
            if self._terminal is not None:
                self._terminal.delete("1.0", "end")  # type: ignore[attr-defined]
            self._current_tags = ()
            return
        if command == "m":
            self._apply_sgr(params)

    def _apply_sgr(self, params: str) -> None:
        codes = params.split(";") if params else ["0"]
        tags = set(self._current_tags)
        for code in codes:
            if code in {"", "0"}:
                tags.clear()
            elif code == "1":
                tags.add("bold")
            elif code == "2":
                tags.add("dim")
            elif code in COLOR_CODES:
                tags = {tag for tag in tags if not tag.startswith("fg_")}
                tags.add(COLOR_CODES[code])
        self._current_tags = tuple(sorted(tags))

    def _configure_terminal_tags(self, terminal: object) -> None:
        colors = {
            "fg_black": "#4b5563",
            "fg_red": "#fb7185",
            "fg_green": "#4ade80",
            "fg_yellow": "#facc15",
            "fg_blue": "#60a5fa",
            "fg_magenta": "#e879f9",
            "fg_cyan": "#22d3ee",
            "fg_white": "#e5e7eb",
            "dim": "#7c8798",
        }
        for tag, color in colors.items():
            terminal.tag_configure(tag, foreground=color)  # type: ignore[attr-defined]
        terminal.tag_configure("bold", font=("Cascadia Mono", self._config.font_size, "bold"))  # type: ignore[attr-defined]

    def _set_status(self, text: str) -> None:
        if self._status_var is not None:
            self._status_var.set(text)  # type: ignore[attr-defined]

    def _request_stop(self, *, close_after_stop: bool = False) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            if close_after_stop:
                self._destroy_root()
            return
        self._close_after_stop = self._close_after_stop or close_after_stop
        if self._stop_requested:
            return
        self._stop_requested = True
        self._shutdown_deadline = perf_counter() + self._config.graceful_shutdown_timeout_s
        self._set_status("stopping voice loop")
        self._set_stop_button("Stopping...", disabled=True)
        self._signal_child_stop()
        if self._root is not None:
            self._root.after(100, self._poll_shutdown)  # type: ignore[attr-defined]

    def _signal_child_stop(self) -> None:
        stop_file = self._config.stop_file
        if stop_file is not None:
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text("stop\n", encoding="utf-8")
                self._set_status("waiting for voice loop to exit")
                return
            except Exception as exc:
                self._set_status(f"stop signal failed: {exc}")

        if sys.platform == "win32":
            try:
                assert self._process is not None
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
                return
            except Exception:
                return
        if self._process is not None:
            self._process.terminate()

    def _poll_shutdown(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            self._finish_stop()
            return
        deadline = self._shutdown_deadline
        if deadline is not None and perf_counter() >= deadline:
            self._set_status("forcing voice loop to stop")
            _terminate_process_tree(process.pid)
            self._shutdown_deadline = perf_counter() + 2.0
        if self._root is not None:
            self._root.after(100, self._poll_shutdown)  # type: ignore[attr-defined]

    def _finish_stop(self) -> None:
        self._set_status("voice loop stopped")
        self._set_stop_button("Stopped", disabled=True)
        self._delete_stop_file()
        self._delete_wake_event_file()
        self._delete_window_command_file()
        if self._close_after_stop:
            self._destroy_root()

    def _set_stop_button(self, text: str, *, disabled: bool) -> None:
        if self._stop_button is None:
            return
        state = "disabled" if disabled else "normal"
        self._stop_button.configure(text=text, state=state)  # type: ignore[attr-defined]

    def _delete_stop_file(self) -> None:
        stop_file = self._config.stop_file
        if stop_file is None:
            return
        try:
            stop_file.unlink(missing_ok=True)
        except Exception:
            return

    def _delete_wake_event_file(self) -> None:
        wake_event_file = self._config.wake_event_file
        if wake_event_file is None:
            return
        try:
            wake_event_file.unlink(missing_ok=True)
        except Exception:
            return

    def _delete_window_command_file(self) -> None:
        window_command_file = self._config.window_command_file
        if window_command_file is None:
            return
        try:
            window_command_file.unlink(missing_ok=True)
        except Exception:
            return

    def _poll_wake_event(self) -> None:
        wake_event_file = self._config.wake_event_file
        if wake_event_file is not None:
            try:
                stat = wake_event_file.stat()
            except FileNotFoundError:
                pass
            except Exception as exc:
                LOGGER.warning("desktop_wake_event_poll_failed error=%s", exc)
            else:
                if self._last_wake_event_mtime_ns != stat.st_mtime_ns:
                    self._last_wake_event_mtime_ns = stat.st_mtime_ns
                    if self._show_window():
                        LOGGER.info("desktop_window_shown reason=wake_event")
                    else:
                        LOGGER.info("desktop_window_show_skipped reason=wake_event")
        if self._root is not None:
            self._root.after(120, self._poll_wake_event)  # type: ignore[attr-defined]

    def _poll_window_command(self) -> None:
        window_command_file = self._config.window_command_file
        if window_command_file is not None:
            try:
                stat = window_command_file.stat()
            except FileNotFoundError:
                pass
            except Exception as exc:
                LOGGER.warning("desktop_window_command_poll_failed error=%s", exc)
            else:
                if self._last_window_command_mtime_ns != stat.st_mtime_ns:
                    self._last_window_command_mtime_ns = stat.st_mtime_ns
                    try:
                        command = window_command_file.read_text(encoding="utf-8").strip()
                    except Exception as exc:
                        LOGGER.warning("desktop_window_command_read_failed error=%s", exc)
                    else:
                        self._handle_window_command(command)
        if self._root is not None:
            self._root.after(120, self._poll_window_command)  # type: ignore[attr-defined]

    def _handle_window_command(self, command: str) -> None:
        command = command.lstrip("\ufeff").strip()
        if command == "hide_to_tray":
            self._hide_to_tray()
            LOGGER.info("desktop_window_hidden reason=window_command")
        elif command:
            LOGGER.warning("desktop_window_command_unknown command=%r", command)
        self._delete_window_command_file()

    def _hide_to_tray(self) -> None:
        root = self._root
        if root is None:
            return
        self._set_status("hidden to tray")
        self._last_show_monotonic = 0.0
        root.withdraw()  # type: ignore[attr-defined]

    def _show_window(self, *, force: bool = False) -> bool:
        root = self._root
        if root is None:
            return False
        now = perf_counter()
        if (
            not force
            and _root_window_visible(root)
            and now - self._last_show_monotonic < WAKE_SHOW_DEBOUNCE_S
        ):
            return False
        self._last_show_monotonic = now
        root.deiconify()  # type: ignore[attr-defined]
        root.state("normal")  # type: ignore[attr-defined]
        self._restore_or_center_window(reason="show")
        root.lift()  # type: ignore[attr-defined]
        try:
            root.focus_force()  # type: ignore[attr-defined]
            root.attributes("-topmost", True)  # type: ignore[attr-defined]
            root.after(250, lambda: root.attributes("-topmost", False))  # type: ignore[attr-defined]
        except Exception:
            return True
        return True

    def _stop_process_now(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()

    def _on_close(self) -> None:
        self._request_stop(close_after_stop=True)

    def _destroy_root(self) -> None:
        if self._root is not None:
            tray_icon = self._tray_icon
            self._tray_icon = None
            if tray_icon is not None:
                tray_icon.stop()  # type: ignore[attr-defined]
            root = self._root
            self._root = None
            root.destroy()  # type: ignore[attr-defined]


def _terminate_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _latest_terminal_frame(text: str) -> str:
    latest_frame_start = text.rfind(FULL_FRAME_CLEAR)
    if latest_frame_start <= 0:
        return text
    return text[latest_frame_start:]


def _estimate_tui_viewport(
    window_width: int,
    window_height: int,
    font_size: int,
) -> tuple[int, int]:
    char_width = max(6.0, font_size * 0.82)
    line_height = max(12.0, font_size * 1.55)
    columns = int((window_width - 120) / char_width)
    rows = int((window_height - 190) / line_height)
    return max(60, min(112, columns)), max(24, min(34, rows))


def _measure_tui_viewport(
    terminal: object | None,
    terminal_font: object | None,
    *,
    fallback_columns: int,
    fallback_rows: int,
) -> tuple[int, int]:
    if terminal is None or terminal_font is None:
        return fallback_columns, fallback_rows
    try:
        terminal.update_idletasks()  # type: ignore[attr-defined]
        width_px = int(terminal.winfo_width())  # type: ignore[attr-defined]
        height_px = int(terminal.winfo_height())  # type: ignore[attr-defined]
        padx = _widget_int(terminal.cget("padx"))  # type: ignore[attr-defined]
        pady = _widget_int(terminal.cget("pady"))  # type: ignore[attr-defined]
        char_width = max(1, int(terminal_font.measure("M")))  # type: ignore[attr-defined]
        line_height = max(1, int(terminal_font.metrics("linespace")))  # type: ignore[attr-defined]
    except Exception as exc:
        LOGGER.debug("desktop_tui_measure_failed error=%s", exc)
        return fallback_columns, fallback_rows

    usable_width = max(1, width_px - (padx * 2))
    usable_height = max(1, height_px - (pady * 2))
    columns = max(60, min(118, int(usable_width / char_width)))
    rows = max(24, min(40, int(usable_height / line_height)))
    return columns, rows


def _widget_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_geometry(placement: WindowPlacement) -> str:
    return f"{placement.width}x{placement.height}+{placement.x}+{placement.y}"


def _center_placement(
    *,
    width: int,
    height: int,
    screen_width: int,
    screen_height: int,
) -> WindowPlacement:
    return WindowPlacement(
        x=max(0, (screen_width - width) // 2),
        y=max(0, (screen_height - height) // 2),
        width=width,
        height=height,
        manual=False,
    )


def _load_window_placement(
    path: Path | None,
    *,
    expected_width: int,
    expected_height: int,
    screen_width: int,
    screen_height: int,
) -> WindowPlacement | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        placement = WindowPlacement(
            x=int(raw["x"]),
            y=int(raw["y"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            manual=bool(raw.get("manual", True)),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if placement.width != expected_width or placement.height != expected_height:
        return None
    return _clamp_placement_to_screen(
        placement,
        screen_width=screen_width,
        screen_height=screen_height,
    )


def _save_window_placement(path: Path | None, placement: WindowPlacement) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "x": placement.x,
                    "y": placement.y,
                    "width": placement.width,
                    "height": placement.height,
                    "manual": placement.manual,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _clamp_placement_to_screen(
    placement: WindowPlacement,
    *,
    screen_width: int,
    screen_height: int,
) -> WindowPlacement:
    max_x = max(0, screen_width - placement.width)
    max_y = max(0, screen_height - placement.height)
    return WindowPlacement(
        x=max(0, min(placement.x, max_x)),
        y=max(0, min(placement.y, max_y)),
        width=placement.width,
        height=placement.height,
        manual=placement.manual,
    )


def _root_window_visible(root: object) -> bool:
    try:
        return str(root.state()) not in {"withdrawn", "iconic"}  # type: ignore[attr-defined]
    except Exception:
        return True
