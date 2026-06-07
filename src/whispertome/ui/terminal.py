from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import textwrap
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from time import time as wall_time
from typing import TextIO

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
MAGENTA = "\x1b[35m"
BLUE = "\x1b[34m"
WHITE = "\x1b[37m"
TUI_WIDTH_ENV = "WHISPERTOME_TUI_WIDTH"
TUI_HEIGHT_ENV = "WHISPERTOME_TUI_HEIGHT"
TUI_FPS_ENV = "WHISPERTOME_TUI_FPS"
TUI_IDLE_FPS_ENV = "WHISPERTOME_TUI_IDLE_FPS"
# When the rendered frame is byte-identical to the last one and nothing is animating, the loop
# backs off to this rate. Real state changes wake it instantly (event-driven), so a slow deep-idle
# tick costs nothing in reactivity — it just stops re-computing a frame that can't have changed.
TUI_DEEP_IDLE_FPS_ENV = "WHISPERTOME_TUI_DEEP_IDLE_FPS"
LOGGER = logging.getLogger(__name__)
DISPLAY_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2026": "...",
        "\u2022": "*",
        "\u00b0": " deg ",
        "\u00d7": "x",
        "\u00f7": "/",
    }
)


class NullVoiceUi:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def status(self, text: str, detail: str = "") -> None:
        return

    def boot(self, phase: str, detail: str = "", progress: float = 0.0) -> None:
        return

    def boot_complete(self) -> None:
        return

    def line(self, text: str) -> None:
        return

    def user_text(self, text: str) -> None:
        return

    def assistant_text(self, text: str) -> None:
        return

    def code_block(self, language: str, code: str) -> None:
        return

    def clear_code_block(self) -> None:
        return

    def activity(self, value: float) -> None:
        return

    def next_event(self, title: str | None, epoch: float | None) -> None:
        return


@dataclass
class TerminalUiState:
    max_lines: int = 10
    status_text: str = "starting"
    status_detail: str = ""
    user: str = ""
    assistant: str = ""
    code_language: str = ""
    code_text: str = ""
    activity_value: float = 0.0
    boot_active: bool = False
    boot_phase: str = "initializing"
    boot_detail: str = ""
    boot_progress: float = 0.0
    next_event_title: str = ""
    next_event_epoch: float | None = None
    stream_lines: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.max_lines = max(1, min(10, self.max_lines))
        self.stream_lines = deque(self.stream_lines, maxlen=self.max_lines)

    def set_status(self, text: str, detail: str = "") -> None:
        self.status_text = display_text(text).strip() or "idle"
        self.status_detail = display_text(detail).strip()
        self.add_line(
            self.status_text
            if not self.status_detail
            else f"{self.status_text} - {self.status_detail}"
        )

    def set_boot(self, phase: str, detail: str = "", progress: float = 0.0) -> None:
        self.boot_active = True
        self.boot_phase = display_text(phase).strip() or "initializing"
        self.boot_detail = display_text(detail).strip()
        self.boot_progress = max(0.0, min(1.0, float(progress)))
        line_detail = f" - {self.boot_detail}" if self.boot_detail else ""
        self.add_line(f"init {self.boot_phase}{line_detail}")

    def clear_boot(self) -> None:
        self.boot_active = False
        self.boot_phase = ""
        self.boot_detail = ""
        self.boot_progress = 0.0

    def set_user(self, text: str) -> None:
        self.user = display_text(text).strip()

    def set_assistant(self, text: str) -> None:
        self.assistant = display_text(text).strip()

    def set_code_block(self, language: str, code: str) -> None:
        self.code_language = display_text(language).strip() or "code"
        self.code_text = display_text(code).strip("\r\n")

    def clear_code_block(self) -> None:
        self.code_language = ""
        self.code_text = ""

    def set_activity(self, value: float) -> None:
        self.activity_value = max(0.0, min(1.0, float(value)))

    def set_next_event(self, title: str | None, epoch: float | None) -> None:
        self.next_event_title = display_text(title or "").strip()
        self.next_event_epoch = epoch if self.next_event_title else None

    def add_line(self, text: str) -> None:
        line = " ".join(display_text(text).strip().split())
        if line:
            self.stream_lines.append(line)

    def copy(self) -> TerminalUiState:
        return TerminalUiState(
            max_lines=self.max_lines,
            status_text=self.status_text,
            status_detail=self.status_detail,
            user=self.user,
            assistant=self.assistant,
            code_language=self.code_language,
            code_text=self.code_text,
            activity_value=self.activity_value,
            boot_active=self.boot_active,
            boot_phase=self.boot_phase,
            boot_detail=self.boot_detail,
            boot_progress=self.boot_progress,
            next_event_title=self.next_event_title,
            next_event_epoch=self.next_event_epoch,
            stream_lines=deque(self.stream_lines, maxlen=self.max_lines),
        )


class TerminalVoiceUi:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        fps: float | None = None,
        idle_fps: float | None = None,
        max_lines: int = 10,
    ) -> None:
        self._stream = stream or sys.stdout
        _configure_stream_for_display(self._stream)
        # Reactions are event-driven (state changes wake the renderer instantly), so the
        # frame rate only governs free-running animation smoothness. Low rates save real
        # energy on a fanless device. Both are env-tunable.
        active = fps if fps is not None else _env_float(TUI_FPS_ENV, 5.0)
        idle = idle_fps if idle_fps is not None else _env_float(TUI_IDLE_FPS_ENV, 2.0)
        deep = _env_float(TUI_DEEP_IDLE_FPS_ENV, 0.5)
        self._fps = max(1.0, active)
        self._idle_fps = max(0.5, min(idle, self._fps))
        self._deep_idle_fps = max(0.1, min(deep, self._idle_fps))
        self._state = TerminalUiState(max_lines=max_lines)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._render_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = 0
        self._render_failures = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._render_event.set()
        self._write_stream("\x1b[?25l")
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._render_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._write_stream("\x1b[?25h" + RESET + "\n")

    def status(self, text: str, detail: str = "") -> None:
        with self._lock:
            self._state.set_status(text, detail)
        self._request_render()

    def boot(self, phase: str, detail: str = "", progress: float = 0.0) -> None:
        with self._lock:
            self._state.set_boot(phase, detail, progress)
        self._request_render()

    def boot_complete(self) -> None:
        with self._lock:
            self._state.clear_boot()
        self._request_render()

    def line(self, text: str) -> None:
        with self._lock:
            self._state.add_line(text)
        self._request_render()

    def user_text(self, text: str) -> None:
        with self._lock:
            self._state.set_user(text)
        self._request_render()

    def assistant_text(self, text: str) -> None:
        with self._lock:
            self._state.set_assistant(text)
        self._request_render()

    def code_block(self, language: str, code: str) -> None:
        with self._lock:
            self._state.set_code_block(language, code)
            self._state.add_line(f"rendered {language or 'code'} block")
        self._request_render()

    def clear_code_block(self) -> None:
        with self._lock:
            self._state.clear_code_block()
        self._request_render()

    def activity(self, value: float) -> None:
        with self._lock:
            self._state.set_activity(value)
        self._request_render()

    def next_event(self, title: str | None, epoch: float | None) -> None:
        with self._lock:
            self._state.set_next_event(title, epoch)
        self._request_render()

    def _request_render(self) -> None:
        self._render_event.set()

    def _render_loop(self) -> None:
        full_interval = 1.0 / self._fps
        idle_interval = 1.0 / self._idle_fps
        deep_idle_interval = 1.0 / self._deep_idle_fps
        interval = full_interval
        last_rendered = 0.0
        last_frame: str | None = None
        while not self._stop_event.is_set():
            timeout = max(0.0, interval - (perf_counter() - last_rendered))
            self._render_event.wait(timeout)
            if self._stop_event.is_set():
                break
            # Clear immediately before snapshotting: any state change (and its set()) that
            # lands after this point survives to trigger the next iteration promptly, so a
            # change is never delayed by the (now slower) idle interval.
            self._render_event.clear()
            try:
                with self._lock:
                    snapshot = self._state.copy()
                width, height = terminal_size()
                frame = render_frame(snapshot, width=width, height=height, frame=self._frame)
                animating = _is_animation_active(snapshot)
                if frame != last_frame:
                    # Only paint when the frame actually changed. Skipping byte-identical
                    # frames avoids the terminal write (and, in desktop mode, the whole
                    # ANSI-parse + widget-insert + drain cascade) every idle tick.
                    self._write_stream("\x1b[H\x1b[2J" + frame)
                    last_frame = frame
                    interval = full_interval if animating else idle_interval
                else:
                    # Nothing changed. Keep ticking for animation; otherwise sink to the
                    # deep-idle rate — a real change wakes us instantly via _render_event.
                    interval = full_interval if animating else deep_idle_interval
                # Advance the animation clock only while something is animating. Freezing it
                # when idle makes the scene byte-stable so the dedup above can actually skip —
                # and stops the slow idle "breathing" from spending cycles for no visible gain.
                if animating:
                    self._frame += 1
                self._render_failures = 0
            except Exception:
                self._render_failures += 1
                LOGGER.exception("terminal_tui_render_failed count=%d", self._render_failures)
                self._write_stream(render_error_frame(*terminal_size()))
            last_rendered = perf_counter()

    def _write_stream(self, text: str) -> None:
        try:
            self._stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe = text.encode(encoding, errors="replace").decode(
                encoding,
                errors="replace",
            )
            self._stream.write(safe)
        self._stream.flush()


def terminal_size() -> tuple[int, int]:
    fallback = shutil.get_terminal_size((88, 29))
    width = _env_int(TUI_WIDTH_ENV, fallback.columns)
    height = _env_int(TUI_HEIGHT_ENV, fallback.lines)
    return width, height


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _configure_stream_for_display(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(errors="replace")
    except (OSError, ValueError):
        return


def display_text(text: str) -> str:
    normalized = str(text).translate(DISPLAY_TRANSLATION)
    return normalized.encode("ascii", errors="replace").decode("ascii")


def render_error_frame(width: int, height: int) -> str:
    width = max(60, width)
    height = max(8, height)
    line = border_line(min(width, 96), "-")
    body = [
        line,
        wrap_panel_line(
            f"{YELLOW}STATUS:{RESET} TUI render recovered",
            min(width, 96),
            YELLOW,
        ),
        wrap_panel_line(
            (
                f"{DIM}The voice loop is still running. "
                f"Check logs for terminal_tui_render_failed.{RESET}"
            ),
            min(width, 96),
            YELLOW,
        ),
        line,
    ]
    while len(body) < min(height, 8):
        body.append("")
    return "\x1b[H\x1b[2J" + "\n".join(body[:height]) + "\n"


def render_frame(state: TerminalUiState, *, width: int, height: int, frame: int) -> str:
    width = max(60, width)
    available_height = max(24, height)
    if state.boot_active:
        return render_boot_frame(state, width=width, height=available_height, frame=frame)

    frame_width = min(width, 118)
    body_height = max(13, available_height - 6)
    compact = frame_width <= 92 or available_height <= 30
    left_width = 29 if compact else 34
    gap = 2
    right_width = max(30, frame_width - left_width - gap)
    lines: list[str] = []
    lines.extend(render_runtime_header(state, frame_width))
    lines.extend(
        render_runtime_columns(
            state,
            left_width=left_width,
            right_width=right_width,
            gap=gap,
            height=body_height,
            frame=frame,
        )
    )
    lines.extend(render_runtime_footer(state, frame_width))
    padded = [center(line, width) for line in lines[:available_height]]
    return "\n".join(padded) + "\n"


def render_boot_frame(
    state: TerminalUiState,
    *,
    width: int,
    height: int,
    frame: int,
) -> str:
    width = max(60, width)
    available_height = max(24, height)
    content_width = min(width - 4, 104)
    phase = state.boot_phase or "initializing"
    detail = state.boot_detail or "loading local voice stack"
    progress = max(0.0, min(1.0, state.boot_progress))
    shimmer = ["-", "\\", "|", "/"][frame % 4]

    frame_width = min(width, 118)
    inner = frame_width - 4
    body: list[str] = []
    body.append(
        pad_visible(
            f"{BOLD}{CYAN}WHISPER TO ME // NPU BOOT{RESET}"
            + " "
            + f"{DIM}STT -> LLM -> TOOLS -> TTS{RESET}",
            inner,
        )
    )
    body.append(pad_visible(f"{CYAN}BOOTSTRAPPING MODULES{RESET}", inner))
    body.extend(
        center(line, inner)
        for line in render_boot_lattice(
            min(70, content_width),
            11 if available_height < 32 else 13,
            frame=frame,
            progress=progress,
        )
    )
    body.append(pad_visible(f"{BOLD}{WHITE}{phase.upper()}{RESET} {DIM}{detail}{RESET}", inner))
    body.append(render_progress_bar(inner, progress, frame=frame))
    body.extend(render_boot_modules(progress, inner))
    if state.stream_lines and available_height >= 32:
        for item in list(state.stream_lines)[-3:]:
            body.append(pad_visible(f"{DIM}> {fit_plain(item, inner - 2)}{RESET}", inner))

    usable_body = max(1, available_height - 2)
    status_line = pad_visible(
        f"{YELLOW}STATUS:{RESET} connecting to model... {CYAN}{shimmer}{RESET}",
        inner,
    )
    body = body[:usable_body]
    while len(body) < usable_body - 1:
        wave = render_waveform(inner, frame=frame + len(body), activity=progress)
        body.append(f"{CYAN}{wave}{RESET}")
    if len(body) < usable_body:
        body.append(status_line)
    else:
        body[-1] = status_line

    top = border_line(frame_width, "-")
    rendered = [top]
    rendered.extend(wrap_panel_line(line, frame_width, CYAN) for line in body)
    rendered.append(top)
    return "\n".join(rendered[:available_height]) + "\n"


def render_boot_lattice(
    width: int,
    height: int,
    *,
    frame: int,
    progress: float,
) -> list[str]:
    width = max(38, width)
    height = max(9, height)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    outer = min(width / 4.3, height / 1.85)
    phase = frame * 0.11
    progress_angle = progress * math.tau

    for y in range(height):
        for x in range(width):
            dx = (x - cx) / 2.1
            dy = y - cy
            radius = math.sqrt(dx * dx + dy * dy)
            angle = (math.atan2(dy, dx) + math.tau) % math.tau
            moving_angle = (angle + phase) % math.tau
            ring = min(
                abs(radius - outer),
                abs(radius - outer * 0.66),
                abs(radius - outer * 0.34),
            )
            spoke = abs(math.sin(moving_angle * 4.0)) < 0.045 and radius < outer * 0.98
            arc = abs(radius - outer) < 0.16 and angle <= progress_angle

            if arc:
                grid[y][x] = "#"
            elif ring < 0.10:
                grid[y][x] = "+" if (x + y + frame) % 5 == 0 else "."
            elif spoke:
                grid[y][x] = "*"

    put(grid, round(cx), round(cy), "@")
    put(grid, round(cx) - 1, round(cy), "[")
    put(grid, round(cx) + 1, round(cy), "]")
    scan_y = int((frame * 0.35) % height)
    for x in range(3, width - 3, 5):
        if grid[scan_y][x] == " ":
            grid[scan_y][x] = "."
    return [CYAN + "".join(row).rstrip() + RESET for row in grid]


def render_progress_bar(width: int, progress: float, *, frame: int) -> str:
    progress = max(0.0, min(1.0, progress))
    inner = max(24, min(64, width - 18))
    filled = min(inner, max(0, round(inner * progress)))
    head = ">" if frame % 2 == 0 else "="
    if filled <= 0:
        body = "." * inner
    elif filled >= inner:
        body = "#" * inner
    else:
        body = "#" * (filled - 1) + head + "." * (inner - filled)
    percent = f"{round(progress * 100):3d}%"
    return f"{GREEN}[{body}]{RESET} {BOLD}{percent}{RESET}"


def render_boot_modules(progress: float, width: int) -> list[str]:
    modules = [
        ("CONFIG", 0.08, "env and runtime policy"),
        ("STT", 0.45, "QNN Whisper session"),
        ("TTS", 0.78, "Kokoro voice session"),
        ("AUDIO", 0.92, "mic, speaker, session ducking"),
    ]
    columns = []
    for name, threshold, detail in modules:
        if progress >= threshold:
            marker = GREEN + "ONLINE " + RESET
        elif progress >= max(0.0, threshold - 0.24):
            marker = YELLOW + "SYNC   " + RESET
        else:
            marker = DIM + "WAIT   " + RESET
        columns.append(f"{marker}{BOLD}{name:<6}{RESET} {DIM}{detail}{RESET}")

    if width >= 92:
        return [
            f"{columns[0]}    {columns[1]}",
            f"{columns[2]}    {columns[3]}",
        ]
    return columns


def format_next_event(state: TerminalUiState) -> str:
    """Compact 'NEXT <title> in Xm' agenda chip; counts down live (computed per frame)."""
    title = (state.next_event_title or "").strip()
    if not title or state.next_event_epoch is None:
        return ""
    delta = state.next_event_epoch - wall_time()
    if delta < 0:
        rel = "now"
    elif delta < 60:
        rel = f"in {int(delta)}s"
    elif delta < 3600:
        rel = f"in {int(delta // 60)}m"
    elif delta < 86400:
        rel = f"in {int(delta // 3600)}h"
    else:
        rel = f"in {int(delta // 86400)}d"
    return f"NEXT {fit_plain(title, 16)} {rel}"


def render_runtime_header(state: TerminalUiState, width: int) -> list[str]:
    status = fit_plain(state.status_text.upper(), 22)
    # Minute resolution (not seconds): keeps the idle frame byte-identical for up to a minute so
    # the dedup/deep-idle render path stops repainting a static screen. The live "in Xs" agenda
    # chip still ticks per-second when an event is imminent.
    clock = datetime.now().strftime("%H:%M")
    pipeline = "STT -> LLM -> TOOLS -> TTS"
    if width < 96:
        content = (
            f"{BOLD}{CYAN}WHISPER TO ME{RESET}"
            f"  |  {CYAN}{pipeline}{RESET}"
            f"  |  {color_for_status(state.status_text)}{status}{RESET}"
        )
    else:
        content = (
            f"{BOLD}{CYAN}WHISPER TO ME{RESET}"
            f"  |  {GREEN}SESSION ACTIVE{RESET}"
            f"  |  {CYAN}{pipeline}{RESET}"
            f"  |  {color_for_status(state.status_text)}{status}{RESET}"
        )
    if width >= 76:
        agenda = format_next_event(state) if width >= 96 else ""
        right_plain = f"{agenda}   {clock}" if agenda else clock
        right_colored = (
            f"{YELLOW}{agenda}{RESET}   {DIM}{clock}{RESET}" if agenda else f"{DIM}{clock}{RESET}"
        )
        content = pad_visible(content, max(0, width - len(right_plain) - 6)) + right_colored
    return [border_line(width, "-"), wrap_panel_line(content, width, CYAN), border_line(width, "-")]


def render_runtime_columns(
    state: TerminalUiState,
    *,
    left_width: int,
    right_width: int,
    gap: int,
    height: int,
    frame: int,
) -> list[str]:
    transcript_height = 7 if height >= 20 else 6
    if state.code_text:
        activity_height = max(7, height - transcript_height - 1)
    else:
        activity_height = max(8, height - transcript_height - 1)

    left = render_pipeline_panel(state, left_width, height=height, frame=frame)
    right = render_live_transcript_panel(
        state,
        right_width,
        height=transcript_height,
        frame=frame,
    )
    right.extend([" " * right_width])
    right.extend(
        render_activity_panel(
            state,
            right_width,
            height=activity_height,
            frame=frame,
        )
    )
    left = fit_block_height(left, height, left_width)
    right = fit_block_height(right, height, right_width)
    return [
        left_line + (" " * gap) + right_line
        for left_line, right_line in zip(left, right, strict=False)
    ]


def render_pipeline_panel(
    state: TerminalUiState,
    width: int,
    *,
    height: int,
    frame: int,
) -> list[str]:
    stage = active_stage(state.status_text)
    steps = [
        ("STT", "speech to text"),
        ("LLM", "thinking"),
        ("TOOLS", "function calling"),
        ("TTS", "text to speech"),
    ]
    body: list[str] = []
    polygon_height = 7 if height <= 24 else 9
    body.extend(
        render_polygon(
            max(20, width - 6),
            polygon_height,
            frame=frame,
            activity=max(state.activity_value, status_activity_boost(state.status_text)),
            status=state.status_text,
        )
    )
    body.append("")
    compact = width <= 30 or height <= 24
    for index, (name, detail) in enumerate(steps):
        state_label = step_state(name, stage)
        marker = "[>]" if state_label == "active" else "[x]" if state_label == "complete" else "[-]"
        color = step_color(name)
        pulse = "..." if state_label == "active" and frame % 2 else "   "
        body.append(f"{color}{index + 1}) {name:<5}{RESET} {marker} {step_copy(state_label)}")
        if not compact:
            body.append(f"   {DIM}{detail}{RESET} {color}{pulse}{RESET}")
        if index < len(steps) - 1:
            body.append(f"{DIM}        v{RESET}")

    panel_height = max(8, height)
    return render_panel("ACTIVE STEP / PIPELINE", body, width, panel_height, color=CYAN)


def render_live_transcript_panel(
    state: TerminalUiState,
    width: int,
    *,
    height: int,
    frame: int,
) -> list[str]:
    activity = max(state.activity_value, status_activity_boost(state.status_text))
    waveform = render_waveform(width - 6, frame=frame, activity=activity)
    body = [
        f"{CYAN}{waveform}{RESET}",
        f"{GREEN}{fit_plain(state.user or '...', width - 6)}{RESET}",
    ]
    wrapped = textwrap.wrap(state.user or "", width=max(20, width - 6))
    if len(wrapped) > 1:
        body.extend(f"{GREEN}{line}{RESET}" for line in wrapped[1:3])
    return render_panel("LIVE TRANSCRIPT / INPUT", body, width, height, color=GREEN)


def render_activity_panel(
    state: TerminalUiState,
    width: int,
    *,
    height: int,
    frame: int,
) -> list[str]:
    if state.code_text:
        body = render_code_activity_body(state, width - 4, max(1, height - 3), frame=frame)
        return render_panel(
            f"{state.code_language.upper() or 'CODE'} VIEW / OUTPUT / SYSTEM STREAM",
            body,
            width,
            height,
            color=YELLOW,
        )

    body: list[str] = []
    assistant_lines = textwrap.wrap(state.assistant or "...", width=max(20, width - 6))[:4]
    body.extend(f"{MAGENTA}{line}{RESET}" for line in assistant_lines)
    if body:
        body.append(f"{DIM}{'-' * max(8, width - 6)}{RESET}")
    body.append(f"{YELLOW}SYSTEM STREAM{RESET}")

    stream_budget = max(1, height - len(body) - 4)
    stream_lines = stream_activity_lines(state, width - 6, stream_budget)
    body.extend(stream_lines)
    return render_panel("ASSISTANT / OUTPUT / TOOLS", body, width, height, color=MAGENTA)


def render_runtime_footer(state: TerminalUiState, width: int) -> list[str]:
    activity = max(state.activity_value, status_activity_boost(state.status_text))
    detail = state.status_detail or "ready"
    if width < 90:
        content = (
            f"{CYAN}STATUS:{RESET} {fit_plain(state.status_text.upper(), 18)}"
            f"  {CYAN}DETAIL:{RESET} {fit_plain(detail, 14)}"
            f"  {GREEN}CTRL+C to stop{RESET}"
        )
    else:
        content = (
            f"{CYAN}STATUS:{RESET} {fit_plain(state.status_text.upper(), 18)}"
            f"  {CYAN}DETAIL:{RESET} {fit_plain(detail, 24)}"
            f"  {CYAN}ACTIVITY:{RESET} {activity:0.2f}"
            f"  {GREEN}CTRL+C to stop{RESET}"
        )
    return [border_line(width, "-"), wrap_panel_line(content, width, CYAN), border_line(width, "-")]


def render_code_activity_body(
    state: TerminalUiState,
    inner_width: int,
    body_height: int,
    *,
    frame: int,
) -> list[str]:
    body: list[str] = []
    if state.assistant:
        body.extend(
            f"{MAGENTA}{line}{RESET}"
            for line in textwrap.wrap(state.assistant, width=inner_width)[:2]
        )
        body.append(f"{DIM}{'-' * max(8, inner_width)}{RESET}")

    code_lines = [line.expandtabs(2) for line in state.code_text.splitlines()] or [""]
    remaining = max(1, body_height - len(body))
    code_width = max(8, inner_width - 4)
    max_offset = max(0, len(code_lines) - remaining)
    offset = 0
    visible = code_lines[offset : offset + remaining]
    line_number = offset + 1
    for line in visible:
        prefix = f"{line_number:>2} "
        body.append(f"{YELLOW}{prefix}{RESET}{fit_code_line(line, code_width)}")
        line_number += 1
    if max_offset:
        end_line = min(offset + remaining, len(code_lines))
        body.append(f"{DIM}{offset + 1}-{end_line}/{len(code_lines)}{RESET}")
    return body[:body_height]


def stream_activity_lines(state: TerminalUiState, width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    for item in list(state.stream_lines)[-max_lines:]:
        wrapped = textwrap.wrap(item, width=max(16, width - 2), max_lines=2, placeholder="...")
        for line in wrapped:
            lines.append(f"{DIM}> {line}{RESET}")
    if not lines:
        lines.append(f"{DIM}> waiting for events{RESET}")
    return lines[-max_lines:]


def active_stage(status: str) -> str:
    lowered = status.lower()
    if "tts" in lowered or "play" in lowered or "speak" in lowered:
        return "TTS"
    if "tool" in lowered or "powershell" in lowered or "capture" in lowered:
        return "TOOLS"
    if (
        "openai" in lowered
        or "cerebras" in lowered
        or "contact" in lowered
        or "llm" in lowered
        or "model" in lowered
    ):
        return "LLM"
    return "STT"


def step_state(name: str, active: str) -> str:
    order = ["STT", "LLM", "TOOLS", "TTS"]
    current = order.index(active) if active in order else 0
    index = order.index(name)
    if index < current:
        return "complete"
    if index == current:
        return "active"
    return "waiting"


def step_copy(value: str) -> str:
    if value == "complete":
        return "complete"
    if value == "active":
        return "running"
    return "waiting"


def step_color(name: str) -> str:
    if name == "STT":
        return CYAN
    if name in {"LLM", "TOOLS"}:
        return YELLOW
    return MAGENTA


def render_waveform(width: int, *, frame: int, activity: float) -> str:
    width = max(12, width)
    activity = max(0.32, min(1.0, activity + 0.25))
    chars = []
    for index in range(width):
        wave = abs(math.sin((index + frame * 0.8) * 0.45))
        ripple = abs(math.sin((index * 0.17) - frame * 0.22))
        value = (wave * 0.72 + ripple * 0.28) * activity
        if value > 0.72:
            chars.append("|")
        elif value > 0.42:
            chars.append(":")
        elif value > 0.20:
            chars.append(".")
        else:
            chars.append(" ")
    return "".join(chars).rstrip() or "."


def render_panel(
    title: str,
    body: list[str],
    width: int,
    height: int,
    *,
    color: str,
) -> list[str]:
    width = max(16, width)
    height = max(3, height)
    inner = width - 4
    body_budget = height - 3
    rendered = [border_line(width, "-")]
    rendered.append(wrap_panel_line(f"{color}{fit_plain(title, inner)}{RESET}", width, color))
    for line in body[:body_budget]:
        rendered.append(wrap_panel_line(line, width, color))
    while len(rendered) < height - 1:
        rendered.append(wrap_panel_line("", width, color))
    rendered.append(border_line(width, "-"))
    return rendered[:height]


def fit_block_height(lines: list[str], height: int, width: int) -> list[str]:
    if len(lines) > height:
        return lines[:height]
    return lines + [" " * width for _ in range(height - len(lines))]


def border_line(width: int, char: str = "-") -> str:
    return "+" + (char * max(0, width - 2)) + "+"


def wrap_panel_line(text: str, width: int, color: str) -> str:
    inner = width - 4
    return color + "| " + RESET + pad_visible(text, inner) + color + " |" + RESET


def pad_visible(text: str, width: int) -> str:
    visible = strip_ansi_len(text)
    if visible >= width:
        return trim_visible(text, width)
    return text + (" " * (width - visible))


def fit_plain(text: str, width: int) -> str:
    text = " ".join(text.strip().split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def fit_code_line(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def trim_visible(text: str, width: int) -> str:
    if width <= 0:
        return ""
    count = 0
    result: list[str] = []
    in_escape = False
    for char in text:
        if char == "\x1b":
            in_escape = True
            result.append(char)
            continue
        if in_escape:
            result.append(char)
            if char == "m":
                in_escape = False
            continue
        if count >= width:
            break
        result.append(char)
        count += 1
    return "".join(result) + (RESET if "\x1b[" in text else "")


def render_polygon(
    width: int,
    height: int,
    *,
    frame: int,
    activity: float,
    status: str,
) -> list[str]:
    width = max(22, width)
    height = max(5, height)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    base_radius = min(width / 2.6, height / 2.2)
    pulse = 1.0 + 0.22 * math.sin(frame * 0.28) + 0.45 * activity
    sides = 6
    angle_offset = frame * 0.075
    vertices = []
    inner_vertices = []
    for index in range(sides):
        angle = angle_offset + (math.tau * index / sides)
        radius = base_radius * pulse * (0.92 + 0.08 * math.sin(frame * 0.17 + index))
        x = cx + math.cos(angle) * radius * 1.85
        y = cy + math.sin(angle) * radius * 0.74
        vertices.append((x, y))
        inner_vertices.append((cx + (x - cx) * 0.48, cy + (y - cy) * 0.48))

    for start, end in pairwise_loop(vertices):
        draw_line(grid, start, end)
    for start, end in pairwise_loop(inner_vertices):
        draw_line(grid, start, end, char=".")
    for outer, inner in zip(vertices, inner_vertices, strict=False):
        draw_line(grid, outer, inner, char=".")

    for x, y in vertices:
        put(grid, round(x), round(y), "*")
    put(grid, round(cx), round(cy), status_marker(status))
    color = color_for_status(status)
    return [color + "".join(row).rstrip() + RESET for row in grid]


def draw_line(
    grid: list[list[str]],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    char: str | None = None,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) * 2))
    for step in range(steps + 1):
        ratio = step / steps
        x = x0 + (x1 - x0) * ratio
        y = y0 + (y1 - y0) * ratio
        if char is None:
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) > abs(dy) * 2:
                line_char = "-"
            elif abs(dy) > abs(dx):
                line_char = "|"
            elif dx * dy < 0:
                line_char = "/"
            else:
                line_char = "\\"
        else:
            line_char = char
        put(grid, round(x), round(y), line_char)


def put(grid: list[list[str]], x: int, y: int, char: str) -> None:
    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
        grid[y][x] = char


def pairwise_loop(points: list[tuple[float, float]]):
    for index, point in enumerate(points):
        yield point, points[(index + 1) % len(points)]


def render_text_row(
    left_title: str,
    left_text: str,
    right_title: str,
    right_text: str,
    width: int,
    *,
    body_lines: int = 3,
) -> list[str]:
    gap = 4
    column_width = max(20, (width - gap) // 2)
    left = render_box(left_title, left_text, column_width, color=GREEN, body_lines=body_lines)
    right = render_box(right_title, right_text, column_width, color=MAGENTA, body_lines=body_lines)
    return [
        left_line + (" " * gap) + right_line
        for left_line, right_line in zip(left, right, strict=False)
    ]


def render_box(
    title: str,
    text: str,
    width: int,
    *,
    color: str,
    body_lines: int = 3,
) -> list[str]:
    inner = width - 4
    body_lines = max(1, body_lines)
    wrapped = textwrap.wrap(text, width=inner, max_lines=body_lines, placeholder="...") or [""]
    wrapped = (wrapped + [""] * body_lines)[:body_lines]
    top = color + "+" + ("-" * (width - 2)) + "+" + RESET
    header = color + "| " + title.ljust(inner) + " |" + RESET
    body = [color + "| " + line.ljust(inner) + " |" + RESET for line in wrapped]
    bottom = color + "+" + ("-" * (width - 2)) + "+" + RESET
    return [top, header, *body, bottom]


def render_code_box(
    state: TerminalUiState,
    width: int,
    *,
    code_height: int,
    frame: int,
) -> list[str]:
    width = max(30, width)
    inner_width = width - 4
    viewport_height = max(1, code_height - 3)
    code_lines = state.code_text.splitlines() or [""]
    wrapped_lines = wrap_code_lines(code_lines, inner_width)
    max_offset = max(0, len(wrapped_lines) - viewport_height)
    offset = 0 if max_offset == 0 else (frame // 12) % (max_offset + 1)
    visible = wrapped_lines[offset : offset + viewport_height]
    title = f"{state.code_language.upper()} VIEW"
    if max_offset:
        end_line = min(offset + viewport_height, len(wrapped_lines))
        title = f"{title} {offset + 1}-{end_line}/{len(wrapped_lines)}"
    top = YELLOW + "+" + ("-" * (width - 2)) + "+" + RESET
    header = YELLOW + "| " + title[:inner_width].ljust(inner_width) + " |" + RESET
    body = [
        YELLOW + "| " + line[:inner_width].ljust(inner_width) + " |" + RESET
        for line in visible
    ]
    while len(body) < viewport_height:
        body.append(YELLOW + "| " + "".ljust(inner_width) + " |" + RESET)
    bottom = YELLOW + "+" + ("-" * (width - 2)) + "+" + RESET
    return [top, header, *body, bottom]


def wrap_code_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        expanded = line.expandtabs(2)
        if not expanded:
            wrapped.append("")
            continue
        while expanded:
            wrapped.append(expanded[:width])
            expanded = expanded[width:]
    return wrapped


def render_stream(
    state: TerminalUiState,
    width: int,
    *,
    max_height: int | None = None,
) -> list[str]:
    if max_height is not None and max_height <= 0:
        return []
    header = f"{BOLD}{WHITE}SYSTEM STREAM{RESET}"
    body_budget = None if max_height is None else max(0, max_height - 1)
    wrapped_lines: list[str] = []
    for item in state.stream_lines:
        for wrapped in textwrap.wrap(item, width=max(20, width - 4)):
            wrapped_lines.append(f"{DIM}> {wrapped}{RESET}")
    if not wrapped_lines:
        wrapped_lines.append(f"{DIM}> waiting for events{RESET}")
    if body_budget is not None:
        wrapped_lines = wrapped_lines[-body_budget:] if body_budget else []
    return [header, *wrapped_lines]


def center(text: str, width: int) -> str:
    visible = strip_ansi_len(text)
    if visible >= width:
        return text
    return " " * ((width - visible) // 2) + text


def strip_ansi_len(text: str) -> int:
    count = 0
    in_escape = False
    for char in text:
        if char == "\x1b":
            in_escape = True
            continue
        if in_escape:
            if char == "m":
                in_escape = False
            continue
        count += 1
    return count


def color_for_status(status: str) -> str:
    lowered = status.lower()
    if "listen" in lowered:
        return GREEN
    if (
        "wake" in lowered
        or "detected" in lowered
        or "interrupted" in lowered
        or "barge" in lowered
    ):
        return GREEN
    if "openai" in lowered or "contact" in lowered:
        return YELLOW
    if "tts" in lowered or "play" in lowered or "speak" in lowered:
        return MAGENTA
    return CYAN


def status_marker(status: str) -> str:
    lowered = status.lower()
    if "interrupted" in lowered or "barge" in lowered:
        return "!"
    if "listen" in lowered:
        return "L"
    if "wake" in lowered or "detected" in lowered:
        return "W"
    if "openai" in lowered or "contact" in lowered:
        return "O"
    if "tts" in lowered or "play" in lowered or "speak" in lowered:
        return "T"
    return "*"


def status_activity_boost(status: str) -> float:
    lowered = status.lower()
    if "interrupted" in lowered or "barge" in lowered:
        return 1.0
    if "wake" in lowered or "detected" in lowered:
        return 0.9
    if "transcribing" in lowered or "speech captured" in lowered:
        return 0.55
    return 0.0


def _is_animation_active(state: TerminalUiState) -> bool:
    """Whether the scene needs full-rate animation. Idle scenes render slowly to save
    energy; any of these conditions restores full FPS (and state changes always wake the
    renderer immediately regardless)."""
    return (
        state.boot_active
        or state.activity_value > 0.05
        or status_activity_boost(state.status_text) > 0.0
    )
