from __future__ import annotations

import math
import os
import shutil
import sys
import textwrap
import threading
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
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
    stream_lines: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.max_lines = max(1, min(10, self.max_lines))
        self.stream_lines = deque(self.stream_lines, maxlen=self.max_lines)

    def set_status(self, text: str, detail: str = "") -> None:
        self.status_text = text.strip() or "idle"
        self.status_detail = detail.strip()
        self.add_line(self.status_text if not detail else f"{self.status_text} - {detail}")

    def set_boot(self, phase: str, detail: str = "", progress: float = 0.0) -> None:
        self.boot_active = True
        self.boot_phase = phase.strip() or "initializing"
        self.boot_detail = detail.strip()
        self.boot_progress = max(0.0, min(1.0, float(progress)))
        line_detail = f" - {self.boot_detail}" if self.boot_detail else ""
        self.add_line(f"init {self.boot_phase}{line_detail}")

    def clear_boot(self) -> None:
        self.boot_active = False
        self.boot_phase = ""
        self.boot_detail = ""
        self.boot_progress = 0.0

    def set_user(self, text: str) -> None:
        self.user = text.strip()

    def set_assistant(self, text: str) -> None:
        self.assistant = text.strip()

    def set_code_block(self, language: str, code: str) -> None:
        self.code_language = language.strip() or "code"
        self.code_text = code.strip("\r\n")

    def clear_code_block(self) -> None:
        self.code_language = ""
        self.code_text = ""

    def set_activity(self, value: float) -> None:
        self.activity_value = max(0.0, min(1.0, float(value)))

    def add_line(self, text: str) -> None:
        line = " ".join(text.strip().split())
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
            stream_lines=deque(self.stream_lines, maxlen=self.max_lines),
        )


class TerminalVoiceUi:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        fps: float = 24.0,
        max_lines: int = 10,
    ) -> None:
        self._stream = stream or sys.stdout
        self._fps = max(4.0, fps)
        self._state = TerminalUiState(max_lines=max_lines)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._render_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._render_event.set()
        self._stream.write("\x1b[?25l")
        self._stream.flush()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._render_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._stream.write("\x1b[?25h" + RESET + "\n")
        self._stream.flush()

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

    def _request_render(self) -> None:
        self._render_event.set()

    def _render_loop(self) -> None:
        interval = 1.0 / self._fps
        last_rendered = 0.0
        while not self._stop_event.is_set():
            timeout = max(0.0, interval - (perf_counter() - last_rendered))
            self._render_event.wait(timeout)
            self._render_event.clear()
            if self._stop_event.is_set():
                break
            with self._lock:
                snapshot = self._state.copy()
            width, height = terminal_size()
            frame = render_frame(snapshot, width=width, height=height, frame=self._frame)
            self._stream.write("\x1b[H\x1b[2J" + frame)
            self._stream.flush()
            self._frame += 1
            last_rendered = perf_counter()


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


def render_frame(state: TerminalUiState, *, width: int, height: int, frame: int) -> str:
    width = max(60, width)
    available_height = max(24, height)
    if state.boot_active:
        return render_boot_frame(state, width=width, height=available_height, frame=frame)

    compact = width <= 92 or available_height <= 30
    content_width = min(width - 4, 96 if compact else 112)
    polygon_width = min(48 if compact else 58, content_width)
    has_code = bool(state.code_text)
    if has_code:
        polygon_height = 7 if compact else 11
    else:
        polygon_height = 9 if compact else 15 if available_height >= 30 else 11
    title = f"{BOLD}{CYAN}WHISPER TO ME{RESET}"
    status = color_for_status(state.status_text) + state.status_text.upper() + RESET
    detail = f" {DIM}{state.status_detail}{RESET}" if state.status_detail else ""
    text_body_lines = 2 if compact and has_code else 3

    lines: list[str] = []
    lines.append(center(title, width))
    lines.append(center(f"{status}{detail}", width))
    lines.append("")
    for polygon_line in render_polygon(
        polygon_width,
        polygon_height,
        frame=frame,
        activity=max(state.activity_value, status_activity_boost(state.status_text)),
        status=state.status_text,
    ):
        lines.append(center(polygon_line, width))
    lines.append("")
    lines.extend(
        render_text_row(
            "INPUT",
            state.user or "...",
            "OUTPUT",
            state.assistant or "...",
            content_width,
            body_lines=text_body_lines,
        )
    )
    lines.append("")
    if has_code:
        code_height = 5 if compact else 8 if available_height >= 34 else 6
        lines.extend(render_code_box(state, content_width, code_height=code_height, frame=frame))
        lines.append("")
    stream_budget = max(1, available_height - len(lines))
    lines.extend(render_stream(state, content_width, max_height=stream_budget))
    return "\n".join(lines[:available_height]) + "\n"


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

    lines: list[str] = []
    lines.append(center(f"{BOLD}{CYAN}WHISPER TO ME // NPU BOOT{RESET}", width))
    lines.append(center(f"{DIM}LOCAL VOICE RUNTIME HANDSHAKE {shimmer}{RESET}", width))
    lines.append("")
    lines.extend(
        center(line, width)
        for line in render_boot_lattice(
            min(70, content_width),
            13 if available_height >= 32 else 11,
            frame=frame,
            progress=progress,
        )
    )
    lines.append("")
    phase_line = f"{BOLD}{WHITE}{phase.upper()}{RESET} {DIM}{detail}{RESET}"
    lines.append(center(phase_line, width))
    lines.append(center(render_progress_bar(content_width, progress, frame=frame), width))
    lines.append("")
    lines.extend(center(line, width) for line in render_boot_modules(progress, content_width))
    if state.stream_lines and available_height >= 32:
        lines.append("")
        lines.append(center(f"{BOLD}{WHITE}BOOT STREAM{RESET}", width))
        for item in list(state.stream_lines)[-4:]:
            lines.append(center(f"{DIM}> {item}{RESET}", width))
    return "\n".join(lines[:available_height]) + "\n"


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


def render_polygon(
    width: int,
    height: int,
    *,
    frame: int,
    activity: float,
    status: str,
) -> list[str]:
    width = max(22, width)
    height = max(9, height)
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
        return BLUE
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
