from __future__ import annotations

import math
import shutil
import sys
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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


class NullVoiceUi:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def status(self, text: str, detail: str = "") -> None:
        return

    def line(self, text: str) -> None:
        return

    def user_text(self, text: str) -> None:
        return

    def assistant_text(self, text: str) -> None:
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
    activity_value: float = 0.0
    stream_lines: deque[str] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.max_lines = max(1, min(10, self.max_lines))
        self.stream_lines = deque(self.stream_lines, maxlen=self.max_lines)

    def set_status(self, text: str, detail: str = "") -> None:
        self.status_text = text.strip() or "idle"
        self.status_detail = detail.strip()
        self.add_line(self.status_text if not detail else f"{self.status_text} - {detail}")

    def set_user(self, text: str) -> None:
        self.user = text.strip()

    def set_assistant(self, text: str) -> None:
        self.assistant = text.strip()

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
            activity_value=self.activity_value,
            stream_lines=deque(self.stream_lines, maxlen=self.max_lines),
        )


class TerminalVoiceUi:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        fps: float = 12.0,
        max_lines: int = 10,
    ) -> None:
        self._stream = stream or sys.stdout
        self._fps = max(2.0, fps)
        self._state = TerminalUiState(max_lines=max_lines)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._stream.write("\x1b[?25l")
        self._stream.flush()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._stream.write("\x1b[?25h" + RESET + "\n")
        self._stream.flush()

    def status(self, text: str, detail: str = "") -> None:
        with self._lock:
            self._state.set_status(text, detail)

    def line(self, text: str) -> None:
        with self._lock:
            self._state.add_line(text)

    def user_text(self, text: str) -> None:
        with self._lock:
            self._state.set_user(text)

    def assistant_text(self, text: str) -> None:
        with self._lock:
            self._state.set_assistant(text)

    def activity(self, value: float) -> None:
        with self._lock:
            self._state.set_activity(value)

    def _render_loop(self) -> None:
        interval = 1.0 / self._fps
        while not self._stop_event.is_set():
            with self._lock:
                snapshot = self._state.copy()
            width, height = shutil.get_terminal_size((100, 34))
            frame = render_frame(snapshot, width=width, height=height, frame=self._frame)
            self._stream.write("\x1b[H\x1b[2J" + frame)
            self._stream.flush()
            self._frame += 1
            time.sleep(interval)


def render_frame(state: TerminalUiState, *, width: int, height: int, frame: int) -> str:
    width = max(60, width)
    available_height = max(24, height)
    content_width = min(width - 4, 112)
    polygon_width = min(58, content_width)
    polygon_height = 15 if available_height >= 30 else 11
    title = f"{BOLD}{CYAN}WHISPER TO ME{RESET}"
    status = color_for_status(state.status_text) + state.status_text.upper() + RESET
    detail = f" {DIM}{state.status_detail}{RESET}" if state.status_detail else ""

    lines: list[str] = []
    lines.append(center(title, width))
    lines.append(center(f"{status}{detail}", width))
    lines.append("")
    for polygon_line in render_polygon(
        polygon_width,
        polygon_height,
        frame=frame,
        activity=state.activity_value,
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
        )
    )
    lines.append("")
    lines.extend(render_stream(state, content_width))
    return "\n".join(lines[:available_height]) + "\n"


def render_polygon(width: int, height: int, *, frame: int, activity: float, status: str) -> list[str]:
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
) -> list[str]:
    gap = 4
    column_width = max(20, (width - gap) // 2)
    left = render_box(left_title, left_text, column_width, color=GREEN)
    right = render_box(right_title, right_text, column_width, color=MAGENTA)
    return [left_line + (" " * gap) + right_line for left_line, right_line in zip(left, right, strict=False)]


def render_box(title: str, text: str, width: int, *, color: str) -> list[str]:
    inner = width - 4
    wrapped = textwrap.wrap(text, width=inner, max_lines=3, placeholder="...") or [""]
    wrapped = (wrapped + ["", ""])[:3]
    top = color + "+" + ("-" * (width - 2)) + "+" + RESET
    header = color + "| " + title.ljust(inner) + " |" + RESET
    body = [color + "| " + line.ljust(inner) + " |" + RESET for line in wrapped]
    bottom = color + "+" + ("-" * (width - 2)) + "+" + RESET
    return [top, header, *body, bottom]


def render_stream(state: TerminalUiState, width: int) -> list[str]:
    lines = [f"{BOLD}{WHITE}SYSTEM STREAM{RESET}"]
    for item in state.stream_lines:
        for wrapped in textwrap.wrap(item, width=max(20, width - 4)):
            lines.append(f"{DIM}> {wrapped}{RESET}")
    if len(lines) == 1:
        lines.append(f"{DIM}> waiting for events{RESET}")
    return lines


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
    if "wake" in lowered or "detected" in lowered:
        return GREEN
    if "openai" in lowered or "contact" in lowered:
        return YELLOW
    if "tts" in lowered or "play" in lowered or "speak" in lowered:
        return MAGENTA
    return CYAN


def status_marker(status: str) -> str:
    lowered = status.lower()
    if "listen" in lowered:
        return "L"
    if "wake" in lowered or "detected" in lowered:
        return "W"
    if "openai" in lowered or "contact" in lowered:
        return "O"
    if "tts" in lowered or "play" in lowered or "speak" in lowered:
        return "T"
    return "*"
