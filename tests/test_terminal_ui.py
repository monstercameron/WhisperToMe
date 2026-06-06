from __future__ import annotations

import unittest

from whispertome.ui.terminal import (
    RESET,
    TerminalUiState,
    render_frame,
    render_polygon,
    status_marker,
    strip_ansi_len,
)


class TerminalUiTests(unittest.TestCase):
    def test_state_limits_stream_to_ten_lines(self) -> None:
        state = TerminalUiState(max_lines=10)

        for index in range(12):
            state.add_line(f"line {index}")

        self.assertEqual(len(state.stream_lines), 10)
        self.assertEqual(state.stream_lines[0], "line 2")
        self.assertEqual(state.stream_lines[-1], "line 11")

    def test_state_clamps_stream_limit(self) -> None:
        self.assertEqual(TerminalUiState(max_lines=0).max_lines, 1)
        self.assertEqual(TerminalUiState(max_lines=99).max_lines, 10)

    def test_render_frame_contains_status_input_output_and_stream(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_status("contacting openai", "gpt-5.2")
        state.set_user("computer summarize the last request")
        state.set_assistant("Working on it.")
        state.add_line("using tools")

        frame = render_frame(state, width=100, height=34, frame=3)

        self.assertIn("WHISPER TO ME", frame)
        self.assertIn("CONTACTING OPENAI", frame)
        self.assertIn("computer summarize the last request", frame)
        self.assertIn("Working on it.", frame)
        self.assertIn("using tools", frame)

    def test_polygon_uses_status_marker(self) -> None:
        polygon = "\n".join(render_polygon(32, 11, frame=0, activity=0.5, status="listening"))

        self.assertIn(status_marker("listening"), polygon)
        self.assertIn(RESET, polygon)

    def test_strip_ansi_len_ignores_color_sequences(self) -> None:
        self.assertEqual(strip_ansi_len("\x1b[31mhello\x1b[0m"), 5)


if __name__ == "__main__":
    unittest.main()
