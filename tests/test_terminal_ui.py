from __future__ import annotations

import unittest

from whispertome.ui.terminal import (
    RESET,
    TerminalUiState,
    display_text,
    render_code_box,
    render_frame,
    render_polygon,
    render_progress_bar,
    status_activity_boost,
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

    def test_render_frame_fits_800_by_600_desktop_viewport(self) -> None:
        state = TerminalUiState(max_lines=10)
        state.set_status("listening", "waiting for computer")
        state.set_user("Computer, check my next important items and keep it short.")
        state.set_assistant("Checking your organizer now.")
        for index in range(10):
            state.add_line(f"event {index}")

        frame = render_frame(state, width=87, height=29, frame=8)
        lines = frame.splitlines()

        self.assertLessEqual(len(lines), 29)
        self.assertTrue(all(strip_ansi_len(line) <= 87 for line in lines))
        self.assertIn("INPUT", frame)
        self.assertIn("OUTPUT", frame)
        self.assertIn("SYSTEM STREAM", frame)
        self.assertIn("event 9", frame)

    def test_render_frame_contains_code_block_view(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_code_block("script", "line one\nline two")

        frame = render_frame(state, width=100, height=40, frame=0)

        self.assertIn("SCRIPT VIEW", frame)
        self.assertIn("line one", frame)
        self.assertIn("line two", frame)

    def test_render_frame_with_code_fits_800_by_600_desktop_viewport(self) -> None:
        state = TerminalUiState(max_lines=10)
        state.set_status("playing speech", "tts")
        state.set_user("Computer, give me a small script.")
        state.set_assistant("Here is one script.")
        state.set_code_block("script", "\n".join(f"line {index}" for index in range(12)))
        for index in range(10):
            state.add_line(f"tool event {index}")

        frame = render_frame(state, width=87, height=29, frame=18)
        lines = frame.splitlines()

        self.assertLessEqual(len(lines), 29)
        self.assertTrue(all(strip_ansi_len(line) <= 87 for line in lines))
        self.assertIn("SCRIPT VIEW", frame)
        self.assertIn("SYSTEM STREAM", frame)

    def test_compact_runtime_keeps_all_pipeline_steps_visible(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_status("playing speech", "tts")

        frame = render_frame(state, width=87, height=29, frame=24)

        self.assertIn("1) STT", frame)
        self.assertIn("2) LLM", frame)
        self.assertIn("3) TOOLS", frame)
        self.assertIn("4) TTS", frame)
        self.assertIn(status_marker("playing speech"), frame)

    def test_code_view_stays_pinned_to_first_line(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_code_block("script", "\n".join(f"line {index}" for index in range(20)))

        frame = render_frame(state, width=87, height=29, frame=96)

        self.assertIn(" 1 ", frame)
        self.assertIn("line 0", frame)
        self.assertNotIn("13-20/20", frame)

    def test_compact_footer_keeps_stop_hint_visible(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_status("contacting openai", "gpt-5.5-mini")

        frame = render_frame(state, width=87, height=29, frame=1)

        self.assertIn("CTRL+C to stop", frame)

    def test_display_text_replaces_unsupported_unicode(self) -> None:
        self.assertEqual(display_text("low\u2011carb and it\u2019s fine"), "low-carb and it's fine")

    def test_frame_with_unicode_model_text_is_ascii_encodable(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_status("streaming cerebras", "gpt-oss-120b")
        state.set_assistant("It\u2019s low\u2011carb, satisfying, and friendly.")

        frame = render_frame(state, width=82, height=26, frame=5)

        frame.encode("ascii")
        self.assertIn("It's low-carb", frame)

    def test_render_frame_fits_safer_800_by_600_desktop_viewport(self) -> None:
        state = TerminalUiState(max_lines=10)
        state.set_status("streaming cerebras", "gpt-oss-120b")
        state.set_user("Computer, help me plan the next big upgrade and keep it short.")
        state.set_assistant("Let's make the desktop shell feel tighter and faster.")
        for index in range(10):
            state.add_line(f"event {index}")

        frame = render_frame(state, width=82, height=26, frame=8)
        lines = frame.splitlines()

        self.assertLessEqual(len(lines), 26)
        self.assertTrue(all(strip_ansi_len(line) <= 82 for line in lines))
        self.assertIn("4) TTS", frame)
        self.assertIn("CTRL+C to stop", frame)

    def test_render_frame_uses_boot_scene_during_initialization(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_boot("loading stt", "QNN Whisper session", 0.35)
        state.set_user("hidden until ready")
        state.set_assistant("also hidden")

        frame = render_frame(state, width=100, height=34, frame=4)

        self.assertIn("WHISPER TO ME // NPU BOOT", frame)
        self.assertIn("LOADING STT", frame)
        self.assertIn("QNN Whisper session", frame)
        self.assertIn("35%", frame)
        self.assertNotIn("INPUT", frame)
        self.assertNotIn("hidden until ready", frame)

    def test_boot_progress_bar_clamps_percent(self) -> None:
        self.assertIn("100%", render_progress_bar(80, 2.0, frame=0))
        self.assertIn("  0%", render_progress_bar(80, -1.0, frame=0))

    def test_code_box_reports_autoscroll_position(self) -> None:
        state = TerminalUiState(max_lines=3)
        state.set_code_block("script", "\n".join(f"line {index}" for index in range(8)))

        box = "\n".join(render_code_box(state, 60, code_height=5, frame=24))

        self.assertIn("SCRIPT VIEW", box)
        self.assertIn("/8", box)

    def test_polygon_uses_status_marker(self) -> None:
        polygon = "\n".join(render_polygon(32, 11, frame=0, activity=0.5, status="listening"))

        self.assertIn(status_marker("listening"), polygon)
        self.assertIn(RESET, polygon)

    def test_wake_and_barge_in_statuses_get_visual_boost(self) -> None:
        self.assertEqual(status_marker("wake detected"), "W")
        self.assertEqual(status_marker("barge-in command"), "!")
        self.assertGreaterEqual(status_activity_boost("wake detected"), 0.9)
        self.assertEqual(status_activity_boost("barge-in command"), 1.0)

    def test_strip_ansi_len_ignores_color_sequences(self) -> None:
        self.assertEqual(strip_ansi_len("\x1b[31mhello\x1b[0m"), 5)


if __name__ == "__main__":
    unittest.main()
