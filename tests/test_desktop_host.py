from __future__ import annotations

from pathlib import Path

from whispertome.cli import build_parser, resolve_log_file
from whispertome.desktop.host import (
    FULL_FRAME_CLEAR,
    VoiceLoopLaunchOptions,
    WindowPlacement,
    _center_placement,
    _estimate_tui_viewport,
    _latest_terminal_frame,
    _load_window_placement,
    _save_window_placement,
    build_voice_loop_command,
)


def test_desktop_parser_accepts_window_and_voice_loop_options() -> None:
    args = build_parser().parse_args(
        [
            "--project-root",
            "C:/project",
            "desktop",
            "--wake",
            "computer",
            "--allow-non-npu",
            "--save-audio",
            "--turns",
            "2",
            "--speech-end-ms",
            "1200",
            "--window-width",
            "1280",
            "--window-height",
            "900",
            "--font-size",
            "12",
            "--show-console",
            "--tui-lines",
            "6",
            "--no-stream-tts",
        ]
    )

    assert args.command == "desktop"
    assert args.project_root == Path("C:/project")
    assert args.wake_phrases == ["computer"]
    assert args.allow_non_npu
    assert args.save_audio
    assert args.turns == 2
    assert args.speech_end_ms == 1200
    assert args.window_width == 1280
    assert args.window_height == 900
    assert args.font_size == 12
    assert args.show_console
    assert args.tui_lines == 6
    assert args.no_stream_tts


def test_desktop_parser_defaults_to_800_by_600() -> None:
    args = build_parser().parse_args(["--project-root", "C:/project", "desktop"])

    assert args.window_width == 800
    assert args.window_height == 600
    assert args.font_size == 10
    assert not args.show_console


def test_desktop_gets_default_host_log_file_under_project_root() -> None:
    args = build_parser().parse_args(["--project-root", "C:/project", "desktop"])

    log_file = resolve_log_file(args)

    assert log_file is not None
    assert log_file.parent == Path("C:/project") / "artifacts" / "logs"
    assert log_file.name.startswith("desktop-")
    assert log_file.suffix == ".log"


def test_build_voice_loop_command_hosts_existing_tui() -> None:
    command = build_voice_loop_command(
        "python",
        VoiceLoopLaunchOptions(
            project_root=Path("C:/project"),
            wake_phrases=("computer", "assistant"),
            max_commands=2,
            allow_non_npu=True,
            save_audio=True,
            warmup=False,
            min_speech_ms=300,
            vad_threshold=0.02,
            speech_end_ms=1400,
            play=False,
            stream_tts=False,
            tui_lines=6,
            stop_file=Path("C:/project/stop.signal"),
            wake_event_file=Path("C:/project/wake.signal"),
        ),
    )

    assert command[:5] == [
        "python",
        "-m",
        "whispertome",
        "--project-root",
        str(Path("C:/project")),
    ]
    assert command[5:7] == ["run", "--turns"]
    assert "--tui" in command
    assert command[command.index("--tui-lines") + 1] == "6"
    assert command.count("--wake") == 2
    assert "--allow-non-npu" in command
    assert "--save-audio" in command
    assert "--no-warmup" in command
    assert command[command.index("--min-speech-ms") + 1] == "300"
    assert command[command.index("--vad-threshold") + 1] == "0.02"
    assert command[command.index("--speech-end-ms") + 1] == "1400"
    assert "--no-play" in command
    assert "--no-stream-tts" in command
    assert command[command.index("--stop-file") + 1] == str(Path("C:/project/stop.signal"))
    assert command[command.index("--wake-event-file") + 1] == str(
        Path("C:/project/wake.signal")
    )


def test_latest_terminal_frame_drops_stale_full_screen_frames() -> None:
    text = f"old log\n{FULL_FRAME_CLEAR}frame one\n{FULL_FRAME_CLEAR}frame two\n"

    assert _latest_terminal_frame(text) == f"{FULL_FRAME_CLEAR}frame two\n"


def test_latest_terminal_frame_keeps_incremental_output() -> None:
    text = "plain crash output\nwithout full frame clear\n"

    assert _latest_terminal_frame(text) == text


def test_estimate_tui_viewport_targets_800_by_600_shell() -> None:
    assert _estimate_tui_viewport(800, 600, 10) == (87, 29)


def test_center_placement_targets_screen_middle() -> None:
    placement = _center_placement(
        width=800,
        height=600,
        screen_width=1920,
        screen_height=1080,
    )

    assert placement == WindowPlacement(x=560, y=240, width=800, height=600)


def test_window_placement_roundtrips_and_clamps_to_screen(tmp_path) -> None:
    state_file = tmp_path / "window-state.json"
    _save_window_placement(
        state_file,
        WindowPlacement(x=2000, y=900, width=800, height=600, manual=True),
    )

    placement = _load_window_placement(
        state_file,
        expected_width=800,
        expected_height=600,
        screen_width=1920,
        screen_height=1080,
    )

    assert placement == WindowPlacement(x=1120, y=480, width=800, height=600, manual=True)


def test_window_placement_ignores_saved_size_mismatch(tmp_path) -> None:
    state_file = tmp_path / "window-state.json"
    _save_window_placement(
        state_file,
        WindowPlacement(x=100, y=100, width=1280, height=720, manual=True),
    )

    placement = _load_window_placement(
        state_file,
        expected_width=800,
        expected_height=600,
        screen_width=1920,
        screen_height=1080,
    )

    assert placement is None
