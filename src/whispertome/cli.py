from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from whispertome.config import AppConfig, load_config, with_wake_phrases
from whispertome.errors import WhisperToMeError
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.runtime.providers import ProviderResolver
from whispertome.stt.base import SpeechToTextModel
from whispertome.tts.base import SpeechSynthesisResult, TextToSpeechModel
from whispertome.wake.router import WakeCommandRouter
from whispertome.wake.sliding_window import SlidingWakeDetector

_DEBUG_KOKORO_CACHE: dict[tuple[str, str], Any] = {}
_DEBUG_STT_CACHE: dict[tuple[str, str, int, str, str], Any] = {}


@dataclass(frozen=True)
class AudioProfile:
    duration_ms: int
    rms: float
    peak: float
    active_ms: int
    clipped_ratio: float


@dataclass(frozen=True)
class RecordingResult:
    audio: Any
    raw_audio: Any
    speech_detected: bool
    reason: str
    speech_start_ms: int | None
    speech_end_ms: int | None


@dataclass(frozen=True)
class QueuedInterruptionCommand:
    command: str
    transcript: str
    stt_wall_ms: float
    stt_latency_ms: float


@dataclass(frozen=True)
class CommandTurnResult:
    stop_requested: bool = False
    queued_command: QueuedInterruptionCommand | None = None


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _resolve_stop_file(project_root: Path, stop_file: Path | None) -> Path | None:
    return _resolve_signal_file(project_root, stop_file)


def _resolve_signal_file(project_root: Path, signal_file: Path | None) -> Path | None:
    if signal_file is None:
        return None
    path = signal_file.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _write_signal_file(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)


def _chunks_until_stop_requested(chunks, stop_requested: Callable[[], bool]):  # type: ignore[no-untyped-def]
    for chunk in chunks:
        if stop_requested():
            return
        yield chunk


def _queued_interruption_command(interruption: Any | None) -> QueuedInterruptionCommand | None:
    if interruption is None:
        return None
    event = getattr(interruption, "event", None)
    if getattr(event, "kind", None) != "command_ready":
        return None
    command = (getattr(event, "command", None) or "").strip()
    if not command:
        return None
    return QueuedInterruptionCommand(
        command=command,
        transcript=getattr(interruption, "transcript", ""),
        stt_wall_ms=float(getattr(interruption, "wall_ms", 0.0)),
        stt_latency_ms=float(getattr(interruption, "stt_latency_ms", 0.0)),
    )


class FallbackTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        primary: TextToSpeechModel,
        fallback: TextToSpeechModel,
        *,
        logger: logging.Logger,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._logger = logger
        self._active: TextToSpeechModel | None = None

    def load(self) -> None:
        if self._active is not None:
            self._active.load()
            return
        try:
            self._primary.load()
            self._active = self._primary
        except WhisperToMeError as exc:
            self._activate_fallback(exc)

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        model = self._active or self._primary
        try:
            result = model.synthesize(text)
            if self._active is None:
                self._active = model
            return result
        except WhisperToMeError as exc:
            if model is self._fallback:
                raise
            self._activate_fallback(exc)
            return self._fallback.synthesize(text)

    def _activate_fallback(self, exc: WhisperToMeError) -> None:
        self._logger.warning(
            "npu_tts_failed=%s debug_non_npu_tts_enabled=true",
            exc,
        )
        self._fallback.load()
        self._active = self._fallback


class DebugKokoroSynthesizer(TextToSpeechModel):
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._kokoro: Any | None = None
        self._provider_names: str | None = None

    def load(self) -> None:
        if self._kokoro is not None:
            return
        self._kokoro = get_kokoro_debug_non_npu(self._config)
        self._provider_names = ", ".join(self._kokoro.sess.get_providers())
        logging.getLogger(__name__).warning(
            "debug_non_npu_tts_provider=%s",
            self._provider_names,
        )

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        from whispertome.audio.types import SynthesizedSpeech

        self.load()
        assert self._kokoro is not None
        assert self._provider_names is not None

        started = perf_counter()
        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self._config.tts.voice,
                speed=self._config.tts.speed,
                lang=self._config.tts.language,
            )
        except AssertionError as exc:
            raise WhisperToMeError(str(exc)) from exc
        return SpeechSynthesisResult(
            speech=SynthesizedSpeech(samples=samples, sample_rate=sample_rate),
            latency_ms=(perf_counter() - started) * 1000.0,
            model=str(self._config.tts.model_path),
            provider=f"debug-non-npu:{self._provider_names}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whispertome")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing .env and model artifacts.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write logs to this file in addition to the console.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check configuration, model paths, and NPU providers.",
    )
    add_wake_args(doctor_parser)

    run_parser = subparsers.add_parser("run", help="Start the live voice loop.")
    add_wake_args(run_parser)
    add_live_loop_args(run_parser)

    desktop_parser = subparsers.add_parser(
        "desktop",
        help="Open a first-class desktop window hosting the live TUI.",
    )
    add_wake_args(desktop_parser)
    add_live_loop_args(desktop_parser, include_tui=False)
    desktop_parser.add_argument(
        "--window-width",
        type=int,
        default=800,
        help="Initial desktop window width in pixels.",
    )
    desktop_parser.add_argument(
        "--window-height",
        type=int,
        default=600,
        help="Initial desktop window height in pixels.",
    )
    desktop_parser.add_argument(
        "--font-size",
        type=int,
        default=10,
        help="Terminal font size inside the desktop window.",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a turn-based microphone -> STT -> OpenAI -> TTS -> speaker demo.",
    )
    demo_parser.add_argument(
        "--record-ms",
        type=int,
        default=10000,
        help="Maximum milliseconds to record for each user turn.",
    )
    demo_parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="Maximum turns to run. 0 means keep going until q/quit.",
    )
    demo_parser.add_argument(
        "--allow-non-npu",
        action="store_true",
        help="Debug only: allow non-NPU STT/TTS so a full conversation can be tested now.",
    )
    demo_parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save each recorded user turn as a WAV file under artifacts/demo.",
    )
    demo_parser.add_argument(
        "--fixed-record",
        action="store_true",
        help="Record exactly --record-ms instead of stopping at a speech pause.",
    )
    demo_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not preload debug STT/TTS models before the first turn.",
    )
    demo_parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Skip STT/OpenAI when detected speech activity is shorter than this.",
    )
    demo_parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override WHISPERTOME_VAD_RMS_THRESHOLD for this demo run.",
    )
    demo_parser.add_argument(
        "--speech-end-ms",
        type=int,
        default=None,
        help="Override silence duration required before ending an utterance.",
    )
    demo_parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play assistant TTS audio.",
    )

    test_wake_parser = subparsers.add_parser(
        "test-wake",
        help="Test wake phrase routing with speech-pause-delimited transcript strings.",
    )
    add_wake_args(test_wake_parser)
    test_wake_parser.add_argument(
        "utterances",
        nargs="+",
        help="Transcript utterances. Each argument is treated as one speech-pause segment.",
    )

    test_tts_parser = subparsers.add_parser(
        "test-tts",
        help="Generate a WAV file from text using the configured local TTS backend.",
    )
    test_tts_parser.add_argument("text", help="Text to synthesize.")
    test_tts_parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/tts-test.wav"),
        help="Output WAV path.",
    )
    test_tts_parser.add_argument(
        "--play",
        action="store_true",
        help="Play the generated WAV after synthesis succeeds.",
    )
    test_tts_parser.add_argument(
        "--allow-non-npu",
        action="store_true",
        help="Debug only: allow non-NPU TTS so voice quality and playback can be tested.",
    )

    test_audio_parser = subparsers.add_parser(
        "test-audio",
        help="Play a short generated tone to verify speaker output.",
    )
    test_audio_parser.add_argument(
        "--duration-ms",
        type=int,
        default=500,
        help="Tone duration in milliseconds.",
    )
    test_audio_parser.add_argument(
        "--frequency",
        type=float,
        default=440.0,
        help="Tone frequency in Hz.",
    )
    test_audio_parser.add_argument(
        "--volume",
        type=float,
        default=0.20,
        help="Tone volume from 0.0 to 1.0.",
    )

    test_stt_parser = subparsers.add_parser(
        "test-stt",
        help="Transcribe a WAV file or short microphone recording with the configured STT backend.",
    )
    stt_input = test_stt_parser.add_mutually_exclusive_group(required=True)
    stt_input.add_argument("--wav", type=Path, help="WAV file to transcribe.")
    stt_input.add_argument(
        "--record-ms",
        type=int,
        help="Record microphone audio for this many milliseconds before transcribing.",
    )
    test_stt_parser.add_argument(
        "--allow-non-npu",
        action="store_true",
        help="Debug only: allow non-NPU STT so transcription quality can be tested.",
    )

    test_openai_parser = subparsers.add_parser(
        "test-openai",
        help="Send a dictation-style transcript to OpenAI through the Responses API.",
    )
    test_openai_parser.add_argument("text", help="Transcript text to send.")
    test_openai_parser.add_argument(
        "--reset",
        action="store_true",
        help="Start a fresh responder conversation for this call.",
    )
    return parser


def add_wake_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wake",
        dest="wake_phrases",
        action="append",
        default=None,
        help="Wake phrase to listen for. Can be repeated. Overrides WHISPERTOME_WAKE_PHRASES.",
    )


def add_live_loop_args(parser: argparse.ArgumentParser, *, include_tui: bool = True) -> None:
    parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="Maximum completed command turns to run. 0 means keep going until Ctrl+C.",
    )
    parser.add_argument(
        "--allow-non-npu",
        action="store_true",
        help="Debug only: allow non-NPU TTS so the wake loop can speak now.",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save every STT utterance and assistant response under artifacts/wake.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not preload STT/TTS models before listening.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Skip STT when detected speech activity is shorter than this.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override WHISPERTOME_VAD_RMS_THRESHOLD for this run.",
    )
    parser.add_argument(
        "--speech-end-ms",
        type=int,
        default=None,
        help="Override silence duration required before ending an utterance.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play assistant TTS audio.",
    )
    parser.add_argument(
        "--no-stream-tts",
        action="store_true",
        help="Use the older batch OpenAI -> TTS path instead of streaming sentence chunks.",
    )
    if include_tui:
        parser.add_argument(
            "--tui",
            action="store_true",
            help="Render a live terminal UI instead of console log lines.",
        )
    parser.add_argument(
        "--tui-lines",
        type=int,
        default=10,
        help="Number of system stream lines to show in the TUI, from 1 to 10.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wake-event-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )


def configure_logging(verbose: bool, log_file: Path | None, *, console: bool = True) -> None:
    handlers: list[logging.Handler] = []
    if console:
        handlers.append(logging.StreamHandler())
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_file = resolve_log_file(args)
    configure_logging(args.verbose, log_file, console=not bool(getattr(args, "tui", False)))
    if log_file is not None:
        logging.getLogger(__name__).info("log_file=%s", log_file)

    try:
        if args.command == "doctor":
            config = apply_cli_overrides(
                load_config(args.project_root, require_openai_key=False),
                args,
            )
            return run_doctor(config)
        if args.command == "run":
            config = apply_cli_overrides(
                load_config(args.project_root, require_openai_key=True),
                args,
            )
            return run_wake_loop(
                config,
                max_commands=args.turns,
                allow_non_npu=args.allow_non_npu,
                save_audio=args.save_audio,
                play=not args.no_play,
                warmup=not args.no_warmup,
                min_speech_ms=args.min_speech_ms,
                vad_threshold=args.vad_threshold,
                use_tui=args.tui,
                tui_lines=args.tui_lines,
                stream_tts=not args.no_stream_tts,
                stop_file=args.stop_file,
                wake_event_file=args.wake_event_file,
            )
        if args.command == "desktop":
            config = apply_cli_overrides(
                load_config(args.project_root, require_openai_key=True),
                args,
            )
            return run_desktop(config, args)
        if args.command == "demo":
            config = apply_cli_overrides(
                load_config(args.project_root, require_openai_key=True),
                args,
            )
            return run_demo(
                config,
                record_ms=args.record_ms,
                max_turns=args.turns,
                allow_non_npu=args.allow_non_npu,
                save_audio=args.save_audio,
                play=not args.no_play,
                fixed_record=args.fixed_record,
                warmup=not args.no_warmup,
                min_speech_ms=args.min_speech_ms,
                vad_threshold=args.vad_threshold,
            )
        if args.command == "test-wake":
            config = apply_cli_overrides(
                load_config(args.project_root, require_openai_key=False),
                args,
            )
            return run_test_wake(config, args.utterances)
        if args.command == "test-tts":
            config = load_config(args.project_root, require_openai_key=False)
            return run_test_tts(
                config,
                args.text,
                args.out,
                play=args.play,
                allow_non_npu=args.allow_non_npu,
            )
        if args.command == "test-audio":
            return run_test_audio(args.duration_ms, args.frequency, args.volume)
        if args.command == "test-stt":
            config = load_config(args.project_root, require_openai_key=False)
            return run_test_stt(
                config,
                wav_path=args.wav,
                record_ms=args.record_ms,
                allow_non_npu=args.allow_non_npu,
            )
        if args.command == "test-openai":
            config = load_config(args.project_root, require_openai_key=True)
            return run_test_openai(config, args.text, reset=args.reset)
    except WhisperToMeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except Exception:
        logging.getLogger(__name__).exception("Unhandled failure")
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def resolve_log_file(args: argparse.Namespace) -> Path | None:
    if args.log_file is not None:
        path = args.log_file
        if not path.is_absolute():
            path = args.project_root / path
        return path

    if args.command in {"demo", "desktop", "run"}:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return args.project_root / "artifacts" / "logs" / f"{args.command}-{stamp}.log"

    return None


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updated = config
    wake_phrases = getattr(args, "wake_phrases", None)
    if wake_phrases:
        updated = with_wake_phrases(updated, tuple(wake_phrases))

    speech_end_ms = getattr(args, "speech_end_ms", None)
    if speech_end_ms is not None:
        if speech_end_ms <= 0:
            raise WhisperToMeError("speech-end-ms must be greater than 0")
        updated = replace(
            updated,
            audio=replace(updated.audio, speech_end_ms=speech_end_ms),
        )
    return updated


def run_doctor(config: AppConfig) -> int:
    logger = logging.getLogger(__name__)
    logger.info("project_root=%s", config.project_root)
    logger.info("openai_api_key_present=%s", bool(config.openai.api_key))
    logger.info("openai_model=%s", config.openai.model)
    logger.info("wake_phrases=%s", ", ".join(config.wake.phrases))
    logger.info(
        "stt_backend=%s model_exists=%s",
        config.stt.backend,
        config.stt.model_path.exists(),
    )
    logger.info("stt_model_path=%s", config.stt.model_path)
    logger.info(
        "tts_backend=%s model_exists=%s",
        config.tts.backend,
        config.tts.model_path.exists(),
    )
    logger.info("tts_model_path=%s", config.tts.model_path)
    logger.info("tts_voice_exists=%s", config.tts.voice_path.exists())
    logger.info("tts_voice_path=%s", config.tts.voice_path)

    factory = NpuOnlyOnnxSessionFactory(config.runtime)
    try:
        available = factory.available_providers()
    except WhisperToMeError as exc:
        logger.error("onnxruntime_check=failed reason=%s", exc)
        return 1

    logger.info("onnx_available_providers=%s", ", ".join(available) or "<none>")
    try:
        provider, rejections = ProviderResolver(config.runtime).resolve(available)
    except WhisperToMeError as exc:
        logger.error("npu_provider_check=failed reason=%s", exc)
        return 1

    for rejection in rejections:
        logger.info("provider_rejected key=%s reason=%s", rejection.key, rejection.reason)
    logger.info("selected_provider=%s proof=%s", provider.onnx_name, provider.proof)
    return 0


def run_desktop(config: AppConfig, args: argparse.Namespace) -> int:
    if args.window_width < 640:
        raise WhisperToMeError("window-width must be at least 640")
    if args.window_height < 480:
        raise WhisperToMeError("window-height must be at least 480")
    if args.font_size < 8:
        raise WhisperToMeError("font-size must be at least 8")
    if args.tui_lines < 1 or args.tui_lines > 10:
        raise WhisperToMeError("tui-lines must be between 1 and 10")

    from whispertome.desktop.host import (
        DesktopHostConfig,
        DesktopTerminalHost,
        VoiceLoopLaunchOptions,
        build_voice_loop_command,
    )

    desktop_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stop_file = (
        config.project_root
        / "artifacts"
        / "desktop"
        / f"stop-{desktop_stamp}.signal"
    )
    wake_event_file = (
        config.project_root
        / "artifacts"
        / "desktop"
        / f"wake-{desktop_stamp}.signal"
    )
    window_command_file = (
        config.project_root
        / "artifacts"
        / "desktop"
        / f"window-{desktop_stamp}.command"
    )
    stop_file.unlink(missing_ok=True)
    wake_event_file.unlink(missing_ok=True)
    window_command_file.unlink(missing_ok=True)
    logging.getLogger(__name__).info(
        "desktop_signal_files stop_file=%s wake_event_file=%s window_command_file=%s",
        stop_file,
        wake_event_file,
        window_command_file,
    )
    options = VoiceLoopLaunchOptions(
        project_root=config.project_root,
        wake_phrases=tuple(args.wake_phrases or ()),
        max_commands=args.turns,
        allow_non_npu=args.allow_non_npu,
        save_audio=args.save_audio,
        warmup=not args.no_warmup,
        min_speech_ms=args.min_speech_ms,
        vad_threshold=args.vad_threshold,
        speech_end_ms=args.speech_end_ms,
        play=not args.no_play,
        stream_tts=not args.no_stream_tts,
        tui_lines=args.tui_lines,
        stop_file=stop_file,
        wake_event_file=wake_event_file,
    )
    command = build_voice_loop_command(sys.executable, options)
    logging.getLogger(__name__).info("desktop_child_command=%r", command)
    return DesktopTerminalHost(
        DesktopHostConfig(
            command=command,
            cwd=config.project_root,
            width=args.window_width,
            height=args.window_height,
            font_size=args.font_size,
            stop_file=stop_file,
            wake_event_file=wake_event_file,
            window_command_file=window_command_file,
        )
    ).run()


def run_test_wake(config: AppConfig, utterances: list[str]) -> int:
    logger = logging.getLogger(__name__)
    router = WakeCommandRouter(SlidingWakeDetector(config.wake))
    logger.info("wake_phrases=%s", ", ".join(config.wake.phrases))

    for index, utterance in enumerate(utterances, start=1):
        event = router.process_utterance(utterance)
        if event.kind == "command_ready":
            logger.info("utterance_%d command_ready=%r", index, event.command)
        elif event.kind == "wake_detected":
            assert event.match is not None
            logger.info(
                "utterance_%d wake_detected phrase=%r score=%.2f",
                index,
                event.match.phrase,
                event.match.score,
            )
        else:
            logger.info("utterance_%d idle", index)
    return 0


def run_wake_loop(
    config: AppConfig,
    *,
    max_commands: int,
    allow_non_npu: bool,
    save_audio: bool,
    play: bool,
    warmup: bool,
    min_speech_ms: int,
    vad_threshold: float | None,
    use_tui: bool = False,
    tui_lines: int = 10,
    stream_tts: bool = True,
    stop_file: Path | None = None,
    wake_event_file: Path | None = None,
) -> int:
    if max_commands < 0:
        raise WhisperToMeError("turns must be greater than or equal to 0")
    if min_speech_ms < 0:
        raise WhisperToMeError("min-speech-ms must be greater than or equal to 0")
    if tui_lines < 1 or tui_lines > 10:
        raise WhisperToMeError("tui-lines must be between 1 and 10")

    logger = logging.getLogger(__name__)
    active_vad_threshold = (
        config.audio.vad_rms_threshold if vad_threshold is None else vad_threshold
    )
    stop_path = _resolve_stop_file(config.project_root, stop_file)
    wake_event_path = _resolve_signal_file(config.project_root, wake_event_file)

    def stop_requested() -> bool:
        return stop_path is not None and stop_path.exists()

    def signal_wake_event(reason: str) -> None:
        try:
            _write_signal_file(
                wake_event_path,
                f"{datetime.now().astimezone().isoformat()} {reason}\n",
            )
        except Exception as exc:
            logger.warning("wake_event_signal_failed reason=%s error=%s", reason, exc)

    if allow_non_npu:
        logger.warning(
            "wake_debug_non_npu_enabled=true production_npu_policy_unchanged=true"
        )
    else:
        logger.info("wake_npu_only_policy=true")

    from whispertome.audio.capture import MicrophoneInput
    from whispertome.audio.playback import SpeakerOutput
    from whispertome.audio.vad import EnergyVad, UtteranceSegmenter
    from whispertome.llm.openai_responses import OpenAIResponder
    from whispertome.organizer.tools import build_organization_tool_registry, build_organizer_store
    from whispertome.system.windows import WindowsBackgroundAudioDucker
    from whispertome.text.markdown import parse_spoken_markdown
    from whispertome.text.streaming import SpokenTextChunker
    from whispertome.tts.streaming import StreamingSpeechPlayer, concatenate_speech
    from whispertome.ui.terminal import NullVoiceUi, TerminalVoiceUi
    from whispertome.wake.interruptions import PlaybackWakeMonitor
    from whispertome.wake.live_preview import InterimWakePreview

    ui = TerminalVoiceUi(max_lines=tui_lines) if use_tui else NullVoiceUi()
    ui.start()
    ui.boot("runtime graph", "building local voice adapters", 0.03)
    stt_model = prepare_stt_model(config, allow_non_npu=allow_non_npu)
    ui.boot("stt adapter", config.stt.backend, 0.10)
    tts_model = prepare_tts_model(
        config,
        allow_non_npu=allow_non_npu,
        prefer_debug_non_npu=allow_non_npu,
    )
    ui.boot("tts adapter", config.tts.backend, 0.16)
    organizer_store = build_organizer_store(config.project_root)
    ui.boot("memory bus", "sqlite organizer and preferences", 0.22)
    responder = OpenAIResponder(
        config.openai,
        tool_registry=build_organization_tool_registry(
            config.project_root,
            store=organizer_store,
        ),
        system_context_provider=organizer_store.preference_prompt_context,
    )
    speaker = SpeakerOutput() if play else None
    audio_ducker = WindowsBackgroundAudioDucker(
        target_percent=config.system.wake_duck_percent,
        enabled=config.system.wake_duck_enabled,
        logger=logger,
    )
    logger.info(
        "wake_audio_ducking enabled=%s target_percent=%d mode=session",
        config.system.wake_duck_enabled,
        config.system.wake_duck_percent,
    )
    router = WakeCommandRouter(SlidingWakeDetector(config.wake))
    segmenter = UtteranceSegmenter(config.audio, EnergyVad(active_vad_threshold))
    stt_lock = Lock()

    def duck_background_audio(reason: str, *, turn_number: int | None = None) -> None:
        result = audio_ducker.duck()
        logger.info(
            (
                "wake_audio_duck_request turn=%s reason=%s ok=%s controlled=%d "
                "lowered=%d target_percent=%s detail=%s"
            ),
            turn_number if turn_number is not None else "n/a",
            reason,
            result.ok,
            result.controlled_count,
            result.lowered_count,
            result.target_percent,
            result.reason,
        )
        if result.ok:
            ui.status(
                "ducking audio",
                f"{result.lowered_count}/{result.controlled_count} apps",
            )

    def restore_background_audio(reason: str, *, turn_number: int | None = None) -> None:
        result = audio_ducker.restore()
        if result.ok or result.reason != "not_ducked":
            logger.info(
                (
                    "wake_audio_restore_request turn=%s reason=%s ok=%s "
                    "controlled=%d restored=%d detail=%s"
                ),
                turn_number if turn_number is not None else "n/a",
                reason,
                result.ok,
                result.controlled_count,
                result.restored_count,
                result.reason,
            )

    try:
        if warmup:
            warmup_started = perf_counter()
            logger.info("wake_warmup_started")
            ui.boot("loading stt", "QNN Whisper session", 0.35)
            stt_model.load()
            ui.boot("loading tts", "Kokoro voice session", 0.70)
            tts_model.load()
            warmup_ms = (perf_counter() - warmup_started) * 1000.0
            logger.info(
                "wake_warmup_complete latency_ms=%.1f",
                warmup_ms,
            )
            ui.boot("systems online", f"warmup {warmup_ms:.0f} ms", 1.0)
            ui.boot_complete()
            ui.status("listening", f"warmup {warmup_ms:.0f} ms")
        else:
            ui.boot_complete()

        audio_dir = config.project_root / "artifacts" / "wake" / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        if save_audio:
            audio_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            (
                "wake_loop_started wake_phrases=%s max_commands=%d allow_non_npu=%s "
                "save_audio=%s play=%s vad_threshold=%.6f min_speech_ms=%d"
            ),
            ", ".join(config.wake.phrases),
            max_commands,
            allow_non_npu,
            save_audio,
            play,
            active_vad_threshold,
            min_speech_ms,
        )
        ui.status("listening", f"wake: {', '.join(config.wake.phrases)}")
        if not use_tui:
            print(
                "Wake loop ready. Say one of: "
                f"{', '.join(config.wake.phrases)}. Press Ctrl+C to quit."
            )

        utterance_count = 0
        command_count = 0
        chunks = MicrophoneInput(config.audio).chunks()

        def on_interim_wake_detected(_result: Any) -> None:
            signal_wake_event("interim_wake_detected")
            duck_background_audio("interim_wake_detected")

        interim_preview = (
            InterimWakePreview(
                config=config,
                stt_model=stt_model,
                stt_lock=stt_lock,
                logger=logger,
                status_callback=ui.status,
                line_callback=ui.line,
                on_wake_detected=on_interim_wake_detected,
            )
            if use_tui
            else None
        )
        utterances = segmenter.utterances(
            _chunks_until_stop_requested(chunks, stop_requested),
            on_speech_start=(
                interim_preview.start if interim_preview is not None else None
            ),
            on_speech_chunk=(
                interim_preview.add_chunk if interim_preview is not None else None
            ),
            on_speech_end=(
                interim_preview.finish if interim_preview is not None else None
            ),
        )

        def run_command_turn(
            command: str,
            *,
            utterance_index: int,
            source: str,
            source_transcript: str,
            utterance_started: float,
            stt_wall_ms: float,
            stt_latency_ms: float,
        ) -> CommandTurnResult:
            nonlocal command_count
            command = command.strip()
            if not command:
                return CommandTurnResult()

            command_count += 1
            ui.user_text(command)
            ui.status("wake detected", f"command {command_count}")
            logger.info(
                "wake_command turn=%d utterance=%d source=%s command=%r transcript=%r",
                command_count,
                utterance_index,
                source,
                command,
                source_transcript,
            )
            if not use_tui:
                print(f"You: {command}")
            if command.lower() in {"q", "quit", "exit", "stop"}:
                logger.info("wake_command_quit turn=%d", command_count)
                restore_background_audio("quit_command", turn_number=command_count)
                return CommandTurnResult(stop_requested=True)
            duck_background_audio("command_turn_start", turn_number=command_count)

            queued_command: QueuedInterruptionCommand | None = None

            def on_agent_tool_event(event) -> None:  # type: ignore[no-untyped-def]
                logger.info(
                    "wake_agent_tool turn=%d name=%s ok=%s latency_ms=%.1f args=%r output=%r",
                    command_count,
                    event.name,
                    event.ok,
                    event.latency_ms,
                    event.arguments,
                    event.output,
                )
                ui.status("using tool", event.name.replace("_", " "))
                ui.line(f"tool {event.name}: {'ok' if event.ok else 'error'}")

            def on_playback_speech_detected(_chunks: tuple[Any, ...]) -> None:
                duck_background_audio(
                    "playback_speech_start",
                    turn_number=command_count,
                )

            def on_playback_wake_detected(_interruption: Any) -> None:
                signal_wake_event("playback_wake_detected")
                duck_background_audio(
                    "playback_wake_confirmed",
                    turn_number=command_count,
                )

            if stream_tts:
                ui.status("streaming openai", config.openai.model)
                chunker = SpokenTextChunker()
                interrupt_monitor = None
                if speaker is not None:
                    interrupt_monitor = PlaybackWakeMonitor(
                        config=config,
                        chunks=chunks,
                        stt_model=stt_model,
                        min_speech_ms=min_speech_ms,
                        vad_threshold=active_vad_threshold,
                        stt_lock=stt_lock,
                        logger=logger,
                        status_callback=ui.status,
                        on_speech_detected=on_playback_speech_detected,
                        on_wake_detected=on_playback_wake_detected,
                    )
                    interrupt_monitor.start()
                player = StreamingSpeechPlayer(
                    tts_model=tts_model,
                    speaker=speaker,
                    interrupt_event=(
                        interrupt_monitor.interrupt_event
                        if interrupt_monitor is not None
                        else None
                    ),
                    logger=logger,
                    status_callback=ui.status,
                )

                def on_openai_delta(delta: str) -> None:
                    for text_chunk in chunker.push(delta):
                        logger.info(
                            "wake_stream_tts_text turn=%d chars=%d text=%r",
                            command_count,
                            len(text_chunk),
                            text_chunk,
                        )
                        ui.line(f"tts queued: {text_chunk}")
                        player.enqueue_text(text_chunk)
                    partial_display = parse_spoken_markdown(chunker.raw_text).display_text
                    if partial_display:
                        ui.assistant_text(partial_display)

                openai_started = perf_counter()
                try:
                    llm_response = responder.generate_stream(
                        command,
                        on_delta=on_openai_delta,
                        on_tool_event=on_agent_tool_event,
                    )
                    openai_wall_ms = (perf_counter() - openai_started) * 1000.0
                    for text_chunk in chunker.finish():
                        logger.info(
                            "wake_stream_tts_text turn=%d chars=%d final=true text=%r",
                            command_count,
                            len(text_chunk),
                            text_chunk,
                        )
                        ui.line(f"tts queued: {text_chunk}")
                        player.enqueue_text(text_chunk)
                    stream_result = player.finish()
                except Exception:
                    player.cancel()
                    raise
                finally:
                    if interrupt_monitor is not None:
                        interrupt_monitor.stop()
                    restore_background_audio(
                        "stream_turn_finished",
                        turn_number=command_count,
                    )

                spoken_response = parse_spoken_markdown(llm_response.text)
                ui.assistant_text(spoken_response.display_text)
                code_block = spoken_response.primary_code_block
                if code_block is not None:
                    ui.code_block(code_block.language, code_block.code)
                    logger.info(
                        "wake_response_code_block turn=%d language=%s chars=%d",
                        command_count,
                        code_block.language,
                        len(code_block.code),
                    )
                else:
                    ui.clear_code_block()
                ui.line(f"openai: {spoken_response.display_text}")
                logger.info(
                    (
                        "wake_openai_stream turn=%d wall_ms=%.1f latency_ms=%.1f "
                        "model=%s response_id=%s chars=%d speech_chars=%d "
                        "code_blocks=%d chunks=%d first_text_ms=%s first_audio_ms=%s "
                        "first_playback_ms=%s text=%r"
                    ),
                    command_count,
                    openai_wall_ms,
                    llm_response.latency_ms,
                    llm_response.model,
                    llm_response.response_id,
                    len(llm_response.text),
                    len(spoken_response.speech_text),
                    len(spoken_response.code_blocks),
                    stream_result.chunk_count,
                    _fmt_ms(stream_result.first_text_ms),
                    _fmt_ms(stream_result.first_audio_ms),
                    _fmt_ms(stream_result.first_playback_ms),
                    llm_response.text,
                )
                if not use_tui:
                    print(f"Assistant: {spoken_response.display_text}")

                tts_wall_ms = stream_result.synthesis_ms
                tts_model_ms = sum(chunk.latency_ms for chunk in stream_result.chunks)
                playback_ms = stream_result.playback_ms
                logger.info(
                    (
                        "wake_streaming_tts turn=%d chunks=%d total_ms=%.1f "
                        "synthesis_ms=%.1f model_ms=%.1f playback_ms=%.1f "
                        "audio_ms=%d provider=%s sample_rate=%s interrupted=%s"
                    ),
                    command_count,
                    stream_result.chunk_count,
                    stream_result.total_ms,
                    stream_result.synthesis_ms,
                    tts_model_ms,
                    stream_result.playback_ms,
                    stream_result.total_audio_ms,
                    stream_result.provider,
                    stream_result.sample_rate,
                    stream_result.interrupted,
                )
                if save_audio:
                    assistant_path = audio_dir / f"turn-{command_count:03d}-assistant.wav"
                    combined_speech = concatenate_speech(stream_result.chunks)
                    if combined_speech is not None:
                        save_audio_buffer(combined_speech, assistant_path)
                        logger.info(
                            "wake_assistant_audio turn=%d path=%s chunks=%d",
                            command_count,
                            assistant_path,
                            stream_result.chunk_count,
                        )
                    else:
                        logger.warning(
                            "wake_assistant_audio_skipped turn=%d reason=%s",
                            command_count,
                            "no compatible streaming chunks",
                        )
                interruption = (
                    interrupt_monitor.result if interrupt_monitor is not None else None
                )
                if stream_result.interrupted and interruption is not None:
                    logger.info(
                        (
                            "wake_playback_interrupted turn=%d latency_ms=%.1f "
                            "transcript=%r kind=%s command=%r"
                        ),
                        command_count,
                        stream_result.playback_ms,
                        interruption.transcript,
                        interruption.event.kind,
                        interruption.event.command,
                    )
                    ui.user_text(interruption.transcript)
                    ui.status("interrupted", interruption.transcript)
                    ui.line(f"interrupted by wake: {interruption.transcript}")
                    queued_command = _queued_interruption_command(interruption)
                elif speaker is not None:
                    logger.info(
                        "wake_stream_played turn=%d latency_ms=%.1f chunks=%d",
                        command_count,
                        playback_ms,
                        stream_result.chunk_count,
                    )
            else:
                ui.status("contacting openai", config.openai.model)
                openai_started = perf_counter()
                llm_response = responder.generate(
                    command,
                    on_tool_event=on_agent_tool_event,
                )
                openai_wall_ms = (perf_counter() - openai_started) * 1000.0
                spoken_response = parse_spoken_markdown(llm_response.text)
                ui.assistant_text(spoken_response.display_text)
                code_block = spoken_response.primary_code_block
                if code_block is not None:
                    ui.code_block(code_block.language, code_block.code)
                    logger.info(
                        "wake_response_code_block turn=%d language=%s chars=%d",
                        command_count,
                        code_block.language,
                        len(code_block.code),
                    )
                else:
                    ui.clear_code_block()
                ui.line(f"openai: {spoken_response.display_text}")
                logger.info(
                    (
                        "wake_openai turn=%d wall_ms=%.1f latency_ms=%.1f model=%s "
                        "response_id=%s chars=%d speech_chars=%d code_blocks=%d text=%r"
                    ),
                    command_count,
                    openai_wall_ms,
                    llm_response.latency_ms,
                    llm_response.model,
                    llm_response.response_id,
                    len(llm_response.text),
                    len(spoken_response.speech_text),
                    len(spoken_response.code_blocks),
                    llm_response.text,
                )
                if not use_tui:
                    print(f"Assistant: {spoken_response.display_text}")

                ui.status("running tts", "Kokoro voice")
                tts_started = perf_counter()
                speech = tts_model.synthesize(spoken_response.speech_text)
                tts_wall_ms = (perf_counter() - tts_started) * 1000.0
                tts_model_ms = speech.latency_ms
                logger.info(
                    "wake_tts turn=%d wall_ms=%.1f latency_ms=%.1f provider=%s sample_rate=%d",
                    command_count,
                    tts_wall_ms,
                    speech.latency_ms,
                    speech.provider,
                    speech.speech.sample_rate,
                )
                if save_audio:
                    assistant_path = audio_dir / f"turn-{command_count:03d}-assistant.wav"
                    save_audio_buffer(speech.speech, assistant_path)
                    logger.info(
                        "wake_assistant_audio turn=%d path=%s",
                        command_count,
                        assistant_path,
                    )
                if speaker is not None:
                    ui.status(
                        "playing speech",
                        (
                            f"{speech.speech.duration_ms} ms - "
                            f"say {config.wake.phrases[0]} to interrupt"
                        ),
                    )
                    interrupt_monitor = PlaybackWakeMonitor(
                        config=config,
                        chunks=chunks,
                        stt_model=stt_model,
                        min_speech_ms=min_speech_ms,
                        vad_threshold=active_vad_threshold,
                        stt_lock=stt_lock,
                        logger=logger,
                        status_callback=ui.status,
                        on_speech_detected=on_playback_speech_detected,
                        on_wake_detected=on_playback_wake_detected,
                    )
                    interrupt_monitor.start()
                    playback_started = perf_counter()
                    try:
                        playback_result = speaker.play_interruptible(
                            speech.speech,
                            interrupt_event=interrupt_monitor.interrupt_event,
                        )
                        playback_ms = (perf_counter() - playback_started) * 1000.0
                    finally:
                        interrupt_monitor.stop()
                        restore_background_audio(
                            "playback_finished",
                            turn_number=command_count,
                        )
                    interruption = interrupt_monitor.result
                    if playback_result.interrupted and interruption is not None:
                        logger.info(
                            (
                                "wake_playback_interrupted turn=%d latency_ms=%.1f "
                                "transcript=%r kind=%s command=%r"
                            ),
                            command_count,
                            playback_result.elapsed_ms,
                            interruption.transcript,
                            interruption.event.kind,
                            interruption.event.command,
                        )
                        ui.user_text(interruption.transcript)
                        ui.status("interrupted", interruption.transcript)
                        ui.line(f"interrupted by wake: {interruption.transcript}")
                        queued_command = _queued_interruption_command(interruption)
                    else:
                        logger.info(
                            "wake_played turn=%d latency_ms=%.1f",
                            command_count,
                            playback_ms,
                        )
                else:
                    playback_ms = 0.0
                    restore_background_audio(
                        "no_play_turn_finished",
                        turn_number=command_count,
                    )

            if queued_command is not None:
                logger.info(
                    "wake_barge_in_command_queued turn=%d command=%r transcript=%r",
                    command_count,
                    queued_command.command,
                    queued_command.transcript,
                )
                ui.status("barge-in command", queued_command.command)
                ui.line(f"barge-in queued: {queued_command.command}")

            logger.info(
                (
                    "wake_turn_profile turn=%d utterance=%d source=%s total_ms=%.1f "
                    "stt_wall_ms=%.1f stt_model_ms=%.1f openai_ms=%.1f "
                    "tts_wall_ms=%.1f tts_model_ms=%.1f playback_ms=%.1f"
                ),
                command_count,
                utterance_index,
                source,
                (perf_counter() - utterance_started) * 1000.0,
                stt_wall_ms,
                stt_latency_ms,
                openai_wall_ms,
                tts_wall_ms,
                tts_model_ms,
                playback_ms,
            )
            return CommandTurnResult(queued_command=queued_command)

        for utterance in utterances:
            if stop_requested():
                logger.info("wake_external_stop_requested phase=before_utterance")
                ui.status("stopping", "desktop stop requested")
                break
            utterance_count += 1
            utterance_started = perf_counter()
            audio_profile = profile_audio(
                utterance,
                active_vad_threshold,
                config.audio.block_ms,
            )
            ui.activity(min(1.0, audio_profile.rms / max(active_vad_threshold * 6.0, 0.001)))
            ui.status("speech captured", f"{audio_profile.duration_ms} ms")
            if save_audio:
                utterance_path = audio_dir / f"utterance-{utterance_count:04d}.wav"
                save_audio_buffer(utterance, utterance_path)
                logger.info(
                    "wake_utterance_audio utterance=%d path=%s",
                    utterance_count,
                    utterance_path,
                )

            if audio_profile.active_ms < min_speech_ms:
                logger.info(
                    (
                        "wake_skip_short_audio utterance=%d active_ms=%d "
                        "min_speech_ms=%d"
                    ),
                    utterance_count,
                    audio_profile.active_ms,
                    min_speech_ms,
                )
                ui.status("listening", "short audio skipped")
                continue

            ui.status("transcribing", "QNN Whisper")
            stt_started = perf_counter()
            with stt_lock:
                transcript = stt_model.transcribe(utterance)
            stt_wall_ms = (perf_counter() - stt_started) * 1000.0
            text = transcript.text.strip()
            ui.user_text(text)
            ui.line(f"transcript: {text}")
            logger.info(
                (
                    "wake_transcript utterance=%d duration_ms=%d active_ms=%d "
                    "wall_ms=%.1f latency_ms=%.1f provider=%s text=%r"
                ),
                utterance_count,
                audio_profile.duration_ms,
                audio_profile.active_ms,
                stt_wall_ms,
                transcript.latency_ms,
                transcript.provider,
                text,
            )
            if not text:
                ui.status("listening", "empty transcript")
                continue

            event = router.process_utterance(text)
            if event.kind == "idle":
                ui.status("listening", "wake phrase not detected")
                continue

            if event.kind == "wake_detected":
                assert event.match is not None
                logger.info(
                    (
                        "wake_detected utterance=%d phrase=%r score=%.3f "
                        "matched_text=%r window=%r"
                    ),
                    utterance_count,
                    event.match.phrase,
                    event.match.score,
                    event.match.matched_text,
                    event.match.transcript_window,
                )
                ui.status("wake detected", event.match.phrase)
                signal_wake_event("wake_detected")
                duck_background_audio("wake_detected")
                if not use_tui:
                    print("Wake detected. Listening for your command.")
                continue

            assert event.command is not None
            result = run_command_turn(
                event.command,
                utterance_index=utterance_count,
                source="wake_loop",
                source_transcript=text,
                utterance_started=utterance_started,
                stt_wall_ms=stt_wall_ms,
                stt_latency_ms=transcript.latency_ms,
            )
            if stop_requested():
                logger.info("wake_external_stop_requested phase=after_turn")
                ui.status("stopping", "desktop stop requested")
                break
            if result.stop_requested:
                break

            while result.queued_command is not None:
                if stop_requested():
                    logger.info("wake_external_stop_requested phase=barge_in_queue")
                    ui.status("stopping", "desktop stop requested")
                    break
                queued = result.queued_command
                if max_commands and command_count >= max_commands:
                    logger.info("wake_max_commands_reached turns=%d", command_count)
                    break
                logger.info(
                    (
                        "wake_barge_in_command_start previous_turn=%d "
                        "command=%r transcript=%r"
                    ),
                    command_count,
                    queued.command,
                    queued.transcript,
                )
                ui.status("running barge-in", queued.command)
                result = run_command_turn(
                    queued.command,
                    utterance_index=utterance_count,
                    source="playback_interruption",
                    source_transcript=queued.transcript,
                    utterance_started=perf_counter(),
                    stt_wall_ms=queued.stt_wall_ms,
                    stt_latency_ms=queued.stt_latency_ms,
                )
                if result.stop_requested:
                    break
            if result.stop_requested:
                break
            if max_commands and command_count >= max_commands:
                logger.info("wake_max_commands_reached turns=%d", command_count)
                break
            ui.status("listening", f"completed turn {command_count}")
    except KeyboardInterrupt:
        logger.info("wake_keyboard_interrupt")
        ui.status("stopping", "keyboard interrupt")
    finally:
        if stop_requested():
            logger.info("wake_external_stop_requested phase=finally")
        close = getattr(locals().get("chunks", None), "close", None)
        if close is not None:
            close()
        restore_background_audio("wake_loop_finished")
        _safe_unlink(stop_path)
        _safe_unlink(wake_event_path)
        ui.stop()

    logger.info(
        "wake_loop_finished utterances=%d commands=%d",
        locals().get("utterance_count", 0),
        locals().get("command_count", 0),
    )
    return 0


def run_demo(
    config: AppConfig,
    *,
    record_ms: int,
    max_turns: int,
    allow_non_npu: bool,
    save_audio: bool,
    play: bool,
    fixed_record: bool,
    warmup: bool,
    min_speech_ms: int,
    vad_threshold: float | None,
) -> int:
    if record_ms <= 0:
        raise WhisperToMeError("record-ms must be greater than 0")
    if max_turns < 0:
        raise WhisperToMeError("turns must be greater than or equal to 0")
    if min_speech_ms < 0:
        raise WhisperToMeError("min-speech-ms must be greater than or equal to 0")

    logger = logging.getLogger(__name__)
    active_vad_threshold = (
        config.audio.vad_rms_threshold if vad_threshold is None else vad_threshold
    )
    if allow_non_npu:
        logger.warning(
            "demo_debug_non_npu_enabled=true production_npu_policy_unchanged=true"
        )
    else:
        logger.info("demo_npu_only_policy=true")

    from whispertome.audio.playback import SpeakerOutput
    from whispertome.llm.openai_responses import OpenAIResponder
    from whispertome.organizer.tools import build_organization_tool_registry, build_organizer_store
    from whispertome.text.markdown import parse_spoken_markdown

    organizer_store = build_organizer_store(config.project_root)
    responder = OpenAIResponder(
        config.openai,
        tool_registry=build_organization_tool_registry(
            config.project_root,
            store=organizer_store,
        ),
        system_context_provider=organizer_store.preference_prompt_context,
    )
    speaker = SpeakerOutput() if play else None
    stt_model = prepare_stt_model(
        config,
        allow_non_npu=allow_non_npu,
        prefer_debug_non_npu=should_prefer_debug_stt(config, allow_non_npu),
    )
    tts_model = prepare_tts_model(
        config,
        allow_non_npu=allow_non_npu,
        prefer_debug_non_npu=allow_non_npu,
    )
    if warmup:
        warmup_started = perf_counter()
        logger.info("demo_warmup_started")
        stt_model.load()
        tts_model.load()
        logger.info(
            "demo_warmup_complete latency_ms=%.1f",
            (perf_counter() - warmup_started) * 1000.0,
        )

    audio_dir = config.project_root / "artifacts" / "demo" / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    if save_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        (
            "demo_started record_ms=%d max_turns=%d allow_non_npu=%s save_audio=%s "
            "play=%s fixed_record=%s vad_threshold=%.6f min_speech_ms=%d"
        ),
        record_ms,
        max_turns,
        allow_non_npu,
        save_audio,
        play,
        fixed_record,
        active_vad_threshold,
        min_speech_ms,
    )
    print("Demo ready. Press Enter to listen for speech, or type q then Enter to quit.")

    turn_profiles: list[dict[str, float]] = []
    turn = 0
    while max_turns == 0 or turn < max_turns:
        try:
            action = input("Press Enter to listen, q to quit> ").strip().lower()
        except EOFError:
            logger.info("demo_stdin_eof")
            break
        if action in {"q", "quit", "exit"}:
            logger.info("demo_user_quit")
            break

        turn += 1
        turn_started = perf_counter()
        logger.info("demo_turn_start turn=%d", turn)
        try:
            if fixed_record:
                print(f"Recording {record_ms / 1000.0:.1f}s...")
            else:
                print(
                    f"Listening until speech pause, max {record_ms / 1000.0:.1f}s..."
                )

            record_started = perf_counter()
            if fixed_record:
                audio = record_audio_buffer(config.audio.sample_rate, record_ms)
                recording = RecordingResult(
                    audio=audio,
                    raw_audio=audio,
                    speech_detected=True,
                    reason="fixed_record",
                    speech_start_ms=None,
                    speech_end_ms=None,
                )
            else:
                recording = record_until_speech_pause(
                    config,
                    max_record_ms=record_ms,
                    vad_threshold=active_vad_threshold,
                )
                audio = recording.audio
            stt_audio = recording.raw_audio if not fixed_record else recording.audio
            stt_audio_source = "raw" if stt_audio is recording.raw_audio else "trimmed"
            record_wall_ms = (perf_counter() - record_started) * 1000.0

            audio_profile = profile_audio(audio, active_vad_threshold, config.audio.block_ms)
            raw_profile = profile_audio(
                recording.raw_audio,
                active_vad_threshold,
                config.audio.block_ms,
            )
            stt_profile = profile_audio(
                stt_audio,
                active_vad_threshold,
                config.audio.block_ms,
            )
            logger.info(
                (
                    "demo_recording turn=%d wall_ms=%.1f reason=%s speech_detected=%s "
                    "speech_start_ms=%s speech_end_ms=%s raw_duration_ms=%d duration_ms=%d "
                    "stt_audio_source=%s stt_duration_ms=%d active_ms=%d rms=%.6f "
                    "peak=%.6f clipped_ratio=%.4f"
                ),
                turn,
                record_wall_ms,
                recording.reason,
                recording.speech_detected,
                recording.speech_start_ms,
                recording.speech_end_ms,
                raw_profile.duration_ms,
                audio_profile.duration_ms,
                stt_audio_source,
                stt_profile.duration_ms,
                stt_profile.active_ms,
                stt_profile.rms,
                stt_profile.peak,
                stt_profile.clipped_ratio,
            )
            if stt_profile.clipped_ratio > 0.001 or stt_profile.peak >= 0.95:
                logger.warning(
                    (
                        "demo_audio_clipping turn=%d peak=%.6f clipped_ratio=%.4f "
                        "lower_microphone_gain_or_move_farther_from_mic=true"
                    ),
                    turn,
                    stt_profile.peak,
                    stt_profile.clipped_ratio,
                )

            if save_audio:
                user_audio_path = audio_dir / f"turn-{turn:03d}-user.wav"
                save_audio_buffer(audio, user_audio_path)
                logger.info("demo_user_audio turn=%d path=%s", turn, user_audio_path)
                if recording.raw_audio.duration_ms != audio.duration_ms:
                    raw_audio_path = audio_dir / f"turn-{turn:03d}-raw.wav"
                    save_audio_buffer(recording.raw_audio, raw_audio_path)
                    logger.info("demo_raw_audio turn=%d path=%s", turn, raw_audio_path)

            if (
                not recording.speech_detected
                or stt_profile.active_ms < min_speech_ms
            ):
                logger.warning(
                    (
                        "demo_skip_short_or_silent_audio turn=%d speech_detected=%s "
                        "active_ms=%d min_speech_ms=%d"
                    ),
                    turn,
                    recording.speech_detected,
                    stt_profile.active_ms,
                    min_speech_ms,
                )
                print("No clear speech detected; skipped.")
                continue

            stt_started = perf_counter()
            transcript = stt_model.transcribe(stt_audio)
            stt_wall_ms = (perf_counter() - stt_started) * 1000.0
            user_text = transcript.text.strip()
            logger.info(
                "demo_transcript turn=%d wall_ms=%.1f latency_ms=%.1f provider=%s text=%r",
                turn,
                stt_wall_ms,
                transcript.latency_ms,
                transcript.provider,
                user_text,
            )
            print(f"You: {user_text or '<empty>'}")
            if not user_text:
                logger.warning("demo_empty_transcript turn=%d", turn)
                continue
            if user_text.lower() in {"q", "quit", "exit", "stop"}:
                logger.info("demo_transcript_quit turn=%d", turn)
                break

            def on_demo_tool_event(event: Any, *, turn_number: int = turn) -> None:
                logger.info(
                    "demo_agent_tool turn=%d name=%s ok=%s latency_ms=%.1f args=%r output=%r",
                    turn_number,
                    event.name,
                    event.ok,
                    event.latency_ms,
                    event.arguments,
                    event.output,
                )

            openai_started = perf_counter()
            llm_response = responder.generate(
                user_text,
                on_tool_event=on_demo_tool_event,
            )
            openai_wall_ms = (perf_counter() - openai_started) * 1000.0
            spoken_response = parse_spoken_markdown(llm_response.text)
            logger.info(
                (
                    "demo_openai turn=%d wall_ms=%.1f latency_ms=%.1f model=%s "
                    "response_id=%s chars=%d speech_chars=%d code_blocks=%d text=%r"
                ),
                turn,
                openai_wall_ms,
                llm_response.latency_ms,
                llm_response.model,
                llm_response.response_id,
                len(llm_response.text),
                len(spoken_response.speech_text),
                len(spoken_response.code_blocks),
                llm_response.text,
            )
            print(f"Assistant: {spoken_response.display_text}")

            tts_started = perf_counter()
            speech = tts_model.synthesize(spoken_response.speech_text)
            tts_wall_ms = (perf_counter() - tts_started) * 1000.0
            logger.info(
                "demo_tts turn=%d wall_ms=%.1f latency_ms=%.1f provider=%s sample_rate=%d",
                turn,
                tts_wall_ms,
                speech.latency_ms,
                speech.provider,
                speech.speech.sample_rate,
            )
            if save_audio:
                assistant_audio_path = audio_dir / f"turn-{turn:03d}-assistant.wav"
                save_audio_buffer(speech.speech, assistant_audio_path)
                logger.info(
                    "demo_assistant_audio turn=%d path=%s",
                    turn,
                    assistant_audio_path,
                )
            if speaker is not None:
                playback_started = perf_counter()
                speaker.play(speech.speech)
                playback_ms = (perf_counter() - playback_started) * 1000.0
                logger.info("demo_played turn=%d latency_ms=%.1f", turn, playback_ms)
            else:
                playback_ms = 0.0
            turn_total_ms = (perf_counter() - turn_started) * 1000.0
            turn_profile = {
                "record_ms": record_wall_ms,
                "stt_ms": stt_wall_ms,
                "openai_ms": openai_wall_ms,
                "tts_ms": tts_wall_ms,
                "playback_ms": playback_ms,
                "total_ms": turn_total_ms,
            }
            turn_profiles.append(turn_profile)
            logger.info(
                (
                    "demo_turn_profile turn=%d total_ms=%.1f record_ms=%.1f "
                    "audio_duration_ms=%d active_ms=%d stt_wall_ms=%.1f "
                    "stt_audio_source=%s stt_duration_ms=%d stt_model_ms=%.1f "
                    "openai_ms=%.1f tts_wall_ms=%.1f tts_model_ms=%.1f "
                    "playback_ms=%.1f response_chars=%d"
                ),
                turn,
                turn_total_ms,
                record_wall_ms,
                audio_profile.duration_ms,
                stt_profile.active_ms,
                stt_wall_ms,
                stt_audio_source,
                stt_profile.duration_ms,
                transcript.latency_ms,
                openai_wall_ms,
                tts_wall_ms,
                speech.latency_ms,
                playback_ms,
                len(llm_response.text),
            )
            logger.info("demo_turn_complete turn=%d", turn)
        except Exception:
            logger.exception("demo_turn_failed turn=%d", turn)
            raise

    if turn_profiles:
        summarize_demo_profiles(logger, turn_profiles)
    logger.info("demo_finished turns=%d", turn)
    return 0


def prepare_stt_model(
    config: AppConfig,
    *,
    allow_non_npu: bool,
    prefer_debug_non_npu: bool | None = None,
) -> SpeechToTextModel:
    logger = logging.getLogger(__name__)
    from whispertome.models.registry import ModelRegistry

    prefer_debug = (
        should_prefer_debug_stt(config, allow_non_npu)
        if prefer_debug_non_npu is None
        else prefer_debug_non_npu
    )
    if allow_non_npu and prefer_debug:
        logger.warning("debug_non_npu_stt_direct=true")
        return get_whisper_debug_non_npu(config)

    return ModelRegistry(config).create_stt()


def prepare_tts_model(
    config: AppConfig,
    *,
    allow_non_npu: bool,
    prefer_debug_non_npu: bool = False,
) -> TextToSpeechModel:
    logger = logging.getLogger(__name__)
    from whispertome.models.registry import ModelRegistry

    debug_tts = DebugKokoroSynthesizer(config)
    if allow_non_npu and prefer_debug_non_npu:
        logger.warning("debug_non_npu_tts_direct=true")
        return debug_tts

    tts = ModelRegistry(config).create_tts()
    if not allow_non_npu:
        return tts

    return FallbackTextToSpeechModel(
        primary=tts,
        fallback=debug_tts,
        logger=logger,
    )


def preload_debug_models(config: AppConfig) -> None:
    prepare_stt_model(
        config,
        allow_non_npu=True,
        prefer_debug_non_npu=should_prefer_debug_stt(config, allow_non_npu=True),
    ).load()
    prepare_tts_model(config, allow_non_npu=True, prefer_debug_non_npu=True).load()


def should_prefer_debug_stt(config: AppConfig, allow_non_npu: bool) -> bool:
    return allow_non_npu and config.stt.backend == "whisper_onnx"


def summarize_demo_profiles(
    logger: logging.Logger,
    profiles: list[dict[str, float]],
) -> None:
    def avg(key: str) -> float:
        return sum(profile[key] for profile in profiles) / len(profiles)

    logger.info(
        (
            "demo_profile_summary turns=%d avg_total_ms=%.1f avg_record_ms=%.1f "
            "avg_stt_ms=%.1f avg_openai_ms=%.1f avg_tts_ms=%.1f avg_playback_ms=%.1f"
        ),
        len(profiles),
        avg("total_ms"),
        avg("record_ms"),
        avg("stt_ms"),
        avg("openai_ms"),
        avg("tts_ms"),
        avg("playback_ms"),
    )


def record_until_speech_pause(
    config: AppConfig,
    *,
    max_record_ms: int,
    vad_threshold: float,
) -> RecordingResult:
    if max_record_ms <= 0:
        raise WhisperToMeError("record-ms must be greater than 0")

    from whispertome.audio.capture import MicrophoneInput
    from whispertome.audio.vad import EnergyVad

    audio_config = config.audio
    vad = EnergyVad(vad_threshold)
    pre_roll: deque[Any] = deque(
        maxlen=max(1, audio_config.pre_roll_ms // audio_config.block_ms)
    )
    raw_chunks: list[Any] = []
    active_chunks: list[Any] = []
    speech_run = 0
    silence_run = 0
    in_speech = False
    speech_detected = False
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None
    reason = "max_record_ms"
    speech_start_chunks = max(1, audio_config.speech_start_ms // audio_config.block_ms)
    speech_end_chunks = max(1, audio_config.speech_end_ms // audio_config.block_ms)

    chunk_iter = MicrophoneInput(audio_config).chunks()
    try:
        for chunk in chunk_iter:
            raw_chunks.append(chunk)
            elapsed_ms = chunk.timestamp_ms + chunk.duration_ms
            decision = vad.classify(chunk)

            if not in_speech:
                pre_roll.append(chunk)
                speech_run = speech_run + 1 if decision.is_speech else 0
                if speech_run >= speech_start_chunks:
                    in_speech = True
                    speech_detected = True
                    active_chunks = list(pre_roll)
                    speech_start_ms = active_chunks[0].timestamp_ms
                    silence_run = 0
                if elapsed_ms >= max_record_ms:
                    reason = "no_speech_timeout"
                    break
                continue

            active_chunks.append(chunk)
            if decision.is_speech:
                silence_run = 0
                speech_end_ms = elapsed_ms
            else:
                silence_run += 1

            if silence_run >= speech_end_chunks:
                reason = "speech_pause"
                break
            if elapsed_ms >= max_record_ms:
                reason = "max_record_ms"
                break
    finally:
        close = getattr(chunk_iter, "close", None)
        if close is not None:
            close()

    raw_audio = merge_audio_chunks(raw_chunks, audio_config.sample_rate)
    if speech_detected and active_chunks:
        audio = merge_audio_chunks(active_chunks, audio_config.sample_rate)
    else:
        audio = raw_audio

    return RecordingResult(
        audio=audio,
        raw_audio=raw_audio,
        speech_detected=speech_detected,
        reason=reason,
        speech_start_ms=speech_start_ms,
        speech_end_ms=speech_end_ms,
    )


def merge_audio_chunks(chunks: list[Any], sample_rate: int):
    import numpy as np

    from whispertome.audio.types import AudioBuffer

    if not chunks:
        return AudioBuffer(samples=np.zeros(1, dtype=np.float32), sample_rate=sample_rate)
    samples = np.concatenate([chunk.samples for chunk in chunks]).astype(
        np.float32,
        copy=False,
    )
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def profile_audio(audio, vad_threshold: float, block_ms: int) -> AudioProfile:
    import numpy as np

    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        return AudioProfile(
            duration_ms=0,
            rms=0.0,
            peak=0.0,
            active_ms=0,
            clipped_ratio=0.0,
        )

    frame_count = max(1, int(audio.sample_rate * block_ms / 1000))
    active_frames = 0
    for start in range(0, samples.size, frame_count):
        chunk = samples[start : start + frame_count]
        if chunk.size and float(np.sqrt(np.mean(np.square(chunk)))) >= vad_threshold:
            active_frames += 1

    peak = float(np.max(np.abs(samples)))
    clipped_ratio = float(np.mean(np.abs(samples) >= 0.98))
    return AudioProfile(
        duration_ms=audio.duration_ms,
        rms=float(np.sqrt(np.mean(np.square(samples)))),
        peak=peak,
        active_ms=active_frames * block_ms,
        clipped_ratio=clipped_ratio,
    )


def run_test_tts(
    config: AppConfig,
    text: str,
    output_path: Path,
    *,
    play: bool,
    allow_non_npu: bool,
) -> int:
    logger = logging.getLogger(__name__)

    output = output_path
    if not output.is_absolute():
        output = config.project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    result = synthesize_text(config, text, allow_non_npu=allow_non_npu)

    try:
        import soundfile as sf
    except ImportError as exc:
        raise WhisperToMeError("soundfile is required to write TTS output") from exc

    sf.write(str(output), result.speech.samples, result.speech.sample_rate)
    if play:
        from whispertome.audio.playback import SpeakerOutput

        SpeakerOutput().play(result.speech)
    logger.info(
        "tts_output=%s sample_rate=%d latency_ms=%.1f provider=%s played=%s",
        output,
        result.speech.sample_rate,
        result.latency_ms,
        result.provider,
        play,
    )
    return 0


def synthesize_text(
    config: AppConfig,
    text: str,
    *,
    allow_non_npu: bool,
    prefer_debug_non_npu: bool = False,
):
    tts = prepare_tts_model(
        config,
        allow_non_npu=allow_non_npu,
        prefer_debug_non_npu=prefer_debug_non_npu,
    )
    return tts.synthesize(text)


def synthesize_kokoro_debug_non_npu(config: AppConfig, text: str):
    return DebugKokoroSynthesizer(config).synthesize(text)


def get_kokoro_debug_non_npu(config: AppConfig):
    if config.tts.backend != "kokoro_onnx":
        raise WhisperToMeError(
            f"debug non-NPU TTS only supports kokoro_onnx, got {config.tts.backend}"
        )
    if not config.tts.model_path.exists():
        raise WhisperToMeError(f"Kokoro model file does not exist: {config.tts.model_path}")
    if not config.tts.voice_path.exists():
        raise WhisperToMeError(f"Kokoro voices file does not exist: {config.tts.voice_path}")

    try:
        from kokoro_onnx import Kokoro
    except ImportError as exc:
        raise WhisperToMeError("kokoro-onnx is required for debug non-NPU TTS") from exc

    cache_key = (str(config.tts.model_path), str(config.tts.voice_path))
    kokoro = _DEBUG_KOKORO_CACHE.get(cache_key)
    if kokoro is None:
        kokoro = Kokoro(str(config.tts.model_path), str(config.tts.voice_path))
        _DEBUG_KOKORO_CACHE[cache_key] = kokoro
    return kokoro


def run_test_audio(duration_ms: int, frequency: float, volume: float) -> int:
    logger = logging.getLogger(__name__)
    if duration_ms <= 0:
        raise WhisperToMeError("duration-ms must be greater than 0")
    if frequency <= 0:
        raise WhisperToMeError("frequency must be greater than 0")
    if volume < 0 or volume > 1:
        raise WhisperToMeError("volume must be between 0.0 and 1.0")

    import numpy as np

    from whispertome.audio.playback import SpeakerOutput
    from whispertome.audio.types import SynthesizedSpeech

    sample_rate = 48_000
    duration_s = duration_ms / 1000.0
    sample_count = max(1, int(sample_rate * duration_s))
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = (volume * np.sin(2.0 * np.pi * frequency * time_axis)).astype(np.float32)

    fade_count = min(int(sample_rate * 0.01), sample_count // 2)
    if fade_count:
        fade = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
        samples[:fade_count] *= fade
        samples[-fade_count:] *= fade[::-1]

    SpeakerOutput().play(SynthesizedSpeech(samples=samples, sample_rate=sample_rate))
    logger.info(
        "audio_test_played duration_ms=%d frequency=%.1f volume=%.2f",
        duration_ms,
        frequency,
        volume,
    )
    return 0


def run_test_stt(
    config: AppConfig,
    *,
    wav_path: Path | None,
    record_ms: int | None,
    allow_non_npu: bool,
) -> int:
    logger = logging.getLogger(__name__)

    if wav_path is not None:
        path = wav_path if wav_path.is_absolute() else config.project_root / wav_path
        audio = load_audio_buffer_from_wav(path)
    else:
        assert record_ms is not None
        audio = record_audio_buffer(config.audio.sample_rate, record_ms)

    transcript = transcribe_audio(config, audio, allow_non_npu=allow_non_npu)

    logger.info(
        "stt_text=%r language=%s sample_rate=%d duration_ms=%d latency_ms=%.1f provider=%s",
        transcript.text,
        transcript.language,
        audio.sample_rate,
        audio.duration_ms,
        transcript.latency_ms,
        transcript.provider,
    )
    return 0


def transcribe_audio(
    config: AppConfig,
    audio,
    *,
    allow_non_npu: bool,
    prefer_debug_non_npu: bool | None = None,
):
    stt = prepare_stt_model(
        config,
        allow_non_npu=allow_non_npu,
        prefer_debug_non_npu=prefer_debug_non_npu,
    )
    return stt.transcribe(audio)


def transcribe_whisper_debug_non_npu(config: AppConfig, audio):
    return get_whisper_debug_non_npu(config).transcribe(audio)


def get_whisper_debug_non_npu(config: AppConfig):
    from whispertome.stt.whisper_onnx import WhisperOnnxDebugTranscriber

    cache_key = (
        str(config.stt.model_path),
        config.stt.language,
        config.stt.max_tokens,
        config.stt.onnx_variant,
        config.stt.prompt,
    )
    debug_stt = _DEBUG_STT_CACHE.get(cache_key)
    if debug_stt is None:
        debug_stt = WhisperOnnxDebugTranscriber(config.stt)
        _DEBUG_STT_CACHE[cache_key] = debug_stt
    return debug_stt


def load_audio_buffer_from_wav(path: Path):
    if not path.exists():
        raise WhisperToMeError(f"WAV file does not exist: {path}")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise WhisperToMeError("soundfile is required to read WAV files") from exc

    from whispertome.audio.types import AudioBuffer

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return AudioBuffer(samples=samples, sample_rate=int(sample_rate))


def save_audio_buffer(audio, path: Path) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise WhisperToMeError("soundfile is required to save WAV files") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.samples, audio.sample_rate)


def record_audio_buffer(sample_rate: int, duration_ms: int):
    if duration_ms <= 0:
        raise WhisperToMeError("record-ms must be greater than 0")
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise WhisperToMeError("sounddevice is required for microphone recording") from exc
    import numpy as np

    from whispertome.audio.types import AudioBuffer

    frame_count = max(1, int(sample_rate * duration_ms / 1000))
    recording = sd.rec(
        frame_count,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return AudioBuffer(
        samples=np.asarray(recording[:, 0], dtype=np.float32),
        sample_rate=sample_rate,
    )


def run_test_openai(config: AppConfig, text: str, *, reset: bool) -> int:
    logger = logging.getLogger(__name__)
    from whispertome.llm.openai_responses import OpenAIResponder
    from whispertome.organizer.tools import build_organization_tool_registry, build_organizer_store

    organizer_store = build_organizer_store(config.project_root)
    responder = OpenAIResponder(
        config.openai,
        tool_registry=build_organization_tool_registry(
            config.project_root,
            store=organizer_store,
        ),
        system_context_provider=organizer_store.preference_prompt_context,
    )
    if reset:
        responder.reset_conversation()

    def on_tool_event(event) -> None:  # type: ignore[no-untyped-def]
        logger.info(
            "openai_agent_tool name=%s ok=%s latency_ms=%.1f args=%r output=%r",
            event.name,
            event.ok,
            event.latency_ms,
            event.arguments,
            event.output,
        )

    response = responder.generate(text, on_tool_event=on_tool_event)
    logger.info(
        "openai_response model=%s response_id=%s latency_ms=%.1f text=%r",
        response.model,
        response.response_id,
        response.latency_ms,
        response.text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
