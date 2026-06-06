from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


def configure_logging(verbose: bool, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

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
    configure_logging(args.verbose, log_file)
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
            from whispertome.pipeline import VoiceLoop

            config = apply_cli_overrides(
                load_config(args.project_root, require_openai_key=True),
                args,
            )
            VoiceLoop.from_config(config).run_forever()
            return 0
        if args.command == "demo":
            config = load_config(args.project_root, require_openai_key=True)
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

    if args.command == "demo":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return args.project_root / "artifacts" / "logs" / f"demo-{stamp}.log"

    return None


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    wake_phrases = getattr(args, "wake_phrases", None)
    if wake_phrases:
        return with_wake_phrases(config, tuple(wake_phrases))
    return config


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

    responder = OpenAIResponder(config.openai)
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

            openai_started = perf_counter()
            llm_response = responder.generate(user_text)
            openai_wall_ms = (perf_counter() - openai_started) * 1000.0
            logger.info(
                (
                    "demo_openai turn=%d wall_ms=%.1f latency_ms=%.1f model=%s "
                    "response_id=%s chars=%d text=%r"
                ),
                turn,
                openai_wall_ms,
                llm_response.latency_ms,
                llm_response.model,
                llm_response.response_id,
                len(llm_response.text),
                llm_response.text,
            )
            print(f"Assistant: {llm_response.text}")

            tts_started = perf_counter()
            speech = tts_model.synthesize(llm_response.text)
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

    responder = OpenAIResponder(config.openai)
    if reset:
        responder.reset_conversation()
    response = responder.generate(text)
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
