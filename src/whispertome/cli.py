from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from whispertome.config import AppConfig, load_config, with_wake_phrases
from whispertome.errors import WhisperToMeError
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.runtime.providers import ProviderResolver
from whispertome.wake.router import WakeCommandRouter
from whispertome.wake.sliding_window import SlidingWakeDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whispertome")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing .env and model artifacts.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check configuration, model paths, and NPU providers.",
    )
    add_wake_args(doctor_parser)

    run_parser = subparsers.add_parser("run", help="Start the live voice loop.")
    add_wake_args(run_parser)

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
    return parser


def add_wake_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wake",
        dest="wake_phrases",
        action="append",
        default=None,
        help="Wake phrase to listen for. Can be repeated. Overrides WHISPERTOME_WAKE_PHRASES.",
    )


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

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
    except WhisperToMeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


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


def run_test_tts(
    config: AppConfig,
    text: str,
    output_path: Path,
    *,
    play: bool,
    allow_non_npu: bool,
) -> int:
    logger = logging.getLogger(__name__)
    from whispertome.models.registry import ModelRegistry

    output = output_path
    if not output.is_absolute():
        output = config.project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    tts = ModelRegistry(config).create_tts()
    try:
        result = tts.synthesize(text)
    except WhisperToMeError as exc:
        if not allow_non_npu:
            raise
        logger.warning(
            "npu_tts_failed=%s debug_non_npu_tts_enabled=true",
            exc,
        )
        result = synthesize_kokoro_debug_non_npu(config, text)

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


def synthesize_kokoro_debug_non_npu(config: AppConfig, text: str):
    logger = logging.getLogger(__name__)
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

    from whispertome.audio.types import SynthesizedSpeech
    from whispertome.tts.base import SpeechSynthesisResult

    started = perf_counter()
    kokoro = Kokoro(str(config.tts.model_path), str(config.tts.voice_path))
    try:
        samples, sample_rate = kokoro.create(
            text,
            voice=config.tts.voice,
            speed=config.tts.speed,
            lang=config.tts.language,
        )
    except AssertionError as exc:
        raise WhisperToMeError(str(exc)) from exc
    latency_ms = (perf_counter() - started) * 1000.0
    providers = ", ".join(kokoro.sess.get_providers())
    logger.warning("debug_non_npu_tts_provider=%s", providers)
    return SpeechSynthesisResult(
        speech=SynthesizedSpeech(samples=samples, sample_rate=sample_rate),
        latency_ms=latency_ms,
        model=str(config.tts.model_path),
        provider=f"debug-non-npu:{providers}",
    )


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
    from whispertome.models.registry import ModelRegistry
    from whispertome.stt.whisper_onnx import WhisperOnnxDebugTranscriber

    if wav_path is not None:
        path = wav_path if wav_path.is_absolute() else config.project_root / wav_path
        audio = load_audio_buffer_from_wav(path)
    else:
        assert record_ms is not None
        audio = record_audio_buffer(config.audio.sample_rate, record_ms)

    stt = ModelRegistry(config).create_stt()
    try:
        transcript = stt.transcribe(audio)
    except WhisperToMeError as exc:
        if not allow_non_npu:
            raise
        logger.warning(
            "npu_stt_failed=%s debug_non_npu_stt_enabled=true",
            exc,
        )
        transcript = WhisperOnnxDebugTranscriber(config.stt).transcribe(audio)

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
    return AudioBuffer(samples=np.asarray(recording[:, 0], dtype=np.float32), sample_rate=sample_rate)


if __name__ == "__main__":
    raise SystemExit(main())
