from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from whispertome.errors import ConfigError


DEFAULT_OPENAI_SYSTEM_PROMPT = """
You are WhisperToMe, a concise spoken assistant in an always-listening voice loop.

The user text you receive is a speech-to-text transcript, not typed text. It may contain
missing punctuation, casing errors, homophones, repeated words, partial phrases, or
small recognition mistakes. Infer the most likely intent from context, but ask one short
clarifying question when a transcript is ambiguous enough that acting would be risky.

Respond for text-to-speech:
- Keep answers brief, natural, and easy to say out loud.
- Default to one short sentence. Use two or three only when needed for clarity.
- Do not mention transcription errors unless they change the meaning.
- Avoid markdown tables, code blocks, bullet-heavy formatting, URLs, and visual layout.
- When the user is dictating text to be written, preserve their wording as much as possible
  and lightly repair punctuation and obvious speech-recognition errors.
- When the user gives a command, answer with the result or the next useful question.
""".strip()


DEFAULT_STT_PROMPT = (
    "Casual spoken voice assistant commands and dictation. Transcribe the exact words, "
    "including slang, profanity, sexual words, unusual phrases, names, and homophones."
)


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _env(name)
    if value is None:
        return default
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    return tuple(part for part in parts if part)


def clean_phrase_list(phrases: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    cleaned = tuple(phrase.strip() for phrase in phrases if phrase.strip())
    if not cleaned:
        raise ConfigError("At least one wake phrase is required.")
    return cleaned


def _env_path(name: str, project_root: Path, default: str) -> Path:
    raw = _env(name, default)
    assert raw is not None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None
    model: str
    system_prompt: str
    max_output_tokens: int | None
    stateful: bool


@dataclass(frozen=True)
class RuntimeConfig:
    provider_order: tuple[str, ...]
    require_npu: bool
    directml_npu_confirmed: bool
    directml_device_id: int
    directml_adapter_name: str | None
    enable_onnx_profiling: bool


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    block_ms: int
    vad_rms_threshold: float
    speech_start_ms: int
    speech_end_ms: int
    pre_roll_ms: int
    max_utterance_ms: int


@dataclass(frozen=True)
class STTConfig:
    backend: str
    model_path: Path
    language: str
    max_tokens: int
    onnx_variant: str
    prompt: str


@dataclass(frozen=True)
class TTSConfig:
    backend: str
    model_path: Path
    voice_path: Path
    voice: str
    language: str
    speed: float
    sample_rate: int


@dataclass(frozen=True)
class WakeConfig:
    phrases: tuple[str, ...]
    fuzzy_threshold: float
    window_words: int
    cooldown_ms: int


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    openai: OpenAIConfig
    runtime: RuntimeConfig
    audio: AudioConfig
    stt: STTConfig
    tts: TTSConfig
    wake: WakeConfig


def with_wake_phrases(config: AppConfig, phrases: tuple[str, ...] | list[str]) -> AppConfig:
    return replace(config, wake=replace(config.wake, phrases=clean_phrase_list(phrases)))


def load_config(project_root: Path | None = None, *, require_openai_key: bool = True) -> AppConfig:
    root = (project_root or Path.cwd()).resolve()
    load_dotenv(root / ".env")

    api_key = _env("OPENAI_API_KEY") or _env("openai")
    if require_openai_key and not api_key:
        raise ConfigError("OPENAI_API_KEY is missing. Put it in .env or the process environment.")

    max_output_tokens = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 64)
    stt_onnx_variant = _env("WHISPERTOME_STT_ONNX_VARIANT", "fp32") or "fp32"
    if stt_onnx_variant not in {"fp32", "int8", "auto"}:
        raise ConfigError("WHISPERTOME_STT_ONNX_VARIANT must be fp32, int8, or auto")

    return AppConfig(
        project_root=root,
        openai=OpenAIConfig(
            api_key=api_key,
            model=_env("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
            system_prompt=_env(
                "OPENAI_SYSTEM_PROMPT",
                DEFAULT_OPENAI_SYSTEM_PROMPT,
            )
            or DEFAULT_OPENAI_SYSTEM_PROMPT,
            max_output_tokens=max_output_tokens,
            stateful=_env_bool("OPENAI_STATEFUL", True),
        ),
        runtime=RuntimeConfig(
            provider_order=_env_list("WHISPERTOME_PROVIDER_ORDER", ("directml", "qnn_htp")),
            require_npu=_env_bool("WHISPERTOME_REQUIRE_NPU", True),
            directml_npu_confirmed=_env_bool("WHISPERTOME_DIRECTML_NPU_CONFIRMED", False),
            directml_device_id=_env_int("WHISPERTOME_DIRECTML_DEVICE_ID", 0),
            directml_adapter_name=_env("WHISPERTOME_DIRECTML_ADAPTER_NAME"),
            enable_onnx_profiling=_env_bool("WHISPERTOME_ONNX_PROFILING", False),
        ),
        audio=AudioConfig(
            sample_rate=_env_int("WHISPERTOME_AUDIO_SAMPLE_RATE", 16000),
            channels=_env_int("WHISPERTOME_AUDIO_CHANNELS", 1),
            block_ms=_env_int("WHISPERTOME_AUDIO_BLOCK_MS", 30),
            vad_rms_threshold=_env_float("WHISPERTOME_VAD_RMS_THRESHOLD", 0.012),
            speech_start_ms=_env_int("WHISPERTOME_SPEECH_START_MS", 150),
            speech_end_ms=_env_int("WHISPERTOME_SPEECH_END_MS", 700),
            pre_roll_ms=_env_int("WHISPERTOME_PRE_ROLL_MS", 600),
            max_utterance_ms=_env_int("WHISPERTOME_MAX_UTTERANCE_MS", 15000),
        ),
        stt=STTConfig(
            backend=_env("WHISPERTOME_STT_BACKEND", "qai_whisper") or "qai_whisper",
            model_path=_env_path(
                "WHISPERTOME_STT_MODEL_PATH",
                root,
                (
                    "models/qai/whisper_small/snapdragon_x2_elite/precompiled_qnn_onnx/"
                    "extracted/whisper_small-precompiled_qnn_onnx-float-"
                    "qualcomm_snapdragon_x2_elite"
                ),
            ),
            language=_env("WHISPERTOME_STT_LANGUAGE", "en") or "en",
            max_tokens=_env_int("WHISPERTOME_STT_MAX_TOKENS", 64),
            onnx_variant=stt_onnx_variant,
            prompt=_env("WHISPERTOME_STT_PROMPT", DEFAULT_STT_PROMPT) or DEFAULT_STT_PROMPT,
        ),
        tts=TTSConfig(
            backend=_env("WHISPERTOME_TTS_BACKEND", "kokoro_onnx") or "kokoro_onnx",
            model_path=_env_path(
                "WHISPERTOME_TTS_MODEL_PATH",
                root,
                "models/kokoro/kokoro-v1.0.onnx",
            ),
            voice_path=_env_path(
                "WHISPERTOME_TTS_VOICE_PATH",
                root,
                "models/kokoro/voices-v1.0.bin",
            ),
            voice=_env("WHISPERTOME_TTS_VOICE", "af_bella") or "af_bella",
            language=_env("WHISPERTOME_TTS_LANGUAGE", "en-us") or "en-us",
            speed=_env_float("WHISPERTOME_TTS_SPEED", 1.0),
            sample_rate=_env_int("WHISPERTOME_TTS_SAMPLE_RATE", 24000),
        ),
        wake=WakeConfig(
            phrases=clean_phrase_list(_env_list("WHISPERTOME_WAKE_PHRASES", ("whisper to me",))),
            fuzzy_threshold=_env_float("WHISPERTOME_WAKE_FUZZY_THRESHOLD", 0.88),
            window_words=_env_int("WHISPERTOME_WAKE_WINDOW_WORDS", 8),
            cooldown_ms=_env_int("WHISPERTOME_WAKE_COOLDOWN_MS", 2500),
        ),
    )
