from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from whispertome.errors import ConfigError


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

    max_output_raw = _env("OPENAI_MAX_OUTPUT_TOKENS")
    max_output_tokens = int(max_output_raw) if max_output_raw else None

    return AppConfig(
        project_root=root,
        openai=OpenAIConfig(
            api_key=api_key,
            model=_env("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
            system_prompt=_env(
                "OPENAI_SYSTEM_PROMPT",
                "You are WhisperToMe: concise, useful, and conversational.",
            )
            or "You are WhisperToMe: concise, useful, and conversational.",
            max_output_tokens=max_output_tokens,
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
            pre_roll_ms=_env_int("WHISPERTOME_PRE_ROLL_MS", 300),
            max_utterance_ms=_env_int("WHISPERTOME_MAX_UTTERANCE_MS", 15000),
        ),
        stt=STTConfig(
            backend=_env("WHISPERTOME_STT_BACKEND", "whisper_onnx") or "whisper_onnx",
            model_path=_env_path(
                "WHISPERTOME_STT_MODEL_PATH",
                root,
                "models/whisper/whisper-tiny",
            ),
            language=_env("WHISPERTOME_STT_LANGUAGE", "en") or "en",
            max_tokens=_env_int("WHISPERTOME_STT_MAX_TOKENS", 96),
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
