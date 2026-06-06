from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from whispertome.audio.types import AudioBuffer
from whispertome.config import STTConfig
from whispertome.errors import ModelNotReadyError, RuntimeUnavailableError
from whispertome.runtime.base import OnnxSessionHandle
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.stt.base import SpeechToTextModel, Transcript


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class WhisperOnnxFiles:
    root: Path
    encoder_path: Path
    decoder_path: Path

    @classmethod
    def resolve(cls, model_path: Path) -> WhisperOnnxFiles:
        if model_path.is_dir():
            onnx_dir = model_path / "onnx"
            return cls(
                root=model_path,
                encoder_path=_first_existing(
                    onnx_dir,
                    (
                        "encoder_model_int8.onnx",
                        "encoder_model_quantized.onnx",
                        "encoder_model.onnx",
                    ),
                ),
                decoder_path=_first_existing(
                    onnx_dir,
                    (
                        "decoder_model_merged_int8.onnx",
                        "decoder_model_merged_quantized.onnx",
                        "decoder_model_merged.onnx",
                    ),
                ),
            )

        raise ModelNotReadyError(
            "Whisper ONNX STT expects a model directory containing Hugging Face "
            "tokenizer/preprocessor files and split ONNX encoder/decoder graphs."
        )


class WhisperOnnxTranscriber(SpeechToTextModel):
    """Whisper ONNX adapter using the production NPU-only runtime policy."""

    def __init__(self, config: STTConfig, session_factory: NpuOnlyOnnxSessionFactory) -> None:
        self._config = config
        self._session_factory = session_factory
        self._files: WhisperOnnxFiles | None = None
        self._encoder: OnnxSessionHandle | None = None
        self._decoder: OnnxSessionHandle | None = None
        self._runner: WhisperOnnxRunner | None = None

    def load(self) -> None:
        if self._runner is not None:
            return

        self._files = WhisperOnnxFiles.resolve(self._config.model_path)
        self._encoder = self._session_factory.create(
            self._files.encoder_path,
            label="whisper-stt-encoder",
        )
        self._decoder = self._session_factory.create(
            self._files.decoder_path,
            label="whisper-stt-decoder",
        )
        self._runner = WhisperOnnxRunner(
            config=self._config,
            files=self._files,
            encoder_session=self._encoder.session,
            decoder_session=self._decoder.session,
            provider=self._encoder.provider.onnx_name,
        )

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        self.load()
        assert self._runner is not None
        return self._runner.transcribe(audio)

    @property
    def provider_name(self) -> str | None:
        if self._encoder is None:
            return None
        return self._encoder.provider.onnx_name


class WhisperOnnxDebugTranscriber(SpeechToTextModel):
    """Debug-only Whisper adapter that uses a caller-selected ONNX Runtime provider."""

    def __init__(self, config: STTConfig, *, provider: str = "CPUExecutionProvider") -> None:
        self._config = config
        self._provider = provider
        self._runner: WhisperOnnxRunner | None = None

    def load(self) -> None:
        if self._runner is not None:
            return

        files = WhisperOnnxFiles.resolve(self._config.model_path)
        ort = self._import_onnxruntime()
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        encoder_session = ort.InferenceSession(
            str(files.encoder_path),
            sess_options=session_options,
            providers=[self._provider],
        )
        decoder_session = ort.InferenceSession(
            str(files.decoder_path),
            sess_options=session_options,
            providers=[self._provider],
        )
        self._runner = WhisperOnnxRunner(
            config=self._config,
            files=files,
            encoder_session=encoder_session,
            decoder_session=decoder_session,
            provider=f"debug-non-npu:{self._provider}",
        )

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        self.load()
        assert self._runner is not None
        return self._runner.transcribe(audio)

    @staticmethod
    def _import_onnxruntime() -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeUnavailableError("onnxruntime is required for debug Whisper STT") from exc
        return ort


class WhisperOnnxRunner:
    def __init__(
        self,
        *,
        config: STTConfig,
        files: WhisperOnnxFiles,
        encoder_session: Any,
        decoder_session: Any,
        provider: str,
    ) -> None:
        self._config = config
        self._files = files
        self._encoder_session = encoder_session
        self._decoder_session = decoder_session
        self._provider = provider
        self._generation_config = self._load_generation_config(files.root)
        self._feature_extractor, self._tokenizer = self._load_transformers(files.root)

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        started = perf_counter()
        samples = _prepare_audio(audio)
        input_features = self._feature_extractor(
            samples,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        ).input_features.astype(np.float32, copy=False)

        encoder_hidden_states = self._encoder_session.run(
            None,
            {"input_features": input_features},
        )[0]
        token_ids = self._generate(encoder_hidden_states)
        text = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return Transcript(
            text=text,
            language=self._config.language,
            latency_ms=(perf_counter() - started) * 1000.0,
            model=str(self._files.root),
            provider=self._provider,
        )

    def _generate(self, encoder_hidden_states: np.ndarray) -> list[int]:
        input_ids = np.array([self._initial_prompt_ids()], dtype=np.int64)
        past: dict[str, np.ndarray] | None = None
        generated: list[int] = []
        eos_token_id = int(self._generation_config["eos_token_id"])
        suppress_tokens = set(self._generation_config.get("suppress_tokens", []))
        begin_suppress_tokens = set(self._generation_config.get("begin_suppress_tokens", []))

        for _ in range(self._config.max_tokens):
            inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "use_cache_branch": np.array([past is not None], dtype=bool),
            }
            for session_input in self._decoder_session.get_inputs():
                if session_input.name in inputs:
                    continue
                if past is None:
                    inputs[session_input.name] = np.zeros((1, 6, 0, 64), dtype=np.float32)
                else:
                    inputs[session_input.name] = past[session_input.name]

            outputs = self._decoder_session.run(None, inputs)
            logits = outputs[0][0, -1].astype(np.float32, copy=True)
            _suppress_logits(logits, suppress_tokens)
            if not generated:
                _suppress_logits(logits, begin_suppress_tokens)

            token_id = int(np.argmax(logits))
            if token_id == eos_token_id:
                break

            generated.append(token_id)
            past = {
                output.name.replace("present", "past_key_values"): value
                for output, value in zip(self._decoder_session.get_outputs()[1:], outputs[1:])
            }
            input_ids = np.array([[token_id]], dtype=np.int64)

        return generated

    def _initial_prompt_ids(self) -> list[int]:
        language_token = f"<|{self._config.language}|>"
        lang_to_id = self._generation_config.get("lang_to_id", {})
        if language_token not in lang_to_id:
            raise ModelNotReadyError(
                f"Whisper language {self._config.language!r} is not available in generation_config.json"
            )

        return [
            int(self._generation_config["decoder_start_token_id"]),
            int(lang_to_id[language_token]),
            int(self._generation_config["task_to_id"]["transcribe"]),
            int(self._generation_config["no_timestamps_token_id"]),
        ]

    @staticmethod
    def _load_generation_config(model_root: Path) -> dict[str, Any]:
        path = model_root / "generation_config.json"
        if not path.exists():
            raise ModelNotReadyError(f"Whisper generation config is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_transformers(model_root: Path) -> tuple[Any, Any]:
        try:
            from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
        except ImportError as exc:
            raise ModelNotReadyError(
                "transformers and tokenizers are required for Whisper STT preprocessing"
            ) from exc

        return (
            WhisperFeatureExtractor.from_pretrained(model_root),
            WhisperTokenizerFast.from_pretrained(
                model_root,
                clean_up_tokenization_spaces=False,
            ),
        )


def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise ModelNotReadyError(
        f"None of the expected Whisper ONNX files exist under {root}: {', '.join(names)}"
    )


def _prepare_audio(audio: AudioBuffer) -> np.ndarray:
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    if audio.sample_rate == TARGET_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)
    return _resample_linear(samples, audio.sample_rate, TARGET_SAMPLE_RATE)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ModelNotReadyError("Audio sample rate must be greater than 0")
    duration_s = samples.shape[0] / source_rate
    output_count = max(1, int(round(duration_s * target_rate)))
    source_axis = np.linspace(0.0, duration_s, num=samples.shape[0], endpoint=False)
    target_axis = np.linspace(0.0, duration_s, num=output_count, endpoint=False)
    return np.interp(target_axis, source_axis, samples).astype(np.float32)


def _suppress_logits(logits: np.ndarray, token_ids: set[int]) -> None:
    for token_id in token_ids:
        if 0 <= token_id < logits.shape[0]:
            logits[token_id] = -np.inf
