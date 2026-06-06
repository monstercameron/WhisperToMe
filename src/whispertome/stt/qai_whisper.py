from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from whispertome.audio.types import AudioBuffer
from whispertome.config import STTConfig
from whispertome.errors import ModelNotReadyError
from whispertome.runtime.base import OnnxSessionHandle
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.stt.base import SpeechToTextModel, Transcript


TARGET_SAMPLE_RATE = 16_000
HF_MODEL_ID = "openai/whisper-small"
MEAN_DECODE_LEN = 200
MASK_NEG = -100.0


@dataclass(frozen=True)
class QaiWhisperFiles:
    root: Path
    encoder_path: Path
    decoder_path: Path

    @classmethod
    def resolve(cls, model_path: Path) -> QaiWhisperFiles:
        if not model_path.is_dir():
            raise ModelNotReadyError(
                "Qualcomm AI Hub Whisper STT expects a directory containing "
                "encoder.onnx and decoder.onnx."
            )
        return cls(
            root=model_path,
            encoder_path=_required_file(model_path / "encoder.onnx"),
            decoder_path=_required_file(model_path / "decoder.onnx"),
        )


class QaiWhisperTranscriber(SpeechToTextModel):
    """Qualcomm AI Hub Whisper-Small adapter using ONNX Runtime QNN plugin NPU."""

    def __init__(self, config: STTConfig, session_factory: NpuOnlyOnnxSessionFactory) -> None:
        self._config = config
        self._session_factory = session_factory
        self._files: QaiWhisperFiles | None = None
        self._encoder: OnnxSessionHandle | None = None
        self._decoder: OnnxSessionHandle | None = None
        self._runner: QaiWhisperRunner | None = None

    def load(self) -> None:
        if self._runner is not None:
            return

        self._files = QaiWhisperFiles.resolve(self._config.model_path)
        self._encoder = self._session_factory.create(
            self._files.encoder_path,
            label="qai-whisper-stt-encoder",
        )
        self._decoder = self._session_factory.create(
            self._files.decoder_path,
            label="qai-whisper-stt-decoder",
        )
        self._runner = QaiWhisperRunner(
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


class QaiWhisperRunner:
    def __init__(
        self,
        *,
        config: STTConfig,
        files: QaiWhisperFiles,
        encoder_session: Any,
        decoder_session: Any,
        provider: str,
    ) -> None:
        self._config = config
        self._files = files
        self._encoder_session = encoder_session
        self._decoder_session = decoder_session
        self._provider = provider
        self._feature_extractor, self._tokenizer, self._model_config, self._generation_config = (
            self._load_transformers(files.root)
        )
        self._forced_decoder_ids = dict(
            self._tokenizer.get_decoder_prompt_ids(
                language=config.language,
                task="transcribe",
                no_timestamps=True,
            )
        )
        self._max_forced_position = max(self._forced_decoder_ids, default=0)
        self._suppress_tokens = _token_set(
            getattr(self._generation_config, "suppress_tokens", None)
        )
        self._begin_suppress_tokens = _token_set(
            getattr(self._generation_config, "begin_suppress_tokens", None)
        )

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        started = perf_counter()
        samples = _prepare_audio(audio)
        input_features = self._feature_extractor(
            samples,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        ).input_features.astype(np.float16, copy=False)

        cross_cache = self._encoder_session.run(None, {"input_features": input_features})
        token_ids = self._generate(cross_cache)
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
            provider=f"{self._provider}:npu-plugin",
        )

    def _generate(self, cross_cache: list[np.ndarray]) -> list[int]:
        input_meta = self._decoder_session.get_inputs()
        output_meta = self._decoder_session.get_outputs()
        cross_by_name = {
            output.name: value for output, value in zip(self._encoder_session.get_outputs(), cross_cache)
        }
        self_cache = [
            np.zeros(tuple(session_input.shape), dtype=np.float16)
            for session_input in input_meta
            if session_input.name.startswith(("k_cache_self", "v_cache_self"))
        ]

        tokens = [int(self._model_config.decoder_start_token_id)]
        eos_token_id = int(self._model_config.eos_token_id)
        attention_mask = np.full(
            (1, 1, 1, MEAN_DECODE_LEN),
            MASK_NEG,
            dtype=np.float16,
        )

        max_steps = min(self._config.max_tokens, MEAN_DECODE_LEN - 1)
        for step in range(max_steps):
            inputs = {
                "input_ids": np.array([[tokens[step]]], dtype=np.int32),
                "attention_mask": attention_mask,
                "position_ids": np.array([step], dtype=np.int32),
            }
            attention_mask[:, :, :, MEAN_DECODE_LEN - step - 1] = 0.0

            self_cache_index = 0
            for session_input in input_meta:
                if session_input.name in inputs:
                    continue
                if session_input.name.startswith(("k_cache_self", "v_cache_self")):
                    inputs[session_input.name] = self_cache[self_cache_index]
                    self_cache_index += 1
                elif session_input.name.startswith(("k_cache_cross", "v_cache_cross")):
                    inputs[session_input.name] = cross_by_name[session_input.name]

            outputs = self._decoder_session.run(None, inputs)
            logits = outputs[0].reshape(1, -1).astype(np.float32, copy=True)
            next_token_id = self._forced_decoder_ids.get(step + 1)
            if next_token_id is None:
                _suppress_logits(logits[0], self._suppress_tokens)
                if step == self._max_forced_position:
                    _suppress_logits(logits[0], self._begin_suppress_tokens)
                next_token_id = int(np.argmax(logits[0]))

            tokens.append(int(next_token_id))
            self_cache = list(outputs[1:])
            if next_token_id == eos_token_id:
                break

        return tokens

    @staticmethod
    def _load_transformers(model_root: Path) -> tuple[Any, Any, Any, Any]:
        try:
            from transformers import (
                GenerationConfig,
                WhisperConfig,
                WhisperFeatureExtractor,
                WhisperTokenizer,
            )
        except ImportError as exc:
            raise ModelNotReadyError(
                "transformers is required for Qualcomm Whisper preprocessing and decoding"
            ) from exc

        source = _transformers_source(model_root)
        return (
            WhisperFeatureExtractor.from_pretrained(source),
            WhisperTokenizer.from_pretrained(source),
            WhisperConfig.from_pretrained(source),
            GenerationConfig.from_pretrained(source),
        )


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise ModelNotReadyError(f"Required Qualcomm Whisper model file does not exist: {path}")
    return path


def _transformers_source(model_root: Path) -> str | Path:
    for candidate in (model_root, model_root / "hf"):
        if (candidate / "preprocessor_config.json").exists():
            return candidate
    return HF_MODEL_ID


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


def _token_set(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(token_id) for token_id in value if int(token_id) >= 0}


def _suppress_logits(logits: np.ndarray, token_ids: set[int]) -> None:
    for token_id in token_ids:
        if 0 <= token_id < logits.shape[0]:
            logits[token_id] = -np.inf
