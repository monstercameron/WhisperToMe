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
    def resolve(cls, model_path: Path, *, variant: str = "fp32") -> WhisperOnnxFiles:
        if model_path.is_dir():
            onnx_dir = model_path / "onnx"
            return cls(
                root=model_path,
                encoder_path=_first_existing(onnx_dir, _encoder_candidates(variant)),
                decoder_path=_first_existing(onnx_dir, _decoder_candidates(variant)),
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

        self._files = WhisperOnnxFiles.resolve(
            self._config.model_path,
            variant=self._config.onnx_variant,
        )
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

        files = WhisperOnnxFiles.resolve(
            self._config.model_path,
            variant=self._config.onnx_variant,
        )
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
                    inputs[session_input.name] = _empty_past_tensor(
                        session_input.shape,
                        batch_size=input_ids.shape[0],
                    )
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
            trimmed = _trim_repeated_suffix(generated)
            if len(trimmed) < len(generated):
                generated = trimmed
                break
            past = {
                output.name.replace("present", "past_key_values"): value
                for output, value in zip(self._decoder_session.get_outputs()[1:], outputs[1:])
            }
            input_ids = np.array([[token_id]], dtype=np.int64)

        return generated

    def _initial_prompt_ids(self) -> list[int]:
        forced_decoder_ids = self._generation_config.get("forced_decoder_ids", [])
        language_token = f"<|{self._config.language}|>"
        lang_to_id = self._generation_config.get("lang_to_id", {})
        prompt_ids = self._context_prompt_ids()
        prompt_ids.append(int(self._generation_config["decoder_start_token_id"]))

        if forced_decoder_ids:
            for _position, token_id in sorted(forced_decoder_ids, key=lambda item: item[0]):
                if token_id is None:
                    if language_token not in lang_to_id:
                        continue
                    token_id = lang_to_id[language_token]
                token = int(token_id)
                if token not in prompt_ids:
                    prompt_ids.append(token)
        else:
            if language_token in lang_to_id:
                prompt_ids.append(int(lang_to_id[language_token]))
            task_to_id = self._generation_config.get("task_to_id", {})
            if "transcribe" in task_to_id:
                prompt_ids.append(int(task_to_id["transcribe"]))

        no_timestamps_token_id = self._generation_config.get("no_timestamps_token_id")
        if no_timestamps_token_id is not None:
            token = int(no_timestamps_token_id)
            if token not in prompt_ids:
                prompt_ids.append(token)

        return prompt_ids

    def _context_prompt_ids(self) -> list[int]:
        prompt = self._config.prompt.strip()
        if not prompt:
            return []
        prev_sot_token_id = self._generation_config.get("prev_sot_token_id")
        if prev_sot_token_id is None:
            return []
        encoded = self._tokenizer.encode(prompt, add_special_tokens=False)
        max_prompt_tokens = 96
        return [int(prev_sot_token_id), *[int(token) for token in encoded[-max_prompt_tokens:]]]

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


def _encoder_candidates(variant: str) -> tuple[str, ...]:
    if variant == "fp32":
        return ("encoder_model.onnx",)
    if variant == "int8":
        return ("encoder_model_int8.onnx", "encoder_model_quantized.onnx")
    if variant == "auto":
        return ("encoder_model.onnx", "encoder_model_int8.onnx", "encoder_model_quantized.onnx")
    raise ModelNotReadyError(f"Unsupported Whisper ONNX variant: {variant}")


def _decoder_candidates(variant: str) -> tuple[str, ...]:
    if variant == "fp32":
        return ("decoder_model_merged.onnx",)
    if variant == "int8":
        return ("decoder_model_merged_int8.onnx", "decoder_model_merged_quantized.onnx")
    if variant == "auto":
        return (
            "decoder_model_merged.onnx",
            "decoder_model_merged_int8.onnx",
            "decoder_model_merged_quantized.onnx",
        )
    raise ModelNotReadyError(f"Unsupported Whisper ONNX variant: {variant}")


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


def _empty_past_tensor(shape: list[Any], *, batch_size: int) -> np.ndarray:
    resolved_shape: list[int] = []
    for dimension in shape:
        if dimension == "batch_size":
            resolved_shape.append(batch_size)
        elif isinstance(dimension, int):
            resolved_shape.append(dimension)
        elif isinstance(dimension, str) and "sequence_length" in dimension:
            resolved_shape.append(0)
        else:
            raise ModelNotReadyError(f"Unsupported dynamic decoder cache dimension: {dimension!r}")
    return np.zeros(tuple(resolved_shape), dtype=np.float32)


def _trim_repeated_suffix(token_ids: list[int]) -> list[int]:
    for ngram_size in range(12, 2, -1):
        if len(token_ids) < ngram_size * 2:
            continue
        previous = token_ids[-(ngram_size * 2) : -ngram_size]
        current = token_ids[-ngram_size:]
        if previous == current:
            return token_ids[:-ngram_size]
    return token_ids


def _suppress_logits(logits: np.ndarray, token_ids: set[int]) -> None:
    for token_id in token_ids:
        if 0 <= token_id < logits.shape[0]:
            logits[token_id] = -np.inf
