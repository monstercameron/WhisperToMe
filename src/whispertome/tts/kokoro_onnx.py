from __future__ import annotations

from time import perf_counter

from whispertome.audio.types import SynthesizedSpeech
from whispertome.config import TTSConfig
from whispertome.errors import ModelNotReadyError
from whispertome.runtime.base import OnnxSessionHandle
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.tts.base import SpeechSynthesisResult, TextToSpeechModel


class KokoroOnnxSynthesizer(TextToSpeechModel):
    """Kokoro ONNX adapter.

    Kokoro requires text normalization, phonemization, token mapping, and voice/style vectors.
    Those pieces stay inside this adapter so the pipeline does not care which TTS model is used.
    """

    def __init__(self, config: TTSConfig, session_factory: NpuOnlyOnnxSessionFactory) -> None:
        self._config = config
        self._session_factory = session_factory
        self._handle: OnnxSessionHandle | None = None
        self._kokoro = None

    def load(self) -> None:
        if not self._config.voice_path.exists():
            raise ModelNotReadyError(f"Kokoro voices file does not exist: {self._config.voice_path}")
        if self._handle is None:
            self._handle = self._session_factory.create(self._config.model_path, label="kokoro-tts")
        if self._kokoro is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise ModelNotReadyError(
                    "kokoro-onnx is required for Kokoro text preprocessing. "
                    "Install the project TTS dependencies first."
                ) from exc
            self._kokoro = Kokoro.from_session(
                self._handle.session,
                str(self._config.voice_path),
            )

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        self.load()
        assert self._handle is not None
        assert self._kokoro is not None

        started = perf_counter()
        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self._config.voice,
                speed=self._config.speed,
                lang=self._config.language,
            )
        except AssertionError as exc:
            raise ModelNotReadyError(str(exc)) from exc

        latency_ms = (perf_counter() - started) * 1000.0
        return SpeechSynthesisResult(
            speech=SynthesizedSpeech(samples=samples, sample_rate=sample_rate),
            latency_ms=latency_ms,
            model=str(self._config.model_path),
            provider=self._handle.provider.onnx_name,
        )

    @property
    def provider_name(self) -> str | None:
        if self._handle is None:
            return None
        return self._handle.provider.onnx_name
