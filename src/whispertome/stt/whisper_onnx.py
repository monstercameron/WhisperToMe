from __future__ import annotations

from whispertome.audio.types import AudioBuffer
from whispertome.config import STTConfig
from whispertome.errors import ModelNotReadyError
from whispertome.runtime.base import OnnxSessionHandle
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.stt.base import SpeechToTextModel, Transcript


class WhisperOnnxTranscriber(SpeechToTextModel):
    """Whisper ONNX adapter.

    This class enforces runtime/session policy now. The actual Whisper decoding adapter is
    intentionally isolated here because exported Whisper graphs differ by toolchain.
    """

    def __init__(self, config: STTConfig, session_factory: NpuOnlyOnnxSessionFactory) -> None:
        self._config = config
        self._session_factory = session_factory
        self._handle: OnnxSessionHandle | None = None

    def load(self) -> None:
        if self._handle is None:
            self._handle = self._session_factory.create(
                self._config.model_path,
                label="whisper-stt",
            )

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        self.load()
        assert self._handle is not None
        _ = audio
        raise ModelNotReadyError(
            "Whisper ONNX session is loaded on an allowed NPU provider, but the "
            "model-specific encoder/decoder/tokenizer adapter is not wired yet. "
            "Add an adapter for the exact Whisper ONNX export format before live STT."
        )

    @property
    def provider_name(self) -> str | None:
        if self._handle is None:
            return None
        return self._handle.provider.onnx_name
