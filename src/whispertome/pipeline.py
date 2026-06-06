from __future__ import annotations

import logging
from dataclasses import dataclass

from whispertome.audio.capture import MicrophoneInput
from whispertome.audio.playback import SpeakerOutput
from whispertome.audio.vad import EnergyVad, UtteranceSegmenter
from whispertome.config import AppConfig
from whispertome.llm.openai_responses import OpenAIResponder
from whispertome.models.registry import ModelRegistry
from whispertome.stt.base import SpeechToTextModel
from whispertome.tts.base import TextToSpeechModel
from whispertome.wake.router import WakeCommandRouter
from whispertome.wake.sliding_window import SlidingWakeDetector


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceLoopComponents:
    microphone: MicrophoneInput
    speaker: SpeakerOutput
    stt: SpeechToTextModel
    tts: TextToSpeechModel
    responder: OpenAIResponder
    wake_router: WakeCommandRouter
    segmenter: UtteranceSegmenter


class VoiceLoop:
    def __init__(self, config: AppConfig, components: VoiceLoopComponents) -> None:
        self._config = config
        self._components = components

    @classmethod
    def from_config(cls, config: AppConfig) -> VoiceLoop:
        registry = ModelRegistry(config)
        vad = EnergyVad(config.audio.vad_rms_threshold)
        return cls(
            config=config,
            components=VoiceLoopComponents(
                microphone=MicrophoneInput(config.audio),
                speaker=SpeakerOutput(),
                stt=registry.create_stt(),
                tts=registry.create_tts(),
                responder=OpenAIResponder(config.openai),
                wake_router=WakeCommandRouter(SlidingWakeDetector(config.wake)),
                segmenter=UtteranceSegmenter(config.audio, vad),
            ),
        )

    def run_forever(self) -> None:
        chunks = self._components.microphone.chunks()
        utterances = self._components.segmenter.utterances(chunks)
        LOGGER.info("Listening with VAD-gated continuous STT")

        for utterance in utterances:
            transcript = self._components.stt.transcribe(utterance)
            text = transcript.text.strip()
            if not text:
                continue

            LOGGER.info(
                "transcript latency_ms=%.1f provider=%s text=%r",
                transcript.latency_ms,
                transcript.provider,
                text,
            )

            event = self._components.wake_router.process_utterance(text)
            if event.kind == "idle":
                continue

            if event.kind == "wake_detected":
                assert event.match is not None
                LOGGER.info(
                    "wake phrase detected phrase=%r score=%.2f; listening until speech pause",
                    event.match.phrase,
                    event.match.score,
                )
                continue

            assert event.command is not None
            self._handle_command(event.command)

    def _handle_command(self, command: str) -> None:
        LOGGER.info("sending command to OpenAI model=%s", self._config.openai.model)
        llm_response = self._components.responder.generate(command)
        LOGGER.info("OpenAI response latency_ms=%.1f", llm_response.latency_ms)
        speech = self._components.tts.synthesize(llm_response.text)
        LOGGER.info(
            "tts latency_ms=%.1f provider=%s",
            speech.latency_ms,
            speech.provider,
        )
        self._components.speaker.play(speech.speech)

