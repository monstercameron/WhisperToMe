from __future__ import annotations

import logging
from dataclasses import dataclass

from whispertome.audio.capture import MicrophoneInput
from whispertome.audio.playback import SpeakerOutput
from whispertome.audio.vad import EnergyVad, UtteranceSegmenter
from whispertome.config import AppConfig, active_llm_model
from whispertome.llm.base import LlmResponder
from whispertome.llm.factory import build_llm_responder
from whispertome.models.registry import ModelRegistry
from whispertome.organizer.tools import build_organization_tool_registry, build_organizer_store
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
    responder: LlmResponder
    wake_router: WakeCommandRouter
    segmenter: UtteranceSegmenter


class VoiceLoop:
    def __init__(self, config: AppConfig, components: VoiceLoopComponents) -> None:
        self._config = config
        self._components = components

    @classmethod
    def from_config(cls, config: AppConfig) -> VoiceLoop:
        from whispertome.tts.voice_tools import TtsVoiceController, build_voice_tools

        registry = ModelRegistry(config)
        vad = EnergyVad(config.audio.vad_rms_threshold)
        organizer_store = build_organizer_store(config.project_root)
        tts = registry.create_tts()
        voice_controller = TtsVoiceController()
        voice_controller.bind(tts)
        return cls(
            config=config,
            components=VoiceLoopComponents(
                microphone=MicrophoneInput(config.audio),
                speaker=SpeakerOutput(),
                stt=registry.create_stt(),
                tts=tts,
                responder=build_llm_responder(
                    config,
                    tool_registry=build_organization_tool_registry(
                        config.project_root,
                        store=organizer_store,
                        extra_tools=build_voice_tools(voice_controller),
                    ),
                    system_context_provider=organizer_store.preference_prompt_context,
                ),
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
        LOGGER.info(
            "sending command to llm provider=%s model=%s",
            self._config.llm_provider,
            active_llm_model(self._config),
        )
        llm_response = self._components.responder.generate(command)
        LOGGER.info("llm response latency_ms=%.1f", llm_response.latency_ms)
        speech = self._components.tts.synthesize(llm_response.text)
        LOGGER.info(
            "tts latency_ms=%.1f provider=%s",
            speech.latency_ms,
            speech.provider,
        )
        self._components.speaker.play(speech.speech)
