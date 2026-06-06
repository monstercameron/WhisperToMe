"""Agent tools that let the LLM list and switch the TTS voice at runtime.

The active TTS model is supplied through a `TtsVoiceController` that the voice loop
binds after it builds the model (the model may be created after the tool registry, so
binding is late). Switching is a live data swap for Supertonic; backends that do not
support it surface a clean error to the model.
"""
from __future__ import annotations

from whispertome.agent.tools import AgentTool, object_schema, string_schema
from whispertome.tts.base import TextToSpeechModel, VoiceNotSupportedError


class TtsVoiceController:
    """Late-bindable handle to the live TTS model for voice tools."""

    def __init__(self) -> None:
        self._tts: TextToSpeechModel | None = None

    def bind(self, tts: TextToSpeechModel) -> None:
        self._tts = tts

    def list_voices(self) -> list[str]:
        return self._tts.list_voices() if self._tts is not None else []

    def current_voice(self) -> str | None:
        return self._tts.current_voice() if self._tts is not None else None

    def set_voice(self, voice: str) -> None:
        if self._tts is None:
            raise VoiceNotSupportedError("No TTS model is active yet")
        self._tts.set_voice(voice)


def build_voice_tools(controller: TtsVoiceController) -> list[AgentTool]:
    return [_tts_list_voices(controller), _tts_set_voice(controller)]


def _tts_list_voices(controller: TtsVoiceController) -> AgentTool:
    def handler(_args: dict) -> dict:
        voices = controller.list_voices()
        if not voices:
            return {"ok": True, "voices": [], "note": "Active TTS backend has no switchable voices."}
        return {"ok": True, "voices": voices, "current": controller.current_voice()}

    return AgentTool(
        name="tts_list_voices",
        description=(
            "List the speaking voices the assistant can switch to right now, and which one "
            "is currently active. Use before changing voice if the user is unsure of names."
        ),
        parameters=object_schema({}),
        handler=handler,
    )


def _tts_set_voice(controller: TtsVoiceController) -> AgentTool:
    def handler(args: dict) -> dict:
        voice = str(args.get("voice", "")).strip()
        if not voice:
            return {"ok": False, "error": "voice is required"}
        try:
            controller.set_voice(voice)
        except VoiceNotSupportedError as exc:
            return {"ok": False, "error": str(exc), "voices": controller.list_voices()}
        return {"ok": True, "voice": controller.current_voice()}

    return AgentTool(
        name="tts_set_voice",
        description=(
            "Change the assistant's speaking voice on direct user request (for example "
            "'use the female voice' or 'switch to M1'). Takes effect on the next spoken reply."
        ),
        parameters=object_schema(
            {"voice": string_schema("Voice name to switch to, e.g. 'F1' or 'M1'.")},
            required=["voice"],
        ),
        handler=handler,
    )
