from __future__ import annotations

import unittest
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from whispertome.audio.types import AudioBuffer
from whispertome.cli import (
    FallbackTextToSpeechModel,
    build_parser,
    prepare_stt_model,
    profile_audio,
    resolve_log_file,
    should_prefer_debug_stt,
)
from whispertome.audio.types import SynthesizedSpeech
from whispertome.errors import WhisperToMeError
from whispertome.tts.base import SpeechSynthesisResult, TextToSpeechModel


class FakeTts(TextToSpeechModel):
    def __init__(self, *, fail_load: bool = False, provider: str = "fake") -> None:
        self.fail_load = fail_load
        self.provider = provider
        self.load_calls = 0
        self.synthesize_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        if self.fail_load:
            raise WhisperToMeError("load failed")

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        self.synthesize_calls += 1
        return SpeechSynthesisResult(
            speech=SynthesizedSpeech(samples=np.zeros(1, dtype=np.float32), sample_rate=24_000),
            latency_ms=1.0,
            model="fake",
            provider=self.provider,
        )


class CliTests(unittest.TestCase):
    def test_demo_gets_default_log_file_under_project_root(self) -> None:
        project_root = Path("C:/project")
        args = SimpleNamespace(
            command="demo",
            project_root=project_root,
            log_file=None,
        )

        log_file = resolve_log_file(args)

        assert log_file is not None
        self.assertEqual(log_file.parent, project_root / "artifacts" / "logs")
        self.assertTrue(log_file.name.startswith("demo-"))
        self.assertEqual(log_file.suffix, ".log")

    def test_relative_log_file_is_resolved_under_project_root(self) -> None:
        project_root = Path("C:/project")
        args = SimpleNamespace(
            command="demo",
            project_root=project_root,
            log_file=Path("artifacts/logs/manual.log"),
        )

        self.assertEqual(
            resolve_log_file(args),
            project_root / "artifacts" / "logs" / "manual.log",
        )

    def test_non_demo_command_has_no_default_log_file(self) -> None:
        args = SimpleNamespace(
            command="doctor",
            project_root=Path("C:/project"),
            log_file=None,
        )

        self.assertIsNone(resolve_log_file(args))

    def test_demo_defaults_to_speech_pause_recording(self) -> None:
        args = build_parser().parse_args(["demo", "--allow-non-npu"])

        self.assertFalse(args.fixed_record)
        self.assertTrue(args.allow_non_npu)
        self.assertEqual(args.record_ms, 10000)
        self.assertFalse(args.no_warmup)

    def test_allow_non_npu_keeps_qai_whisper_on_npu(self) -> None:
        config = SimpleNamespace(stt=SimpleNamespace(backend="qai_whisper"))

        self.assertFalse(should_prefer_debug_stt(config, allow_non_npu=True))  # type: ignore[arg-type]

    def test_allow_non_npu_uses_debug_for_legacy_whisper_onnx(self) -> None:
        config = SimpleNamespace(stt=SimpleNamespace(backend="whisper_onnx"))

        self.assertTrue(should_prefer_debug_stt(config, allow_non_npu=True))  # type: ignore[arg-type]

    def test_prepare_stt_keeps_qai_whisper_on_registry_with_allow_non_npu(self) -> None:
        config = SimpleNamespace(stt=SimpleNamespace(backend="qai_whisper"))
        stt = object()
        with patch("whispertome.models.registry.ModelRegistry") as registry_cls:
            registry_cls.return_value.create_stt.return_value = stt

            result = prepare_stt_model(config, allow_non_npu=True)  # type: ignore[arg-type]

        self.assertIs(result, stt)

    def test_tts_fallback_stays_active_after_primary_load_failure(self) -> None:
        primary = FakeTts(fail_load=True, provider="primary")
        fallback = FakeTts(provider="fallback")
        model = FallbackTextToSpeechModel(
            primary=primary,
            fallback=fallback,
            logger=logging.getLogger("test"),
        )

        model.load()
        first = model.synthesize("hello")
        second = model.synthesize("again")

        self.assertEqual(primary.load_calls, 1)
        self.assertEqual(primary.synthesize_calls, 0)
        self.assertEqual(fallback.load_calls, 1)
        self.assertEqual(fallback.synthesize_calls, 2)
        self.assertEqual(first.provider, "fallback")
        self.assertEqual(second.provider, "fallback")

    def test_audio_profile_reports_active_and_clipped_audio(self) -> None:
        samples = np.array([0.0, 0.02, -0.02, 1.0, -1.0], dtype=np.float32)
        audio = AudioBuffer(samples=samples, sample_rate=1000)

        profile = profile_audio(audio, vad_threshold=0.01, block_ms=1)

        self.assertEqual(profile.duration_ms, 5)
        self.assertEqual(profile.active_ms, 4)
        self.assertEqual(profile.peak, 1.0)
        self.assertGreater(profile.clipped_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
