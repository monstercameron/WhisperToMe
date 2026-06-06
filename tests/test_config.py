from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from whispertome.config import load_config, with_wake_phrases


class ConfigTests(unittest.TestCase):
    def test_loads_env_without_requiring_key_for_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "OPENAI_MODEL=gpt-5.2",
                        "WHISPERTOME_WAKE_PHRASES=whisper to me,hey assistant",
                    ]
                ),
                encoding="utf-8",
            )

            old_key = os.environ.pop("OPENAI_API_KEY", None)
            try:
                config = load_config(root, require_openai_key=False)
            finally:
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key

            self.assertIsNone(config.openai.api_key)
            self.assertEqual(config.openai.model, "gpt-5.2")
            self.assertEqual(config.wake.phrases, ("whisper to me", "hey assistant"))

    def test_can_override_wake_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp), require_openai_key=False)

            updated = with_wake_phrases(config, ["computer", "hello dashboard"])

            self.assertEqual(updated.wake.phrases, ("computer", "hello dashboard"))

    def test_voice_defaults_are_latency_oriented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_openai_max = os.environ.pop("OPENAI_MAX_OUTPUT_TOKENS", None)
            old_stt_max = os.environ.pop("WHISPERTOME_STT_MAX_TOKENS", None)
            old_stt_variant = os.environ.pop("WHISPERTOME_STT_ONNX_VARIANT", None)
            old_stt_prompt = os.environ.pop("WHISPERTOME_STT_PROMPT", None)
            old_speech_end = os.environ.pop("WHISPERTOME_SPEECH_END_MS", None)
            try:
                config = load_config(Path(tmp), require_openai_key=False)
            finally:
                if old_openai_max is not None:
                    os.environ["OPENAI_MAX_OUTPUT_TOKENS"] = old_openai_max
                if old_stt_max is not None:
                    os.environ["WHISPERTOME_STT_MAX_TOKENS"] = old_stt_max
                if old_stt_variant is not None:
                    os.environ["WHISPERTOME_STT_ONNX_VARIANT"] = old_stt_variant
                if old_stt_prompt is not None:
                    os.environ["WHISPERTOME_STT_PROMPT"] = old_stt_prompt
                if old_speech_end is not None:
                    os.environ["WHISPERTOME_SPEECH_END_MS"] = old_speech_end

            self.assertEqual(config.openai.max_output_tokens, 64)
            self.assertEqual(config.stt.backend, "qai_whisper")
            self.assertEqual(config.stt.max_tokens, 64)
            self.assertEqual(config.stt.onnx_variant, "fp32")
            self.assertIn("exact words", config.stt.prompt)
            self.assertEqual(
                config.stt.model_path,
                Path(tmp)
                / "models"
                / "qai"
                / "whisper_small"
                / "snapdragon_x2_elite"
                / "precompiled_qnn_onnx"
                / "extracted"
                / "whisper_small-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite",
            )
            self.assertEqual(config.audio.pre_roll_ms, 600)
            self.assertEqual(config.audio.speech_end_ms, 1200)


if __name__ == "__main__":
    unittest.main()
