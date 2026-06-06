from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from whispertome.config import active_llm_model, load_config, with_wake_phrases


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

    def test_can_select_cerebras_llm_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "WHISPERTOME_LLM_PROVIDER=cerebras",
                        "cerebras=test-key",
                        "CEREBRAS_MODEL=gpt-oss-120b",
                    ]
                ),
                encoding="utf-8",
            )

            old_provider = os.environ.pop("WHISPERTOME_LLM_PROVIDER", None)
            old_key = os.environ.pop("CEREBRAS_API_KEY", None)
            old_alias = os.environ.pop("cerebras", None)
            try:
                config = load_config(root, require_openai_key=True)
            finally:
                if old_provider is not None:
                    os.environ["WHISPERTOME_LLM_PROVIDER"] = old_provider
                if old_key is not None:
                    os.environ["CEREBRAS_API_KEY"] = old_key
                if old_alias is not None:
                    os.environ.update({"cerebras": old_alias})

            self.assertEqual(config.llm_provider, "cerebras")
            self.assertEqual(config.cerebras.model, "gpt-oss-120b")
            self.assertEqual(config.cerebras.base_url, "https://api.cerebras.ai/v1")
            self.assertEqual(config.cerebras.reasoning_effort, "low")
            self.assertEqual(config.cerebras.timeout_seconds, 20.0)
            self.assertEqual(config.cerebras.max_retries, 0)
            self.assertEqual(active_llm_model(config), "gpt-oss-120b")

    def test_can_override_wake_phrases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp), require_openai_key=False)

            updated = with_wake_phrases(config, ["computer", "hello dashboard"])

            self.assertEqual(updated.wake.phrases, ("computer", "hello dashboard"))

    def test_voice_defaults_are_latency_oriented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_openai_max = os.environ.pop("OPENAI_MAX_OUTPUT_TOKENS", None)
            old_openai_model = os.environ.pop("OPENAI_MODEL", None)
            old_llm_provider = os.environ.pop("WHISPERTOME_LLM_PROVIDER", None)
            old_stt_max = os.environ.pop("WHISPERTOME_STT_MAX_TOKENS", None)
            old_stt_variant = os.environ.pop("WHISPERTOME_STT_ONNX_VARIANT", None)
            old_stt_prompt = os.environ.pop("WHISPERTOME_STT_PROMPT", None)
            old_speech_end = os.environ.pop("WHISPERTOME_SPEECH_END_MS", None)
            old_duck_enabled = os.environ.pop("WHISPERTOME_WAKE_DUCK_VOLUME", None)
            old_duck_percent = os.environ.pop("WHISPERTOME_WAKE_DUCK_VOLUME_PERCENT", None)
            try:
                config = load_config(Path(tmp), require_openai_key=False)
            finally:
                if old_openai_max is not None:
                    os.environ["OPENAI_MAX_OUTPUT_TOKENS"] = old_openai_max
                if old_openai_model is not None:
                    os.environ["OPENAI_MODEL"] = old_openai_model
                if old_llm_provider is not None:
                    os.environ["WHISPERTOME_LLM_PROVIDER"] = old_llm_provider
                if old_stt_max is not None:
                    os.environ["WHISPERTOME_STT_MAX_TOKENS"] = old_stt_max
                if old_stt_variant is not None:
                    os.environ["WHISPERTOME_STT_ONNX_VARIANT"] = old_stt_variant
                if old_stt_prompt is not None:
                    os.environ["WHISPERTOME_STT_PROMPT"] = old_stt_prompt
                if old_speech_end is not None:
                    os.environ["WHISPERTOME_SPEECH_END_MS"] = old_speech_end
                if old_duck_enabled is not None:
                    os.environ["WHISPERTOME_WAKE_DUCK_VOLUME"] = old_duck_enabled
                if old_duck_percent is not None:
                    os.environ["WHISPERTOME_WAKE_DUCK_VOLUME_PERCENT"] = old_duck_percent

            self.assertEqual(config.openai.model, "gpt-5.4-mini")
            self.assertEqual(config.openai.max_output_tokens, 512)
            self.assertIn("Use at most one fenced block", config.openai.system_prompt)
            self.assertIn("Organization tools", config.openai.system_prompt)
            self.assertIn("reminders", config.openai.system_prompt)
            self.assertIn("daily plan", config.openai.system_prompt)
            self.assertIn("time-sensitive check", config.openai.system_prompt)
            self.assertIn("short but sweet", config.openai.system_prompt)
            self.assertIn("plainspoken", config.openai.system_prompt)
            self.assertIn("not literary", config.openai.system_prompt)
            self.assertIn("ask one short follow-up question", config.openai.system_prompt)
            self.assertIn("what's up next", config.openai.system_prompt)
            self.assertIn("people/contact notes", config.openai.system_prompt)
            self.assertIn("Use preferences", config.openai.system_prompt)
            self.assertIn("call me X", config.openai.system_prompt)
            self.assertIn("one compact prompt-ready sentence", config.openai.system_prompt)
            self.assertIn("Keep stored resources clean", config.openai.system_prompt)
            self.assertIn("mark stale", config.openai.system_prompt)
            self.assertIn("no longer appear in active lists", config.openai.system_prompt)
            self.assertIn("System control tools", config.openai.system_prompt)
            self.assertIn("directly asks", config.openai.system_prompt)
            self.assertIn("volume and brightness to 0-100", config.openai.system_prompt)
            self.assertIn("assistant-window minimize tool", config.openai.system_prompt)
            self.assertIn(
                "get the assistant/app/window out of the way",
                config.openai.system_prompt,
            )
            self.assertIn("we're done", config.openai.system_prompt)
            self.assertIn("system tray", config.openai.system_prompt)
            self.assertIn("desktop capture tool", config.openai.system_prompt)
            self.assertIn("attached", config.openai.system_prompt)
            self.assertIn("PowerShell tool", config.openai.system_prompt)
            self.assertIn("read-only commands", config.openai.system_prompt)
            self.assertIn("explicit confirmation", config.openai.system_prompt)
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
            self.assertTrue(config.system.wake_duck_enabled)
            self.assertEqual(config.system.wake_duck_percent, 25)


if __name__ == "__main__":
    unittest.main()
