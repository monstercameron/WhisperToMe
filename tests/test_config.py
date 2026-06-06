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


if __name__ == "__main__":
    unittest.main()
