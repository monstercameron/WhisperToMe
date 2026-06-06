from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whispertome.config import load_config
from whispertome.models.registry import ModelRegistry
from whispertome.stt.qai_whisper import QaiWhisperFiles, QaiWhisperTranscriber


class QaiWhisperTests(unittest.TestCase):
    def test_resolves_compiled_qai_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "encoder.onnx").write_bytes(b"encoder")
            (root / "decoder.onnx").write_bytes(b"decoder")

            files = QaiWhisperFiles.resolve(root)

            self.assertEqual(files.encoder_path, root / "encoder.onnx")
            self.assertEqual(files.decoder_path, root / "decoder.onnx")

    def test_registry_creates_qai_whisper_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "WHISPERTOME_STT_BACKEND=qai_whisper",
                        "WHISPERTOME_STT_MODEL_PATH=models/qai/test",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(root, require_openai_key=False)

            stt = ModelRegistry(config).create_stt()

            self.assertIsInstance(stt, QaiWhisperTranscriber)


if __name__ == "__main__":
    unittest.main()
