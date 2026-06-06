from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from whispertome.audio.types import AudioBuffer
from whispertome.stt.whisper_onnx import WhisperOnnxFiles, _prepare_audio


class WhisperOnnxTests(unittest.TestCase):
    def test_resolves_split_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_dir = root / "onnx"
            onnx_dir.mkdir()
            encoder = onnx_dir / "encoder_model_int8.onnx"
            decoder = onnx_dir / "decoder_model_merged_int8.onnx"
            encoder.write_bytes(b"encoder")
            decoder.write_bytes(b"decoder")

            files = WhisperOnnxFiles.resolve(root)

            self.assertEqual(files.root, root)
            self.assertEqual(files.encoder_path, encoder)
            self.assertEqual(files.decoder_path, decoder)

    def test_prepare_audio_resamples_to_whisper_rate(self) -> None:
        source = np.linspace(-1.0, 1.0, 24_000, dtype=np.float32)
        prepared = _prepare_audio(AudioBuffer(samples=source, sample_rate=24_000))

        self.assertEqual(prepared.dtype, np.float32)
        self.assertEqual(prepared.shape, (16_000,))


if __name__ == "__main__":
    unittest.main()
