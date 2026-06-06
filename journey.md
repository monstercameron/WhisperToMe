# WhisperToMe Journey

This file records practical wins, losses, and decisions while building the local voice assistant.

## Milestone 1: Initial Voice Assistant Scaffold

Commit: `7e3e19f` (`Milestone: initial voice assistant scaffold`)

### Wins

- Created a modular Python package with separate audio, STT, TTS, runtime, wake, LLM, and pipeline layers.
- Added `.env` loading, including support for the existing `openai` key alias.
- Added arbitrary wake phrases from `.env` or repeated `--wake` CLI flags.
- Built VAD-gated utterance segmentation so wake detection runs over speech-pause-delimited transcripts.
- Added strict NPU-only ONNX Runtime provider selection.
- Verified QNN plugin discovery on this Windows ARM64 machine.
- Verified `QNNExecutionProvider` exposes a Qualcomm NPU device.
- Added `doctor`, `test-wake`, `test-tts`, and `test-audio` commands.
- Proved speaker playback works with `test-audio`.
- Wired Kokoro TTS preprocessing, voice loading, and synthesis behind the TTS adapter.
- Added an explicit `--allow-non-npu` TTS debug path and heard real Kokoro speech.

### Losses

- Stock Kokoro v1.0 ONNX does not initialize on QNN HTP because QNN rejects dynamic internal shapes.
- Freezing Kokoro's public token input to length 512 did not remove all dynamic internal tensors.
- Supertonic2 Qualcomm source ONNX subgraphs either assigned nodes to CPU or still had QNN-rejected dynamic tensors.
- Supertonic2 precompiled QNN context artifacts were not compatible with the current Windows ONNX Runtime QNN path.
- DirectML cannot be treated as NPU execution unless we can prove the selected adapter is NPU-backed.
- Proper NPU TTS still requires a static-shape or compatible QNN-context model artifact.

### Decisions

- Production local AI inference remains NPU-only with no CPU or GPU fallback.
- Debug-only non-NPU paths may exist, but must be explicit and visibly logged.
- QNN HTP is the currently verified NPU path on this Snapdragon target.
- Model adapters should stay isolated because ONNX exports differ substantially by toolchain.

## Milestone 2: STT Bring-Up

### Wins

- Started with `onnx-community/whisper-tiny` because it provides separate ONNX encoder/decoder files and Hugging Face tokenizer/preprocessor metadata.
- Downloaded ignored local model artifacts under `models/whisper/whisper-tiny`.
- Inspected the encoder and decoder ONNX contracts:
  - encoder input: `input_features` shaped as 80x3000 Whisper log-mel features.
  - decoder input: merged decoder with past-key-value cache and `use_cache_branch`.
- Added a real Whisper ONNX adapter with feature extraction, greedy decoding, tokenizer decoding, and max-token control.
- Added `test-stt` for WAV-file and short microphone transcription checks.
- Added an explicit `--allow-non-npu` STT debug path.
- Verified debug STT can run end-to-end through `CPUExecutionProvider` on `artifacts/tts-debug.wav`.
- Verified microphone STT smoke test with `test-stt --record-ms 5000 --allow-non-npu`; it transcribed `Testing,`.

### Losses

- ONNX Runtime QNN does not claim every node in the `onnx-community/whisper-tiny` int8 encoder graph.
- ONNX Runtime QNN does not claim every node in the `onnx-community/whisper-tiny` int8 merged decoder graph.
- Strict NPU STT therefore fails fast under the no-fallback policy.
- Debug Whisper Tiny transcription of synthetic Kokoro speech was rough: `"Hello, I'm the one who is in the"`.
- The live assistant cannot complete a production NPU-only spoken turn until Whisper uses a QNN-compatible export.

### Next Steps

- Try a better-quality Whisper model for debug transcription, likely `whisper-base` or `whisper-small`, while keeping the NPU-only production policy strict.
- Search for or generate a static/QNN-compatible Whisper export for Snapdragon X.
- Add provider profiling once a QNN-compatible STT model initializes successfully.
- Feed microphone `test-stt --record-ms` output into wake phrase tests.
