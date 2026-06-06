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

## Milestone 3: OpenAI Responses Layer

### Wins

- Upgraded the OpenAI layer from a thin call wrapper into a stateful Responses API client.
- Added a dictation-aware default system prompt for speech-to-text transcripts.
- Wrapped user input as an explicit speech transcript so the model knows it may contain STT errors.
- Added `previous_response_id` chaining when `OPENAI_STATEFUL=true`.
- Added `test-openai` to verify API key, model, prompt, response parsing, and latency without running the full voice loop.
- Added unit tests with a fake OpenAI client so Responses API behavior is covered without network calls.
- Verified a live Responses API call with the transcript `rite a very short reminder to call alex tomorrow`; the model returned `Reminder: Call Alex tomorrow.`

### Losses

- Realtime/WebSocket is not wired yet; it is reserved for a future lower-latency speech-to-speech or realtime transcription path.
- Dictation prompt quality still needs real conversational testing once STT quality improves.

### Decisions

- Keep the current architecture as local STT -> OpenAI Responses API text turn -> local TTS.
- Re-send `instructions` every Responses API turn because previous-response chaining does not carry new instructions forward.
- Keep responses short and TTS-friendly by default.

## Milestone 4: Full Conversation Demo Harness

### Wins

- Added a `demo` CLI command that runs microphone recording, STT, OpenAI Responses API, TTS, and speaker playback in one human-testable loop.
- Made the demo turn-based with Enter-to-record and `q`/`quit` to exit, so failed turns can be reproduced without starting the live always-listening loop.
- Added automatic per-demo log files under `artifacts/logs`.
- Added optional `--save-audio` capture under `artifacts/demo` for debugging bad STT results against the original microphone audio and verifying assistant TTS output when playback fails.
- Logged every major stage: recording, transcript, OpenAI response metadata, TTS provider, playback, completed turns, and exception stack traces.
- Cached debug STT and TTS model instances during CLI runs so repeated demo turns avoid unnecessary model reloads.
- Made the explicit debug demo path skip the known-failing QNN probes, which avoids low-level provider log spam during human conversation tests.
- Replaced fixed-duration demo recording with speech-pause-delimited recording, which removes several seconds of trailing silence before STT.
- Added model warmup before the first debug demo turn.
- Added per-turn stage profiling and summary profiling for recording, STT, OpenAI, TTS, playback, and total turn latency.
- Added audio energy and clipping logs so microphone gain problems are visible.

### Losses

- The full audible demo still needs `--allow-non-npu` because the current Whisper and Kokoro ONNX artifacts do not satisfy strict QNN NPU placement.
- The first human demo used fixed 5-second capture, but speech activity was only about 0.4 to 2.2 seconds per turn. That wasted latency and gave Whisper trailing silence to hallucinate into.
- Saved user WAVs showed peaks near full scale, so microphone clipping may be contributing to bad words.
- Debug STT quality is still limited by Whisper Tiny and the current microphone conditions.

### Decisions

- Keep the production NPU-only policy unchanged.
- Keep the full conversation harness explicit and debuggable while NPU-compatible model exports are still unresolved.
- Use log files and saved turn WAVs as the first failure artifacts before changing model/runtime code.
- Use speech-pause recording as the default demo path, with `--fixed-record` only for controlled tests.
- Default spoken OpenAI replies to a shorter output cap unless `.env` overrides it.

## Milestone 5: STT Quality Tier Bump

### Wins

- Downloaded `onnx-community/whisper-base` split ONNX artifacts into `models/whisper/whisper-base`.
- Switched the default STT model path from Whisper Tiny to Whisper Base.
- Kept the adapter contract unchanged because Base uses the same encoder/decoder filenames.
- Fixed the decoder cache bootstrap so Whisper variants with different attention head counts work.
- Added repeated-token-loop trimming to stop pathological Base outputs like repeated "allowed to be allowed".
- Downloaded `onnx-community/whisper-small.en` for English-only comparison testing.
- Changed the demo to transcribe the raw speech-pause capture instead of the trimmed diagnostic clip; this fixed cases where Base understood raw audio but failed on the trimmed version.
- Increased default pre-roll to 600 ms to reduce first-syllable loss in VAD-segmented audio.

### Losses

- Whisper Base should improve recognition quality, but it is larger and may increase CPU debug STT latency.
- This still does not solve strict QNN NPU placement; it is a debug-quality upgrade while the NPU-compatible export remains unresolved.
- On the 10:06 Base demo logs, Base still produced poor transcripts on several turns despite healthy mic levels and no clipping.
- `whisper-small.en` avoided some repetition but was slower and still changed meaning on at least one saved test clip.

### Decisions

- Use Whisper Base as the default human-demo STT model.
- Keep Whisper Tiny available locally for latency comparison and fallback experiments.
- Keep `whisper-small.en` available for targeted comparison rather than making it the default yet.
- Treat STT accuracy as a decoder/model-selection problem now that fixed-length recording, trailing silence, and clipping have been reduced.

## Milestone 6: Better STT Candidate Trial

### Wins

- Verified the saved bad-turn audio was intelligible by comparing local Whisper outputs with OpenAI audio transcription.
- Tried Moonshine Tiny/Base model downloads and confirmed its Python API shape for future use.
- Found that Moonshine's Python package/source path is blocked on this Windows ARM64 setup by a missing native `moonshine.dll`.
- Downloaded Qualcomm AI Hub Whisper-Small precompiled QNN ONNX artifacts for Snapdragon X2 Elite.
- Installed and registered the `onnxruntime-qnn` plugin execution provider.
- Confirmed ONNX Runtime sees a Qualcomm NPU EP device and can load both Qualcomm Whisper encoder and decoder sessions with `session.disable_cpu_ep_fallback=1`.
- Ran Qualcomm Whisper-Small against the exact failed saved WAVs. It transcribed `I need you to get sexy for me.` correctly on raw and trimmed clips.
- Measured roughly 325-360 ms STT latency for the short saved clips using the NPU path.
- Added a `qai_whisper` STT backend so the working Qualcomm path is available through the normal model registry and CLI.

### Losses

- `qai_hub_models` does not install cleanly on this Windows ARM64 Python environment because its dependency resolver falls into unsupported local builds.
- Qualcomm's export/demo scripts import PyTorch, and no prebuilt PyTorch wheel was available for this Windows ARM64 Python.
- The Qualcomm static asset helper also imports PyTorch, so the compiled artifact had to be fetched directly from the published S3 release URL.
- ONNX Runtime plugin-QNN sessions still report `CPUExecutionProvider` in metadata even when CPU fallback is disabled, so provider-list checks alone are not enough proof.
- TTS is still not on a verified NPU path; the full audible conversation demo still needs the explicit debug TTS override unless/until TTS gets a QNN-compatible artifact.

### Decisions

- Make `qai_whisper` the default STT backend on this Snapdragon X2 Elite machine.
- Treat Qualcomm AI Hub's precompiled QNN ONNX Whisper-Small artifact as the current production STT candidate.
- Keep Moonshine parked until a native Windows ARM64 library is available or we choose to build it.
- Continue enforcing no CPU/GPU fallback by selecting the NPU EP device and setting `session.disable_cpu_ep_fallback=1`.
