# WhisperToMe

WhisperToMe is a new project for a hands-free voice assistant that continuously listens, converts speech to text, sends the text to the OpenAI API, and speaks the response back to the user.

## Objectives

- Continuously listen from the microphone with low latency.
- Detect useful speech while avoiding unnecessary processing during silence.
- Run Whisper speech-to-text locally on the NPU.
- Send the recognized text to the OpenAI API for reasoning and response generation.
- Run text-to-speech locally on the NPU.
- Stream the spoken response back through the speakers.
- Keep the interaction loop fast enough to feel conversational.

## Target Flow

```text
Microphone
  -> voice activity detection
  -> Whisper speech-to-text
  -> OpenAI API request
  -> response text
  -> local text-to-speech
  -> speakers
```

## Current Decisions

- STT: Whisper.
- TTS: Kokoro-82M v1.0 ONNX as the first candidate.
- TTS fallback: KittenTTS Nano ONNX if Kokoro is too heavy for the target device.
- Runtime: ONNX Runtime-based local inference.
- Acceleration order for the first Windows prototype:
  1. Verified Windows NPU path.
  2. QNN HTP path for Qualcomm Snapdragon NPU targets.
- No CPU or GPU fallback for local AI inference.

## NPU-First Design

The local inference parts of the pipeline should target the NPU:

- Whisper speech-to-text model
- Voice activity detection model, if applicable
- Text-to-speech model

The OpenAI API call itself is a network request to OpenAI's hosted models, so that part does not run on the local NPU. The hard target is that all local AI workloads run on the NPU. If an STT, VAD, or TTS model cannot run on the NPU, the app should fail fast instead of silently falling back to CPU or GPU.

DirectML by itself is not a guaranteed NPU path; it is a broad Windows hardware acceleration path and may route work to the GPU. For this project, a DirectML or Windows runtime path is acceptable only when the selected adapter/provider is verified to be NPU-backed. QNN HTP is the Qualcomm-specific NPU path and should be used when the target machine has a Snapdragon NPU available.

## Model Direction

### Speech-to-Text

Use Whisper locally for transcription. Start with the smallest Whisper variant that is accurate enough for the microphone setup, then scale up only if accuracy is poor.

Initial candidates:

- `whisper-tiny` for latency testing.
- `whisper-base` for a better quality baseline.
- ONNX-exported Whisper model for verified Windows NPU and QNN experiments.

### Text-to-Speech

Use Kokoro-82M v1.0 ONNX first. It is small enough for local use, has quantized ONNX variants, and should give better naturalness than ultra-tiny models.

Keep KittenTTS Nano ONNX as the fallback when the priority is the smallest possible model footprint. It is much smaller than Kokoro, but should be evaluated for voice quality before making it the default assistant voice.

## Initial MVP

1. Capture microphone audio continuously.
2. Add voice activity detection to decide when speech starts and stops.
3. Transcribe speech to text with Whisper.
4. Send the transcript to the OpenAI API.
5. Convert the API response to speech with Kokoro-82M ONNX.
6. Play the generated audio.
7. Confirm each local model is running on the NPU.
8. Measure latency for each stage.
9. Move STT and TTS onto a verified Windows NPU path.
10. Test QNN HTP on compatible Snapdragon hardware.

## Runtime Strategy

The app should treat acceleration as a provider selection problem, but it should accept only NPU execution for local AI inference.

```text
Try verified Windows NPU provider
  -> if unavailable or unsupported, try QNN HTP on Snapdragon
  -> if unavailable or unsupported, fail fast
```

For each local model, log the selected provider, model variant, first-token or first-audio latency, total stage latency, and proof that the run used the NPU. Any CPU or GPU execution should be treated as a configuration error.

## Implementation

This repo is now structured as a Python package under `src/whispertome`.

```text
src/whispertome/
  audio/      microphone capture, playback, VAD, utterance segmentation
  llm/        OpenAI Responses API client
  models/     STT/TTS model registry
  runtime/    NPU-only ONNX Runtime provider selection
  stt/        Whisper STT adapter interface
  tts/        Kokoro TTS adapter interface
  wake/       sliding transcript wake phrase detector and command router
  pipeline.py live voice loop orchestration
  cli.py      doctor and run commands
```

The code is intentionally modular:

- Runtime selection is isolated from model code.
- STT and TTS use abstract interfaces so new models can be added without changing the voice loop.
- The OpenAI API client is isolated behind `OpenAIResponder`.
- Audio capture and playback are replaceable components.
- Wake detection is text-window based, so it works with any STT backend.
- A wake phrase can be supplied from `.env` or repeated `--wake` CLI flags.

## Setup

Create a Python environment and install the project with the runtime extra that matches the target device.

```powershell
cd C:\Users\mreca\Desktop\whispertome
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[directml]
```

For Snapdragon/QNN testing, use a Windows ARM64 Python 3.11 environment and install:

```powershell
pip install -e .[qnn]
```

The `.env` file is read automatically. The app only reports whether `OPENAI_API_KEY` is present; it does not print the key.

Before running the live loop, check the environment:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome doctor
```

Start the live loop:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run
```

Override wake phrases at runtime with repeated `--wake` flags:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --wake "computer"
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --wake "computer" --wake "hello dashboard"
```

Test wake phrase behavior without a microphone, model files, or OpenAI calls:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-wake --wake "computer" "computer tell me the time"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-wake --wake "whisper to me" "whisper to me" "summarize my calendar"
```

Each `test-wake` argument is treated as one speech-pause-delimited utterance. In the live loop, VAD decides those utterance boundaries from microphone audio.

Test TTS model initialization and WAV generation:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-tts "Hello from WhisperToMe." --out artifacts\tts-test.wav
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-tts "Hello from WhisperToMe." --out artifacts\tts-test.wav --play
```

Under the NPU-only policy this command must fail if ONNX Runtime cannot place the entire TTS graph on the selected NPU execution provider.

For an audible voice-quality smoke test while the NPU export is still blocked, use the explicit debug override:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-tts "Hello from WhisperToMe." --out artifacts\tts-debug.wav --play --allow-non-npu
```

That command intentionally does not prove NPU execution; it exists only so playback, voices, and text preprocessing can be tested.

Verify speaker playback separately from TTS:

```powershell
whispertome test-audio --duration-ms 500 --frequency 440 --volume 0.20
```

## Current Build Status

Implemented:

- `.env` configuration loading.
- OpenAI Responses API client using the official Python SDK.
- Microphone capture and speaker playback.
- Energy-based VAD and utterance segmentation.
- Continuous STT-oriented voice loop.
- Sliding transcript wake phrase detection with arbitrary wake phrases.
- Speech-pause-delimited command capture after wake detection.
- Strict NPU-only ONNX Runtime provider selection.
- `doctor` command for safe config and runtime checks.
- `test-wake` command for wake phrase routing tests without microphone/model access.
- `test-tts` command for TTS initialization and WAV generation.
- `test-audio` command for speaker playback checks without any model inference.
- Kokoro ONNX TTS adapter using the real `kokoro-onnx` tokenizer, phonemizer, voices, and injected NPU-only ONNX session.
- Unit tests for config, wake phrase detection, and provider policy.

Still required before the live assistant can complete an end-to-end spoken turn:

- Choose the exact Whisper ONNX export format and wire its encoder/decoder/tokenizer adapter.
- Produce or acquire a TTS export that QNN HTP can load with no CPU/GPU-assigned nodes.
- Add provider-specific profiling to prove no local model runs on CPU or GPU.

### TTS Status

The project is currently configured for the downloaded Kokoro-82M v1.0 ONNX artifacts, but the stock export does not initialize on this Snapdragon/QNN NPU path. ONNX Runtime QNN reports:

```text
Dynamic shape is not supported yet
```

Freezing Kokoro's public token input to length 512 was not enough because dynamic internal tensors remain after the encoder/decoder shape operations. Supertonic2 Qualcomm source ONNX subgraphs were also tested; ONNX Runtime QNN either assigned unsupported nodes to CPU or rejected dynamic tensors, which violates this project's no-fallback rule.

The next TTS artifact must be one of:

- A Kokoro export compiled specifically for Windows ONNX Runtime QNN HTP with static graph shapes.
- A QNN context ONNX generated from a context binary compatible with the target Snapdragon X NPU and this ONNX Runtime QNN build.
- A different TTS architecture with fixed-shape subgraphs that QNN can claim completely.

## Research Notes

- Whisper is the STT baseline because it is a general-purpose speech recognition model with multilingual recognition, speech translation, and language identification support.
- Kokoro-82M v1.0 ONNX is the first TTS candidate because it provides ONNX and quantized model files, including 8-bit and mixed-precision variants.
- KittenTTS Nano ONNX is the small-footprint fallback because it is a 15M-parameter ONNX TTS model with WebGPU and WASM-oriented exports.
- ONNX Runtime DirectML supports broad DirectX 12 hardware acceleration on Windows, but it should not be treated as NPU execution unless the selected adapter/provider proves that work is on the NPU.
- ONNX Runtime QNN can target Qualcomm Snapdragon devices, and its HTP backend is the NPU path.

## Open Questions

- What is the reliable Windows API signal for verifying that DirectML or WinML execution is actually using the NPU?
- Should QNN HTP be first whenever a Qualcomm NPU is detected?
- Which Whisper size is the best latency and accuracy tradeoff?
- Does Kokoro-82M meet real-time latency targets on the target hardware?
- Should responses be interruptible while the assistant is speaking?
- Should the assistant require a wake word, push-to-talk, or always-on voice activity detection?
- Should the app stream partial transcriptions and partial TTS, or wait for complete utterances?

## Guiding Constraint

Prioritize a working voice loop first, then optimize model placement and NPU acceleration stage by stage.

## References

- Whisper: https://github.com/openai/whisper
- Kokoro-82M v1.0 ONNX: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- KittenTTS Nano ONNX: https://huggingface.co/onnx-community/KittenTTS-Nano-v0.8-ONNX
- ONNX Runtime DirectML Execution Provider: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX Runtime QNN Execution Provider: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
