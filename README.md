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
- STT NPU path: Qualcomm AI Hub Whisper-Small precompiled QNN ONNX for Snapdragon X2 Elite.
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

Use Whisper locally for transcription. The current working NPU path is Qualcomm AI Hub's compiled Whisper-Small artifact for this Snapdragon X2 Elite machine.

Initial candidates:

- `qai_whisper` with Qualcomm AI Hub Whisper-Small precompiled QNN ONNX. This is the current STT default.
- `whisper-base` as the current debug-quality baseline.
- `whisper-tiny` for latency testing when accuracy is acceptable.
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
  agent/      Responses function-tool loop and tool registry
  audio/      microphone capture, playback, VAD, utterance segmentation
  llm/        OpenAI Responses API client
  models/     STT/TTS model registry
  organizer/  local SQLite organizer tools
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
- Agent tools are isolated behind a local registry so new tools can be added without changing the voice/audio loop.
- Active SQLite preferences are injected into the OpenAI system prompt as compact context each turn.
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

For Snapdragon/QNN testing, use a Windows ARM64 Python environment with an
`onnxruntime-qnn` wheel available. This machine is using Python 3.12 ARM64.

```powershell
pip install -e .[qnn]
```

The current STT artifact lives under:

```text
models/qai/whisper_small/snapdragon_x2_elite/precompiled_qnn_onnx/extracted/whisper_small-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite
```

That folder must contain `encoder.onnx`, `decoder.onnx`, `encoder_qairt_context.bin`, and `decoder_qairt_context.bin`.

The `.env` file is read automatically. The app only reports whether `OPENAI_API_KEY` is present; it does not print the key.

Before running the live loop, check the environment:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome doctor
```

Start the live loop:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run
```

That command is strict NPU-only. For manual wake-word testing with audible TTS while the TTS NPU artifact is still blocked, use the explicit debug TTS override:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio
```

The `run` command continuously segments microphone speech with VAD, transcribes each utterance with STT, checks a sliding transcript window for a wake phrase, and only sends a command to OpenAI after wake detection. It writes a log under `artifacts\logs\run-*.log`; `--save-audio` writes each STT utterance and assistant WAV under `artifacts\wake\...`.

For the live terminal UI, add `--tui`:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200 --tui
```

The TUI opens with a full-screen NPU boot animation while the local STT and TTS models warm up. Once the voice stack is ready, it switches to the normal central animated polygon whose pulse follows input activity, live input/output text panels, and a bounded 1-10 line system stream for states like `listening`, `wake detected`, `transcribing`, `contacting openai`, `running tts`, and `playing speech`. UI state changes wake the renderer immediately so wake/interruption animations do not wait for the next fixed tick. Use `--tui-lines 6` to change the stream height.

For the first desktop-host integration, use `desktop`:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome desktop --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200
```

The desktop command opens an owned fixed-size `WhisperToMe` Windows window at 800x600 and embeds the existing terminal TUI inside it. The host passes a compact character viewport to the child voice loop so the polygon, input/output panels, optional code viewport, and newest system-stream lines fit the shell without bottom clipping. It also creates a Windows tray icon with a right-click menu for `Show Window`, `Stop Listening`, and `Close Program`. The first launch centers the window on the current screen; if you manually move it, the host persists that position in `artifacts\desktop\window-state.json` and restores to that position on later wake/tray shows. Wake detection writes a host signal so the desktop window restores and comes forward when the wake word is triggered. Voice requests such as "minimize yourself", "hide", or "we're done talking for now" route through the host and hide the window to the system tray instead of minimizing it to the taskbar; clicking the normal title-bar minimize button does the same tray hide. The tray `Show Window` command or the next wake activation restores it. The host process writes `desktop-*` logs, while the embedded voice loop still writes the normal `run-*` logs. Stop and window close request a cooperative child shutdown first, then fall back to process-tree termination only if the voice loop does not exit within the grace window. This first host uses the built-in Tk window stack and an ANSI terminal-frame renderer; later native shell work can replace the terminal surface with richer panels without changing the voice loop.

The desktop command hides host and child console windows by default, so the only visible UI should be the `WhisperToMe` desktop window and tray icon. Use `--show-console` only when developing or debugging the host process itself.

When the TUI is enabled, the wake loop also runs an interim wake preview during active speech. This is not true Whisper token streaming; the local Whisper/QNN path is still batch STT. Instead, the app periodically transcribes a rolling audio window while the user is still speaking and uses that interim transcript only to update the TUI wake state sooner. The final pause-delimited transcript remains the source of truth for command execution.

The live wake loop streams OpenAI response deltas by default. Spoken-safe sentence chunks are sent to TTS as soon as they are complete, and audio chunks play in order while later tokens and TTS chunks are still being produced. Use `--no-stream-tts` to fall back to the older batch path for debugging.

Markdown fenced code blocks in OpenAI responses are parsed before TTS. The voice does not read triple-backtick fences or code contents aloud; it says that the code/script is on screen. The system prompt asks for one fenced block per response by default, unless you explicitly ask for multiple files or examples. The TUI renders the latest fenced block, preferring ```script blocks, inside an auto-scrolling code viewport. The parser also catches obvious unfenced code, such as `package main`, as a defensive fallback so raw code is not read aloud.

While assistant audio is playing, the wake loop keeps consuming the microphone stream in a background wake monitor. Say the wake phrase, for example `computer`, to interrupt playback. The monitor runs STT and wake matching concurrently with speaker output, stops audio when the wake phrase is detected, and logs `wake_playback_interrupted`. If the interruption transcript also contains a command, such as `computer I don't care`, the command is queued and immediately sent to OpenAI as the next turn.

When wake activation starts, WhisperToMe ducks Windows audio sessions for other apps, such as the browser, while excluding the assistant process so TTS stays audible. Ducking can begin from interim wake detection, final wake detection, command turn start, or playback-interruption speech. Volumes are restored to their exact previous per-app levels when the turn finishes or the app exits. This is controlled by:

```powershell
WHISPERTOME_WAKE_DUCK_VOLUME=true
WHISPERTOME_WAKE_DUCK_VOLUME_PERCENT=25
```

The duck target only lowers other app sessions; if an app is already below the target, it is not raised.

The default end-of-speech silence is `WHISPERTOME_SPEECH_END_MS=1200`, which gives room for short thinking pauses, "um", and "ah" without cutting a sentence in half. For live tuning:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200
```

Override wake phrases at runtime with repeated `--wake` flags:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --wake "computer"
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio --wake "computer" --wake "hello dashboard"
```

Test wake phrase behavior without a microphone, model files, or OpenAI calls:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-wake --wake "computer" "computer tell me the time"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-wake --wake "whisper to me" "whisper to me" "summarize my calendar"
```

Each `test-wake` argument is treated as one speech-pause-delimited utterance. In the live loop, VAD decides those utterance boundaries from microphone audio. Wake matching is token-window based, supports phrases split across adjacent STT utterances, ignores stale matches that are only in the old window, and preserves the original command text after the wake phrase when possible.

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

Test STT model initialization and transcription:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-stt --wav artifacts\tts-debug.wav
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-stt --record-ms 3000
```

Under the NPU-only policy these commands must fail if ONNX Runtime cannot place the Qualcomm Whisper graph on the QNN NPU plugin execution provider. ONNX Runtime may still list `CPUExecutionProvider` in the session metadata; the app disables CPU fallback during session creation.

For the older `whisper_onnx` baseline, use the explicit debug override:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-stt --wav artifacts\tts-debug.wav --allow-non-npu
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-stt --record-ms 3000 --allow-non-npu
```

The debug override intentionally does not prove NPU execution; it exists only so microphone/file input, Whisper preprocessing, decoding, and tokenizer output can be tested.

Test the OpenAI Responses API layer with a dictation-style transcript:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "rite a short reminder to call alex tomorrow"
```

The OpenAI layer uses the Responses API by default. It sends a dictation-aware system prompt every turn, wraps STT text as a speech transcript, and keeps `previous_response_id` when `OPENAI_STATEFUL=true`.

The OpenAI layer also exposes a local function-tool loop for organization commands. When the model calls a tool, WhisperToMe executes it locally, sends a `function_call_output` item back to the Responses API, and continues until the assistant has a normal spoken response. The default SQLite-backed tools are:

- Notes for loose memory, ideas, facts, and dictated snippets.
- Preferences for stable personal defaults like what to call the user, preferred units, tone, formatting, and default assistant behavior.
- Reminders for time-bound nudges and follow-ups.
- Checklists for task lists, shopping lists, packing lists, and procedures.
- Itinerary for dated or place-based plans, stops, appointments, and agendas.
- Tasks for single actionable items with status, due date, priority, and project.
- Projects for goal buckets that group tasks, notes, and decisions.
- Daily plan for a date-focused view of tasks, reminders, and itinerary.
- Time-sensitive check for "up next", "next ups", important items, due reminders,
  due tasks, and itinerary in a short lookahead window.
- Decision log for choices, rationale, dates, and project links.
- People/contact notes for names, roles, preferences, and follow-ups.

It also exposes narrow Windows system-control tools on direct user request:

- `system_volume_get`, `system_volume_set`, `system_volume_change`, and `system_volume_mute`.
- `screen_brightness_get`, `screen_brightness_set`, and `screen_brightness_change`.
- `agent_window_minimize`.
- `desktop_capture`.

Volume uses Windows Core Audio through `pycaw`. Agent volume tools control the default speaker endpoint; wake ducking controls per-app audio sessions so browser/background audio can drop without lowering the assistant's own TTS. Built-in screen brightness uses Windows WMI/CIM classes; external monitor brightness may not be available unless Windows exposes it through those classes. The assistant-window minimize tool targets the owned desktop host window passed to the voice loop by process ID, so requests like "minimize yourself" or "we're done talking" affect only WhisperToMe and hide it to the system tray. The desktop capture tool saves a full PNG under `artifacts\captures\`, temporarily excludes the WhisperToMe window when possible, and attaches a downscaled PNG as an OpenAI Responses `input_image` so the model can inspect what is behind the assistant. The prompt tells the model to clamp values to 0-100, use small relative changes for vague commands, use desktop capture for visual-context requests, and confirm briefly.

Organizer data is stored locally under ignored `artifacts\organizer\organizer.sqlite`. If an older `artifacts\organizer\store.json` exists, it is migrated into SQLite once and left in place as a legacy artifact. Preferences are upserted by stable keys such as `preferred_name` or `units`; only compact active preference sentences are injected into the system prompt, while raw user wording can be stored as evidence for debugging. The TUI system stream shows `using tool` while a tool is running, and logs include `wake_agent_tool`, `demo_agent_tool`, or `openai_agent_tool` entries with the tool name, arguments, result, and latency.

Examples:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "take a note that the wake word is computer"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "make a checklist called demo prep with charge laptop and pack adapter"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "add prototype demo tomorrow at 10 AM in the lab to my itinerary"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "remind me tomorrow to review the demo notes"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "what is my daily plan for tomorrow"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "what time sensitive things do I have tomorrow"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "computer call me Marcus"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "computer I prefer metric values"
```

Next organization tools worth adding are recurring routines, calendar export/sync, notifications, templates, review mode, and specialized reading/packing/shopping lists. Reminders currently persist and appear in daily-plan queries; a true alarm needs a scheduler or notification layer.

Run a full turn-based conversation demo:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome demo --allow-non-npu --save-audio
```

Press Enter to start listening, then speak. The app records until a speech pause or the `--record-ms` maximum, transcribes the speech, sends the transcript through OpenAI, synthesizes the assistant response, and plays it through the speakers. Type `q` or `quit` at the prompt to exit.

The `demo` command writes a log file automatically under `artifacts\logs\demo-*.log`. Use `--log-file artifacts\logs\my-demo.log` to choose a stable path. `--save-audio` writes each user recording and assistant TTS WAV under `artifacts\demo\...` so failed transcriptions and playback issues can be debugged later.

The current full audible demo requires `--allow-non-npu` because TTS is still on the explicit debug Kokoro path. STT stays on the verified Qualcomm QNN/NPU `qai_whisper` backend by default. The older `whisper_onnx` backend is the only STT path that opts into debug non-NPU mode with this flag. Without `--allow-non-npu`, the demo keeps the production policy strict and fails instead of silently using CPU or GPU execution.

Performance-oriented demo options:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome demo --allow-non-npu --save-audio --record-ms 8000
whispertome --project-root C:\Users\mreca\Desktop\whispertome demo --allow-non-npu --save-audio --vad-threshold 0.02
whispertome --project-root C:\Users\mreca\Desktop\whispertome demo --allow-non-npu --save-audio --fixed-record --record-ms 3000
```

The default demo prepares and warms persistent STT/TTS objects before the first turn, skips silent or very short captures, logs audio energy and clipping, and emits a `demo_turn_profile` line for each turn plus a `demo_profile_summary` at the end. On the Qualcomm Whisper-Small QNN path, warmup is about 2.5 seconds once and saved short-turn transcriptions run in about 300 ms wall time. If logs show `demo_audio_clipping`, lower the microphone input gain or move farther from the microphone; clipped speech will hurt Whisper accuracy.

For STT quality testing, the demo transcribes the raw speech-pause capture instead of the trimmed diagnostic clip. This keeps the front of short utterances intact while still avoiding the old fixed five-second recording delay. `turn-*-user.wav` and `turn-*-raw.wav` are both saved when `--save-audio` is enabled.

## Current Build Status

Implemented:

- `.env` configuration loading.
- OpenAI Responses API client using the official Python SDK.
- Microphone capture and speaker playback.
- Energy-based VAD and utterance segmentation.
- Continuous STT-oriented voice loop.
- Sliding transcript wake phrase detection with arbitrary wake phrases.
- Speech-pause-delimited command capture after wake detection.
- Token-window wake matching with fuzzy spans, stale-window protection, and original command-text extraction.
- OpenAI Responses API client with a dictation-aware system prompt and stateful response chaining.
- Responses API function-tool loop for local SQLite organizer tools.
- SQLite-backed preference memory with compact real-time system prompt injection.
- Windows volume and built-in screen brightness agent tools.
- Per-app Windows audio-session ducking during wake turns and playback interruption.
- Strict NPU-only ONNX Runtime provider selection.
- `doctor` command for safe config and runtime checks.
- `test-wake` command for wake phrase routing tests without microphone/model access.
- `test-tts` command for TTS initialization and WAV generation.
- `test-audio` command for speaker playback checks without any model inference.
- `test-stt` command for WAV-file and short microphone transcription checks.
- `test-openai` command for Responses API smoke tests.
- `demo` command for microphone -> STT -> OpenAI -> TTS -> speaker conversation testing with per-run logs.
- `run` command for the continuous wake phrase loop with per-run logs and optional utterance audio capture.
- Optional `run --tui` terminal UI with a startup NPU boot animation, central polygon animation, input/output text, and a bounded system state stream.
- `desktop` command that opens a first-class Windows window and hosts the existing terminal TUI process.
- Markdown-aware TTS text preparation that suppresses fenced and obvious unfenced code blocks and renders script/code blocks in the TUI.
- Interruptible assistant playback with concurrent STT wake detection during spoken output.
- Streaming OpenAI deltas into sentence-level TTS chunks for lower perceived assistant latency.
- Speech-pause-delimited demo recording with stage-level profiling and clipping warnings.
- Whisper ONNX split encoder/decoder adapter with Hugging Face feature extraction and tokenizer decoding.
- Kokoro ONNX TTS adapter using the real `kokoro-onnx` tokenizer, phonemizer, voices, and injected NPU-only ONNX session.
- Unit tests for config, wake phrase detection, and provider policy.

Still required before the live assistant can complete a fully NPU-only spoken turn:

- Produce or acquire a TTS export that QNN HTP can load with no CPU/GPU-assigned nodes.
- Add provider-specific profiling for TTS once a compatible artifact is available.

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

### STT Status

The project is currently configured for Qualcomm AI Hub Whisper-Small precompiled QNN ONNX artifacts on this Snapdragon X2 Elite machine:

```text
models/qai/whisper_small/snapdragon_x2_elite/precompiled_qnn_onnx/extracted/whisper_small-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite/encoder.onnx
models/qai/whisper_small/snapdragon_x2_elite/precompiled_qnn_onnx/extracted/whisper_small-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite/decoder.onnx
```

The `qai_whisper` adapter loads encoder and decoder sessions through the ONNX Runtime QNN plugin EP device with `session.disable_cpu_ep_fallback=1`. It transcribed the saved failed phrase `I need you to get sexy for me.` correctly and runs short saved turns in about 300 ms after warmup. The older `whisper_onnx` adapter remains available for model-quality experiments, but it is not the current production STT path.

### OpenAI Status

The OpenAI layer is wired through the Responses API:

- `instructions` carries the dictation-aware system prompt.
- `input` is a user message containing the speech-to-text transcript.
- `previous_response_id` is used for stateful turns when enabled.
- `tools` exposes local organization functions when the agent registry is enabled.
- `function_call_output` feeds local tool results back into the model until a final answer is ready.
- Active saved preferences are appended to `instructions` as a compact `User preferences` block every request, including immediately after a preference-saving tool call.
- `test-openai` verifies the API key, selected model, prompt, and response parsing.
- The default model is `gpt-5.4-mini` for lower voice-loop latency.
- Spoken replies stay brief by prompt; `OPENAI_MAX_OUTPUT_TOKENS=512` leaves room for short fenced code blocks unless overridden.
- The wake loop uses `responses.create(..., stream=True)` for the streaming TTS path and logs `wake_openai_stream` / `wake_streaming_tts` timing.

Realtime/WebSocket is not the default path yet. It remains the right future option if we switch to lower-latency cloud speech-to-speech or realtime audio transcription, but the current architecture keeps local STT/TTS plus a text Responses API turn.

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
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Realtime API: https://platform.openai.com/docs/guides/realtime
- ONNX Runtime DirectML Execution Provider: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- ONNX Runtime QNN Execution Provider: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
