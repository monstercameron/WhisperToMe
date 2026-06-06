# Getting Started

This guide covers the practical setup and run commands for WhisperToMe. The root README is now the product overview; this file is the manual.

## Prerequisites

- Windows.
- Python 3.11 or newer.
- A project `.env` file with an OpenAI key or Cerebras key.
- Local model artifacts under `models/`.
- A microphone and speakers.

For Snapdragon/QNN testing, use a Windows ARM64 Python environment with an `onnxruntime-qnn` wheel available. This development machine has been using Python 3.12 ARM64.

## Install

```powershell
cd C:\Users\mreca\Desktop\whispertome
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[directml]
```

For QNN provider work:

```powershell
pip install -e .[qnn]
```

For development tools:

```powershell
pip install -e .[dev]
```

## Environment

Copy `.env.example` to `.env`, then fill in the keys and model choices.

Important keys:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
WHISPERTOME_LLM_PROVIDER=openai

CEREBRAS_API_KEY=
CEREBRAS_MODEL=gpt-oss-120b

WHISPERTOME_STT_BACKEND=qai_whisper
WHISPERTOME_WAKE_PHRASES=computer
WHISPERTOME_SPEECH_END_MS=1200
WHISPERTOME_WAKE_DUCK_VOLUME=true
WHISPERTOME_WAKE_DUCK_VOLUME_PERCENT=25

# TTS: supertonic (NPU, default) or kokoro_onnx (CPU debug voice)
WHISPERTOME_TTS_BACKEND=supertonic
WHISPERTOME_SUPERTONIC_VOICE=M1
WHISPERTOME_SUPERTONIC_SPEED=1.15

# Scheduled events / reminders (in-process firing)
WHISPERTOME_SCHEDULER_ENABLED=true

# Auto-compaction: idle (1h) + usage threshold (Codex-style, cache-aware)
WHISPERTOME_IDLE_COMPACT_SECONDS=3600
WHISPERTOME_CONTEXT_WINDOW_TOKENS=128000
WHISPERTOME_COMPACT_THRESHOLD_PCT=0.75
```

The app reads `.env` automatically. It only reports whether a key is present; it does not print secrets.

## Model Artifacts

The current STT default is Qualcomm AI Hub Whisper-Small precompiled QNN ONNX for the Snapdragon X2 Elite target:

```text
models/qai/whisper_small/snapdragon_x2_elite/precompiled_qnn_onnx/extracted/whisper_small-precompiled_qnn_onnx-float-qualcomm_snapdragon_x2_elite
```

That folder must contain:

```text
encoder.onnx
decoder.onnx
encoder_qairt_context.bin
decoder_qairt_context.bin
```

The default TTS is **Supertonic**, which runs on the Hexagon NPU. Its source ONNX stages live under:

```text
models/supertonic2/model/onnx/        (text_encoder, duration_predictor, vector_estimator, vocoder + tts.json, unicode_indexer.json)
models/supertonic2/model/voice_styles/  (M1.json, F1.json, ...)
```

On first run the adapter static-fixes the source ONNX and compiles QNN HTP context binaries into `artifacts/supertonic_static/` (one-time ~20s warmup; later loads are fast). Set `WHISPERTOME_TTS_BACKEND=kokoro_onnx` (with `--allow-non-npu`) to use the Kokoro CPU debug voice instead — Kokoro's stock ONNX is still blocked on strict QNN HTP (dynamic shapes).

## Runtime Policy

WhisperToMe treats local inference as NPU-only by default. If an STT, VAD, or TTS graph cannot run on the selected NPU provider, the app should fail instead of silently falling back to CPU or GPU.

Debug commands that include `--allow-non-npu` intentionally bypass that production policy for development.

## Health Check

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome doctor
```

Use `doctor` before a live run when changing providers, model paths, or `.env` values.

## Run The Desktop App

This is the main manual-test command during development:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome desktop --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200
```

The desktop command opens an 800x600 `WhisperToMe` window and embeds the terminal TUI. It also creates a tray icon with a right-click menu for showing the window, stopping listening, and closing the program.

Normal desktop mode hides host and child console windows. Use this only when debugging the desktop host:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome desktop --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200 --show-console
```

Useful desktop behavior:

- The first launch centers the window.
- If the user manually moves it, the position is stored under `artifacts\desktop\window-state.json`.
- Wake detection restores the window and brings it forward.
- "Minimize yourself", "hide", or "we're done" should hide the app to the system tray instead of the taskbar.
- Stop and close request a cooperative shutdown before force-killing the child process.

## Run The Terminal UI

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200 --tui
```

The TUI includes:

- Startup animation while STT/TTS warm.
- Central animated polygon.
- Live input and output text.
- Bounded system stream for states such as listening, wake detected, contacting model, using tool, running TTS, and playing speech.
- Code/script viewport for fenced code blocks.

Change the stream height with:

```powershell
--tui-lines 6
```

## Run The Continuous Wake Loop

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200
```

The live loop continuously segments microphone speech with VAD, transcribes each utterance, checks a sliding transcript window for wake phrases, and sends a command to the model only after wake detection.

Wake phrase overrides:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --wake "computer"
whispertome --project-root C:\Users\mreca\Desktop\whispertome run --allow-non-npu --save-audio --wake "computer" --wake "hello dashboard"
```

The default end-of-speech pause is currently tuned around:

```text
WHISPERTOME_SPEECH_END_MS=1200
```

Raise it if sentences are cut off after "um", "ah", or short thinking pauses.

## Streaming And Interruptions

The wake loop streams OpenAI response deltas by default. Spoken-safe sentence chunks are sent to TTS as soon as they are ready.

While assistant audio is playing, a background wake monitor keeps listening. Say the wake phrase to interrupt playback. If the interruption also includes a command, such as `computer I don't care`, it is queued as the next turn.

Disable streaming TTS for debugging:

```powershell
--no-stream-tts
```

## Markdown And Code

Assistant responses are parsed before TTS:

- Fenced code blocks are not read aloud.
- The TUI renders the latest fenced block, preferring script-tagged blocks.
- The system prompt asks for one fenced code block per response unless more are explicitly requested.
- Defensive parsing also catches obvious unfenced code such as `package main`.

## Audio Ducking

Wake turns can lower other Windows app audio sessions while keeping WhisperToMe audible:

```text
WHISPERTOME_WAKE_DUCK_VOLUME=true
WHISPERTOME_WAKE_DUCK_VOLUME_PERCENT=25
```

The duck target only lowers other apps. If another app is already below the target, WhisperToMe does not raise it.

## LLM Providers

OpenAI:

```powershell
$env:WHISPERTOME_LLM_PROVIDER="openai"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "give me a short hello"
```

Cerebras:

```powershell
$env:WHISPERTOME_LLM_PROVIDER="cerebras"
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "give me a short hello"
```

The provider layer is swappable at runtime. Keep both key families in `.env` if you want to compare latency and behavior.

## Test Commands

Wake phrase routing without microphone, model files, or network:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-wake --wake "computer" "computer tell me the time"
```

Speaker playback without TTS:

```powershell
whispertome test-audio --duration-ms 500 --frequency 440 --volume 0.20
```

TTS smoke test:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-tts "Hello from WhisperToMe." --out artifacts\tts-test.wav
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-tts "Hello from WhisperToMe." --out artifacts\tts-debug.wav --play --allow-non-npu
```

STT smoke test:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-stt --record-ms 3000
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-stt --wav artifacts\tts-debug.wav
```

OpenAI/Cerebras layer smoke test:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome test-openai "rite a short reminder to call alex tomorrow"
```

Turn-based full demo:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome demo --allow-non-npu --save-audio
```

## Organizer Tools

The local agent tool loop currently supports:

- Notes.
- Preferences.
- Reminders.
- Checklists.
- Itinerary.
- Tasks.
- Projects.
- Daily plans.
- Time-sensitive "up next" checks.
- Decision log.
- People/contact notes.

Organizer data lives under:

```text
artifacts\organizer\organizer.sqlite
```

That folder is intentionally ignored by git.

## Windows Tools

The agent can use narrow local tools on direct user request:

- System volume get/set/change/mute.
- Screen brightness get/set/change.
- Agent window minimize.
- Desktop capture.
- Guarded PowerShell command execution.

Desktop capture saves a PNG under `artifacts\captures\`, tries to exclude the WhisperToMe window, and can send a downscaled image to the model as visual context.

## Logs

Logs are written under:

```text
artifacts\logs\
```

Useful log families:

- `desktop-*` for host window and tray behavior.
- `run-*` for continuous wake-loop behavior.
- `demo-*` for turn-based manual tests.

Use `--save-audio` when debugging STT or TTS. It stores user captures and assistant WAV files under `artifacts\wake\...` or `artifacts\demo\...`.

## Test Suite

```powershell
pytest
ruff check .
```

Run the full suite before milestone commits that touch code. For docs-only edits, a file/link smoke check is usually enough.
