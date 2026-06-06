# WhisperToMe

WhisperToMe is a Windows voice assistant that lives on your desktop, listens for your chosen wake phrase, understands what you say with local Whisper STT, thinks through a hosted LLM, and speaks back with local TTS.

It is built for the Snapdragon/NPU era: keep the speech stack local, keep the assistant quick, keep the UI alive, and make the whole thing feel more like a cockpit instrument than a chatbot tab.

## The Pitch

Say `computer`, keep talking, and WhisperToMe handles the rest.

- It listens continuously with a wake phrase instead of a button.
- It transcribes locally through a modular Whisper backend.
- It calls OpenAI Responses or Cerebras through a runtime-swappable LLM layer.
- It streams text into speech so replies start sooner.
- It keeps listening during playback so you can interrupt it naturally.
- It stores notes, preferences, reminders, tasks, checklists, itinerary items, projects, decisions, and people notes in SQLite.
- It can answer "what is up next?" from local time-sensitive organizer data.
- It can adjust Windows volume, duck other apps, change brightness, minimize itself to the tray, run guarded PowerShell commands, and capture the desktop for visual context.
- It can switch to other open windows ("switch to Chrome") and open web searches, maps, and sites in the browser.
- It schedules future events and reminders that fire on their own — at a time or on a recurrence — to speak a reminder or run a recorded tool workflow autonomously.
- It auto-compacts the conversation after an idle hour or when context fills up (a Codex-style, cache-aware threshold), keeping replies fast over long sessions.
- It runs speech on the Snapdragon NPU: both Whisper STT and Supertonic TTS execute on the Hexagon NPU through ONNX Runtime QNN.
- It ships with a desktop-hosted TUI: central animated polygon, voice-reactive waveform, a live "next up" agenda chip, input/output panes, status stream, code/script viewport, startup animation, tray menu, and wake-triggered restore.

## Why It Is Fun

WhisperToMe is not trying to be another passive assistant bubble. It is a local-first voice rig for a power user desktop:

- The wake loop is always warm.
- The UI shows what the system is doing right now.
- The assistant can save useful structure instead of dropping everything into a chat transcript.
- Preferences are injected compactly into the system prompt as they are saved.
- Code blocks are rendered on screen instead of read aloud.
- Background audio ducks when the assistant needs the stage.
- The agent can look at the desktop when asked, then talk about what is actually on screen.

## Current Shape

The project is intentionally modular:

```text
src/whispertome/
  agent/      local tool loop and system prompts
  audio/      microphone, playback, VAD, ducking
  llm/        OpenAI and Cerebras provider adapters
  models/     model registry and runtime wiring
  organizer/  SQLite-backed memory and planning tools
  runtime/    NPU-only provider selection + async runtime event bus
  scheduler/  in-process scheduled events that fire (reminders + recorded workflows)
  stt/        Whisper adapters (Qualcomm QNN Whisper-Small runs on the NPU)
  tts/        Kokoro + Supertonic adapters (Supertonic runs on the NPU)
  tui/        animated terminal UI
  wake/       wake phrase detection and command routing
```

The production policy is still strict: local AI inference runs on verified NPU paths, with no quiet CPU or GPU fallback. Both halves of the speech stack now meet it — Whisper STT and Supertonic TTS run on the Hexagon NPU via QNN HTP. Explicit debug flags can still bypass the policy (e.g. Kokoro on CPU) for voice-quality testing.

## Quick Start

Developer setup, environment variables, model paths, live commands, desktop host details, and test commands now live in:

[Getting Started](docs/GETTING_STARTED.md)

The shortest current desktop dev run is now fully NPU (STT + Supertonic TTS), no debug flag:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome desktop --wake "computer"
```

Add `--allow-non-npu` only to fall back to the Kokoro CPU debug voice.

Use `--show-console` only when debugging the desktop host itself. Normal desktop mode should show the WhisperToMe window and tray icon without extra console windows.

## Project Website

A basic static project website lives in:

[docs/index.html](docs/index.html)

Open it directly in a browser. No build step is required.

## Research

Project history and research notes were moved out of the root so the README can stay readable:

- [Journey](research/journey.md)
- [Notes](research/notes.md)

The TTS/NPU research thread resolved to a working NPU voice: **Supertonic** runs end-to-end on the Hexagon NPU (~10x realtime) by static-fixing the source ONNX and compiling fresh on-device. Kokoro stays as a CPU debug voice — its stock export is blocked on QNN dynamic shapes and a correct static export needs an attention-mask re-export (no ARM64 Windows PyTorch wheel). Details and the full milestone log are in the journey.

## North Star

Fast wake. Accurate dictation. Short spoken replies. Useful local memory. Desktop-native controls. Speech models on the NPU.

That is the product.
