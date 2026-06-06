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
- It ships with a desktop-hosted TUI: central animated polygon, live input/output panes, status stream, code/script viewport, startup animation, tray menu, and wake-triggered restore.

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
  runtime/    NPU-only provider selection
  stt/        Whisper adapters
  tts/        Kokoro adapters and speech preparation
  tui/        animated terminal UI
  wake/       wake phrase detection and command routing
```

The production policy is still strict: local AI inference should run on verified NPU paths, with no quiet CPU or GPU fallback. During feature development, explicit debug flags can bypass that policy so the voice loop, UI, tools, and LLM behavior can be tested while NPU TTS artifacts are still being researched.

## Quick Start

Developer setup, environment variables, model paths, live commands, desktop host details, and test commands now live in:

[Getting Started](docs/GETTING_STARTED.md)

The shortest current desktop dev run is:

```powershell
whispertome --project-root C:\Users\mreca\Desktop\whispertome desktop --allow-non-npu --save-audio --wake "computer" --speech-end-ms 1200
```

Use `--show-console` only when debugging the desktop host itself. Normal desktop mode should show the WhisperToMe window and tray icon without extra console windows.

## Project Website

A basic static project website lives in:

[docs/index.html](docs/index.html)

Open it directly in a browser. No build step is required.

## Research

Project history and research notes were moved out of the root so the README can stay readable:

- [Journey](research/journey.md)
- [Notes](research/notes.md)

The active TTS/NPU research thread is still tracked there. In short: Kokoro is the preferred local voice target, but the current stock ONNX export is blocked on dynamic-shape support for the Snapdragon/QNN NPU path. Debug TTS remains available for building the product loop while the proper NPU artifact is worked out.

## North Star

Fast wake. Accurate dictation. Short spoken replies. Useful local memory. Desktop-native controls. Speech models on the NPU.

That is the product.
