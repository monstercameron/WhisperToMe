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

## Milestone 7: Local Latency Cleanup

### Wins

- Committed the QNN Whisper demo path as milestone `2cd2906`.
- Found that demo STT wall time was much higher than model time because each turn rebuilt the STT object and reloaded Hugging Face tokenizer/config assets.
- Changed the demo to prepare persistent STT and TTS model objects once before the turn loop.
- Warmed the Qualcomm Whisper STT model once before the first turn instead of letting the first spoken turn pay setup cost.
- Kept `qai_whisper` strict on the QNN/NPU path even when the demo uses `--allow-non-npu` for TTS.
- Verified saved demo clips after the refactor: warmup was about 2518 ms once, then QNN STT wall time matched model time at 294-318 ms per short turn.
- Added unit tests covering the STT debug-routing policy and TTS fallback activation.

### Losses

- OpenAI Responses API latency is still a hosted network/model round trip and is outside local NPU optimization.
- TTS is still debug non-NPU Kokoro, so the full audible demo is not fully NPU-only yet.
- Playback time is proportional to response length, so long assistant replies still make the conversation feel slower even if synthesis is fast.

### Decisions

- Keep the current local latency target focused on hot model reuse and short spoken replies.
- Treat OpenAI streaming or Realtime/WebSocket as the next architectural step if time-to-first-audio becomes the main blocker.

## Milestone 8: Wake Word Detection Hardening

### Wins

- Reworked wake detection from whole-window string matching to token-window matching.
- Added fuzzy matching against candidate token spans instead of comparing one phrase to the entire transcript history.
- Prevented stale wake phrases from re-triggering when the old phrase remains in the sliding window but the current utterance does not participate in the match.
- Added span metadata to wake matches so the router can preserve original command text after the wake phrase.
- Supported wake phrases split across adjacent STT utterances, such as `whisper` followed by `to me open diagnostics`.
- Added a logged continuous `run` wake loop for manual testing with QNN STT, wake routing, OpenAI, TTS, playback, and optional saved utterance WAVs.
- Increased the default end-of-speech silence from 700 ms to 1200 ms after manual testing cut off a sentence with a short thinking pause.
- Added `--speech-end-ms` so the pause can be tuned live without editing `.env`.
- Reviewed wake-loop logs after TTS appeared to skip; found TTS was not failing, the wake router failed to emit `wake_command` for long utterances where `computer` was at the start.
- Fixed the detector to check the previous context plus the full current transcript before trimming to the long-term sliding window.
- Added tests for fuzzy wake extraction, stale-window protection, split-utterance wake phrases, parser options, and run log defaults.

### Losses

- This is still STT-derived wake detection, so wake reliability depends on Whisper hearing the wake phrase correctly.
- Continuous local STT costs one QNN Whisper inference for each VAD-delimited speech utterance, even when the user is not talking to the assistant.
- A longer speech-end pause improves sentence capture but adds about 500 ms before STT starts after each user utterance.
- The audible wake loop still needs explicit debug TTS until a QNN-compatible TTS artifact exists.

### Decisions

- Keep arbitrary wake phrases as plain text configured through `.env` or repeated `--wake` flags.
- Prefer robust transcript-window matching over a separate wake-word model for now because it stays model-agnostic and runs on the verified STT path.
- Save utterance audio during manual wake tests so missed or false wake events can be traced back to the exact STT input.

## Milestone 9: Terminal UI Design

### Wins

- Added an optional `run --tui` terminal UI without introducing a new dependency.
- Built a central polygonal ASCII visualization that animates continuously and pulses with input activity.
- Added live input and output text panels for the latest transcript/command and assistant response.
- Added a bounded 1-10 line system stream for state transitions such as listening, speech captured, transcribing, wake detected, contacting OpenAI, running TTS, and playing speech.
- Routed wake-loop events into the UI while keeping detailed logs in the run log file.
- Suppressed console log streaming while the TUI is active so the visualization is not corrupted.
- Added unit tests for stream bounds, status rendering, polygon markers, and CLI options.

### Losses

- This is a terminal-rendered visualization, not a full Textual/curses app yet; resizing and high-frame animation are intentionally simple.
- PowerShell/terminal ANSI support is assumed for the live visual mode.

### Decisions

- Keep the TUI optional via `--tui` so plain log-oriented runs remain available.
- Keep the renderer pure Python and event-driven so the voice loop does not depend on UI internals.

## Milestone 10: Playback Interruption

### Wins

- Replaced blocking speaker playback with an interruptible playback path backed by a `threading.Event`.
- Added a playback-time wake monitor that consumes the existing microphone stream while assistant audio is playing, avoiding a second input stream.
- Kept the interruption detector on the same STT and wake-router path as normal listening.
- Added TUI states for interruption STT and interruption detected.
- Logged playback-time transcripts and `wake_playback_interrupted` events for debugging false positives and missed interruptions.
- Added tests for interruptible speaker stop behavior and wake detection during playback.
- Follow-up: when playback-time STT produces a `command_ready` event, the command is now queued and immediately executed as the next OpenAI turn.

### Losses

- The first interruption implementation stopped playback and returned to listening even when STT had already parsed a command such as `Computer, I don't care.`
- Echo from speakers may be captured by the microphone, so interruption wake phrases should be chosen carefully and tested with real speaker volume.
- Playback-time wake detection still waits for a VAD-delimited utterance before STT can confirm the wake phrase.

### Decisions

- Use the existing microphone iterator during playback rather than opening a second input stream, because some audio devices reject parallel input streams.
- Treat `wake_detected` interruptions as stop-only, but treat `command_ready` interruptions as immediate barge-in commands.

## Milestone 11: Markdown-Aware TTS and Script Display

### Wins

- Added a markdown parser for fenced code blocks before the TTS stage.
- Prevented TTS from reading triple-backtick fences, language tags, and code contents aloud.
- Replaced fenced code in spoken text with a short spoken placeholder such as `I put the script on screen.`
- Added a TUI code viewport that renders the latest fenced block, preferring `script` blocks when several are present.
- Made the code viewport auto-scroll when the block is longer than the available box height.
- Added tests for fenced script parsing, code-only responses, preferred script block selection, and TUI code rendering.
- Refined the OpenAI prompt so code is display content, not spoken content, and limited normal replies to one fenced code block unless multiple are explicitly requested.
- Changed the default OpenAI model from `gpt-5.2` to `gpt-5.5`.
- Raised the default OpenAI output cap from 64 to 512 tokens so short fenced snippets are less likely to be truncated before the closing fence.
- Added a defensive parser fallback for obvious unfenced code, including the raw `package main` Go FizzBuzz failure shape seen in the TUI.

### Losses

- The TUI code viewport auto-scrolls over time; it does not yet support keyboard-controlled scrolling.
- Unfenced-code detection is heuristic. It is a TTS safety net, not a replacement for the model following the fenced-block prompt.

### Decisions

- Keep raw OpenAI text in logs, but send only spoken-safe text to TTS.
- Keep prose in the assistant output panel and move code/script contents to the dedicated TUI viewport.
- Prefer one larger, copyable block over several smaller blocks unless the user asks for multiple files or examples.

## Milestone 12: Streaming Tokens to Streaming TTS

### Wins

- Added a streaming Responses API path that consumes `response.output_text.delta` events while preserving stateful `previous_response_id` behavior.
- Added a markdown-aware spoken chunker that emits sentence-complete text outside fenced code blocks.
- Added a queued streaming speech player with separate TTS and playback workers so later chunks can synthesize while earlier audio is playing.
- Made the wake loop use streaming TTS by default and added `--no-stream-tts` for the older batch path.
- Kept final markdown parsing for the TUI code viewport, so streamed speech and final display stay aligned.
- Logged first text, first audio, first playback, chunk count, synthesis time, and playback time for profiling.

### Losses

- Kokoro is still a batch TTS model, so streaming means chunked synthesis rather than true acoustic-frame streaming.
- Code-heavy replies may still wait until the final parse before speaking the placeholder, because the chunker avoids reading ambiguous pre-code lead-ins too early.
- The streaming path is implemented for the live wake loop first; the older `demo` command still uses batch OpenAI/TTS.

### Decisions

- Stream only sentence-complete spoken chunks to avoid choppy TTS and mid-sentence prosody problems.
- Keep `--no-stream-tts` as a practical fallback while tuning chunking, interruption, and audio device behavior.
- Continue using the local TTS model for audio; OpenAI streaming is only for text deltas.

## Milestone 13: Faster TUI Wake Sync

### Wins

- Changed the TUI renderer from a fixed polling loop to an event-driven loop that wakes immediately on state changes.
- Increased the default TUI render rate from 12 FPS to 24 FPS for smoother polygon motion.
- Added wake, interruption, and barge-in visual boosts so the central polygon reacts as soon as those states are set.

### Losses

- The TUI still depends on STT completion for true wake-word confirmation; this only removes render-loop lag after the wake state is known.

### Decisions

- Keep the animation event-driven instead of adding sleeps or polling in the wake loop.

## Milestone 14: Interim Wake Preview During Speech

### Wins

- Added speech-start, speech-chunk, and speech-end callbacks to the VAD segmenter.
- Added an interim wake preview worker that periodically transcribes a rolling audio window while the user is still speaking.
- Updates the TUI to `wake detected` as soon as rolling STT sees the wake phrase, instead of waiting for the final speech pause.
- Added a shared STT lock so interim preview, final STT, and playback interruption STT do not race the same model instance.
- Scoped interim preview to TUI runs so non-visual console runs do not pay the extra STT cost.

### Losses

- This is not true streaming-token Whisper. The current QNN Whisper path remains batch inference over short rolling audio windows.
- Interim preview can only accelerate the visual wake state; command execution still waits for the final pause-delimited transcript.

### Decisions

- Treat rolling interim transcripts as UI hints only. The final transcript remains the source of truth for router state and command dispatch.

## Milestone 15: Agentic Organizer Loop

### Wins

- Added a local Responses function-tool loop around `OpenAIResponder`.
- Tool calls are executed locally and returned to the model as `function_call_output` items.
- Kept support for streamed final assistant text after tool calls, so streaming TTS still works for the live path.
- Added a modular organizer store under ignored `artifacts/organizer/store.json`.
- Implemented notes, checklist, and itinerary tools with local persistence.
- Revised the system prompt so the assistant knows when to save notes, create/list/complete checklists, and add/list itinerary items.
- Added TUI/log visibility for agent tool use through `wake_agent_tool`, `demo_agent_tool`, and `openai_agent_tool` log entries.
- Added tests for organizer persistence and fake Responses tool loops.
- Verified both non-streaming and streaming Responses API smoke tests with the real OpenAI endpoint and local `notes_add` execution.

### Losses

- Reminders are not real yet because they need a scheduler or notification surface.
- Checklist completion by spoken text is heuristic when several items match.
- Itinerary date parsing depends on the model converting natural language to fields; the local store does not parse dates itself.
- `ruff` was not installed in the local virtual environment, so verification used pytest and compile checks.

### Decisions

- Keep organization data in ignored artifacts for local-first testing and no accidental commits.
- Use function names with underscores, such as `notes_add`, for API compatibility.
- Add only the core organizer tools now, then leave reminders, projects/goals, decision logs, contacts, and recurring routines as the next candidates.

## Milestone 16: SQLite Organizer Expansion

### Wins

- Replaced the organizer JSON store with a SQLite database at `artifacts/organizer/organizer.sqlite`.
- Added one-time migration from the legacy `artifacts/organizer/store.json` file.
- Expanded the organizer from notes/checklists/itinerary to the first nine organization categories:
  notes, reminders, checklists, itinerary, tasks, projects, daily plan, decision log, and people/contact notes.
- Added add/list/complete-style tools where they make sense, while preserving the existing note/checklist/itinerary tool names.
- Updated the system prompt so the model can choose the right organizer bucket for spoken requests.
- Added tests for SQLite persistence, legacy JSON migration, and registry coverage across all nine categories.
- Refined the prompt and `time_sensitive_check` tool for "up next", "next ups", and "important" queries with short lookahead windows and overdue open-loop inclusion.

### Losses

- Reminder alarms are still persistence-only; there is no background scheduler or notification loop yet.
- Daily plan is a persisted plan plus a date query over tasks, reminders, and itinerary, not a full planner UI.
- The tool list is larger now, which adds prompt/tool-schema payload to every OpenAI turn until we add deferred or intent-gated tool loading.

### Decisions

- Keep SQLite under ignored artifacts for local-first durability without committing personal organizer data.
- Leave the legacy JSON file in place after migration rather than deleting user data automatically.
- Prefer category-specific tool names for model reliability, even though a future latency pass may consolidate or defer rarely used tools.

## Milestone 17: Preference Memory Injection

### Wins

- Added SQLite-backed preference storage for stable user defaults like preferred name, units, tone, and formatting.
- Exposed `preferences_save` and `preferences_list` through the local Responses tool registry.
- Active preferences are injected into the OpenAI system prompt on every request as a compact `User preferences` block.
- Preference saves are visible immediately in the follow-up model call after the tool executes, without restarting the app.
- Raw user wording can be stored as evidence, while only the compressed preference sentence is injected to control token cost.
- Added tests for preference upsert/persistence, registry coverage, and dynamic prompt-context recomputation.

### Losses

- Preferences add two more tool schemas to the OpenAI request payload.
- There is no explicit preference deletion/deactivation tool yet; changes currently rely on upserting the same stable key.

### Decisions

- Store prompt-ready preference sentences separately from raw evidence so runtime context stays small.
- Sort injected preferences by priority and recency, then cap both item count and total characters.

## Milestone 18: Windows System Controls And Wake Ducking

### Wins

- Added narrow agent tools for Windows master volume, mute state, and built-in screen brightness.
- Added prompt guardrails so the model only changes volume or brightness on direct user request.
- Added wake-time volume ducking during assistant playback interruption, with exact restore afterward.
- Moved playback ducking earlier so it triggers on user-speech start, before batch STT confirms the wake phrase.
- Reworked ducking to use per-application Windows audio sessions, lowering background apps like Chrome while excluding the assistant process.
- Brightness read support was verified through the Windows WMI/CIM brightness classes on this machine.
- Core Audio volume read support was verified after adding `pycaw` to the environment.
- Added tests for system-control tools, percent clamping, relative changes, and volume duck/restore behavior.

### Losses

- External monitor brightness may not be controllable through the Windows WMI brightness classes.
- The first `pycaw` integration assumed the older COM activation shape; the installed version exposes `EndpointVolume` directly, so the backend had to support both.
- Wake ducking changes live Windows audio-session volumes, so the restore path has to stay conservative and robust.
- Ducking after confirmed wake detection was too late for short assistant replies because confirmation waits on a pause-delimited STT segment.
- Master-volume ducking and playback-only triggers did not match real testing with browser audio; the browser stayed normal because the app was in the normal wake-listening path, not playback interruption.

### Decisions

- Keep system controls narrow: no arbitrary desktop automation, no shell commands chosen by the model.
- Duck volume only downward and restore the previous exact value after the interruption path.
- Duck on user-speech start during assistant playback, then restore after playback if the speech was not a confirmed wake interruption.
- Prefer per-app session ducking for live wake turns so the assistant voice stays audible while background audio drops.
- Use built-in WMI/CIM for brightness before considering DDC/CI monitor-specific control.

## Milestone 19: Per-App Wake Audio Ducking Verified

### Wins

- Replaced the live wake-loop ducker with Windows per-application audio-session ducking.
- Confirmed `chrome.exe` could be lowered from full session volume to 25% and restored to full volume through `pycaw`.
- Verified the live run ducks background sessions on interim wake detection before final pause-delimited STT completes.
- Restores exact per-session volumes after the assistant turn finishes.
- Kept the assistant process excluded so local TTS remains audible while browser/background audio drops.
- Full test suite passed with 87 tests.

### Losses

- The original playback-only trigger missed the normal wake-listening path, so browser audio stayed loud when the user woke the assistant outside assistant playback.
- Per-session ducking affects sessions that exist at duck time; a new audio session opened mid-turn may need another duck trigger.

### Decisions

- Use per-app audio-session ducking for automatic wake behavior.
- Keep default speaker endpoint volume tools only for direct user requests like "turn the volume down."
- Trigger ducking from interim wake detection, final wake detection, command turn start, and playback-interruption speech.

## Milestone 20: TUI NPU Boot Animation

### Wins

- Added a dedicated full-screen TUI boot scene that covers startup until STT and TTS warmup finishes.
- Shows a tech-style animated lattice, progress rail, and module states for config, STT, TTS, and audio readiness.
- Keeps the normal input/output panels hidden until the voice loop is actually ready to listen.
- Wired CLI warmup steps into explicit boot phases for adapter setup, STT loading, TTS loading, and systems-online.

### Losses

- Startup progress is phase-based rather than true model-load percentage because the underlying model loaders do not expose incremental progress callbacks.
- Runs with `--no-warmup` clear the boot screen quickly because models are intentionally not loaded at startup.

### Decisions

- Keep the boot animation inside the existing terminal renderer instead of adding a separate splash subsystem.
- Use ASCII-only rendering so the TUI remains stable in plain Windows terminal sessions.
