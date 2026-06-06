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

## Milestone 21: Desktop Terminal Host Prototype

### Wins

- Added a `desktop` command that opens an owned `WhisperToMe` Windows window.
- The desktop host launches the existing `run --tui` voice loop as a child process, preserving the current audio, wake, OpenAI, TTS, and logging behavior.
- Renders the child process ANSI terminal frames inside the window with a dark terminal surface and basic color support.
- Closing the window or pressing Stop requests child shutdown from the host window.
- Refined Stop and window close to request cooperative child shutdown through a per-run stop file before falling back to force termination.
- Forced the desktop shell to an 800x600 target and passed a compact TUI viewport to the child process.
- Reworked the TUI frame budget so the polygon, input/output panels, code viewport, and newest system-stream lines fit in the smaller window.
- Coalesced queued ANSI frames in the desktop host so stale animation frames do not make the window feel behind.
- Added an agent tool that hides the owned desktop host window to the system tray on direct user request.
- Added a Windows tray icon with a right-click menu for show window, stop listening, and close program.
- Added a wake-event signal file so the host restores, centers, and lifts the desktop window when wake detection fires.
- Added a child-to-host window command signal so "minimize", "hide", and "we're done talking" can withdraw the Tk window instead of creating a taskbar-minimized window.
- Tightened the system prompt and tool description so conversational dismissal phrases steer to the tray-hide tool.
- Added tests for desktop argument parsing, host log resolution, and child command construction.

### Losses

- This first slice is an ANSI frame host, not a real ConPTY terminal emulator yet.
- Keyboard input is intentionally not wired through because the current app is voice-first and does not need terminal input for normal use.
- The host window is still a shell around the TUI rather than a native multi-panel Windows interface.
- Cooperative stop is checked promptly while listening; a child stuck in a long external request can still require the fallback timeout.
- The 800x600 shell requires tighter stream/code budgets, so older stream entries are intentionally dropped from view first.
- The tray menu depends on `pystray` and `Pillow`; the desktop host logs and continues without a tray icon if those packages are unavailable.
- If the tray icon cannot start, hiding the window would make recovery awkward, so logs should be checked before relying on tray-only operation on a new machine.

### Decisions

- Start with a no-new-dependency Tk host to validate first-class window ownership quickly.
- Keep the embedded voice loop as a child process so crashes and logs stay isolated from the host window.
- Prefer cooperative stop-file shutdown before any process-tree termination.
- Favor a fixed small window while validating the app feel, then revisit resizable/native panels once the hosted loop is stable.
- Pass host-window identity through environment variables so child-process tools can target the owned window precisely.
- Use a small signal-file protocol for child-to-host window activation instead of parsing ANSI TUI frames.
- Use a second signal-file command for tray-hide requests so the host process owns Tk `withdraw()` and restore behavior.
- Leave WebView2/xterm.js or ConPTY as the next deeper integration step once the first window-host behavior is proven.

## Follow-up: Desktop Placement and Visual Context

### Wins

- Centered the desktop host on first launch instead of relying on Tk's default window placement.
- Added persisted window placement so manual user moves survive tray hide/show and future launches.
- Added clamping for saved placement so stale monitor layouts cannot restore the window fully off-screen.
- Added a `desktop_capture` tool that saves a PNG artifact and sends a downscaled PNG to the Responses API as an `input_image`.
- Desktop capture temporarily excludes the WhisperToMe window when possible so the model can inspect what is behind the assistant.

### Losses

- Capturing behind the assistant requires a brief hide/show of the host window when it is visible, so there can be a small visual blink on capture requests.
- The screenshot is visual context only; it does not grant arbitrary desktop control or OCR outside what the model can infer from the image.

### Decisions

- Keep placement state local in `artifacts\desktop\window-state.json` rather than introducing global Windows settings.
- Send screenshots as Responses API image inputs, not as text-only file paths, because the model needs actual pixels to reason about the desktop.

## Follow-up: Lower-Latency OpenAI Model

### Wins

- Switched the default OpenAI Responses model from `gpt-5.5` to `gpt-5.4-mini` for lower voice-loop latency, after confirming `gpt-5.5-mini` is not available in the account.
- Added an `.env` override so the currently running app picks the mini model without relying on code defaults.

### Losses

- The mini model may be less capable on visual or nuanced reasoning than the regular model, so screen-inspection quality should be watched in logs.

### Decisions

- Prefer lower latency for the spoken loop unless a task clearly needs the regular model.

## Milestone 22: Provider Swap and Agent-Loop Hardening

### Wins

- Added a runtime-selectable LLM provider layer so the app can switch between OpenAI Responses and Cerebras Chat Completions without deleting either path.
- Wired Cerebras `gpt-oss-120b` into the same dictation-aware system prompt, organizer tools, TUI events, and TTS flow.
- Hardened Cerebras rate-limit behavior by disabling long SDK retries and surfacing `429` as a short spoken fallback instead of freezing the loop.
- Added recoverable wake-turn guards so provider and tool-loop failures restore ducked audio, log the traceback, and return to listening.
- Added a bounded `powershell_run` agent tool for explicit Windows inspection commands such as time, services, processes, and local configuration.
- Verified the 429 fallback live: the assistant read the rate-limit message aloud and the app stayed usable.

### Losses

- Cerebras tool follow-up calls can still rate limit during dense agentic turns, so OpenAI remains the steadier manual-test provider for now.
- The Cerebras path is text-only for desktop-capture tool results, while OpenAI Responses can receive the actual screenshot image.
- Generic PowerShell access required conservative blocking and output limits; destructive or broad state-changing commands intentionally do not run through this tool.

### Decisions

- Fail fast on Cerebras provider errors in the voice loop instead of letting SDK retry sleeps block interruption handling.
- Treat LLM/tool-loop errors as recoverable turn failures with short spoken feedback.
- Prefer dedicated system tools for volume, brightness, window, and desktop capture; reserve PowerShell for explicit local inspection gaps.

## Milestone 23: Kokoro NPU TTS — Root-Causing the Dynamic Shape (In Progress)

Work isolated under `research/kokoro_npu/` (diagnostic-only scripts; no changes to
the `whispertome` package). Targets the Kokoro path only; the supertonic2 QNN
effort is owned elsewhere and left untouched.

### Wins

- Built three reusable diagnostic tools that run on the verified ARM64 QNN venv:
  - `diagnose.py` — enumerates graph inputs/outputs, every value_info tensor with a
    symbolic/dynamic dim, the distinct symbolic dim params, and attempts a real QNN
    HTP session with `session.disable_cpu_ep_fallback=1` to capture the exact error.
  - `trace.py` — walks backwards from any tensor to its producers/inputs, printing
    inferred shapes, to locate where dynamism originates.
  - `make_static.py` — symbolic-shape-inference + batch-axis forcing pass (WIP).
- Captured the exact QNN rejection on `kokoro-v1.0-seq512-inferred.onnx`:
  `EP_FAIL : Dynamic shape is not supported yet, for input: /encoder/shared/Transpose_output_0`.
- Proved the "1609 dynamic internal tensors" are actually TWO different things:
  - The vast majority are the **batch axis** left symbolic (`<unk__357>`, `<unk__344>`,
    `<unk__315>` …). The sequence axis is already static at 512 after the seq-fix.
    These are statically fixable (batch = 1).
  - One family, `<unk__368>`, is the **data-dependent output frame count** and is the
    real blocker.
- Traced `<unk__368>` to its origin. It is the StyleTTS2 length regulator:
  `predictor.ReduceSum(durations)` → `Div by speed` input → `Round` → `Clip` →
  `CumSum` → take last element (total frames) → `Range(0, total_frames)`. That length
  then flows UP through the alignment `MatMul` (`/encoder/MatMul_output_0`
  `[<unk__357>,640,<unk__368>]`) and into the decoder/vocoder, so it is not a tail-only
  dynamic dim — it is baked through the second half of the graph.

### Losses

- Freezing `tokens` to `[1,512]` can never make Kokoro fully static: it fixes the
  input phoneme axis but the output time axis is `sum(round(predicted_durations/speed))`,
  computed at inference time from the text and the `speed` input.
- ONNX Runtime `SymbolicShapeInference` crashes on this graph
  (`_infer_Gather: object of type 'NoneType' has no len()`) because an upstream tensor
  has unknown rank, so the easy "just re-infer shapes" route does not finish.
- Therefore the README's existing seq-512 artifacts (`-seq512`, `-seq512-inferred`)
  cannot load on QNN as-is, which matches the QNN error above.

### Findings / Decisions

- The only QNN-loadable Kokoro paths are: (a) graph surgery to replace the
  duration-driven `Range` length with a constant `MAX_FRAMES` (pad the alignment,
  trim trailing silence on CPU using the real total) — feasible here with no PyTorch;
  or (b) a fixed-max-length + mask re-export from the PyTorch source — blocked because
  there is still no PyTorch wheel for this Windows ARM64 Python; or (c) split the graph
  at the length regulator (encoder+duration on NPU, expansion on CPU, decoder on NPU
  with fixed frames) — the supertonic-style approach.
- Chosen next step: attempt path (a), constant-`MAX_FRAMES` graph surgery, because it
  is the only fully-NPU option that does not require the blocked PyTorch toolchain.

### Next Steps

- Force the batch family (`<unk__357>` et al.) to 1 across inputs and value_info.
- Replace the `Range` limit feeding `<unk__368>` with a constant `MAX_FRAMES`; keep the
  real total-frame scalar as a second output so CPU can trim the padded audio.
- Re-run `diagnose.py` to confirm zero remaining dynamic tensors, then re-attempt the
  QNN HTP load and verify no CPU-assigned nodes.

### Update: shapes are solvable, correctness is not (the real wall)

The `surgery.py` constant-`MAX_FRAMES` pass works mechanically (`<unk__368>` is gone),
but running the seq-512 graph on CPU exposed a deeper, fatal problem that pure ONNX
surgery cannot fix:

- Measured the real frame geometry: hop = 600 samples/frame at 24 kHz. The reference
  utterance "Hello from WhisperToMe." (29 wrapped tokens) is 57 frames / 1.43 s.
- The same text right-padded to 512 tokens produces **578 frames / 14.45 s** — the 483
  pad tokens (token id `0`, which is also bos/eos) each receive a nonzero predicted
  duration. Kokoro has no padding/attention-mask input, so padding is not ignored.
- Energy profile of the padded output: 402 of 578 frames are audibly loud, spread
  across the whole clip — not "content then silence." `corr(reference, padded) =
  0.0046` (essentially uncorrelated). Right-padding corrupts the entire utterance.
- Isolated the cause: compared the BERT/ALBERT encoder output for the real tokens,
  unpadded vs padded. `max|diff| = 2.58`, `mean = 0.31` over `[1,29,768]` (~57% of
  signal magnitude). So the encoder self-attention AND the bidirectional text-encoder
  LSTM mix pad tokens into the real tokens. The contamination is upstream of durations,
  so masking durations alone cannot recover correct audio.
- Important correction: there are two `Range` nodes; only `/encoder/Range` is the
  duration-driven frame axis. `Range_6494` is a tiny indexing arange inside the iSTFT
  (output shape `(1,)`); pinning it to `MAX_FRAMES` is what broke the vocoder iSTFT with
  `Incompatible dimensions`. Surgery must target `/encoder/Range` only.

### Revised conclusion / decision

- Freezing tokens to 512 cannot yield correct NPU Kokoro by ONNX surgery alone. A static
  512 graph is only correct if pad tokens are truly masked, which requires: BERT
  attention masking, the ONNX LSTM `sequence_lens` input wired to a real length, and a
  duration mask — i.e. effectively a re-export with a proper attention mask and a fixed
  max length from the PyTorch source.
- Re-checked torch on this machine (2026-06-06): still no `torch` wheel for Windows
  ARM64 cp312 (`pip` finds no distribution), so the clean re-export cannot be done here.
- Therefore the realistic options are: (a) re-export Kokoro with mask + fixed max length
  on an x64/Linux box that has PyTorch, then bring the static ONNX back; (b) high-effort
  in-graph mask surgery (attention mask + LSTM `sequence_lens` + duration mask), correctness
  verified per-stage against the dynamic model — risky and slow; or (c) keep TTS on the
  fast CPU path (already real-time) and accept it is not on the NPU, treating the all-day
  power goal as satisfied by STT-on-NPU for now. All diagnostic tooling lives in
  `research/kokoro_npu/` for whoever picks this up.

### Existing-export survey (does a right-shaped ONNX already exist?)

- The publicly downloadable Kokoro ONNX models — `onnx-community/Kokoro-82M-ONNX` and
  `-v1.0-ONNX`, `thewh1teagle/kokoro-onnx` (our current source), `NeuML/kokoro-*-onnx` —
  all use a **dynamic token axis with no mask/length input**. Same problem as ours; none
  are the right shape for QNN, and none can be safely right-padded (see contamination
  measurement above).
- The correct approach is published as **export pipelines, not prebuilt files**:
  - `NimbleEdge/kokoro` rebuilds Kokoro as a static graph with an `input_lengths` input
    and recomputes attention masks after every upsample — i.e. it adds exactly the
    padding mask our graph lacks. `export_onnx.py` produces `kokoro_batched_quantized.onnx`.
    No prebuilt ONNX is published; the README does not state whether the token axis is
    pinned to a fixed size, so the exporter would need to be run with a fixed seq (512).
  - `adrianlyjak/kokoro-onnx-export` is a second, similar mask-based export script set.
- Both require PyTorch to run, which is unavailable on this ARM64 Windows box. Recommended
  path: run `NimbleEdge/kokoro`'s exporter on any torch-capable machine with a fixed
  token length + `input_lengths`, then bring the ONNX back here and QNN-load it. This is
  strictly better than in-graph surgery because the mask correctness comes from the source.

> NOTE: root `journey.md` was deleted by a parallel agent's docs restructure at ~16:21
> (a new `docs/` site appeared). This copy under `research/journey.md` preserves the
> Kokoro/Supertonic NPU-TTS findings (Milestones 23-24) so they are not lost. Merge back
> into the canonical journey/docs once the restructure settles.

## Milestone 24: Supertonic NPU Load-Probe (Isolated, Read-Only)

Goal: answer the open question from Milestone 1 — do the Supertonic precompiled QNN
context artifacts load on THIS machine's onnxruntime-qnn 2.2.0 / QnnHtp build? Done as
an isolated read-only probe (`research/supertonic_npu/`, on copies); the live Supertonic
work in `models/supertonic2/` belongs to another effort and was not modified.

### Wins (signal gathered)

- Confirmed Supertonic's text_encoder is architecturally NPU-ready: the `_net.json`
  `converter_command` shows a clean QAIRT `qnn-onnx-converter` run — INT8 QDQ (8-bit
  act/weight/bias, asymmetric, min-max calibration with a real calibration list), fully
  static input dims (`text_ids 1,128`, `style_ttl 1,50,256`, `text_mask 1,1,128`),
  `unroll_lstm_time_steps=True`. It has a proper `text_mask` input — the mask design
  Kokoro lacks. This is the right approach.
- The context-ONNX I/O is static and sensible:
  `text_ids[1,128]`, `text_mask_dq[1,128,1]`, `style_ttl_dq[1,256,50]` -> `text_emb_dq[1,128,256]`.

### Losses (both ctx ONNX fail to load on this build)

- `text_encoder_htp_net_qnn_ctx.onnx` is **opset 26**; ORT 1.24.4 rejects it at
  `ValidateOpsetForDomain` (opset too new). The `_op20` variant (opset 20) passes that check.
- `text_encoder_htp_net_qnn_ctx_op20.onnx` then fails inside QNN with
  `LoadQnnCtxFromOnnxGraph: Failed to get context binary info` — the same error class as
  Milestone 1, already special-cased in `runtime/onnx_session.py`. The EPContext wrapper
  reports `ep_sdk_version = unknown`; ORT-QNN uses that field to validate the embedded
  binary, so an unstamped/mismatched version makes the context unreadable.
- Online ORT-QNN compile of the *dynamic source* `text_encoder.onnx` fails with
  "nodes assigned to the default CPU EP" under no-fallback — the online partitioner will
  not claim every node. Confirms the offline QAIRT route was the correct choice; online
  compilation is a dead end here.

### Environment facts

- `onnxruntime-qnn 2.2.0` bundles HTP stubs/skels for **V68, V73, V81** (plus a V79 stub).
  The compiled binary's target HTP arch must be one of these AND match the X2 Elite's
  Hexagon version or it will not load.

### Findings / hand-off for the Supertonic effort

- The `.bin` looks correctly built; the failure is in the **EPContext wrapper / context-
  binary version metadata**, not the model graph. Two concrete fixes to try:
  1. Regenerate the context with a QAIRT SDK version matched to onnxruntime-qnn 2.2.0's
     QNN libs (confirm HTP arch is V68/V73/V81) so `ep_sdk_version` is stamped/readable.
  2. Or have ORT-QNN itself produce the EPContext (`ep.context_enable=1`) from a static
     QDQ ONNX, so the wrapper metadata is written by the same stack that reads it.
- Only `text_encoder` is wrapped to a ctx ONNX so far; `duration_predictor`,
  `vector_estimator`, `vocoder` have `.bin`+`.json` but no ctx ONNX yet — the full 4-stage
  pipeline still needs wrapping + a CPU glue runner between stages.

### BREAKTHROUGH: all 4 stages run on the X2 NPU from the fp source (no version-matched binary needed)

The precompiled-binary version mismatch turned out to be a non-issue: the dynamic-shape
failure was the *only* real blocker. Static-fixing each source ONNX to the QAIRT dims and
letting **this machine's** ORT-QNN compile it fresh on HTP works for every stage, and the
fp16 HTP output matches CPU fp32:

| stage | HTP load (no CPU fallback) | CPU↔NPU corr | rel L2 |
|-------|----------------------------|--------------|--------|
| text_encoder       | OK | 0.99998 | 0.0066 |
| duration_predictor | OK | 1.00000 | 0.0007 |
| vector_estimator   | OK | 1.00000 | 0.0020 |
| vocoder            | OK | 0.99978 | 0.0220 |

Pipeline contract (fixed dims chosen by the QAIRT compile; masks handle padding):
- text_encoder: `text_ids[1,128]`, `style_ttl[1,50,256]`, `text_mask[1,1,128]` -> `text_emb[1,256,128]`
- duration_predictor: `text_ids[1,128]`, `style_dp[1,8,16]`, `text_mask[1,1,128]` -> `duration[128]`
- vector_estimator (flow-matching velocity field, iterated `total_step` times):
  `noisy_latent[1,144,192]`, `text_emb[1,256,128]`, `style_ttl[1,50,256]`,
  `latent_mask[1,1,192]`, `text_mask[1,1,128]`, `current_step`(f32), `total_step`(f32)
  -> `denoised_latent[1,144,192]`
- vocoder: `latent[1,144,192]` -> `wav_tts` (589,824 samples for 192 frames -> hop 3072)
- Fixed maxima: text_length=128, latent/frame_length=192, latent_dim=144.

Implication: do NOT reuse the version-mismatched `.bin`/ctx artifacts. The correct
on-device recipe is static-fix source -> ORT-QNN fresh compile (optionally cache via
`ep.context_enable=1` so warmup is paid once). fp16 is accurate enough; INT8 QDQ is a
later optimization, not required to run.

### Remaining for end-to-end NPU TTS (CPU glue, ~not started)

- Tokenization + style: `model/onnx/unicode_indexer.json`, `tts.json` (config), style vectors.
- Length regulation on CPU: expand `text_emb` by predicted `duration` into the 192-frame
  latent layout and build `latent_mask` (this is the variable-length step Kokoro couldn't
  make static; Supertonic keeps it on CPU between static NPU stages — the key design win).
- Flow-matching integration loop on CPU: init `noisy_latent` from noise, call
  vector_estimator for `current_step` in 0..total_step, integrate to `denoised_latent`.
- Feed final latent to vocoder -> audio. All four NPU calls are static; only the glue is CPU.
- App integration (TTS backend) is the parallel agent's domain — keep this as a standalone
  isolated runner in `research/supertonic_npu/` and hand them the verified recipe.

### END-TO-END NPU TTS WORKING (`run_e2e.py`, `correctness_e2e.py`)

Built the full pipeline on the NPU using the reference glue from
`models/supertonic2/supertonic_inference.py` (tokenize via `unicode_indexer[ord(c)]` +
`<lang>` tags; precomputed `voice_styles/M1.json` etc.; CPU length-regulation; 10-step
flow-matching loop; vocoder; trim). Inputs padded to text=128 / frames=192 with masks.
Audio params: sample_rate 44100, chunk 512*6=3072 samples/frame, latent_dim 144.

Latency (test sentence, 6.32 s of audio, 10 diffusion steps), steady-state:
| engine | synth time | RTF | note |
|--------|-----------|-----|------|
| Supertonic on **NPU (HTP)**   | **~605 ms** | **0.096** (~10x realtime) | dp 3ms, te 7ms, ve_loop(10) 547ms, vo 43ms |
| Supertonic on CPU (reference) | ~2457 ms | 0.389 | ~4x slower than NPU |
| Kokoro on CPU (current app)   | ~949 ms (5.1 s audio) | 0.186 | ~1.6x slower than NPU |

NPU session init+compile (one-time warmup) ~17.7 s — should be cached via
`ep.context_enable=1` so it is paid once, not per run.

Correctness (apples-to-apples, identical noise — waveform corr is meaningless across
different random init, so seed both sides):
- NPU-padded vs CPU-padded = **0.98** (fp16 EP fidelity through the 10-step loop)
- CPU-padded vs CPU-reference = **0.95** (padding+masking == dynamic reference)
- NPU-padded vs CPU-reference = **0.92** (end-to-end)
Audio played back fine on speakers; the diffusion loop (`vector_estimator` x steps) is
~90% of NPU time, so fewer steps trade quality for latency.

Net: Supertonic is the TTS that runs on this X2 NPU today — fastest of the three AND
off-CPU (the all-day-power goal). Remaining: cache the HTP context (warmup), then the
parallel agent can wire it in as a `TextToSpeechModel` backend.

### Milestone 25: Supertonic wired into the app as the NPU TTS backend

Integrated through the existing TTS abstraction so switching is config-only:
- New `tts/supertonic_onnx.py` `SupertonicOnnxSynthesizer` implements `TextToSpeechModel`.
  It static-fixes the source ONNX to the fixed dims (text=128, frames=192) into
  `artifacts/supertonic_static/` (cached, regenerated if source is newer), creates the 4
  stages via the shared `NpuOnlyOnnxSessionFactory` (QNN HTP, no CPU fallback), and keeps
  tokenization / length-regulation / the diffusion loop / trim on the CPU. Frontend ported
  in-package (no coupling to the parallel agent's `supertonic_inference.py`).
- `models/registry.py`: `create_tts()` now dispatches `supertonic` -> SupertonicOnnxSynthesizer,
  `kokoro_onnx` -> KokoroOnnxSynthesizer.
- `config.py`: default `WHISPERTOME_TTS_BACKEND` is now `supertonic`; added
  `WHISPERTOME_SUPERTONIC_DIR`, `WHISPERTOME_SUPERTONIC_VOICE` (M1), `WHISPERTOME_SUPERTONIC_STEPS` (10).
- `cli.py`: added `should_prefer_debug_tts()` (mirrors the STT pattern) so only `kokoro_onnx`
  routes to the debug CPU synth under `--allow-non-npu`; `supertonic` stays on the NPU with
  Kokoro as the `FallbackTextToSpeechModel` backup.

Switching: `WHISPERTOME_TTS_BACKEND=supertonic` (NPU, default) | `=kokoro_onnx` (CPU backup).
Verified: full suite 112 passed; `ModelRegistry.create_tts()` -> NPU synth 595 ms / 5.95 s
audio (`provider=QNNExecutionProvider`); backend switch returns the right adapter both ways;
audio plays correctly. HTP compile ~18.5 s at load (still uncached — `ep.context_enable=1`
is the next optimization).

NOTE on collision: the parallel agent is committing with `git add -A`, which swept my
`registry.py` edit into their commit `8a56dfe`. My `config.py`/`cli.py`/`tts/supertonic_onnx.py`
changes are currently in the working tree; they may likewise get absorbed. Functionally fine,
but the canonical history will attribute these to mixed commits.

### Milestone 26: HTP context caching (warmup once)

`SupertonicOnnxSynthesizer` now compiles each stage to a QNN EPContext ONNX on first load
(`ep.context_enable=1` + `ep.context_file_path`, embedded binary) into
`artifacts/supertonic_static/<stage>_ctx.onnx`, then loads those on subsequent runs.
Generating the context with this machine's ORT-QNN stack avoids the version-mismatch that
broke the parallel agent's prebuilt binaries. Measured: **cold load 18.8 s -> warm load 0.5 s**
(across process restarts). Falls back to on-the-fly compile if QNN/devices are unavailable.

### Milestone 27: Model-load event API

New `models/loading.py`: `ModelLoadEvent` + `LoadListener` + `ModelLoadReporter`. The core
loader (`ModelRegistry`) owns the listener list (`add_load_listener`) and attaches a reporter
to every model it builds (`set_load_reporter`, default no-op on the STT/TTS base classes, so
existing adapters are unaffected). Supertonic emits `start` / per-stage `stage`
(cached vs compiling, timing) / final `loaded` (with provider). `cli._build_registry` attaches
a default logging listener (`model_load component=... status=... stage=...`); richer observers
(boot UI, voice loop) can register their own. Verified events fire end-to-end.

### Milestone 28: Runtime voice toggle via an LLM tool

Voices in Supertonic are just style-embedding files fed to the same NPU graphs, so switching
is a live data swap — no recompile/reload. Added to the `TextToSpeechModel` base:
`list_voices()`, `current_voice()`, `set_voice()` (default `VoiceNotSupportedError`).
Supertonic implements them against `model/voice_styles/*.json`. `FallbackTextToSpeechModel`
forwards these to the active model. New `tts/voice_tools.py`: a late-bindable
`TtsVoiceController` + `build_voice_tools()` exposing agent tools `tts_list_voices` and
`tts_set_voice`, wired into the wake-loop and `pipeline.VoiceLoop` tool registries via the new
`extra_tools` arg on `build_organization_tool_registry`. Verified: tool lists `[F1, M1]`,
switches both ways, rejects unknown voices with the available list; F1/M1 render distinct audio.
Only F1 and M1 ship today; dropping more `voice_styles/*.json` adds more (custom cloning needs
the style-encoder, which is not exported). 118 tests pass.

### Milestone 29: Long-text repetition + short-clip slowness fixes

Two issues reported on longer/streamed use: random word repetition, and ~0.8-0.9x RTF on
short replies. Root causes + fixes:

- **Repetition**: the adapter hard-truncated tokens at 128, which dropped text AND the closing
  `</lang>` tag — causing the model to ramble/repeat; and audio beyond the 192-frame cap was
  clipped. Verified: a 250-token sentence was cut to 128 (16.6s predicted -> 13.4s cap).
  Fix: `_split_text()` chunks input at sentence -> clause -> word boundaries so every segment
  fits the text window with its own tags; `synthesize()` renders each segment and concatenates
  with a small inter-segment gap. The 250-token sentence now splits into 3 segments, full
  17.6s audio, no truncation.
- **Slowness on short clips**: the diffusion loop always ran the full 192 frames, so a 1s reply
  paid the same ~600ms. Fix: **frame bucketing** — `FRAME_BUCKETS=(96,192)`; only
  vector_estimator/vocoder depend on frame count, so each is compiled per bucket (cached as
  `*_ctx_f{frames}.onnx`) and each segment routes to the smallest bucket that fits its predicted
  length. Short reply (1.8s audio) dropped from ~600ms to **325ms (RTF 0.18, ~5.5x realtime)**;
  medium 4.5s audio RTF 0.074.

Cost: cold compile now ~26s (6 graphs: 2 text + 2 stages x 2 buckets), warm load ~0.9s (cached).
120 tests pass.

### Milestone 30: Async STT event bus -> dynamic UI reaction

New `runtime/events.py`: `RuntimeEvent` + `EventBus` (non-blocking `publish()`, background
daemon dispatch thread, listener-error isolation, `NULL_BUS` default; publishes inline if not
started so events are never dropped). Mirrors the load-event API but for repeated runtime events.

- STT base gains `set_event_bus()` (no-op default). `QaiWhisperTranscriber.transcribe` publishes
  `stt/start` (before) and `stt/final` (after, with text + provider + latency), `stt/error` on failure.
- `attach_speech_reactions(bus, ui)` subscribes a listener that maps STT events onto the voice-UI
  protocol (`status`/`activity`/`user_text`) — so the activity waveform + status line react
  *off the STT thread*: jump to activity 0.65 + "transcribing" on start, settle to 0.0 + show the
  transcript on final. Uses the duck-typed UI protocol, so it needed no edit to the peer's `terminal.py`.
- Wired into the live wake loop (`run`): create bus, `attach_speech_reactions(bus, ui)`,
  `stt_model.set_event_bus(bus)`.

Verified end-to-end: STT (NPU) transcribing the TTS output fired reactions on a *different thread*
than main (+0.1ms transcribing/0.65, +696ms "heard"/transcript/0.0). 5 new unit tests; 125 pass.
Natural next step: feed live mic RMS as `stt/level` events so the waveform tracks real loudness
while speaking (the VAD/capture path is the producer).

### Milestone 31: Supertonic speaking-rate control (fix "drunk"/sluggish pacing)

Default Supertonic pacing sounded slow/slurred. Supertonic controls rate by dividing the
predicted duration by a speed factor (more speed -> fewer frames -> tighter speech). Added a
dedicated, env-tunable knob so this is adjustable without code changes and without affecting
Kokoro:

- `TTSConfig.supertonic_speed`, read from **`WHISPERTOME_SUPERTONIC_SPEED`** (default **1.15**).
- Adapter uses `dur_s = predicted / supertonic_speed` (was the shared `config.speed`).
- A/B at fixed text: 1.0 -> 6.11s audio, 1.15 -> 5.32s (~13% tighter; frame-bucket quantization
  rounds the exact ratio). Kokoro still uses `WHISPERTOME_TTS_SPEED`.

Verified env override (unset->1.15, =1.0 original, =1.3 snappier); 126 tests pass.

### Milestone 32: Desktop TUI fit and centered 800x600 shell

The 800x600 desktop shell looked good but wasted too much right and bottom space. A direct
window screenshot showed the terminal content was rendered as a smaller virtual grid than the
Tk text widget could actually display. The first screenshot attempt grabbed pixels behind the
window, so `PrintWindow` became the reliable visual-check path for the embedded desktop app.

Fix: the host now waits for Tk layout, measures the actual text widget and Cascadia Mono cell
size, advertises that measured grid to the child TUI, and removes extra text-widget padding.
The viewport moved from `95x30` to `99x32` in the same fixed 800x600 window, which tightens the
right and bottom margins without clipping. The host also keeps full-screen TUI frames pinned to
the top so old scrollback cannot hide the current frame.

Verified with a fresh app run, screenshot comparison, focused desktop/TUI tests, and ruff.

### Milestone 33: Conversation compaction and new-chat tools

Added provider-neutral conversation tools so the spoken agent can manage its own chat context:
`conversation_compact` keeps a concise carry-forward summary, and
`conversation_start_new` clears the active chat thread. Both preserve organizer data such as
notes, preferences, reminders, tasks, and checklists.

Win: OpenAI Responses and Cerebras Chat now share the same `LlmResponder` surface:
`reset_conversation()` and `compact_conversation(summary)`. OpenAI drops the next
`previous_response_id`; Cerebras clears local `_history`; both inject compacted summaries into
system instructions. The tricky loss/risk was tool-loop timing: the current tool call still
needs one provider follow-up, so reset/compact triggered from a tool is applied for the next
user turn instead of breaking the in-flight function-call response.

Verified with direct responder tests, tool-triggered reset tests, registry tests, ruff, and the
full suite.

## Milestone 32: App-wide latency + energy pass (modular, critic-reviewed)

Goal: tighter/faster/lower-energy without breaking things, kept modular, with a critiquing
subagent guarding each change. Measured baseline first (data-driven): STT 465ms (26 decode
tokens x 11.1ms), TTS short-reply floor ~321ms; vector_estimator cost is ~linear in frames
(48->179ms, 96->293ms, 192->534ms for the 10-step loop). LLM is network (out of local scope).

Wins (each verified, 136 tests passing):
- **TTS smaller frame bucket**: `FRAME_BUCKETS (96,192) -> (48,96,192)`. Short/tiny replies (the
  common case) route to the 48-frame graph: **321ms -> ~200ms (~38% faster)**; medium unchanged.
  Masking makes bucket size output-irrelevant (critic-verified). Cost: cold compile builds 8
  graphs (2 text + 2 stages x 3 buckets), one-time (cached); warm load unaffected.
- **Event bus idle CPU eliminated**: dispatch loop changed from `queue.get(timeout=0.1)`
  (10 Hz idle polling) to a blocking `get()` with a `_STOP` sentinel. Critic caught a stale-
  sentinel restart bug -> fixed by recreating the queue in `start()`; added a stop->restart
  regression test.
- **TUI render idle/active throttle**: was a constant 24 FPS free-run. Now env-tunable
  `WHISPERTOME_TUI_FPS` (default 5) / `WHISPERTOME_TUI_IDLE_FPS` (default 2), switching via
  `_is_animation_active()` (boot / activity>0.05 / active status). ~5x fewer active redraws,
  ~12x fewer idle. Responsiveness preserved: every state setter calls `_request_render()` so
  changes wake the renderer instantly regardless of FPS. Tightened `clear()` ordering per critic.

Abstractions kept clean: bucketing lives in the TTS adapter; the event bus is the existing
runtime API; FPS is config/env like the other knobs. Two critic subagent passes (changes 1-2,
then 3) found 1 real bug (fixed) + the FPS-ordering nit (tightened); no regressions.

Remaining levers (not done; risk/coordination): STT per-token decode (IO-binding cross caches,
QNN-support-uncertain), even smaller TTS buckets for one-word replies, and the wake-loop running
full STT on every utterance (would need a cheap pre-gate). Render loop lives in the peer's
actively-edited terminal.py — changes may get reworked by them.

## Milestone 33: "Switch to <app>" via list-and-match window tools

User wanted "switch to chrome" to alt-tab to Chrome. From the desktop run log, the model
could already open URLs via powershell Start-Process, but had no way to focus existing windows
and sometimes refused. Chosen design (user's suggestion): let the MODEL do the fuzzy match
against the real window list, rather than alias-matching inside a tool.

- `system/windows.py`: `WindowInfo`, `WindowSwitcher` protocol, `WindowsWindowSwitcher` with
  `list_windows()` (EnumWindows + per-window exe stem via OpenProcess/QueryFullProcessImageNameW,
  excludes the assistant's own window) and `focus_window(hwnd)` (AttachThreadInput workaround for
  the Windows foreground lock: attach to target+foreground threads, SW_RESTORE if minimized,
  SetForegroundWindow/BringWindowToTop/SetActiveWindow, verify via GetForegroundWindow). Pure
  ctypes, no new deps; HANDLE argtypes/restypes set so 64-bit handles are not truncated on ARM64.
- `system/tools.py`: two agent tools `list_windows` (returns [{hwnd, app, title}]) and
  `focus_window(hwnd)`, added to `build_system_control_tools` via a `window_switcher` param.
- System prompt: instructs the model to call list_windows, pick the closest app/title to the
  user's words (loose match: "email"->Outlook, "browser"->Chrome/Edge), then focus_window(hwnd);
  say which apps are open if nothing matches. Distinguished from the browser-open web instructions.

Verified live on this machine: list_windows returned 12 real windows; "switch to chrome" matched
and focus_window brought Chrome to the foreground (ok=True). 4 new unit tests (fake switcher);
140 tests pass.

Also (prior turn) Milestone for web/browser: system prompt now tells the model it CAN open
Google search / Maps / sites in the browser via Start-Process '<url>' and that opening links is
not state-changing (no confirmation) — fixing the logged "I can't browse the web" refusal and the
"need your okay first" stall.

## Milestone 34: Scheduled events & reminders that fire

Closed the journey-M16 gap (reminders persisted but never fired). New `scheduler/` package
(model/clock/executor/recurrence/scheduler/tools) — core is audio/NPU-free and unit-tested.
Decisions (user): in-process scheduler + catch-up; full autonomous execution; structured
recorded tool sequences (deterministic replay, NO LLM at fire time).

- Persistence: `scheduled_events` table on OrganizerStore with an atomic claim CAS
  (`UPDATE ... WHERE status='pending' AND next_fire_at=?`) that makes double-fire impossible;
  crash orphans (`firing` at startup) -> `needs_review`, never replayed.
- Action = ordered steps {speak|tool}; ActionExecutor replays deterministically, never raises,
  per-step tool watchdog, `{stepN.key}` literal templating for dynamic readback.
- SchedulerThread: catch-up + recover on start, wakeable Condition loop (lost-wakeup fixed via
  `_pending_wake`), fires under a shared `turn_lock` so audio never overlaps a live turn.
- Recurrence: daily@HH:MM, weekly@<dow>@HH:MM, every@N@minutes|hours (local-tz for daily/weekly).
- Tools: schedule_event/list_scheduled_events/cancel_scheduled_event with save-time tool-name
  validation (AgentToolRegistry.tool_names()); wired into run_wake_loop (turn_lock + scheduler_speak
  + start/stop) and pipeline creation tools; SchedulerConfig + env; system-prompt guidance.

Verified: 11 scheduler unit tests (double-fire, crash recovery w/o replay, recurring advance +
backlog collapse, catch-up, tool-failure paths, step-timeout bound, templating, validation,
recurrence) + end-to-end with the REAL organizer registry (scheduled workflow ran notes_add and
spoke a confirmation). 151 tests pass. A critic subagent confirmed all 8 invariants hold and
flagged 1 MEDIUM (timed-out tool side effects — mitigated: speak is inline, tools emit no audio)
+ 1 LOW (lost wakeup — fixed).

## Milestone 35: Scheduler eval fixes + "next up" TUI chip

Eval found: a scheduled "open Chrome -> facebook" reminder fired but did nothing. Root cause:
the LLM authored the powershell_run step with allow_mutation=False, so the PowerShell tool
blocked Start-Process (a mutation). Scheduler/executor were correct; the action was self-blocked.
Fix: scheduling IS authorization and fired events run unattended, so `schedule_event` now
pre-authorizes powershell_run steps (`_preauthorize_steps` -> allow_mutation=True). Truly
destructive patterns stay hard-blocked regardless (independent of allow_mutation).

Also (prior request): the TUI header now shows a live "NEXT <title> in Xm" agenda chip
(top-right, before the clock), counting down each frame. Plumbing: TerminalUiState
next_event_title/epoch + set_next_event + TerminalVoiceUi.next_event; SchedulerThread.on_view +
_refresh_view (called on start, after each fire pass, and on wake) reading
OrganizerStore.next_pending_event(); wired in run_wake_loop as on_view -> ui.next_event. 151 pass.

## Milestone 36: Idle + usage ("Codex") auto-compaction

Two triggers feeding one compaction action (ConversationController.auto_compact: summarize via
the model -> compact_conversation(summary), reset fallback):
- Idle: after WHISPERTOME_IDLE_COMPACT_SECONDS (default 3600 = 1 hour) with no turn, compact.
  Background daemon thread; activity resets the timer.
- Usage (Codex strategy): after a turn whose NON-CACHED context tokens cross
  WHISPERTOME_COMPACT_THRESHOLD_PCT (0.75) of WHISPERTOME_CONTEXT_WINDOW_TOKENS (128000),
  compact proactively. Cached prompt tokens excluded ("context % less the context cache").
  To enable this, LlmResponse now carries input/cached/output tokens, captured from the
  Responses API usage (batch, streaming, and tool paths) via _usage_from().

New `llm/compaction.py` ConversationCompactionManager (note_turn + idle thread), `ConversationConfig`
in config, wired into run_wake_loop (note_turn per turn, start/stop, on the shared turn_lock).
CRITICAL fix found during impl: note_turn runs inside a turn that already holds turn_lock, and the
usage trigger re-acquires it -> self-deadlock. Made turn_lock an RLock (reentrant for same-thread,
still mutually exclusive across the scheduler/idle threads); added a regression test. 160 tests pass.
