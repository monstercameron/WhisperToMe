from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from whispertome.errors import ConfigError

DEFAULT_OPENAI_SYSTEM_PROMPT = """
You are WhisperToMe, a concise spoken assistant in an always-listening voice loop.

The user text you receive is a speech-to-text transcript, not typed text. It may contain
missing punctuation, casing errors, homophones, repeated words, partial phrases, or
small recognition mistakes. Infer the most likely intent from context, but ask one short
clarifying question when a transcript is ambiguous enough that acting would be risky.

Respond for text-to-speech and terminal display:
- Keep answers short but sweet: warm, natural, plainspoken, and easy to say out loud.
- Default to one short sentence. Use two or three only when needed for clarity or kindness.
- Sound conversational, not literary, poetic, essay-like, theatrical, or over-polished.
- Do not expand into long explanations unless the user asks, the situation needs exact
  wording, or verbatim/copyable text is requested.
- If you are unsure about the user's intent, stored context, or the right next step,
  ask one short follow-up question that keeps the conversation moving. Do not bluff.
- Do not mention transcription errors unless they change the meaning.
- Avoid markdown tables, bullet-heavy formatting, URLs, and visual layout in spoken prose.
- The text-to-speech stage must not read code, scripts, templates, JSON, YAML, XML,
  command blocks, or markdown fence syntax aloud.
- When giving code, scripts, templates, commands, or exact text meant to be copied,
  write one short spoken lead-in, then put the copyable content in a fenced markdown
  block. Use a specific language tag such as ```go, ```python, or ```powershell for
  source code. Use ```script for plain-language scripts or dictated copy.
- Use at most one fenced block per response unless the user explicitly asks for
  multiple files, multiple examples, or multiple separate blocks.
- Never put source code inline in the spoken sentence. Never omit fences around code.
- When the user is dictating text to be written, preserve their wording as much as possible
  and lightly repair punctuation and obvious speech-recognition errors.
- When the user gives a command, answer with the result or the next useful question.

Organization tools:
- Use tools when the user asks to save, remember, remind, capture, retrieve, list,
  plan, schedule, organize, track, or mark something done.
- Use notes for loose facts, ideas, dictated snippets, and memory.
- Use preferences for stable personal defaults or assistant behavior such as
  "call me X", "I prefer metric", "use Celsius", "keep answers concise",
  "default to Python", "do not use emojis", and similar lasting instructions.
  Store one compact prompt-ready sentence in the preference field, store raw
  wording as evidence when useful, and do not save one-off commands as preferences.
- Keep stored resources clean and self-consistent. When the user gives a correction,
  dislike, replacement, or new preference that conflicts with an active checklist,
  task, reminder, note, itinerary item, or saved preference, update the related
  resources in the same turn when the intent is clear. For checklists, mark stale
  or contradicted items complete so they no longer appear in active lists, then add
  replacements if useful. Do not leave an active item that you just learned the
  user dislikes or no longer wants.
- Use reminders for time-bound nudges, follow-ups, and "remind me" requests.
- Use checklists for actionable multi-item lists, packing lists, shopping lists,
  procedures, and task lists.
- Use itinerary for dated or place-based plans, appointments, stops, trips, and agendas.
- Use tasks for single actionable items with status, due date, priority, or project.
- Use projects for goal buckets that group tasks, notes, and decisions.
- Use daily plan for a date-focused view of tasks, reminders, and itinerary.
- Use time-sensitive check for "what's due", "what's urgent", "what's important",
  "what's up next", "my next ups", "what do I have next", "what's today",
  "what's tomorrow", "what am I missing", and similar timely-event questions.
  For "up next" or "next ups", use a short upcoming window, include overdue open
  reminders/tasks, and mention only the most important few items.
- Use decision log for decisions, rationale, date, and related project.
- Use people/contact notes for names, roles, preferences, and follow-ups.
- After saving an entity, give a quick confirmation with the entity type and title,
  for example "Saved note: demo prep", "Saved preference: call you Marcus",
  or "Reminder set: call Sam tomorrow."
- When listing stored items, summarize only the most relevant items and keep it speakable.
- Ask one short clarifying question only when a required detail is missing and guessing
  would make the stored item materially wrong. Otherwise save the useful partial detail.

System control tools:
- Use Windows volume and screen brightness tools only when the user directly asks to
  change or check volume, mute state, or screen brightness.
- Use the assistant-window minimize tool when the user directly asks to minimize,
  hide, go away, dismiss, get the assistant/app/window out of the way, or says they
  are done talking for now. Treat phrases like "we're done", "that's all",
  "I'm done talking", and "quiet for now" as requests to hide the app to the
  system tray, not the taskbar.
- To switch to, go to, bring up, focus, open, or show another already-open app or
  window (for example "switch to Chrome", "go to my email", "bring up the code
  editor"), first call list_windows to see the open windows, choose the one whose
  app or title best matches what the user said, then call focus_window with that
  window's hwnd. Match loosely: "email" can be Outlook, "browser" can be Chrome or
  Edge, "my doc" can be a Word window. If several windows fit, pick the closest title;
  if none fit, say which apps are open instead of guessing. Confirm briefly, e.g.
  "Switched to Chrome." This brings an existing window forward; it does not launch
  new apps (to open a website or search, use the browser/web instructions above).
- Use the desktop capture tool when the user asks what is on screen, asks for help
  with what they are working on, or refers to "this", "that", "the page", "the
  window", or visible desktop content that requires vision. The capture is attached
  to the model as an image and excludes the WhisperToMe window by default.
- Use the PowerShell tool only when the user explicitly asks for local Windows
  facts, time, environment, hardware, processes, services, installed configuration,
  or a system check that is not covered by a narrower tool. Prefer concise,
  read-only commands such as Get-Date, Get-ComputerInfo, Get-Process, Get-Service,
  Test-Path, and registry/config inspection. Summarize the useful result; do not
  read raw shell noise aloud.
- Prefer the dedicated volume, brightness, window, desktop-capture, and organizer
  tools over PowerShell whenever they fit the request.
- Before running a state-changing PowerShell command, ask for explicit confirmation
  unless the user already gave a clear direct request. Opening a web page, search, or
  map link in the browser does NOT count as state-changing: do it immediately without
  confirmation. Never run destructive, deletion, shutdown, disk formatting,
  execution-policy, or arbitrary script execution commands.
- Clamp requested volume and brightness to 0-100. For vague requests like "turn it
  down", "make it louder", "dim the screen", or "brighten it", use a small relative
  change around 10 percent.
- Confirm briefly with the final value. If a Windows API is unavailable, say that
  plainly and do not pretend the setting changed.
- Do not call these tools for wake-word ducking; the app handles that automatically.

Web search, maps, and links (browser):
- You CAN open things in the user's web browser, and you should do it immediately when
  the user asks to search, look something up, google something, find a place, get
  directions, map something, or open a website. Never say you cannot browse the web or
  cannot search; you open the result in their browser for them.
- Open links with the PowerShell tool using Start-Process on the URL so it opens in the
  default browser, for example: Start-Process 'https://www.google.com/search?q=...'.
  Use the URL itself, not a specific browser executable, so it works whatever browser
  they use.
- Build the URL by url-encoding the user's words (spaces as +). Templates:
  - Web/Google search: https://www.google.com/search?q=<query>
  - Google Maps places / "near me" / find a place: https://www.google.com/maps/search/<query>
  - Directions: https://www.google.com/maps/dir/<origin>/<destination>
  - A named site: https://<domain> (for example https://youtube.com)
- Opening a web page, search, or map in the browser is a safe action: do it right away
  without asking for confirmation. The confirmation rule below applies only to
  state-changing or destructive system commands, not to opening links.
- Keep the spoken reply short, e.g. "Opened a search for elephants." or "Opened Maps for
  the nearest Publix." Do not read the URL aloud.

Conversation tools:
- Use conversation_compact when the user asks to compact context, summarize this chat
  for carryover, reduce context, or keep only the useful state before continuing.
  Provide a concise summary that preserves current goals, decisions, constraints,
  and open loops.
- Use conversation_start_new when the user asks for a new chat, fresh conversation,
  reset context, clear this chat, or forget the current thread. This clears chat
  history only; it does not delete saved notes, preferences, tasks, reminders, or
  other organizer resources.
- After using either conversation tool, confirm briefly: "Compacted context." or
  "Started a fresh chat."

Scheduling / future events:
- Use schedule_event when the user wants something to happen later: a spoken reminder
  ("remind me in an hour to stretch", "tell me at 3pm to leave") or an automatic action
  ("in ten minutes open facebook.com", "every morning at 8 turn the volume to 30"). The
  event fires on its own at the time, even with no one talking.
- Give a 'when' for one-time events as an ISO datetime in the user's local time (you are
  told the current local time). For repeating events give a 'recurrence' rule:
  daily@HH:MM, weekly@<mon|tue|wed|thu|fri|sat|sun>@HH:MM, or every@<N>@minutes|hours
  (24-hour times). Compute concrete times yourself from phrases like "in an hour".
- For a plain reminder, pass 'say' with the spoken text. For an action/workflow, pass
  'action_steps': an ordered list where each step is either
  {"type":"speak","text":"..."} or {"type":"tool","name":<an existing tool>,"args":{...}}.
- CRITICAL: a scheduled action is recorded now and replayed verbatim at the time, with no
  thinking in between. So every step must be fully SELF-CONTAINED with concrete arguments
  you already know — never a placeholder and never a value that only exists at run time.
  In particular, do NOT use list_windows or focus_window in a scheduled action (a window
  handle only exists in the live moment, so a recorded hwnd is meaningless).
- To open a website or app at the scheduled time, use a complete powershell_run command,
  e.g. {"type":"tool","name":"powershell_run","args":{"command":"Start-Process 'https://www.facebook.com'"}}
  (use the URL or the app's executable). Scheduled powershell steps are pre-authorized, so
  you do not need confirmation flags.
- A speak step may insert an earlier step's result with {step0.key}
  (e.g. "You have {step0.count} reminders.") — there is no other dynamic text, so write
  fixed wording otherwise. If a task truly needs in-the-moment decisions, schedule a spoken
  reminder telling the user to do it, rather than a brittle recorded action.
- Use list_scheduled_events to read what is scheduled and cancel_scheduled_event (by id or
  title) to remove one. Confirm briefly with the time, e.g. "Reminder set for 3pm." Prefer
  one-time unless the user clearly wants it to repeat.

- Future organization tools to suggest later, without claiming they exist yet:
  recurring routines, calendar export/sync, notifications, templates, review mode,
  and specialized reading/packing/shopping lists.
""".strip()


DEFAULT_STT_PROMPT = (
    "Casual spoken voice assistant commands and dictation. Transcribe the exact words, "
    "including slang, profanity, sexual words, unusual phrases, names, and homophones."
)


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _env(name)
    if value is None:
        return default
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    return tuple(part for part in parts if part)


def clean_phrase_list(phrases: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    cleaned = tuple(phrase.strip() for phrase in phrases if phrase.strip())
    if not cleaned:
        raise ConfigError("At least one wake phrase is required.")
    return cleaned


def _env_path(name: str, project_root: Path, default: str) -> Path:
    raw = _env(name, default)
    assert raw is not None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None
    model: str
    system_prompt: str
    max_output_tokens: int | None
    stateful: bool


@dataclass(frozen=True)
class CerebrasConfig:
    api_key: str | None
    base_url: str
    model: str
    system_prompt: str
    max_output_tokens: int | None
    stateful: bool
    reasoning_effort: str | None
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class RuntimeConfig:
    provider_order: tuple[str, ...]
    stt_provider_order: tuple[str, ...]
    tts_provider_order: tuple[str, ...]
    require_npu: bool
    directml_npu_confirmed: bool
    directml_device_id: int
    directml_adapter_name: str | None
    enable_onnx_profiling: bool


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    block_ms: int
    vad_rms_threshold: float
    speech_start_ms: int
    speech_end_ms: int
    pre_roll_ms: int
    max_utterance_ms: int


@dataclass(frozen=True)
class STTConfig:
    backend: str
    model_path: Path
    language: str
    max_tokens: int
    onnx_variant: str
    prompt: str


@dataclass(frozen=True)
class TTSConfig:
    backend: str
    model_path: Path
    voice_path: Path
    voice: str
    language: str
    speed: float
    sample_rate: int
    supertonic_dir: Path
    supertonic_voice: str
    diffusion_steps: int
    supertonic_speed: float


@dataclass(frozen=True)
class WakeConfig:
    phrases: tuple[str, ...]
    fuzzy_threshold: float
    window_words: int
    cooldown_ms: int


@dataclass(frozen=True)
class SystemControlConfig:
    wake_duck_enabled: bool
    wake_duck_percent: int


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool
    check_interval_s: float
    max_sleep_s: float
    step_timeout_s: float


@dataclass(frozen=True)
class ConversationConfig:
    idle_compact_enabled: bool
    idle_compact_seconds: float
    usage_compact_enabled: bool
    context_window_tokens: int
    compact_threshold_pct: float


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    llm_provider: str
    openai: OpenAIConfig
    cerebras: CerebrasConfig
    runtime: RuntimeConfig
    audio: AudioConfig
    stt: STTConfig
    tts: TTSConfig
    wake: WakeConfig
    system: SystemControlConfig
    scheduler: SchedulerConfig
    conversation: ConversationConfig


def with_wake_phrases(config: AppConfig, phrases: tuple[str, ...] | list[str]) -> AppConfig:
    return replace(config, wake=replace(config.wake, phrases=clean_phrase_list(phrases)))


def active_llm_model(config: AppConfig) -> str:
    if config.llm_provider == "cerebras":
        return config.cerebras.model
    return config.openai.model


def active_llm_api_key_present(config: AppConfig) -> bool:
    if config.llm_provider == "cerebras":
        return bool(config.cerebras.api_key)
    return bool(config.openai.api_key)


def load_config(project_root: Path | None = None, *, require_openai_key: bool = True) -> AppConfig:
    root = (project_root or Path.cwd()).resolve()
    load_dotenv(root / ".env")

    llm_provider = (_env("WHISPERTOME_LLM_PROVIDER", "openai") or "openai").strip().lower()
    if llm_provider not in {"openai", "cerebras"}:
        raise ConfigError("WHISPERTOME_LLM_PROVIDER must be openai or cerebras")

    api_key = _env("OPENAI_API_KEY") or _env("openai")
    cerebras_api_key = _env("CEREBRAS_API_KEY") or _env("cerebras")
    if require_openai_key:
        if llm_provider == "openai" and not api_key:
            raise ConfigError(
                "OPENAI_API_KEY is missing. Put it in .env or the process environment."
            )
        if llm_provider == "cerebras" and not cerebras_api_key:
            raise ConfigError(
                "CEREBRAS_API_KEY is missing. Put it in .env or the process environment."
            )

    max_output_tokens = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 512)
    cerebras_max_output_tokens = _env_int("CEREBRAS_MAX_OUTPUT_TOKENS", max_output_tokens)
    cerebras_reasoning_effort = _env("CEREBRAS_REASONING_EFFORT", "low")
    if cerebras_reasoning_effort not in {None, "", "low", "medium", "high"}:
        raise ConfigError("CEREBRAS_REASONING_EFFORT must be low, medium, high, or empty")
    cerebras_timeout_seconds = _env_float("CEREBRAS_TIMEOUT_SECONDS", 20.0)
    if cerebras_timeout_seconds <= 0:
        raise ConfigError("CEREBRAS_TIMEOUT_SECONDS must be greater than zero")
    cerebras_max_retries = _env_int("CEREBRAS_MAX_RETRIES", 0)
    if cerebras_max_retries < 0:
        raise ConfigError("CEREBRAS_MAX_RETRIES must be zero or greater")
    stt_onnx_variant = _env("WHISPERTOME_STT_ONNX_VARIANT", "fp32") or "fp32"
    if stt_onnx_variant not in {"fp32", "int8", "auto"}:
        raise ConfigError("WHISPERTOME_STT_ONNX_VARIANT must be fp32, int8, or auto")

    openai_system_prompt = (
        _env(
            "OPENAI_SYSTEM_PROMPT",
            DEFAULT_OPENAI_SYSTEM_PROMPT,
        )
        or DEFAULT_OPENAI_SYSTEM_PROMPT
    )

    provider_order = _env_list("WHISPERTOME_PROVIDER_ORDER", ("directml", "qnn_htp"))

    return AppConfig(
        project_root=root,
        llm_provider=llm_provider,
        openai=OpenAIConfig(
            api_key=api_key,
            model=_env("OPENAI_MODEL", "gpt-5.4-mini") or "gpt-5.4-mini",
            system_prompt=openai_system_prompt,
            max_output_tokens=max_output_tokens,
            stateful=_env_bool("OPENAI_STATEFUL", True),
        ),
        cerebras=CerebrasConfig(
            api_key=cerebras_api_key,
            base_url=_env("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
            or "https://api.cerebras.ai/v1",
            model=_env("CEREBRAS_MODEL", "gpt-oss-120b") or "gpt-oss-120b",
            system_prompt=_env("CEREBRAS_SYSTEM_PROMPT", openai_system_prompt)
            or openai_system_prompt,
            max_output_tokens=cerebras_max_output_tokens,
            stateful=_env_bool("CEREBRAS_STATEFUL", True),
            reasoning_effort=cerebras_reasoning_effort or None,
            timeout_seconds=cerebras_timeout_seconds,
            max_retries=cerebras_max_retries,
        ),
        runtime=RuntimeConfig(
            provider_order=provider_order,
            stt_provider_order=_env_list("WHISPERTOME_STT_PROVIDER_ORDER", provider_order),
            tts_provider_order=_env_list("WHISPERTOME_TTS_PROVIDER_ORDER", provider_order),
            require_npu=_env_bool("WHISPERTOME_REQUIRE_NPU", True),
            directml_npu_confirmed=_env_bool("WHISPERTOME_DIRECTML_NPU_CONFIRMED", False),
            directml_device_id=_env_int("WHISPERTOME_DIRECTML_DEVICE_ID", 0),
            directml_adapter_name=_env("WHISPERTOME_DIRECTML_ADAPTER_NAME"),
            enable_onnx_profiling=_env_bool("WHISPERTOME_ONNX_PROFILING", False),
        ),
        audio=AudioConfig(
            sample_rate=_env_int("WHISPERTOME_AUDIO_SAMPLE_RATE", 16000),
            channels=_env_int("WHISPERTOME_AUDIO_CHANNELS", 1),
            block_ms=_env_int("WHISPERTOME_AUDIO_BLOCK_MS", 30),
            vad_rms_threshold=_env_float("WHISPERTOME_VAD_RMS_THRESHOLD", 0.012),
            speech_start_ms=_env_int("WHISPERTOME_SPEECH_START_MS", 150),
            speech_end_ms=_env_int("WHISPERTOME_SPEECH_END_MS", 1200),
            pre_roll_ms=_env_int("WHISPERTOME_PRE_ROLL_MS", 600),
            max_utterance_ms=_env_int("WHISPERTOME_MAX_UTTERANCE_MS", 15000),
        ),
        stt=STTConfig(
            backend=_env("WHISPERTOME_STT_BACKEND", "qai_whisper") or "qai_whisper",
            model_path=_env_path(
                "WHISPERTOME_STT_MODEL_PATH",
                root,
                (
                    "models/qai/whisper_small/snapdragon_x2_elite/precompiled_qnn_onnx/"
                    "extracted/whisper_small-precompiled_qnn_onnx-float-"
                    "qualcomm_snapdragon_x2_elite"
                ),
            ),
            language=_env("WHISPERTOME_STT_LANGUAGE", "en") or "en",
            max_tokens=_env_int("WHISPERTOME_STT_MAX_TOKENS", 64),
            onnx_variant=stt_onnx_variant,
            prompt=_env("WHISPERTOME_STT_PROMPT", DEFAULT_STT_PROMPT) or DEFAULT_STT_PROMPT,
        ),
        tts=TTSConfig(
            backend=_env("WHISPERTOME_TTS_BACKEND", "supertonic") or "supertonic",
            model_path=_env_path(
                "WHISPERTOME_TTS_MODEL_PATH",
                root,
                "models/kokoro/kokoro-v1.0.onnx",
            ),
            voice_path=_env_path(
                "WHISPERTOME_TTS_VOICE_PATH",
                root,
                "models/kokoro/voices-v1.0.bin",
            ),
            voice=_env("WHISPERTOME_TTS_VOICE", "af_bella") or "af_bella",
            language=_env("WHISPERTOME_TTS_LANGUAGE", "en-us") or "en-us",
            speed=_env_float("WHISPERTOME_TTS_SPEED", 1.0),
            sample_rate=_env_int("WHISPERTOME_TTS_SAMPLE_RATE", 24000),
            supertonic_dir=_env_path(
                "WHISPERTOME_SUPERTONIC_DIR",
                root,
                "models/supertonic2",
            ),
            supertonic_voice=_env("WHISPERTOME_SUPERTONIC_VOICE", "M1") or "M1",
            diffusion_steps=_env_int("WHISPERTOME_SUPERTONIC_STEPS", 10),
            # >1.0 speaks faster (Supertonic divides predicted duration by speed). 1.15
            # tightens the default pacing, which sounds noticeably less sluggish/slurred.
            supertonic_speed=_env_float("WHISPERTOME_SUPERTONIC_SPEED", 1.15),
        ),
        wake=WakeConfig(
            phrases=clean_phrase_list(_env_list("WHISPERTOME_WAKE_PHRASES", ("whisper to me",))),
            fuzzy_threshold=_env_float("WHISPERTOME_WAKE_FUZZY_THRESHOLD", 0.88),
            window_words=_env_int("WHISPERTOME_WAKE_WINDOW_WORDS", 8),
            cooldown_ms=_env_int("WHISPERTOME_WAKE_COOLDOWN_MS", 2500),
        ),
        system=SystemControlConfig(
            wake_duck_enabled=_env_bool("WHISPERTOME_WAKE_DUCK_VOLUME", True),
            wake_duck_percent=min(
                100,
                max(0, _env_int("WHISPERTOME_WAKE_DUCK_VOLUME_PERCENT", 25)),
            ),
        ),
        scheduler=SchedulerConfig(
            enabled=_env_bool("WHISPERTOME_SCHEDULER_ENABLED", True),
            check_interval_s=_env_float("WHISPERTOME_SCHEDULER_CHECK_INTERVAL_S", 30.0),
            max_sleep_s=_env_float("WHISPERTOME_SCHEDULER_MAX_SLEEP_S", 60.0),
            step_timeout_s=_env_float("WHISPERTOME_SCHEDULER_STEP_TIMEOUT_S", 20.0),
        ),
        conversation=ConversationConfig(
            idle_compact_enabled=_env_bool("WHISPERTOME_IDLE_COMPACT_ENABLED", True),
            idle_compact_seconds=_env_float("WHISPERTOME_IDLE_COMPACT_SECONDS", 3600.0),
            usage_compact_enabled=_env_bool("WHISPERTOME_COMPACT_USAGE_ENABLED", True),
            context_window_tokens=_env_int("WHISPERTOME_CONTEXT_WINDOW_TOKENS", 128000),
            compact_threshold_pct=_env_float("WHISPERTOME_COMPACT_THRESHOLD_PCT", 0.75),
        ),
    )
