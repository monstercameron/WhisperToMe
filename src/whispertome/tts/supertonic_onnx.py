from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from unicodedata import normalize as unicode_normalize

import numpy as np

from whispertome.audio.types import SynthesizedSpeech
from whispertome.config import TTSConfig
from whispertome.errors import ModelNotReadyError
from whispertome.models.loading import NULL_REPORTER, ModelLoadReporter
from whispertome.runtime.base import OnnxSessionHandle
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.tts.base import SpeechSynthesisResult, TextToSpeechModel, VoiceNotSupportedError

# Fixed graph dimensions the Supertonic stages are compiled for. The variable-length
# pieces (real token count, real frame count) are handled by padding + masks, exactly
# as the model was trained, so a static graph is correct on the NPU.
TEXT_MAX = 128
LATENT_DIM = 144  # ttl.latent_dim (24) * chunk_compress_factor (6)

# Frame-length buckets (latent time steps). Each utterance routes to the smallest bucket
# that fits its predicted length, so short replies don't pay the full-length diffusion cost.
# Only vector_estimator/vocoder depend on frame count; text stages are frame-independent.
FRAME_BUCKETS = (96, 192)
FRAME_MAX = max(FRAME_BUCKETS)

# Frame-independent stages (depend only on the text window).
_TEXT_STAGES = {
    "text_encoder": {"text_ids": [1, TEXT_MAX], "style_ttl": [1, 50, 256], "text_mask": [1, 1, TEXT_MAX]},
    "duration_predictor": {"text_ids": [1, TEXT_MAX], "style_dp": [1, 8, 16], "text_mask": [1, 1, TEXT_MAX]},
}
_STAGES = _TEXT_STAGES


def _frame_stage_shapes(frames: int) -> dict:
    return {
        "vector_estimator": {
            "noisy_latent": [1, LATENT_DIM, frames], "text_emb": [1, 256, TEXT_MAX],
            "style_ttl": [1, 50, 256], "latent_mask": [1, 1, frames], "text_mask": [1, 1, TEXT_MAX],
            "current_step": [1], "total_step": [1],
        },
        "vocoder": {"latent": [1, LATENT_DIM, frames]},
    }


class SupertonicOnnxSynthesizer(TextToSpeechModel):
    """Supertonic-2 TTS on the NPU.

    Four ONNX stages (text_encoder, duration_predictor, vector_estimator, vocoder) run
    on the QNN HTP NPU through the shared NPU-only session factory. The source graphs are
    dynamic; this adapter static-fixes their shapes once (cached under artifacts) so QNN
    HTP can claim every node. The variable-length glue (tokenization, length regulation,
    the flow-matching loop, trimming) stays on the CPU between NPU calls.
    """

    def __init__(
        self,
        config: TTSConfig,
        session_factory: NpuOnlyOnnxSessionFactory,
        *,
        project_root: Path,
    ) -> None:
        self._config = config
        self._session_factory = session_factory
        self._project_root = project_root
        self._onnx_dir = config.supertonic_dir / "model" / "onnx"
        self._cache_dir = project_root / "artifacts" / "supertonic_static"
        self._handles: dict[str, OnnxSessionHandle] = {}  # frame-independent text stages
        self._frame_handles: dict[tuple[str, int], OnnxSessionHandle] = {}  # (stage, frames)
        self._unicode_indexer: list[int] | None = None
        self._sample_rate = config.sample_rate
        self._chunk = 0
        self._provider = "?"
        self._voice = config.supertonic_voice
        self._reporter: ModelLoadReporter = NULL_REPORTER

    def set_load_reporter(self, reporter: ModelLoadReporter) -> None:
        self._reporter = reporter

    # ---- setup -------------------------------------------------------------

    def load(self) -> None:
        if self._handles:
            return
        if not self._onnx_dir.exists():
            raise ModelNotReadyError(f"Supertonic model dir not found: {self._onnx_dir}")

        try:
            cfg = json.loads((self._onnx_dir / "tts.json").read_text())
            self._sample_rate = int(cfg["ae"]["sample_rate"])
            ccf = int(cfg["ttl"]["chunk_compress_factor"])
            self._chunk = int(cfg["ae"]["base_chunk_size"]) * ccf
            self._unicode_indexer = json.loads((self._onnx_dir / "unicode_indexer.json").read_text())
        except (OSError, KeyError, ValueError) as exc:
            self._reporter.error(f"config/tokenizer unreadable: {exc}")
            raise ModelNotReadyError(f"Supertonic config/tokenizer unreadable: {exc}") from exc

        # Build list of (stage, frames) units: text stages once + frame stages per bucket.
        units: list[tuple[str, int | None]] = [(s, None) for s in _TEXT_STAGES]
        units += [(s, f) for f in FRAME_BUCKETS for s in _frame_stage_shapes(f)]
        self._reporter.start(
            detail=f"voice={self._voice}, {len(_TEXT_STAGES)} text + "
                   f"{len(FRAME_BUCKETS)} frame buckets {FRAME_BUCKETS}"
        )
        for index, (stage, frames) in enumerate(units, start=1):
            started = perf_counter()
            label = stage if frames is None else f"{stage}@{frames}"
            ctx = self._cache_dir / self._ctx_name(stage, frames)
            cached = ctx.exists() and ctx.stat().st_mtime >= (self._onnx_dir / f"{stage}.onnx").stat().st_mtime
            self._reporter.stage(
                label, detail=("cached" if cached else "compiling HTP graph"),
                progress=(index - 1) / len(units), cached=cached,
            )
            model_path = self._ensure_compiled(stage, frames)
            handle = self._session_factory.create(model_path, label=f"supertonic-{label}")
            if frames is None:
                self._handles[stage] = handle
            else:
                self._frame_handles[(stage, frames)] = handle
            self._reporter.stage(
                label, detail=f"loaded ({'cached' if cached else 'compiled'})",
                progress=index / len(units), cached=cached,
                elapsed_ms=(perf_counter() - started) * 1000.0,
            )
        self._provider = next(iter(self._handles.values())).provider.onnx_name
        self._reporter.loaded(detail=f"voice={self._voice}", provider=self._provider)

    @staticmethod
    def _ctx_name(stage: str, frames: int | None) -> str:
        return f"{stage}_ctx.onnx" if frames is None else f"{stage}_ctx_f{frames}.onnx"

    @staticmethod
    def _static_name(stage: str, frames: int | None) -> str:
        return f"{stage}_static.onnx" if frames is None else f"{stage}_static_f{frames}.onnx"

    @staticmethod
    def _bucket_for(latent_len: int) -> int:
        for frames in FRAME_BUCKETS:
            if latent_len <= frames:
                return frames
        return FRAME_MAX

    def _ensure_compiled(self, stage: str, frames: int | None = None) -> Path:
        """Return a QNN EPContext ONNX for the stage, compiling+caching it on first use.

        The HTP graph compile is the ~5s/stage warmup. Generating the context with this
        machine's ORT-QNN stack (ep.context_enable) makes it both version-matched and
        reusable: later loads deserialize the embedded binary instead of recompiling.
        """
        static = self._ensure_static(stage, frames)
        ctx = self._cache_dir / self._ctx_name(stage, frames)
        if ctx.exists() and ctx.stat().st_mtime >= static.stat().st_mtime:
            return ctx
        if self._compile_ctx(static, ctx) and ctx.exists():
            return ctx
        return static  # caching unavailable -> fall back to on-the-fly compile

    @staticmethod
    def _compile_ctx(static: Path, ctx: Path) -> bool:
        import onnxruntime as ort

        try:
            import onnxruntime_qnn as qnn
        except ImportError:
            return False
        try:
            ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
        except Exception as exc:  # noqa: BLE001 — tolerate already-registered
            if "already" not in str(exc).lower() and "duplicate" not in str(exc).lower():
                return False
        devices = [
            d for d in ort.get_ep_devices()
            if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not devices:
            return False
        if ctx.exists():
            ctx.unlink()
        so = ort.SessionOptions()
        so.log_severity_level = 4
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_file_path", str(ctx))
        so.add_session_config_entry("ep.context_embed_mode", "1")
        so.add_provider_for_devices(devices, {"backend_path": qnn.get_qnn_htp_path()})
        # Session creation compiles the HTP graph and writes the context ONNX to ctx.
        ort.InferenceSession(str(static), sess_options=so)
        return True

    def _ensure_static(self, stage: str, frames: int | None = None) -> Path:
        """Static-fix the source ONNX to the fixed NPU dims, cached on disk."""
        src = self._onnx_dir / f"{stage}.onnx"
        if not src.exists():
            raise ModelNotReadyError(f"Supertonic stage missing: {src}")
        dst = self._cache_dir / self._static_name(stage, frames)
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst

        import onnx
        from onnx import shape_inference

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        model = onnx.load(str(src))
        shapes = _TEXT_STAGES[stage] if frames is None else _frame_stage_shapes(frames)[stage]
        for inp in model.graph.input:
            if inp.name in shapes:
                dims = inp.type.tensor_type.shape.dim
                for axis, value in enumerate(shapes[inp.name]):
                    dims[axis].ClearField("dim_param")
                    dims[axis].dim_value = value
        del model.graph.value_info[:]
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
        onnx.save(model, str(dst))
        return dst

    # ---- text frontend (ported from Supertonic's reference preprocessing) --

    @staticmethod
    def _preprocess(text: str) -> str:
        try:
            text = unicode_normalize("NFKD", text)
        except Exception:  # noqa: BLE001
            pass
        for old, new in {
            "–": "-", "—": "-", "‑": "-", "_": " ",
            "“": '"', "”": '"', "‘": "'", "’": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
        }.items():
            text = text.replace(old, new)
        text = re.sub(r"[̂-̯]", "", text)
        text = text.replace("@", " at ")
        for punct in [",", ".", "!", "?", ";", ":", "'"]:
            text = re.sub(f" \\{punct}", punct, text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and not re.search(r"[.!?;:,'\")\]}]$", text):
            text += "."
        return text

    def _text_to_ids(self, text: str, lang: str) -> tuple[np.ndarray, np.ndarray, int]:
        assert self._unicode_indexer is not None
        tagged = f"<{lang}>{self._preprocess(text)}</{lang}>"
        ids: list[int] = []
        for ch in tagged:
            code = ord(ch)
            if code < len(self._unicode_indexer):
                idx = self._unicode_indexer[code]
                if idx != -1:
                    ids.append(idx)
        ids = ids[:TEXT_MAX]
        real = len(ids)
        padded = np.zeros((1, TEXT_MAX), dtype=np.int64)
        padded[0, :real] = ids
        mask = np.zeros((1, 1, TEXT_MAX), dtype=np.float32)
        mask[0, 0, :real] = 1.0
        return padded, mask, real

    def _voices_dir(self) -> Path:
        return self._onnx_dir.parent / "voice_styles"

    def _load_voice(self) -> tuple[np.ndarray, np.ndarray]:
        path = self._voices_dir() / f"{self._voice}.json"
        if not path.exists():
            raise ModelNotReadyError(f"Supertonic voice style not found: {path}")
        data = json.loads(path.read_text())
        style_ttl = np.array(data["style_ttl"]["data"], dtype=np.float32).reshape(1, 50, 256)
        style_dp = np.array(data["style_dp"]["data"], dtype=np.float32).reshape(1, 8, 16)
        return style_ttl, style_dp

    # ---- runtime voice control --------------------------------------------

    def list_voices(self) -> list[str]:
        directory = self._voices_dir()
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.json"))

    def current_voice(self) -> str | None:
        return self._voice

    def set_voice(self, voice: str) -> None:
        # A voice is just a style-embedding file fed to the same NPU graphs, so switching
        # is a live data swap — no recompile or session reload needed.
        available = self.list_voices()
        if available and voice not in available:
            raise VoiceNotSupportedError(
                f"Unknown Supertonic voice '{voice}'. Available: {', '.join(available)}"
            )
        if not (self._voices_dir() / f"{voice}.json").exists():
            raise VoiceNotSupportedError(f"Voice style file not found for '{voice}'")
        self._voice = voice
        self._reporter.emit("loaded", detail=f"voice switched to {voice}", provider=self._provider)

    # ---- synthesis ---------------------------------------------------------

    def _token_count(self, text: str, lang: str) -> int:
        assert self._unicode_indexer is not None
        tagged = f"<{lang}>{self._preprocess(text)}</{lang}>"
        return sum(
            1 for ch in tagged
            if ord(ch) < len(self._unicode_indexer) and self._unicode_indexer[ord(ch)] != -1
        )

    def _split_text(self, text: str, lang: str, budget: int = 115) -> list[str]:
        """Split into segments that each tokenize within the fixed text window, so no text
        (or the closing language tag) is ever truncated. Prefers sentence, then clause, then
        word boundaries."""
        import re

        if self._token_count(text, lang) <= budget:
            return [text]
        sentences = re.findall(r"[^.!?]+[.!?]+|\S[^.!?]*$", text) or [text]
        segments: list[str] = []
        current = ""
        for sentence in sentences:
            piece = sentence.strip()
            if not piece:
                continue
            if self._token_count(piece, lang) > budget:
                # sentence itself too long -> greedily pack words
                if current:
                    segments.append(current); current = ""
                words, buf = piece.split(), ""
                for word in words:
                    trial = f"{buf} {word}".strip()
                    if buf and self._token_count(trial, lang) > budget:
                        segments.append(buf); buf = word
                    else:
                        buf = trial
                if buf:
                    current = buf
                continue
            trial = f"{current} {piece}".strip()
            if current and self._token_count(trial, lang) > budget:
                segments.append(current); current = piece
            else:
                current = trial
        if current:
            segments.append(current)
        return segments

    def _synth_segment(self, text: str, lang: str, style_ttl, style_dp) -> np.ndarray:
        ids, mask, real = self._text_to_ids(text, lang)
        if real == 0:
            return np.zeros(0, dtype=np.float32)

        text = self._handles["text_encoder"].session
        dp = self._handles["duration_predictor"].session

        duration = dp.run(None, {"text_ids": ids, "style_dp": style_dp, "text_mask": mask})[0]
        dur_s = float(np.asarray(duration).reshape(-1)[0]) / max(self._config.supertonic_speed, 1e-3)
        wav_len = int(dur_s * self._sample_rate)
        needed = (wav_len + self._chunk - 1) // max(self._chunk, 1)
        frames = self._bucket_for(needed)            # smallest static bucket that fits
        latent_len = min(frames, needed)
        ve = self._frame_handles[("vector_estimator", frames)].session
        voc = self._frame_handles[("vocoder", frames)].session

        text_emb = text.run(None, {"text_ids": ids, "style_ttl": style_ttl, "text_mask": mask})[0]

        rng = np.random.default_rng()
        latent = np.zeros((1, LATENT_DIM, frames), dtype=np.float32)
        latent[:, :, :latent_len] = rng.standard_normal((1, LATENT_DIM, latent_len)).astype(np.float32)
        latent_mask = (np.arange(frames) < latent_len).astype(np.float32).reshape(1, 1, -1)
        latent *= latent_mask

        steps = max(1, self._config.diffusion_steps)
        total = np.array([steps], dtype=np.float32)
        for i in range(steps):
            latent = ve.run(None, {
                "noisy_latent": latent, "text_emb": text_emb, "style_ttl": style_ttl,
                "text_mask": mask, "latent_mask": latent_mask,
                "current_step": np.array([i], dtype=np.float32), "total_step": total,
            })[0]
        return np.asarray(voc.run(None, {"latent": latent})[0]).reshape(-1)[:wav_len].astype(np.float32)

    def synthesize(self, text: str) -> SpeechSynthesisResult:
        self.load()
        started = perf_counter()

        style_ttl, style_dp = self._load_voice()
        lang = self._config.language.split("-")[0]  # "en-us" -> "en" for the <lang> tag
        segments = self._split_text(text, lang)

        gap = np.zeros(int(0.06 * self._sample_rate), dtype=np.float32)  # small inter-segment pause
        parts: list[np.ndarray] = []
        for seg in segments:
            wav = self._synth_segment(seg, lang, style_ttl, style_dp)
            if wav.size:
                if parts:
                    parts.append(gap)
                parts.append(wav)
        if not parts:
            raise ModelNotReadyError("Supertonic produced no tokens for the given text")
        wav = np.concatenate(parts)

        latency_ms = (perf_counter() - started) * 1000.0
        return SpeechSynthesisResult(
            speech=SynthesizedSpeech(samples=wav, sample_rate=self._sample_rate),
            latency_ms=latency_ms,
            model=str(self._onnx_dir),
            provider=self._provider,
        )

    @property
    def provider_name(self) -> str | None:
        return self._provider if self._handles else None
