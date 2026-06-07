from __future__ import annotations

import zipfile
from pathlib import Path

from whispertome.config import load_config
from whispertome.models import downloader as dl
from whispertome.models.downloader import build_groups, missing_groups


def _group(cfg, name_part):
    return next(g for g in build_groups(cfg) if name_part in g.name)


def test_missing_then_present(tmp_path):
    cfg = load_config(tmp_path, require_openai_key=False)

    # Fresh project root: both NPU model groups are missing.
    assert {g.name for g in missing_groups(cfg)} == {
        "Whisper-Small (NPU STT)",
        "Supertonic (NPU TTS)",
    }

    # Create the Supertonic required files (short paths) -> that group becomes present,
    # Whisper stays missing. (Avoids creating Whisper's very deep path under pytest tmp.)
    for path in _group(cfg, "Supertonic").required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    names = {g.name for g in missing_groups(cfg)}
    assert "Supertonic (NPU TTS)" not in names
    assert "Whisper-Small (NPU STT)" in names


def test_group_download_specs(tmp_path):
    cfg = load_config(tmp_path, require_openai_key=False)
    whisper = _group(cfg, "Whisper")
    supertonic = _group(cfg, "Supertonic")

    # Whisper is a single public S3 zip extracted next to the model dir.
    assert whisper.zip_url and whisper.zip_url.endswith("x2_elite.zip")
    assert whisper.zip_extract_to == cfg.stt.model_path.parent

    # Supertonic is per-file from the public HF repo, mapped into model/onnx + voice_styles.
    onnx_dir = cfg.tts.supertonic_dir / "model" / "onnx"
    dests = {f.dest for f in supertonic.files}
    assert onnx_dir / "vector_estimator.onnx" in dests
    assert (cfg.tts.supertonic_dir / "model" / "voice_styles" / "M1.json") in dests
    assert all(
        f.url.startswith("https://huggingface.co/Supertone/supertonic/") for f in supertonic.files
    )


def test_download_group_zip_and_files(tmp_path):
    # Exercise the real download + extract path with local file:// URLs (no network).
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.bin").write_bytes(b"x" * 2048)
    zpath = src / "pkg.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("leaf/encoder.onnx", b"E" * 256)
        zf.writestr("leaf/decoder.onnx", b"D" * 256)

    extract = tmp_path / "out" / "extract"
    file_dest = tmp_path / "out" / "hello.bin"
    group = dl.ModelGroup(
        name="syn",
        approx_mb=1,
        required=[extract / "leaf" / "encoder.onnx", file_dest],
        files=[dl._FileDL(Path(src / "hello.bin").as_uri(), file_dest)],
        zip_url=zpath.as_uri(),
        zip_extract_to=extract,
    )
    assert not group.is_present()

    events = []
    dl.download_group(group, lambda label, frac: events.append(frac))

    assert group.is_present()
    assert (extract / "leaf" / "decoder.onnx").exists()
    assert file_dest.read_bytes() == b"x" * 2048
    assert events and events[-1] == 1.0


def test_ensure_models_present_confirm_yes_then_no(tmp_path, monkeypatch):
    src = tmp_path / "s.bin"
    src.write_bytes(b"y" * 1024)
    dest = tmp_path / "o" / "hi.bin"
    syn = dl.ModelGroup(
        name="syn", approx_mb=1, required=[dest],
        files=[dl._FileDL(src.as_uri(), dest)],
    )
    monkeypatch.setattr(dl, "build_groups", lambda config: [syn])

    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    assert dl.ensure_models_present(None, gui=False) is True
    assert dest.exists()

    dest.unlink()
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    assert dl.ensure_models_present(None, gui=False) is False
    assert not dest.exists()
