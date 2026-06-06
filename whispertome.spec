# -*- mode: python ; coding: utf-8 -*-
"""One-file PyInstaller build for WhisperToMe (Snapdragon X2 Elite / ARM64).

Bundles the Python deps and native libraries (onnxruntime-qnn QNN HTP DLLs, PortAudio,
libsndfile, Tcl/Tk). Model artifacts under models/ stay external — pass --project-root at
runtime so the exe finds them and .env.
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# Packages that ship data files and/or native libraries PyInstaller must be told about.
_COLLECT = [
    "onnxruntime",       # core ORT + EP plumbing
    "onnxruntime_qnn",   # QNN HTP DLLs + skels (get_library_path resolves inside this pkg)
    "onnx",
    "transformers",      # Whisper tokenizer/feature-extractor + data
    "kokoro_onnx",       # CPU debug voice
    "sounddevice",       # PortAudio
    "soundfile",         # libsndfile
    "comtypes",          # pycaw COM
    "pycaw",
    "pystray",
    "PIL",
]
for _pkg in _COLLECT:
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# The app imports its own submodules lazily inside functions, so collect them all.
hiddenimports += collect_submodules("whispertome")
hiddenimports += ["numpy"]

a = Analysis(
    ["packaging/whispertome_main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "matplotlib", "pytest", "tkinter.test"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="whispertome",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
