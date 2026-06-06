from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_PROVIDERS = frozenset(
    {
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "ROCMExecutionProvider",
        "TensorrtExecutionProvider",
        "CoreMLExecutionProvider",
    }
)


@dataclass(frozen=True)
class RuntimeProvider:
    key: str
    onnx_name: str
    options: dict[str, Any]
    npu_verified: bool
    proof: str
    ep_devices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class OnnxSessionHandle:
    session: Any
    model_path: Path
    provider: RuntimeProvider
    session_providers: tuple[str, ...]
