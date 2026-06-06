from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from whispertome.config import RuntimeConfig
from whispertome.errors import NpuVerificationError, RuntimeUnavailableError
from whispertome.runtime.base import RuntimeProvider


@dataclass(frozen=True)
class ProviderRejection:
    key: str
    reason: str


class ProviderResolver:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def resolve(
        self,
        available_onnx_providers: Iterable[str],
    ) -> tuple[RuntimeProvider, tuple[ProviderRejection, ...]]:
        available = set(available_onnx_providers)
        rejections: list[ProviderRejection] = []

        for key in self._config.provider_order:
            normalized = key.strip().lower()
            try:
                provider = self._provider_for_key(normalized)
            except RuntimeUnavailableError as exc:
                rejections.append(ProviderRejection(key=normalized, reason=str(exc)))
                continue

            if provider.onnx_name not in available:
                rejections.append(
                    ProviderRejection(
                        key=normalized,
                        reason=f"{provider.onnx_name} is not available in this onnxruntime build",
                    )
                )
                continue

            if self._config.require_npu and not provider.npu_verified:
                rejections.append(
                    ProviderRejection(key=normalized, reason="provider is not verified as NPU")
                )
                continue

            return provider, tuple(rejections)

        reasons = "; ".join(f"{item.key}: {item.reason}" for item in rejections)
        raise NpuVerificationError(f"No verified NPU provider is available. {reasons}")

    def _provider_for_key(self, key: str) -> RuntimeProvider:
        if key == "qnn_htp":
            return RuntimeProvider(
                key="qnn_htp",
                onnx_name="QNNExecutionProvider",
                options={"backend_type": "htp"},
                npu_verified=True,
                proof="QNN HTP backend selected; HTP is the Qualcomm NPU path.",
            )

        if key == "directml":
            if not self._config.directml_npu_confirmed:
                raise NpuVerificationError(
                    "DirectML was requested but WHISPERTOME_DIRECTML_NPU_CONFIRMED is false. "
                    "DirectML can route to GPU, so this project refuses it without explicit "
                    "NPU verification."
                )
            options = {"device_id": self._config.directml_device_id}
            adapter = self._config.directml_adapter_name or "operator-confirmed NPU adapter"
            return RuntimeProvider(
                key="directml",
                onnx_name="DmlExecutionProvider",
                options=options,
                npu_verified=True,
                proof=f"DirectML adapter asserted as NPU-backed: {adapter}",
            )

        raise RuntimeUnavailableError(f"Unknown runtime provider key: {key}")

