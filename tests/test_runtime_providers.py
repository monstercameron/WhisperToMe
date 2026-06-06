from __future__ import annotations

import unittest

from whispertome.config import RuntimeConfig
from whispertome.errors import NpuVerificationError
from whispertome.runtime.providers import ProviderResolver


def runtime_config(**overrides) -> RuntimeConfig:  # type: ignore[no-untyped-def]
    values = {
        "provider_order": ("directml", "qnn_htp"),
        "require_npu": True,
        "directml_npu_confirmed": False,
        "directml_device_id": 0,
        "directml_adapter_name": None,
        "enable_onnx_profiling": False,
    }
    values.update(overrides)
    return RuntimeConfig(**values)


class RuntimeProviderTests(unittest.TestCase):
    def test_directml_requires_explicit_npu_confirmation(self) -> None:
        resolver = ProviderResolver(runtime_config(provider_order=("directml",)))

        with self.assertRaises(NpuVerificationError):
            resolver.resolve(("DmlExecutionProvider",))

    def test_qnn_htp_is_accepted_as_npu(self) -> None:
        resolver = ProviderResolver(runtime_config(provider_order=("qnn_htp",)))

        provider, rejections = resolver.resolve(("QNNExecutionProvider",))

        self.assertEqual(provider.onnx_name, "QNNExecutionProvider")
        self.assertTrue(provider.npu_verified)
        self.assertEqual(rejections, ())

    def test_directml_can_be_selected_when_confirmed(self) -> None:
        resolver = ProviderResolver(
            runtime_config(
                provider_order=("directml",),
                directml_npu_confirmed=True,
                directml_adapter_name="Verified NPU",
            )
        )

        provider, _ = resolver.resolve(("DmlExecutionProvider",))

        self.assertEqual(provider.onnx_name, "DmlExecutionProvider")
        self.assertEqual(provider.options["device_id"], 0)


if __name__ == "__main__":
    unittest.main()

