from __future__ import annotations

import unittest

from whispertome.config import RuntimeConfig
from whispertome.errors import NpuVerificationError, RuntimeUnavailableError
from whispertome.runtime.base import RuntimeProvider
from whispertome.runtime.onnx_session import NpuOnlyOnnxSessionFactory
from whispertome.runtime.providers import ProviderResolver


def runtime_config(**overrides) -> RuntimeConfig:  # type: ignore[no-untyped-def]
    values = {
        "provider_order": ("directml", "qnn_htp"),
        "stt_provider_order": ("directml", "qnn_htp"),
        "tts_provider_order": ("directml", "qnn_htp"),
        "require_npu": True,
        "directml_npu_confirmed": False,
        "directml_device_id": 0,
        "directml_adapter_name": None,
        "enable_onnx_profiling": False,
        "ort_intra_op_threads": 1,
        "ort_inter_op_threads": 1,
        "htp_performance_mode": "burst",
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

    def test_qnn_plugin_session_may_report_cpu_with_fallback_disabled(self) -> None:
        provider = RuntimeProvider(
            key="qnn_htp",
            onnx_name="QNNExecutionProvider",
            options={},
            npu_verified=True,
            proof="test",
            ep_devices=(object(),),
        )

        NpuOnlyOnnxSessionFactory._assert_no_forbidden_fallback(
            ("QNNExecutionProvider", "CPUExecutionProvider"),
            provider,
            "test-model",
        )

    def test_non_plugin_session_rejects_cpu_provider_listing(self) -> None:
        provider = RuntimeProvider(
            key="directml",
            onnx_name="DmlExecutionProvider",
            options={},
            npu_verified=True,
            proof="test",
        )

        with self.assertRaises(RuntimeUnavailableError):
            NpuOnlyOnnxSessionFactory._assert_no_forbidden_fallback(
                ("DmlExecutionProvider", "CPUExecutionProvider"),
                provider,
                "test-model",
            )


if __name__ == "__main__":
    unittest.main()
