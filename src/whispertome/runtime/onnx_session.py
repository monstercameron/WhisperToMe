from __future__ import annotations

from pathlib import Path
from typing import Any

from whispertome.config import RuntimeConfig
from whispertome.errors import ModelNotReadyError, RuntimeUnavailableError
from whispertome.runtime.base import FORBIDDEN_PROVIDERS, OnnxSessionHandle, RuntimeProvider
from whispertome.runtime.providers import ProviderResolver


class NpuOnlyOnnxSessionFactory:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._resolver = ProviderResolver(config)

    def available_providers(self) -> tuple[str, ...]:
        ort = self._import_onnxruntime()
        providers = set(ort.get_available_providers())
        if self._qnn_npu_devices(ort):
            providers.add("QNNExecutionProvider")
        return tuple(sorted(providers))

    def create(self, model_path: Path, *, label: str) -> OnnxSessionHandle:
        if not model_path.exists():
            raise ModelNotReadyError(f"{label} model file does not exist: {model_path}")

        ort = self._import_onnxruntime()
        available = self.available_providers()
        provider, _ = self._resolver.resolve(available)

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        session_options.enable_profiling = self._config.enable_onnx_profiling
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        if provider.onnx_name == "QNNExecutionProvider":
            qnn_ep = self._import_onnxruntime_qnn()
            devices = self._qnn_npu_devices(ort)
            if not devices:
                raise RuntimeUnavailableError("QNNExecutionProvider has no NPU EP device")
            provider = RuntimeProvider(
                key=provider.key,
                onnx_name=provider.onnx_name,
                options={"backend_path": qnn_ep.get_qnn_htp_path()},
                npu_verified=provider.npu_verified,
                proof=provider.proof,
                ep_devices=devices,
            )
            session_options.add_provider_for_devices(list(provider.ep_devices), provider.options)
            session = self._create_session(
                ort,
                model_path,
                label=label,
                provider=provider,
                session_options=session_options,
            )
        elif provider.onnx_name == "DmlExecutionProvider":
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers: list[Any] = [(provider.onnx_name, provider.options)]
            session = self._create_session(
                ort,
                model_path,
                label=label,
                provider=provider,
                session_options=session_options,
                providers=providers,
            )
        else:
            providers = [(provider.onnx_name, provider.options)]
            session = self._create_session(
                ort,
                model_path,
                label=label,
                provider=provider,
                session_options=session_options,
                providers=providers,
            )
        session_providers = tuple(session.get_providers())
        self._assert_no_forbidden_fallback(session_providers, provider.onnx_name, label)

        return OnnxSessionHandle(
            session=session,
            model_path=model_path,
            provider=provider,
            session_providers=session_providers,
        )

    @staticmethod
    def _assert_no_forbidden_fallback(
        session_providers: tuple[str, ...],
        selected_provider: str,
        label: str,
    ) -> None:
        if selected_provider not in session_providers:
            raise RuntimeUnavailableError(
                f"{label} session did not use requested provider {selected_provider}. "
                f"Session providers: {session_providers}"
            )

        forbidden = (FORBIDDEN_PROVIDERS - {selected_provider}).intersection(session_providers)
        if forbidden:
            raise RuntimeUnavailableError(
                f"{label} session includes forbidden fallback providers: {sorted(forbidden)}"
        )

    @staticmethod
    def _create_session(
        ort: Any,
        model_path: Path,
        *,
        label: str,
        provider: RuntimeProvider,
        session_options: Any,
        providers: list[Any] | None = None,
    ) -> Any:
        try:
            if providers is None:
                return ort.InferenceSession(str(model_path), sess_options=session_options)
            return ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            message = str(exc)
            detail = NpuOnlyOnnxSessionFactory._summarize_onnxruntime_error(message)
            if "Dynamic shape is not supported yet" in message:
                raise RuntimeUnavailableError(
                    f"{label} could not initialize on {provider.onnx_name}: "
                    "the selected NPU runtime rejected dynamic shapes in this ONNX graph. "
                    f"{detail}"
                ) from exc
            if "assigned to the default CPU EP" in message:
                raise RuntimeUnavailableError(
                    f"{label} cannot run under the NPU-only policy: "
                    f"{provider.onnx_name} did not claim every node in the graph. {detail}"
                ) from exc
            if "Failed to get context binary info" in message:
                raise RuntimeUnavailableError(
                    f"{label} could not initialize on {provider.onnx_name}: "
                    "the QNN context wrapper is not compatible with this ONNX Runtime QNN path. "
                    f"{detail}"
                ) from exc
            raise RuntimeUnavailableError(
                f"{label} could not initialize on {provider.onnx_name}: {detail}"
            ) from exc

    @staticmethod
    def _summarize_onnxruntime_error(message: str, *, limit: int = 500) -> str:
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        if not lines:
            return "ONNX Runtime failed without an error message."

        for line in reversed(lines):
            if "ONNXRuntimeError" in line or "Dynamic shape is not supported" in line:
                return line[:limit]
        return lines[-1][:limit]

    @staticmethod
    def _import_onnxruntime() -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "onnxruntime is not installed. Install the matching NPU runtime extra, "
                "for example `pip install -e .[qnn]` or `pip install -e .[directml]`."
            ) from exc
        return ort

    @staticmethod
    def _import_onnxruntime_qnn() -> Any:
        try:
            import onnxruntime_qnn as qnn_ep
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "onnxruntime-qnn is not installed. Install `pip install -e .[qnn]`."
            ) from exc
        return qnn_ep

    @classmethod
    def _qnn_npu_devices(cls, ort: Any) -> tuple[Any, ...]:
        try:
            qnn_ep = cls._import_onnxruntime_qnn()
            cls._register_qnn_plugin(ort, qnn_ep)
            device_type = ort.OrtHardwareDeviceType.NPU
            return tuple(
                ep_device
                for ep_device in ort.get_ep_devices()
                if ep_device.ep_name == "QNNExecutionProvider"
                and ep_device.device.type == device_type
            )
        except RuntimeUnavailableError:
            return ()

    @staticmethod
    def _register_qnn_plugin(ort: Any, qnn_ep: Any) -> None:
        registration_name = "QNNExecutionProvider"
        try:
            ort.register_execution_provider_library(
                registration_name,
                qnn_ep.get_library_path(),
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already" not in message and "duplicate" not in message:
                raise RuntimeUnavailableError(f"Failed to register QNN plugin EP: {exc}") from exc
