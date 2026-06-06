class WhisperToMeError(Exception):
    """Base project exception."""


class ConfigError(WhisperToMeError):
    """Raised when configuration is missing or invalid."""


class RuntimeUnavailableError(WhisperToMeError):
    """Raised when no allowed NPU runtime can be selected."""


class NpuVerificationError(RuntimeUnavailableError):
    """Raised when a provider cannot prove NPU execution."""


class ModelNotReadyError(WhisperToMeError):
    """Raised when a model backend lacks required artifacts or adapter code."""


class AudioError(WhisperToMeError):
    """Raised for microphone or speaker failures."""

