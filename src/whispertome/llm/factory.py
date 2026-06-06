from __future__ import annotations

from collections.abc import Callable

from whispertome.agent.tools import AgentToolRegistry
from whispertome.config import AppConfig
from whispertome.errors import ConfigError
from whispertome.llm.base import LlmResponder
from whispertome.llm.cerebras_chat import CerebrasResponder
from whispertome.llm.openai_responses import OpenAIResponder


def build_llm_responder(
    config: AppConfig,
    *,
    tool_registry: AgentToolRegistry | None = None,
    system_context_provider: Callable[[], str | None] | None = None,
) -> LlmResponder:
    if config.llm_provider == "openai":
        return OpenAIResponder(
            config.openai,
            tool_registry=tool_registry,
            system_context_provider=system_context_provider,
        )
    if config.llm_provider == "cerebras":
        return CerebrasResponder(
            config.cerebras,
            tool_registry=tool_registry,
            system_context_provider=system_context_provider,
        )
    raise ConfigError(f"Unsupported LLM provider: {config.llm_provider}")
