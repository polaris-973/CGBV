from __future__ import annotations

from cgbv.config.settings import LLMConfig
from cgbv.llm.base import LLMClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the appropriate LLM client based on config.provider."""
    provider = config.provider.lower()

    if provider == "openai":
        from cgbv.llm.openai_client import OpenAIClient
        return OpenAIClient(config)
    elif provider == "deepseek":
        from cgbv.llm.deepseek_client import DeepSeekClient
        return DeepSeekClient(config)
    elif provider == "qwen":
        from cgbv.llm.qwen_client import QwenClient
        return QwenClient(config)
    elif provider == "glm":
        from cgbv.llm.glm_client import GLMClient
        return GLMClient(config)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Supported: openai, deepseek, qwen, glm"
        )
