from __future__ import annotations

from cgbv.config.settings import LLMConfig
from cgbv.llm.openai_client import OpenAIClient

_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class GLMClient(OpenAIClient):
    """Zhipu GLM API client (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        if not config.base_url:
            object.__setattr__(config, "base_url", _DEFAULT_BASE_URL)
        super().__init__(config)
