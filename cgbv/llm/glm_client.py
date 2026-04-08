from __future__ import annotations

from typing import Any

from cgbv.config.settings import LLMConfig
from cgbv.llm.openai_client import OpenAIClient

_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


class GLMClient(OpenAIClient):
    """Zhipu GLM API client (OpenAI-compatible)."""

    def __init__(self, config: LLMConfig):
        if not config.base_url:
            object.__setattr__(config, "base_url", _DEFAULT_BASE_URL)
        super().__init__(config)

    def _build_request_kwargs(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        params = super()._build_request_kwargs(messages, **kwargs)
        temperature = params.get("temperature")
        extra_body = dict(params.get("extra_body") or {})

        # BigModel's OpenAI-compatible docs note that temperature=0 is not
        # applicable on this endpoint; switch to do_sample=False instead.
        if isinstance(temperature, (int, float)) and temperature <= 0:
            params.pop("temperature", None)
            extra_body.setdefault("do_sample", False)

        if extra_body:
            params["extra_body"] = extra_body
        else:
            params.pop("extra_body", None)
        return params
