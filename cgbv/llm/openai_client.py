from __future__ import annotations

import asyncio
import json
from inspect import isawaitable

from openai import AsyncOpenAI

from cgbv.llm.base import LLMClient
from cgbv.config.settings import LLMConfig


class OpenAIClient(LLMClient):
    """OpenAI API client (also works for OpenAI-compatible endpoints)."""

    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.extra_body = config.extra_body
        self._request_semaphore = (
            asyncio.Semaphore(config.max_parallel_requests)
            if config.max_parallel_requests and config.max_parallel_requests > 0
            else None
        )
        self._request_gate = asyncio.Lock()
        self._min_request_interval = max(0.0, config.min_request_interval)
        self._next_request_at = 0.0
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = AsyncOpenAI(**kwargs)
        self._init_retry(config)

    @staticmethod
    def _stringify_content(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [OpenAIClient._stringify_content(item) for item in value]
            return "".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                text = OpenAIClient._stringify_content(value.get(key))
                if text:
                    return text
        return ""

    @classmethod
    def _extract_text_from_payload(cls, payload: dict) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = cls._stringify_content(message.get("content"))
                    if content:
                        return content
                    reasoning = cls._stringify_content(message.get("reasoning_content"))
                    if reasoning:
                        return reasoning
                delta = first.get("delta")
                if isinstance(delta, dict):
                    content = cls._stringify_content(delta.get("content"))
                    if content:
                        return content

        output_text = cls._stringify_content(payload.get("output_text"))
        if output_text:
            return output_text

        output = payload.get("output")
        if isinstance(output, dict):
            output_text = cls._stringify_content(output.get("text"))
            if output_text:
                return output_text
            output_text = cls._stringify_content(output.get("content"))
            if output_text:
                return output_text
        elif isinstance(output, list):
            output_text = cls._stringify_content(output)
            if output_text:
                return output_text

        raise ValueError(
            "No assistant text found in chat completion payload. "
            f"Top-level keys: {sorted(payload.keys())}. "
            f"Payload: {json.dumps(payload, ensure_ascii=False)[:2000]}"
        )

    @staticmethod
    async def _read_raw_payload(response: object) -> dict:
        if hasattr(response, "json"):
            payload = response.json()
            if isawaitable(payload):
                payload = await payload
            if isinstance(payload, dict):
                return payload

        raw_text = getattr(response, "text", None)
        if isawaitable(raw_text):
            raw_text = await raw_text
        if isinstance(raw_text, str):
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return payload

        raw_bytes = getattr(response, "content", None)
        if isawaitable(raw_bytes):
            raw_bytes = await raw_bytes
        if isinstance(raw_bytes, (bytes, bytearray)):
            payload = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(payload, dict):
                return payload

        if hasattr(response, "parse"):
            payload = response.parse()
            if isawaitable(payload):
                payload = await payload
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump(mode="json", exclude_none=False)
            if isinstance(payload, dict):
                return payload

        raise TypeError(
            "Unable to decode raw chat completion response into a JSON object. "
            f"Response type: {type(response).__name__}"
        )

    async def _wait_for_request_slot(self) -> None:
        if self._min_request_interval <= 0:
            return
        async with self._request_gate:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next_request_at:
                await asyncio.sleep(self._next_request_at - now)
                now = loop.time()
            self._next_request_at = now + self._min_request_interval

    def _merge_extra_body(
        self,
        base: dict | None,
        override: dict | None,
    ) -> dict | None:
        merged: dict = {}
        if base:
            merged.update(base)
        if override:
            merged.update(override)
        return merged or None

    def _build_request_kwargs(self, messages: list[dict], **kwargs) -> dict:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            params["temperature"] = temperature
        extra_body = self._merge_extra_body(self.extra_body, kwargs.get("extra_body"))
        if extra_body is not None:
            params["extra_body"] = extra_body
        return params

    async def _send_request(self, messages: list[dict], **kwargs) -> str:
        await self._wait_for_request_slot()
        request_kwargs = self._build_request_kwargs(messages, **kwargs)
        response = await self._client.chat.completions.with_raw_response.create(
            **request_kwargs,
        )
        payload = await self._read_raw_payload(response)
        return self._extract_text_from_payload(payload)

    async def complete(self, messages: list[dict], **kwargs) -> str:
        if self._request_semaphore is None:
            return await self._send_request(messages, **kwargs)
        async with self._request_semaphore:
            return await self._send_request(messages, **kwargs)
