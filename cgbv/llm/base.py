from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for all LLM backends."""

    # Set by subclass __init__ via LLMConfig
    api_retry_count: int = 3
    api_retry_delay: float = 2.0

    def _init_retry(self, config: "LLMConfig") -> None:  # noqa: F821
        """Store network-level retry parameters from config."""
        self.api_retry_count = config.api_retry_count
        self.api_retry_delay = config.api_retry_delay

    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request, return the text response."""
        ...

    async def complete_with_retry(self, messages: list[dict], **kwargs) -> str:
        """complete() with exponential-backoff retry on transient errors.

        Retry budget and backoff base are read from self.api_retry_count /
        self.api_retry_delay, which are set from LLMConfig at construction time.
        """
        last_exc: Exception | None = None
        for attempt in range(self.api_retry_count):
            try:
                return await self.complete(messages, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = self.api_retry_delay * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, self.api_retry_count, exc, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"LLM call failed after {self.api_retry_count} attempts"
        ) from last_exc
