from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    coro: Coroutine,
    sample_id: str = "",
) -> Any:
    """
    Run a coroutine under a semaphore.
    Catches all exceptions and returns them as values (so gather doesn't short-circuit).
    """
    async with semaphore:
        try:
            return await coro
        except Exception as e:
            logger.error("Sample %s failed with unhandled exception: %s", sample_id, e, exc_info=True)
            return e


async def run_concurrent(
    tasks: list[tuple[str, Coroutine]],
    max_concurrency: int,
) -> list[Any]:
    """
    Run a list of (sample_id, coroutine) pairs with bounded concurrency.

    Returns a list of results in the same order as input.
    Exceptions are returned as values (not raised) so the full batch completes.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    wrapped = [
        run_with_semaphore(semaphore, coro, sample_id)
        for sample_id, coro in tasks
    ]
    return await asyncio.gather(*wrapped)
