"""
Background evaluation tasks.

`schedule()` keeps a reference to each task (asyncio only holds weak references, so
an unreferenced task can be garbage-collected mid-flight) and logs failures.
`bounded()` caps how many evaluations call the judge LLM at the same time
(AITRAJECTIQ_EVAL_CONCURRENCY, default 4), so an OTLP batch of hundreds of spans
doesn't hit the provider's rate limits all at once.
"""

import asyncio
import logging
import os
from typing import Any, Awaitable

log = logging.getLogger("aitrajectiq.eval")

_semaphore = asyncio.Semaphore(max(1, int(os.getenv("AITRAJECTIQ_EVAL_CONCURRENCY", "4"))))
_tasks: set = set()


async def bounded(awaitable: Awaitable[Any]) -> Any:
    async with _semaphore:
        return await awaitable


async def _run(awaitable: Awaitable[Any]) -> None:
    try:
        await awaitable
    except Exception:
        log.exception("Background evaluation failed")


def schedule(awaitable: Awaitable[Any]) -> asyncio.Task:
    """Run `awaitable` in the background. Wrap single evaluations in `bounded()`."""
    task = asyncio.create_task(_run(awaitable))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
