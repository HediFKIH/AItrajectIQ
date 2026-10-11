"""
In-process publish/subscribe of project events, consumed by the dashboard stream
(GET /api/v1/stream).

Event types:
    trace.started    a trace was created            {trace}
    trace.completed  a trace was closed             {trace_id, status}
    span.created     spans were stored in a trace   {trace_id, count, span?}
    metrics.updated  an evaluation finished         {scope: span|trace, trace_id, span_id?}

Publishing never blocks or fails the caller: a subscriber whose queue is full loses
events and is told to resync. Subscribers only see events of the worker process that
published them, so the stream is exact with a single uvicorn worker.
"""

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from typing import Any, Iterator, Optional

log = logging.getLogger("aitrajectiq.realtime")

DEFAULT_QUEUE_SIZE = 256


class Subscription:
    def __init__(self, project_id: str, queue_size: int):
        self.project_id = project_id
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.dropped = 0

    def offer(self, event: dict) -> bool:
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    async def next(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Next event, or None after `timeout` seconds without one."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def take_dropped(self) -> int:
        dropped, self.dropped = self.dropped, 0
        return dropped


class EventBus:
    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE):
        self._queue_size = queue_size
        self._subscribers: dict = defaultdict(set)

    def publish(self, project_id: Any, event_type: str, data: Optional[dict] = None) -> int:
        """Deliver an event to every subscriber of the project. Returns the delivery count."""
        subscribers = self._subscribers.get(str(project_id))
        if not subscribers:
            return 0
        event = {"type": event_type, "data": data or {}, "ts": time.time()}
        return sum(1 for sub in list(subscribers) if sub.offer(event))

    @contextlib.contextmanager
    def subscribe(self, project_id: Any) -> Iterator[Subscription]:
        key = str(project_id)
        sub = Subscription(key, self._queue_size)
        self._subscribers[key].add(sub)
        try:
            yield sub
        finally:
            subscribers = self._subscribers.get(key)
            if subscribers is not None:
                subscribers.discard(sub)
                if not subscribers:
                    del self._subscribers[key]

    def subscriber_count(self, project_id: Any = None) -> int:
        if project_id is not None:
            return len(self._subscribers.get(str(project_id), ()))
        return sum(len(subs) for subs in self._subscribers.values())


bus = EventBus()


def publish(project_id: Any, event_type: str, data: Optional[dict] = None) -> int:
    """Publish on the process-wide bus; never raises into the ingestion path."""
    try:
        return bus.publish(project_id, event_type, data)
    except Exception:
        log.exception("Could not publish %s for project %s", event_type, project_id)
        return 0
