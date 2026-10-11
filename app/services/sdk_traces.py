"""
Closing SDK traces: explicitly (POST /ingest/trace/end) or, when the application never
ended them (killed process, long-running server using auto-traces), after
AITRAJECTIQ_SDK_IDLE_S seconds without new spans (default 600), with status `timeout`.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from models.trace import Trace, TraceStatus
from realtime import publish

log = logging.getLogger("aitrajectiq.ingest")

IDLE_SECONDS = float(os.getenv("AITRAJECTIQ_SDK_IDLE_S", "600"))


async def close_trace(trace: Trace, status: TraceStatus, ended_at: Optional[datetime] = None) -> None:
    """Mark the trace ended, notify the dashboard and schedule its evaluation."""
    from evaluation.runner import evaluate_trace
    from evaluation.scheduler import bounded, schedule

    trace.status = status
    trace.ended_at = ended_at or datetime.utcnow()
    trace.duration_ms = max(0, int((trace.ended_at - trace.started_at).total_seconds() * 1000))
    # $set rather than save(): spans still in flight increment the counters concurrently.
    await Trace.get_motor_collection().update_one(
        {"_id": trace.id},
        {"$set": {
            "status": trace.status.value,
            "ended_at": trace.ended_at,
            "duration_ms": trace.duration_ms,
        }},
    )
    publish(trace.project_id, "trace.completed", {"trace_id": str(trace.id), "status": trace.status.value})
    schedule(bounded(evaluate_trace(trace.id)))


async def sweep_idle_traces(idle_seconds: float = IDLE_SECONDS, now: Optional[datetime] = None) -> int:
    """Close running SDK traces that received nothing for `idle_seconds`."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(seconds=idle_seconds)
    stale = await Trace.find({
        "source": {"$ne": "otel"},
        "status": TraceStatus.running.value,
        "$or": [
            {"last_activity_at": {"$lt": cutoff}},
            {"last_activity_at": None, "started_at": {"$lt": cutoff}},
        ],
    }).to_list()
    for trace in stale:
        try:
            await close_trace(trace, TraceStatus.timeout, ended_at=trace.last_activity_at or trace.started_at)
        except Exception:
            log.exception("Could not close idle SDK trace %s", trace.id)
    return len(stale)


async def run_sweeper(idle_seconds: float = IDLE_SECONDS) -> None:
    interval = max(5.0, min(60.0, idle_seconds / 4))
    while True:
        await asyncio.sleep(interval)
        try:
            count = await sweep_idle_traces(idle_seconds)
            if count:
                log.info("Closed %d idle SDK trace(s) as timeout", count)
        except Exception:
            log.exception("SDK trace sweeper failed")
