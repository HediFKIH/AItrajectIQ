"""
Judge coverage of a project, and evaluation on demand from the dashboard.

Verdicts of an older judge version are left out of the aggregates (insights), so after a
judge change the quality metrics and their recommendations stay empty until the traces are
evaluated again. `status()` says how many ended traces of a period have a verdict of the
current judge and how many are waiting for one; `evaluate_pending()` evaluates a bounded
batch of them in the background (no command line needed).

A trace is waiting when it has no verdict of the current judge and either holds a verdict of
an older one or is an agent run (agent runs always get judged metrics).
"""

from datetime import datetime

from beanie import PydanticObjectId

from evaluation.runner import evaluate_trace, judge_tag
from evaluation.scheduler import bounded, schedule
from models.trace import Trace, TraceStatus

MAX_BATCH = 30

_in_flight: dict[str, str] = {}     # trace id -> project id


def _ended(pid: PydanticObjectId, since: datetime) -> dict:
    return {"project_id": pid, "started_at": {"$gte": since}, "status": {"$ne": TraceStatus.running.value}}


def pending_query(pid: PydanticObjectId, since: datetime, tag: str) -> dict:
    return {
        **_ended(pid, since),
        "metrics.judge": {"$ne": tag},
        "$or": [
            {"metrics": {"$elemMatch": {"judge": {"$exists": True, "$nin": [None, tag]}}}},
            {"total_tool_spans": {"$gt": 0}},
        ],
    }


def _running(pid: PydanticObjectId) -> int:
    return sum(1 for project in _in_flight.values() if project == str(pid))


async def status(pid: PydanticObjectId, since: datetime) -> dict:
    tag = judge_tag()
    collection = Trace.get_motor_collection()
    ended = await collection.count_documents(_ended(pid, since))
    if tag is None:
        return {"judge": None, "ended_traces": ended, "evaluated": 0, "pending": 0, "running": 0}
    return {
        "judge": tag,
        "ended_traces": ended,
        "evaluated": await collection.count_documents({**_ended(pid, since), "metrics.judge": tag}),
        "pending": await collection.count_documents(pending_query(pid, since, tag)),
        "running": _running(pid),
    }


def trace_status(trace) -> dict:
    tag = judge_tag()
    evaluated = tag is not None and any(getattr(m, "judge", None) == tag for m in trace.metrics or [])
    return {
        "judge": tag,
        "evaluated": evaluated,
        "running": str(trace.id) in _in_flight,
        "trace_running": trace.status == TraceStatus.running,
    }


async def _evaluate(trace_id: PydanticObjectId) -> None:
    try:
        await bounded(evaluate_trace(trace_id))
    finally:
        _in_flight.pop(str(trace_id), None)


def _start(trace_id: PydanticObjectId, pid: PydanticObjectId) -> bool:
    if str(trace_id) in _in_flight:
        return False
    _in_flight[str(trace_id)] = str(pid)
    schedule(_evaluate(trace_id))
    return True


async def evaluate_pending(pid: PydanticObjectId, since: datetime, limit: int = MAX_BATCH) -> int:
    """Start the evaluation of up to `limit` waiting traces, latest first. Returns how many
    were started (traces already being evaluated are skipped)."""
    tag = judge_tag()
    if tag is None:
        return 0
    docs = await Trace.get_motor_collection().find(pending_query(pid, since, tag), {"_id": 1}) \
        .sort("started_at", -1).limit(limit + len(_in_flight)).to_list(None)
    started = 0
    for doc in docs:
        if started >= limit:
            break
        started += _start(doc["_id"], pid)
    return started


def evaluate_one(trace) -> bool:
    """Start the evaluation of one ended trace. False when no judge is configured, the trace
    is still running or already being evaluated."""
    if judge_tag() is None or trace.status == TraceStatus.running:
        return False
    return _start(trace.id, trace.project_id)


def judge_missing_message() -> str:
    return ("No evaluation model is configured: set EVAL_LLM_API_KEY (and EVAL_LLM_MODEL) "
            "in .env, then restart the backend.")
