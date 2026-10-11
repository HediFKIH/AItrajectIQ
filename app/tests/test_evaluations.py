from datetime import datetime
from types import SimpleNamespace

import pytest
from bson import ObjectId

from models.trace import TraceStatus
from services import evaluations

TAG = "gpt-4.1@v4"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    evaluations._in_flight.clear()
    started = []
    monkeypatch.setattr(evaluations, "schedule", lambda awaitable: started.append(awaitable) or awaitable.close())
    yield started
    evaluations._in_flight.clear()


def trace(status=TraceStatus.completed, judges=()):
    return SimpleNamespace(id=ObjectId(), project_id=ObjectId(), status=status,
                           metrics=[SimpleNamespace(judge=j) for j in judges])


def test_pending_query_targets_stale_verdicts_and_agent_runs():
    pid, since = ObjectId(), datetime(2026, 1, 1)
    query = evaluations.pending_query(pid, since, TAG)
    assert query["project_id"] == pid and query["status"] == {"$ne": "running"}
    assert query["metrics.judge"] == {"$ne": TAG}
    stale, agent = query["$or"]
    assert stale["metrics"]["$elemMatch"]["judge"]["$nin"] == [None, TAG]
    assert agent == {"total_tool_spans": {"$gt": 0}}


def test_trace_status(monkeypatch):
    monkeypatch.setattr(evaluations, "judge_tag", lambda: TAG)
    assert evaluations.trace_status(trace(judges=[None, TAG]))["evaluated"] is True
    assert evaluations.trace_status(trace(judges=[None, "gpt-4.1@v3"]))["evaluated"] is False
    assert evaluations.trace_status(trace(status=TraceStatus.running))["trace_running"] is True

    monkeypatch.setattr(evaluations, "judge_tag", lambda: None)
    status = evaluations.trace_status(trace(judges=[TAG]))
    assert status["judge"] is None and status["evaluated"] is False


def test_evaluate_one_skips_running_traces_duplicates_and_missing_judge(monkeypatch, isolated):
    monkeypatch.setattr(evaluations, "judge_tag", lambda: TAG)
    ended = trace()
    assert evaluations.evaluate_one(ended) is True
    assert evaluations.evaluate_one(ended) is False
    assert evaluations.trace_status(ended)["running"] is True
    assert evaluations.evaluate_one(trace(status=TraceStatus.running)) is False
    assert len(isolated) == 1

    monkeypatch.setattr(evaluations, "judge_tag", lambda: None)
    assert evaluations.evaluate_one(trace()) is False
