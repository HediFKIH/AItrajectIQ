from datetime import datetime, timedelta, timezone

import pytest

from services.dashboard import (
    RANGES,
    attribute_agents,
    bucket_expr,
    bucket_starts,
    change_ratio,
    epoch_ms,
    from_epoch_ms,
    get_range,
    iso_utc,
    percentile,
    span_start,
    top_series,
)

T0 = datetime(2026, 1, 1, 12, 0, 0)


def test_ranges_and_bucket_sizes():
    assert get_range("1h").bucket_ms == 60_000
    assert get_range("24h").bucket_ms == 3_600_000
    assert get_range("30d").days == 30
    assert set(RANGES) == {"1h", "24h", "7d", "30d"}
    with pytest.raises(ValueError):
        get_range("2h")


def test_epoch_round_trip_treats_naive_as_utc():
    ms = epoch_ms(T0)
    assert ms == epoch_ms(T0.replace(tzinfo=timezone.utc))
    assert from_epoch_ms(ms) == T0


def test_iso_utc():
    assert iso_utc(None) is None
    assert iso_utc(T0) == "2026-01-01T12:00:00.000Z"
    aware = datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    assert iso_utc(aware) == "2026-01-01T12:00:00.000Z"


def test_bucket_expr_floors_on_epoch_ms():
    expr = bucket_expr("started_at", 60_000)
    as_long = {"$toLong": "$started_at"}
    assert expr == {"$toDate": {"$subtract": [as_long, {"$mod": [as_long, 60_000]}]}}


def test_bucket_starts_are_aligned_and_inclusive():
    since = T0 + timedelta(seconds=30)
    until = T0 + timedelta(minutes=3, seconds=10)
    starts = bucket_starts(since, until, 60_000)
    assert starts == [T0 + timedelta(minutes=m) for m in range(4)]
    assert all(epoch_ms(s) % 60_000 == 0 for s in starts)


def test_bucket_starts_hourly_count_for_24h():
    rng = get_range("24h")
    until = T0 + timedelta(minutes=5)
    starts = bucket_starts(until - rng.span, until, rng.bucket_ms)
    assert len(starts) == 25
    assert starts[0] == T0 - timedelta(hours=24)


def test_percentile_nearest_rank():
    values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    assert percentile(values, 50) == 500
    assert percentile(values, 95) == 1000
    assert percentile([42], 95) == 42
    assert percentile([3, None, 1], 50) == 1
    assert percentile([], 50) is None


def test_change_ratio():
    assert change_ratio(150, 100) == 0.5
    assert change_ratio(50, 100) == -0.5
    assert change_ratio(10, 0) is None
    assert change_ratio(0, 0) is None


def test_span_start_depends_on_source():
    sdk = {"timestamp": T0, "latency_ms": 250, "source": "sdk"}
    otel = {"timestamp": T0, "latency_ms": 250, "source": "otel"}
    assert span_start(sdk) == T0 - timedelta(milliseconds=250)
    assert span_start(otel) == T0
    assert span_start({"timestamp": T0}) == T0
    assert span_start({"latency_ms": 10}) is None


def test_attribute_agents_prefers_nearest_agent_ancestor():
    agents = [
        {"trace_id": "t", "sdk_span_id": "root", "parent_span_id": None, "agent_name": "orchestrator"},
        {"trace_id": "t", "sdk_span_id": "sub", "parent_span_id": "root", "name": "researcher"},
    ]
    llms = [
        {"_id": 1, "trace_id": "t", "sdk_span_id": "l1", "parent_span_id": "sub"},
        {"_id": 2, "trace_id": "t", "sdk_span_id": "l2", "parent_span_id": "root"},
        {"_id": 3, "trace_id": "t", "sdk_span_id": "l3", "parent_span_id": None, "agent_name": "solo"},
        {"_id": 4, "trace_id": "t", "sdk_span_id": "l4", "parent_span_id": None, "caller_chain": ["main", "plan_step"]},
        {"_id": 5, "trace_id": "t", "sdk_span_id": "l5", "parent_span_id": None},
        # Same span ids in another trace must not match.
        {"_id": 6, "trace_id": "other", "sdk_span_id": "l1", "parent_span_id": "sub"},
    ]
    assert attribute_agents(llms, agents) == {
        1: "researcher",
        2: "orchestrator",
        3: "solo",
        4: "plan_step",
        5: "unattributed",
        6: "unattributed",
    }


def test_attribute_agents_survives_parent_cycles():
    llms = [
        {"_id": 1, "trace_id": "t", "sdk_span_id": "a", "parent_span_id": "b"},
        {"_id": 2, "trace_id": "t", "sdk_span_id": "b", "parent_span_id": "a", "agent_name": "fallback"},
    ]
    assert attribute_agents(llms, []) == {1: "unattributed", 2: "fallback"}


def test_top_series_keeps_largest():
    totals = {"gpt-4o": 5.0, "mini": 0.5, "claude": 2.0}
    assert top_series(totals, limit=2) == ["gpt-4o", "claude"]
    assert top_series({}, limit=3) == []
