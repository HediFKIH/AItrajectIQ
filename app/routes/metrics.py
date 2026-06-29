"""
Metrics aggregation routes — time series for the dashboard.
"""

from fastapi import APIRouter, HTTPException, Query
from beanie import PydanticObjectId
from typing import Optional
from datetime import datetime, timedelta

from models.span import Span, SpanType
from models.trace import Trace
from models.metric import MetricName

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/aggregate/spans")
async def aggregate_span_metrics(
    project_id: str = Query(...),
    metric: str = Query(..., description="Metric name, e.g. general_faithfulness"),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    granularity: str = Query("day", description="day | hour"),
):
    """
    Time-series of a span-level metric averaged per day or hour.
    Used for dashboard trend charts.
    """
    try:
        pid = PydanticObjectId(project_id)
        metric_name = MetricName(metric)
    except Exception:
        raise HTTPException(400, "Invalid project_id or metric name")

    if not to_date:
        to_date = datetime.utcnow()
    if not from_date:
        from_date = to_date - timedelta(days=30)

    date_trunc = {
        "hour": {
            "year": {"$year": "$timestamp"},
            "month": {"$month": "$timestamp"},
            "day": {"$dayOfMonth": "$timestamp"},
            "hour": {"$hour": "$timestamp"},
        },
        "day": {
            "year": {"$year": "$timestamp"},
            "month": {"$month": "$timestamp"},
            "day": {"$dayOfMonth": "$timestamp"},
        },
    }.get(granularity, {
        "year": {"$year": "$timestamp"},
        "month": {"$month": "$timestamp"},
        "day": {"$dayOfMonth": "$timestamp"},
    })

    pipeline = [
        {"$match": {
            "project_id": pid,
            "span_type": "llm",
            "timestamp": {"$gte": from_date, "$lte": to_date},
            "metrics": {"$elemMatch": {"name": metric}},
        }},
        {"$unwind": "$metrics"},
        {"$match": {"metrics.name": metric}},
        {"$group": {
            "_id": date_trunc,
            "avg_value": {"$avg": {"$toDouble": "$metrics.value"}},
            "min_value": {"$min": {"$toDouble": "$metrics.value"}},
            "max_value": {"$max": {"$toDouble": "$metrics.value"}},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]

    results = await Span.aggregate(pipeline).to_list()

    return {
        "metric": metric,
        "project_id": project_id,
        "granularity": granularity,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "data": [
            {
                "period": _format_period(r["_id"], granularity),
                "avg": round(r["avg_value"], 4) if r["avg_value"] is not None else None,
                "min": round(r["min_value"], 4) if r["min_value"] is not None else None,
                "max": round(r["max_value"], 4) if r["max_value"] is not None else None,
                "count": r["count"],
            }
            for r in results
        ],
    }


@router.get("/aggregate/traces")
async def aggregate_trace_metrics(
    project_id: str = Query(...),
    metric: str = Query(..., description="Metric name, e.g. task_completion"),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    granularity: str = Query("day"),
):
    """
    Time-series of a trace-level metric averaged per day or hour.
    """
    try:
        pid = PydanticObjectId(project_id)
        metric_name = MetricName(metric)
    except Exception:
        raise HTTPException(400, "Invalid project_id or metric name")

    if not to_date:
        to_date = datetime.utcnow()
    if not from_date:
        from_date = to_date - timedelta(days=30)

    date_trunc = {
        "year": {"$year": "$started_at"},
        "month": {"$month": "$started_at"},
        "day": {"$dayOfMonth": "$started_at"},
    }
    if granularity == "hour":
        date_trunc["hour"] = {"$hour": "$started_at"}

    pipeline = [
        {"$match": {
            "project_id": pid,
            "started_at": {"$gte": from_date, "$lte": to_date},
            "metrics": {"$elemMatch": {"name": metric}},
        }},
        {"$unwind": "$metrics"},
        {"$match": {"metrics.name": metric}},
        {"$group": {
            "_id": date_trunc,
            "avg_value": {"$avg": {"$toDouble": "$metrics.value"}},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]

    results = await Trace.aggregate(pipeline).to_list()

    return {
        "metric": metric,
        "project_id": project_id,
        "granularity": granularity,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "data": [
            {
                "period": _format_period(r["_id"], granularity),
                "avg": round(r["avg_value"], 4) if r["avg_value"] is not None else None,
                "count": r["count"],
            }
            for r in results
        ],
    }


@router.get("/summary")
async def project_metrics_summary(
    project_id: str = Query(...),
    days: int = Query(7, description="Look-back window in days"),
):
    """
    Quick summary: average of each metric over the last N days.
    """
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, "Invalid project_id")

    since = datetime.utcnow() - timedelta(days=days)

    span_pipeline = [
        {"$match": {"project_id": pid, "timestamp": {"$gte": since}, "metrics.0": {"$exists": True}}},
        {"$unwind": "$metrics"},
        {"$group": {
            "_id": "$metrics.name",
            "avg_value": {"$avg": {"$toDouble": "$metrics.value"}},
            "count": {"$sum": 1},
        }},
    ]

    trace_pipeline = [
        {"$match": {"project_id": pid, "started_at": {"$gte": since}, "metrics.0": {"$exists": True}}},
        {"$unwind": "$metrics"},
        {"$group": {
            "_id": "$metrics.name",
            "avg_value": {"$avg": {"$toDouble": "$metrics.value"}},
            "count": {"$sum": 1},
        }},
    ]

    span_results = await Span.aggregate(span_pipeline).to_list()
    trace_results = await Trace.aggregate(trace_pipeline).to_list()

    cost_pipeline = [
        {"$match": {"project_id": pid, "started_at": {"$gte": since}}},
        {"$group": {
            "_id": None,
            "total_cost": {"$sum": "$total_cost_usd"},
            "total_traces": {"$sum": 1},
            "total_tokens": {"$sum": {"$add": ["$total_tokens_in", "$total_tokens_out"]}},
            "avg_duration_ms": {"$avg": "$duration_ms"},
        }},
    ]
    cost_results = await Trace.aggregate(cost_pipeline).to_list()
    cost_data = cost_results[0] if cost_results else {}

    return {
        "project_id": project_id,
        "period_days": days,
        "span_metrics": {
            r["_id"]: {"avg": round(r["avg_value"], 4), "evaluated_count": r["count"]}
            for r in span_results
        },
        "trace_metrics": {
            r["_id"]: {"avg": round(r["avg_value"], 4), "evaluated_count": r["count"]}
            for r in trace_results
        },
        "cost": {
            "total_usd": round(cost_data.get("total_cost", 0), 4),
            "total_traces": cost_data.get("total_traces", 0),
            "total_tokens": cost_data.get("total_tokens", 0),
            "avg_duration_ms": round(cost_data.get("avg_duration_ms", 0)),
        },
    }


def _format_period(id_doc: dict, granularity: str) -> str:
    y = id_doc.get("year", 2025)
    m = id_doc.get("month", 1)
    d = id_doc.get("day", 1)
    if granularity == "hour":
        h = id_doc.get("hour", 0)
        return f"{y}-{m:02d}-{d:02d}T{h:02d}:00"
    return f"{y}-{m:02d}-{d:02d}"
