"""
Metrics aggregation routes — time series for the dashboard.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from beanie import PydanticObjectId
from typing import Optional
from datetime import datetime, timedelta

from models.span import Span, SpanType
from models.trace import Trace
from models.metric import MetricName
from security.api_keys import READ, authenticate
from services.insights import current_verdicts_stage, metrics_summary

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/aggregate/spans")
async def aggregate_span_metrics(
    request: Request,
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
    await authenticate(request, READ, project_id=project_id, allow_legacy=True)

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
        *current_verdicts_stage(),
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
    request: Request,
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
    await authenticate(request, READ, project_id=project_id, allow_legacy=True)

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
        *current_verdicts_stage(),
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
    request: Request,
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
    await authenticate(request, READ, project_id=project_id, allow_legacy=True)

    summary = await metrics_summary(pid, days)
    return {"project_id": project_id, **summary}


def _format_period(id_doc: dict, granularity: str) -> str:
    y = id_doc.get("year", 2025)
    m = id_doc.get("month", 1)
    d = id_doc.get("day", 1)
    if granularity == "hour":
        h = id_doc.get("hour", 0)
        return f"{y}-{m:02d}-{d:02d}T{h:02d}:00"
    return f"{y}-{m:02d}-{d:02d}"
