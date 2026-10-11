"""
Re-run the trace-level evaluation (task completion, step efficiency...) on recent ended traces,
with the judge configured like the backend (EVAL_LLM_* variables in ../.env).

Use it after changing the judge model or its inputs: stored metrics are only computed
when a trace ends.

    cd app
    python -m scripts.reevaluate_traces                  # ended traces of the last 7 days
    python -m scripts.reevaluate_traces --days 30 --limit 200
"""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv

from config.db import init_db
from evaluation.runner import configure_eval_llm_from_env, evaluate_trace
from models.trace import Trace, TraceStatus


async def reevaluate(days: int, limit: int) -> None:
    await init_db()
    if not configure_eval_llm_from_env():
        raise SystemExit("No judge configured: set EVAL_LLM_API_KEY (or OPENAI_API_KEY) in .env")
    since = datetime.utcnow() - timedelta(days=days)
    traces = await (
        Trace.find(Trace.started_at >= since, Trace.status != TraceStatus.running)
        .sort("-started_at").limit(limit).to_list()
    )
    print(f"Re-evaluating {len(traces)} trace(s)...")
    for trace in traces:
        await evaluate_trace(trace.id)
        updated = await Trace.get(trace.id)
        verdict = next((m for m in updated.metrics if getattr(m.name, "value", m.name) == "task_completion"), None)
        label = "-" if verdict is None else ("OK " if verdict.value else "NON")
        print(f"  {label} {(trace.name or str(trace.id))[:70]:<70} {verdict.reason if verdict else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    load_dotenv(dotenv_path="../.env")
    asyncio.run(reevaluate(args.days, args.limit))


if __name__ == "__main__":
    main()
