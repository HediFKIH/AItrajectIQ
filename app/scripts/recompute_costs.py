"""
Recompute the cost of LLM spans stored before the price-lookup fix in utils/cost.py.

The old lookup billed "mini"/"nano" models at the price of their larger sibling
(gpt-4o-mini as gpt-4o, o1-mini as o1, ...) and gave Claude ids written with dashes
(claude-opus-4-5-...) the default price. This script recomputes those costs, then
refreshes each affected trace's total and its cost-based metrics (cost_efficiency,
total_cost_summary). No evaluation LLM is called.

Only SDK spans are recomputed: OpenTelemetry spans may carry a cost reported by
their instrumentation, which must not be overwritten.

    cd app
    python -m scripts.recompute_costs            # dry run: shows what would change
    python -m scripts.recompute_costs --apply    # writes the corrected costs
"""

import argparse
import asyncio
from collections import defaultdict

from pymongo import UpdateOne

from config.db import init_db
from models.span import Span, SpanType
from services.pricing import load_cache, refresh_trace_cost
from utils.cost import compute_cost


async def recompute(apply: bool) -> None:
    await init_db()
    load_cache()

    updates = []
    traces = set()
    per_model: dict = defaultdict(lambda: {"spans": 0, "old": 0.0, "new": 0.0})
    query = {"span_type": SpanType.llm.value, "source": {"$ne": "otel"}}
    async for span in Span.find(query):
        # Same condition as /ingest/span: spans without both token counts have no cost.
        if not (span.model and span.tokens_in and span.tokens_out):
            continue
        new_cost = compute_cost(span.model, span.tokens_in, span.tokens_out)
        if span.cost_usd is not None and abs(new_cost - span.cost_usd) < 1e-9:
            continue
        updates.append(UpdateOne({"_id": span.id}, {"$set": {"cost_usd": new_cost}}))
        traces.add(span.trace_id)
        stats = per_model[span.model]
        stats["spans"] += 1
        stats["old"] += span.cost_usd or 0.0
        stats["new"] += new_cost

    if not updates:
        print("All stored costs already match the current price table.")
        return

    print(f"{'Model':<40} {'Spans':>7} {'Stored cost':>14} {'Corrected cost':>16}")
    for model, stats in sorted(per_model.items(), key=lambda item: -item[1]["old"]):
        print(f"{model:<40} {stats['spans']:>7} {stats['old']:>13.6f}$ {stats['new']:>15.6f}$")
    old_total = sum(s["old"] for s in per_model.values())
    new_total = sum(s["new"] for s in per_model.values())
    print(f"{'Total':<40} {len(updates):>7} {old_total:>13.6f}$ {new_total:>15.6f}$")
    print(f"{len(traces)} trace(s) affected.")

    if not apply:
        print("\nDry run: nothing was written. Re-run with --apply to save these costs.")
        return

    await Span.get_motor_collection().bulk_write(updates, ordered=False)
    for trace_id in traces:
        await refresh_trace_cost(trace_id)
    print(f"\nUpdated {len(updates)} span(s) and {len(traces)} trace(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="write the corrected costs")
    args = parser.parse_args()
    asyncio.run(recompute(args.apply))


if __name__ == "__main__":
    main()
