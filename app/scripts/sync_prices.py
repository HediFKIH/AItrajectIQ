"""
Fetch the current model prices from the web now (LiteLLM and OpenRouter), cache them for the
backend, and re-price the spans billed at the default price whose model now has a price.

    cd app
    python -m scripts.sync_prices                 # fetch, cache and re-price
    python -m scripts.sync_prices --no-reprice    # fetch and cache only (no database needed)
    python -m scripts.sync_prices --model claude-opus-4-8-thinking-high   # show a model's price

A running backend reloads the cache at its next start; it also refreshes on its own every
AITRAJECTIQ_PRICE_SYNC_HOURS hours.
"""

import argparse
import asyncio

from services import pricing
from utils.cost import price_key, price_of


async def main_async(reprice: bool, models: list[str]) -> None:
    if reprice:
        from config.db import init_db
        await init_db()
    status = await pricing.refresh(reprice=reprice)
    if status["source"] != "web":
        pricing.load_cache()
        print(f"Price refresh failed: {status.get('error')}")
    else:
        print(f"{status['models']} model prices from the web ({status['sources']}), updated {status['updated_at']}.")
        if status.get("error"):
            print(f"Warning: {status['error']}")
    for model in models:
        key = price_key(model)
        price = price_of(key) if key else None
        if price is None:
            print(f"{model}: unknown price (billed at the default price)")
        else:
            print(f"{model} -> {key}: in {price['in']} $, out {price['out']} $, "
                  f"cache read {price['cache_read']} $, cache write {price['cache_write']} $ per 1M tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-reprice", action="store_true", help="don't touch the database")
    parser.add_argument("--model", action="append", default=[], help="print the price a model id resolves to")
    args = parser.parse_args()
    asyncio.run(main_async(not args.no_reprice, args.model))


if __name__ == "__main__":
    main()
