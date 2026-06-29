"""Verifie toutes les traces et spans dans MongoDB."""
import requests, json

BASE = "http://127.0.0.1:8085"
PROJECT_ID = "69e8a32d96636a4e1390d273"

print("=" * 70)
print("  TOUTES LES TRACES DU PROJET")
print("=" * 70)

r = requests.get(f"{BASE}/traces/", params={"project_id": PROJECT_ID})
data = r.json()

for t in data["traces"]:
    tid = t["id"]
    print(f"\nTrace {tid}")
    print(f"  name={t['name']}  status={t['status']}  spans={t['total_spans']}  cost=${t['total_cost_usd']:.4f}")

    if t["total_spans"] == 0:
        print("  >>> TRACE VIDE (ancien run rate) — normal qu'il n'y ait pas de metriques")
        continue

    detail = requests.get(f"{BASE}/traces/{tid}").json()

    print(f"  Trace-level metrics: {detail['metrics']}")

    for s in detail["spans"]:
        print(f"\n  Span {s['step_index']} [{s.get('provider','')}/{s.get('model','')}]")
        if s["metrics"]:
            for m in s["metrics"]:
                print(f"    {m['name']:35s} = {m['value']}")
        else:
            print(f"    (pas de metriques)")

print("\n" + "=" * 70)
print("  VERIFICATION DIRECTE DANS MONGODB — collection 'spans'")
print("=" * 70)

from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client.llm_insight_db2

spans_with_metrics = list(db.spans.find({"metrics": {"$ne": []}}, {"_id": 1, "step_index": 1, "model": 1, "metrics": 1}))
print(f"\nSpans avec metriques dans MongoDB: {len(spans_with_metrics)}")

for s in spans_with_metrics:
    print(f"\n  Span {s['_id']} (step {s.get('step_index', '?')}, model={s.get('model', '?')})")
    for m in s.get("metrics", []):
        print(f"    {m['name']:35s} = {m['value']}")

print(f"\n\nTraces dans MongoDB:")
for t in db.traces.find({}, {"_id": 1, "name": 1, "total_spans": 1, "metrics": 1}):
    print(f"  Trace {t['_id']} name={t.get('name','')} spans={t.get('total_spans',0)} metrics={len(t.get('metrics', []))}")
