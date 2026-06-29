from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client.llm_insight_db2

print("Collections:", db.list_collection_names())
print()
print("Traces count:", db.traces.count_documents({}))
print("Spans count:", db.spans.count_documents({}))
print()

print("--- Tous les spans ---")
for s in db.spans.find():
    metrics = s.get("metrics", [])
    print(f"  {s['_id']}  model={s.get('model')}  metrics_count={len(metrics)}")
    for m in metrics:
        print(f"    {m['name']}: {m['value']}")
