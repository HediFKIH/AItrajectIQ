from pymongo import MongoClient
import json

client = MongoClient("mongodb://localhost:27017")
db = client.llm_insight_db2

print("=== Collection 'spans' — champ 'metrics' ===\n")

for s in db.spans.find({"metrics.0": {"$exists": True}}):
    print(f"Span _id: {s['_id']}")
    print(f"  model: {s.get('model')}")
    print(f"  response: {str(s.get('response',''))[:80]}")
    print(f"  metrics:")
    for m in s["metrics"]:
        print(f"    {m['name']}: {m['value']}")
    print()

print("Total spans avec metriques:", db.spans.count_documents({"metrics.0": {"$exists": True}}))
print("Total spans sans metriques:", db.spans.count_documents({"metrics": []}))
