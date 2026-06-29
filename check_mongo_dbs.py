from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")

print("=== TOUTES LES BASES MONGODB ===")
for db_name in client.list_database_names():
    db = client[db_name]
    collections = db.list_collection_names()
    print(f"\n  {db_name}")
    for c in collections:
        count = db[c].count_documents({})
        print(f"    {c}: {count} documents")
