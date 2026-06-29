import requests

BASE = "http://127.0.0.1:8085"
TRACE_ID = "69e8a485ec9f6deb7833fd63"

detail = requests.get(f"{BASE}/traces/{TRACE_ID}").json()

print("=" * 70)
print(f"  TRACE: {TRACE_ID}")
print(f"  Trace metrics: {detail['metrics']}")
print("=" * 70)

for s in detail["spans"]:
    provider = s.get("provider", "")
    model = s.get("model", "")
    preview = s.get("response_preview", "")
    print(f"\nSpan {s['step_index']} [{provider}/{model}]")
    print(f"  Reponse: {preview[:90]}")
    print(f"  Cout:    ${s.get('cost_usd', 0) or 0:.6f}")

    if s["metrics"]:
        for m in s["metrics"]:
            name = m["name"]
            val = m["value"]
            reason = m.get("reason", "")
            r_short = f"  ({reason[:70]})" if reason else ""
            print(f"  {name:35s} = {val}{r_short}")
    else:
        print("  (metriques en attente)")
