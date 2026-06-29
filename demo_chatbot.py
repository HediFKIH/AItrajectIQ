"""
Demo: app client generative utilisant le SDK LLMInsight.
Simule un chatbot qui repond a des questions factuelles.
"""

import time
import requests
import json
from unittest.mock import MagicMock

PROJECT_ID = "69e8aa8744bc8782ddf5a2b8"
API_KEY = "chatbot-key-002"
BASE = "http://127.0.0.1:8085"

# ── 1. Mock le client OpenAI AVANT le patch SDK ──
# On remplace la methode au niveau classe AVANT que le SDK la patche,
# comme ca le wrapper du SDK intercepte notre mock.

def fake_create(self, *args, **kwargs):
    question = kwargs.get("messages", [{}])[-1].get("content", "")
    answers = {
        "Quelle est la capitale de la France ?":
            "La capitale de la France est Paris. C'est la plus grande ville du pays avec environ 2,1 millions d'habitants intra-muros.",
        "Qui a invente le telephone ?":
            "Le telephone a ete invente par Alexander Graham Bell en 1876. Il a brevete son invention le 7 mars 1876.",
        "Quel est le plus grand ocean du monde ?":
            "Le plus grand ocean du monde est l'ocean Pacifique. Il couvre environ 165,25 millions de km2, soit plus d'un tiers de la surface terrestre.",
    }
    answer = answers.get(question, "Je ne suis pas sur de la reponse.")

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = answer
    response.usage = MagicMock()
    response.usage.prompt_tokens = 45
    response.usage.completion_tokens = 60
    response.usage.total_tokens = 105
    response.model = kwargs.get("model", "gpt-4.1")
    return response

import openai.resources.chat.completions
openai.resources.chat.completions.Completions.create = fake_create


# ── 2. Initialiser LLMInsight APRES le mock ──
import llminsight

llminsight.init(
    project_id=PROJECT_ID,
    api_key=API_KEY,
    base_url=BASE,
)
llminsight.patch()  # patche par-dessus le mock → intercepte les appels


# ── 3. Lancer le chatbot ──
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="fake-key",
    azure_endpoint="https://fake.openai.azure.com/",
    api_version="2024-12-01-preview",
)

questions = [
    "Quelle est la capitale de la France ?",
    "Qui a invente le telephone ?",
    "Quel est le plus grand ocean du monde ?",
]

print("=" * 60)
print("  DEMO CHATBOT avec LLMInsight SDK")
print("=" * 60)

with llminsight.trace("session-chat-demo", {"user": "demo-user"}):
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"User: {question}")

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        )

        print(f"Bot:  {response.choices[0].message.content}")
        time.sleep(0.3)

print("\n" + "=" * 60)
print("  3 spans envoyes dans 1 trace")
print("  Metriques en cours d'evaluation...")
print("=" * 60)

time.sleep(5)

# ── 4. Verifier les resultats ──
print("\n>>> RESULTATS <<<\n")

r = requests.get(f"{BASE}/traces/", params={"project_id": PROJECT_ID})
data = r.json()
print(f"Traces trouvees: {data['total']}")

for t in data["traces"]:
    print(f"\n{'='*60}")
    print(f"  Trace:   {t['id']}")
    print(f"  Nom:     {t['name']}")
    print(f"  Status:  {t['status']}")
    print(f"  Spans:   {t['total_spans']}")
    print(f"  Tokens:  {t['total_tokens_in']} in / {t['total_tokens_out']} out")
    print(f"  Cout:    ${t['total_cost_usd']:.4f}")
    print(f"{'='*60}")

    detail = requests.get(f"{BASE}/traces/{t['id']}").json()
    for s in detail["spans"]:
        print(f"\n  Span {s['step_index']} [{s['span_type']}] {s.get('provider','')}/{s.get('model','')}")
        print(f"    Tokens:  {s.get('tokens_in',0)} in / {s.get('tokens_out',0)} out")
        print(f"    Cout:    ${s.get('cost_usd', 0) or 0:.6f}")
        print(f"    Latence: {s.get('latency_ms', 0)}ms")
        preview = s.get('response_preview', '')
        if preview:
            print(f"    Reponse: {preview[:100]}")
        if s.get("metrics"):
            print(f"    --- Metriques ---")
            for m in s["metrics"]:
                val = m['value']
                reason = m.get('reason', '')
                print(f"      {m['name']}: {val}" + (f"  ({reason[:80]})" if reason else ""))
        else:
            print(f"    --- Metriques: en attente ---")
