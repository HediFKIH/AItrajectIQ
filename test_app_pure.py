"""
App pure -- ZERO reference a llminsight.
Mock au niveau transport HTTP (comme une vraie app).
"""
import time
import json
import random
import httpx


class FakeTransport(httpx.BaseTransport):
    """Simule les reponses d'un LLM sans toucher au SDK OpenAI."""

    def handle_request(self, request):
        body = json.loads(request.content)
        messages = body.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), "")

        time.sleep(random.uniform(0.05, 0.15))

        if "manager" in system.lower():
            answer = "Plan: recherche -> calcul -> redaction"
        elif "recherche" in system.lower():
            answer = "Tesla Model 3: 42k EUR, autonomie 500km"
        elif "calcul" in system.lower():
            answer = "Cout 5 ans: 52k EUR. Economie: 18k EUR"
        elif "redaction" in system.lower():
            answer = "Tesla Model 3 rentable sur 5 ans avec 18k EUR d'economie"
        else:
            answer = "Reponse generique"

        resp_body = {
            "id": "fake-1",
            "object": "chat.completion",
            "model": body.get("model", "gpt-4.1"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": random.randint(200, 800), "completion_tokens": random.randint(50, 200), "total_tokens": 500},
        }
        return httpx.Response(200, json=resp_body)


from openai import OpenAI

client = OpenAI(
    api_key="fake-key",
    http_client=httpx.Client(transport=FakeTransport()),
)


def worker_recherche(question):
    r = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "Worker RECHERCHE"}, {"role": "user", "content": question}],
    )
    return r.choices[0].message.content


def worker_calcul(data):
    r = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "Worker CALCUL"}, {"role": "user", "content": data}],
    )
    return r.choices[0].message.content


def worker_redaction(recherche, calcul):
    r = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "Worker REDACTION"}, {"role": "user", "content": f"{recherche} | {calcul}"}],
    )
    return r.choices[0].message.content


def manager(question):
    plan = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": "MANAGER"}, {"role": "user", "content": question}],
    )
    print(f"  Plan: {plan.choices[0].message.content}")
    rech = worker_recherche(question)
    print(f"  Recherche: {rech}")
    calc = worker_calcul(rech)
    print(f"  Calcul: {calc}")
    redac = worker_redaction(rech, calc)
    print(f"  Redaction: {redac}")
    return redac


print("=== APP PURE (zero llminsight dans le code) ===")
result = manager("Cout Tesla Model 3 sur 5 ans?")
print(f"\n  Resultat: {result}")
print("=== FIN ===")
