"""
Test — Mode zéro-code : pas de trace, pas de tool_span.
Le SDK crée automatiquement une trace.
Pas besoin de clé OpenAI — on simule les appels.

Lancer : python tests/test_no_trace.py
"""

import llminsight
from unittest.mock import MagicMock
import time

# ── Init + patch ──
llminsight.init(project_id="mon-chatbot", api_key="test-key-123", base_url="http://localhost:8090")
llminsight.patch()

# ── Simuler un client OpenAI (pas de clé nécessaire) ──
from openai import OpenAI

client = OpenAI(api_key="fake-key-for-testing", base_url="http://localhost:1/v1")

# On remplace la méthode create par un mock qui renvoie une fausse réponse
def fake_openai_create(self=None, **kwargs):
    time.sleep(0.2)

    class Usage:
        prompt_tokens = 25
        completion_tokens = 15
        total_tokens = 40

    class Message:
        content = f"[Mock response for: {kwargs.get('messages', [{}])[-1].get('content', '?')[:50]}]"
        role = "assistant"

    class Choice:
        message = Message()
        index = 0
        finish_reason = "stop"

    class Response:
        choices = [Choice()]
        usage = Usage()
        model = kwargs.get("model", "gpt-4o-mini")
        id = "mock-123"

    return Response()


# On injecte le mock APRÈS le patch (il wrappera notre mock aussi)
import openai
original_patched = openai.resources.chat.completions.Completions.create
def combined_create(self, *args, **kwargs):
    # Le wrapper LLMInsight appellera cette fonction, qui retourne un mock
    return fake_openai_create(**kwargs)

# Remplace la fonction originale dans le wrapper
openai.resources.chat.completions.Completions.create.__wrapped__ = combined_create

# Plus simple : on unpatch, on met le mock, et on re-patch
llminsight.unpatch()
openai.resources.chat.completions.Completions.create = fake_openai_create
llminsight.patch()

# ── Code applicatif normal — AUCUN llminsight.trace(), AUCUN tool_span() ──

print("Appel 1...")
r1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Résume le livre '1984' en 2 phrases."}],
    max_tokens=100,
)
print(f"  {r1.choices[0].message.content}\n")

print("Appel 2...")
r2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Qui est l'auteur de '1984' ?"}],
    max_tokens=50,
)
print(f"  {r2.choices[0].message.content}\n")

print("Appel 3...")
r3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Donne 3 livres similaires à '1984'."}],
    max_tokens=100,
)
print(f"  {r3.choices[0].message.content}\n")

# Laisser le temps aux threads daemon d'envoyer les spans
time.sleep(1)

print("Terminé. Vérifiez le mock backend :")
print("  -> 1 trace AUTO créée automatiquement")
print("  -> 3 spans LLM rattachés à cette trace")
print("  -> Aucune ligne de code LLMInsight ajoutée au-delà de init+patch")
