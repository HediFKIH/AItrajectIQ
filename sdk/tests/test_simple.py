"""
Test 1 — Mode simple : auto-patch OpenAI.
Zéro modification de code, tout est capturé automatiquement.

Prérequis : OPENAI_API_KEY dans l'environnement
Lancer : python tests/test_simple.py
"""

import aitrajectiq
from openai import OpenAI

# 2 lignes pour tout configurer
aitrajectiq.init(project_id="flow-ai-mobile", api_key="test-key-123")
aitrajectiq.patch()

# Code applicatif normal — AUCUNE modification
client = OpenAI()

print("--- Appel 1 : question simple ---")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Quelle est la capitale de la France ?"}],
    max_tokens=50,
)
print(f"Réponse: {resp.choices[0].message.content}\n")

print("--- Appel 2 : question de suivi ---")
resp2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Quelle est la capitale de la France ?"},
        {"role": "assistant", "content": resp.choices[0].message.content},
        {"role": "user", "content": "Et celle de l'Allemagne ?"},
    ],
    max_tokens=50,
)
print(f"Réponse: {resp2.choices[0].message.content}\n")

print("Terminé. Vérifiez le terminal du mock backend pour voir les spans capturés.")
