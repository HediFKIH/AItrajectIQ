"""
Test — Agent complet : simule Flow AI Mobile.
3 appels LLM (Action, Reflection, Planning) + 4 outils (screenshot, OCR, tap).
Pas besoin de clé OpenAI — tout est simulé.

Lancer : python tests/test_agent.py
"""

import llminsight
import time
import openai

# ── Fake OpenAI ──
call_count = 0
FAKE_RESPONSES = [
    "Action: tap sur le champ Email (coordonnées 234, 456)",
    "Réflexion: L'action a réussi, le curseur est dans le champ email.",
    "Planning: Étapes restantes: 1) saisir email 2) saisir password 3) tap Login",
]

def fake_openai_create(self=None, **kwargs):
    global call_count
    time.sleep(0.3)

    class Usage:
        prompt_tokens = 80 + call_count * 10
        completion_tokens = 40 + call_count * 5
        total_tokens = prompt_tokens + completion_tokens

    class Message:
        content = FAKE_RESPONSES[call_count % len(FAKE_RESPONSES)]
        role = "assistant"

    class Choice:
        message = Message()
        index = 0
        finish_reason = "stop"

    class Response:
        choices = [Choice()]
        usage = Usage()
        model = kwargs.get("model", "gpt-4o")
        id = f"mock-{call_count}"

    call_count += 1
    return Response()


# ── Setup SDK ──
llminsight.init(project_id="flow-ai-mobile", api_key="test-key-123", base_url="http://localhost:8090")
openai.resources.chat.completions.Completions.create = fake_openai_create
llminsight.patch()

client = openai.OpenAI(api_key="fake", base_url="http://localhost:1/v1")


# ── Un run complet d'agent ──
print("=== Flow AI Mobile — Scénario de test ===\n")

with llminsight.trace("mobile-scenario", {
    "instruction": "Se connecter à l'application bancaire",
    "device_id": "pixel-7",
    "platform": "android",
}):
    # Step 1 : capture d'écran
    print("Step 0: Screenshot...")
    with llminsight.tool_span("screenshot", {"device_id": "pixel-7"}) as cap:
        time.sleep(0.2)
        cap.set_output("screenshot_pixel7_1280x720.png")

    # Step 2 : perception OCR
    print("Step 1: Perception OCR + GroundingDINO...")
    with llminsight.tool_span("perception", {"model": "GroundingDINO + EasyOCR"}) as cap:
        time.sleep(0.5)
        cap.set_output("ocr=12 icons=3 fields=2 [button:Login, field:Email, field:Password]")

    # Step 3 : Action Agent (LLM)
    print("Step 2: Action Agent (LLM)...")
    action = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Tu es un agent de navigation mobile."},
            {"role": "user", "content": "Écran: [button:Login, field:Email, field:Password]. Instruction: Se connecter."},
        ],
    )
    print(f"  -> {action.choices[0].message.content}")

    # Step 4 : exécution sur device
    print("Step 3: Device tap...")
    with llminsight.tool_span("device_tap", {"element": "Email", "coords": [234, 456]}) as cap:
        time.sleep(0.1)
        cap.set_output("tap success")

    # Step 5 : Reflection Agent (LLM)
    print("Step 4: Reflection Agent (LLM)...")
    reflection = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Compare avant/après. L'action a-t-elle réussi ?"},
            {"role": "user", "content": "Avant: écran login. Après: curseur dans email."},
        ],
    )
    print(f"  -> {reflection.choices[0].message.content}")

    # Step 6 : Planning Agent (LLM)
    print("Step 5: Planning Agent (LLM)...")
    planning = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Mets à jour les étapes restantes."},
            {"role": "user", "content": "Fait: tap email. Reste: saisir email, password, login."},
        ],
    )
    print(f"  -> {planning.choices[0].message.content}")

# Laisser le temps aux threads daemon
time.sleep(1)

print("\n=== Résumé ===")
print("1 trace 'mobile-scenario' avec :")
print("  - 3 spans LLM  (Action Agent, Reflection, Planning)")
print("  - 4 spans Tool  (screenshot, perception, device_tap)")
print("  - Coût calculé par span LLM")
print("  - Timeline ordonnée par step_index (0→6)")
print("\nVérifiez le terminal du mock backend pour voir tout ça.")
