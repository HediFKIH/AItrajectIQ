"""
Multi-Agent Séquentiel — Support Client
Pipeline : Classifier → Specialist → Summarizer
Chaque agent passe son output au suivant.
"""

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
import llminsight

load_dotenv()  # Charger les variables d'environnement depuis un fichier .env
llminsight_api_key = os.getenv("API_KEY")
llminsight_project_id = os.getenv("LLMINSIGHT_PROJECT_ID") 
llminsight_base_url = os.getenv("LLMINSIGHT_BASE_URL") 
azure_endpoint = os.getenv("AZURE_ENDPOINT")   # ex: https://mon-resource.openai.azure.com
azure_api_version = os.getenv("API_VERSION", "2025-01-01-preview")
azure_deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")

# ── LLMInsight ───────────────────────────────────
llminsight.init(
    project_id=llminsight_project_id,
    api_key=llminsight_api_key,
    base_url=llminsight_base_url,
)
llminsight.patch()   # ← intercepte OpenAI automatiquement

# ── Client Azure OpenAI ──────────────────────────
client = AzureOpenAI(
    api_key=llminsight_api_key,
    azure_endpoint=azure_endpoint,   # ex: https://mon-resource.openai.azure.com
    api_version=azure_api_version,
)
MODEL = azure_deployment  # nom du déploiement Azure


def llm(system: str, user: str) -> str:
    """Appel LLM simple — capturé automatiquement par llminsight.patch()."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Agent 1 : Classifier ─────────────────────────
def agent_classifier(ticket: str) -> dict:
    """
    Classifie le ticket client.
    Retourne : category, priority, language
    """
    print("\n[1/3] 🔍 Classifier...")

    result = llm(
        system="""Tu es un agent de classification de tickets support.
Analyse le ticket et retourne UNIQUEMENT un JSON valide avec ces champs :
{
  "category": "billing|technical|account|shipping|other",
  "priority": "low|medium|high|urgent",
  "language": "fr|en|ar|other",
  "summary": "résumé en 1 phrase"
}
Aucun texte en dehors du JSON.""",
        user=ticket
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        # Fallback si le modèle ajoute du texte autour
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        return json.loads(match.group()) if match else {
            "category": "other", "priority": "medium",
            "language": "fr", "summary": ticket[:100]
        }


# ── Agent 2 : Specialist ─────────────────────────
def agent_specialist(ticket: str, classification: dict) -> str:
    """
    Rédige une réponse adaptée à la catégorie et priorité.
    Reçoit le ticket + la classification de l'agent précédent.
    """
    print("[2/3] 💬 Specialist...")

    specialists = {
        "billing":   "expert en facturation et paiements",
        "technical": "expert technique niveau 2",
        "account":   "expert gestion de comptes",
        "shipping":  "expert logistique et livraisons",
        "other":     "agent support généraliste",
    }
    role = specialists.get(classification.get("category", "other"), "agent support")
    priority = classification.get("priority", "medium")
    lang = classification.get("language", "fr")

    lang_instruction = {
        "fr": "Réponds en français.",
        "en": "Reply in English.",
        "ar": "أجب باللغة العربية.",
    }.get(lang, "Réponds dans la langue du client.")

    return llm(
        system=f"""Tu es un {role} dans un service support client.
Priorité du ticket : {priority}.
{lang_instruction}
Rédige une réponse professionnelle, empathique et utile.
Si priorité urgent/high : commence par rassurer le client immédiatement.""",
        user=f"Ticket client :\n{ticket}\n\nContexte :\n{json.dumps(classification, ensure_ascii=False)}"
    )


# ── Agent 3 : Summarizer ─────────────────────────
def agent_summarizer(ticket: str, classification: dict, response: str) -> dict:
    """
    Produit un résumé structuré de l'interaction complète.
    Reçoit les outputs des 2 agents précédents.
    """
    print("[3/3] 📋 Summarizer...")

    result = llm(
        system="""Tu es un agent de synthèse support client.
Produis un résumé structuré de l'interaction. Retourne UNIQUEMENT un JSON valide :
{
  "ticket_id": "généré aléatoirement ex: TKT-XXXX",
  "category": "catégorie du ticket",
  "priority": "priorité",
  "issue": "problème en 1 phrase",
  "resolution": "réponse fournie en 1 phrase",
  "follow_up_required": true/false,
  "estimated_resolution_time": "immédiat|24h|48h|72h"
}""",
        user=f"Ticket:\n{ticket}\n\nClassification:\n{json.dumps(classification, ensure_ascii=False)}\n\nRéponse fournie:\n{response}"
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        return json.loads(match.group()) if match else {"issue": ticket[:100], "resolution": response[:100]}


# ── Pipeline séquentiel ──────────────────────────

def run_sequential_pipeline(ticket: str) -> dict:
    """
    Exécute le pipeline séquentiel :
    ticket → Classifier → Specialist → Summarizer → résultat final
    """
    print(f"\n{'═'*60}")
    print(f"  PIPELINE SÉQUENTIEL — Support Client")
    print(f"{'═'*60}")
    print(f"  Ticket : {ticket[:80]}{'...' if len(ticket) > 80 else ''}")
    print(f"{'─'*60}")

    # Étape 1
    classification = agent_classifier(ticket)
    print(f"  → Catégorie: {classification.get('category')} | Priorité: {classification.get('priority')}")

    # Étape 2 — reçoit l'output de l'étape 1
    response = agent_specialist(ticket, classification)
    print(f"  → Réponse rédigée ({len(response)} chars)")

    # Étape 3 — reçoit les outputs des étapes 1 et 2
    summary = agent_summarizer(ticket, classification, response)
    print(f"  → Résumé généré")

    print(f"{'═'*60}\n")

    return {
        "ticket":         ticket,
        "classification": classification,
        "response":       response,
        "summary":        summary,
    }


# ── CLI ──────────────────────────────────────────

SAMPLE_TICKETS = [
    "Je n'arrive pas à me connecter à mon compte depuis 2 jours, j'ai un message d'erreur 403.",
    "J'ai été débité deux fois pour ma commande #4521, je veux un remboursement urgent.",
    "My package was supposed to arrive last week but I still haven't received it.",
]

def main():
    print("\n🤖 Multi-Agent Séquentiel — Support Client")
    print("   Pipeline : Classifier → Specialist → Summarizer")
    print("   Tapez 'demo' pour les tickets d'exemple, 'quitter' pour arrêter\n")

    while True:
        try:
            ticket = input("📩 Ticket client : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if not ticket:
            continue
        if ticket.lower() in ("quitter", "exit", "quit", "q"):
            print("Au revoir !")
            break
        if ticket.lower() == "demo":
            for t in SAMPLE_TICKETS:
                result = run_sequential_pipeline(t)
                print(f"✅ Réponse :\n{result['response']}\n")
                print(f"📋 Résumé : {json.dumps(result['summary'], ensure_ascii=False, indent=2)}\n")
            continue

        result = run_sequential_pipeline(ticket)
        print(f"\n✅ Réponse au client :\n{result['response']}\n")
        print(f"📋 Résumé :\n{json.dumps(result['summary'], ensure_ascii=False, indent=2)}\n")

if __name__ == "__main__":
    main()
