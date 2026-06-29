"""
Multi-Agent Hiérarchique — Support Client
Orchestrateur décide quels sous-agents appeler et dans quel ordre.
Agents : Classifier, Specialist, Summarizer (mêmes rôles, contrôle différent)
"""

import os
import json
from openai import AzureOpenAI
import llminsight
from dotenv import load_dotenv

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
llminsight.patch() 

# ── Client Azure OpenAI ──────────────────────────
client = AzureOpenAI(
    api_key=llminsight_api_key,
    azure_endpoint=azure_endpoint,   # ex: https://mon-resource.openai.azure.com
    api_version=azure_api_version,
)
MODEL = azure_deployment  # nom du déploiement Azure


def llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Sous-agents ──────────────────────────────────

def sub_classifier(ticket: str) -> dict:
    print("    ├─ [classifier] analyse...")
    result = llm(
        system="""Classifie ce ticket support. Retourne UNIQUEMENT un JSON :
{
  "category": "billing|technical|account|shipping|other",
  "priority": "low|medium|high|urgent",
  "language": "fr|en|ar|other",
  "needs_specialist": true/false,
  "needs_escalation": true/false
}""",
        user=ticket
    )
    try:
        return json.loads(result)
    except Exception:
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        return json.loads(match.group()) if match else {
            "category": "other", "priority": "medium",
            "language": "fr", "needs_specialist": True, "needs_escalation": False
        }


def sub_specialist(ticket: str, classification: dict) -> str:
    print("    ├─ [specialist] rédaction réponse...")
    specialists = {
        "billing": "expert facturation", "technical": "expert technique N2",
        "account": "expert comptes",     "shipping": "expert logistique",
        "other":   "agent généraliste",
    }
    role = specialists.get(classification.get("category", "other"), "agent support")
    lang_map = {"fr": "Réponds en français.", "en": "Reply in English.", "ar": "أجب باللغة العربية."}
    lang_instr = lang_map.get(classification.get("language", "fr"), "Réponds dans la langue du client.")

    return llm(
        system=f"Tu es {role}. {lang_instr} Sois professionnel, empathique et concis.",
        user=f"Ticket : {ticket}\nContexte : {json.dumps(classification, ensure_ascii=False)}"
    )


def sub_escalation(ticket: str, classification: dict) -> str:
    print("    ├─ [escalation] traitement prioritaire...")
    return llm(
        system="""Tu es un responsable support senior. Ce ticket nécessite une escalade.
Rédige une réponse prioritaire avec :
1. Accusé de réception immédiat
2. Engagement sur un délai de résolution précis
3. Nom d'un référent dédié (fictif)
Sois rassurant et professionnel.""",
        user=f"Ticket urgent : {ticket}\nPriorité : {classification.get('priority')}"
    )


def sub_summarizer(ticket: str, classification: dict, response: str) -> dict:
    print("    └─ [summarizer] synthèse finale...")
    result = llm(
        system="""Synthèse d'interaction support. Retourne UNIQUEMENT un JSON :
{
  "ticket_id": "TKT-XXXX",
  "category": "...",
  "priority": "...",
  "issue": "problème en 1 phrase",
  "resolution": "réponse en 1 phrase",
  "escalated": true/false,
  "follow_up_required": true/false,
  "estimated_resolution_time": "immédiat|24h|48h|72h"
}""",
        user=f"Ticket:\n{ticket}\n\nClassification:\n{json.dumps(classification, ensure_ascii=False)}\n\nRéponse:\n{response}"
    )
    try:
        return json.loads(result)
    except Exception:
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        return json.loads(match.group()) if match else {"issue": ticket[:80], "resolution": response[:80]}


# ── Orchestrateur ────────────────────────────────

# Déclaration des sous-agents comme outils pour l'orchestrateur
SUBAGENTS_DECL = [
    {"type": "function", "function": {
        "name": "sub_classifier",
        "description": "Classifie le ticket : catégorie, priorité, langue, besoin d'escalade",
        "parameters": {"type": "object",
            "properties": {"ticket": {"type": "string", "description": "Texte du ticket client"}},
            "required": ["ticket"]}
    }},
    {"type": "function", "function": {
        "name": "sub_specialist",
        "description": "Rédige une réponse client selon la catégorie. Nécessite la classification.",
        "parameters": {"type": "object",
            "properties": {
                "ticket": {"type": "string"},
                "classification": {"type": "string", "description": "JSON de classification (output de sub_classifier)"}
            },
            "required": ["ticket", "classification"]}
    }},
    {"type": "function", "function": {
        "name": "sub_escalation",
        "description": "Traitement prioritaire pour tickets urgent/high avec needs_escalation=true",
        "parameters": {"type": "object",
            "properties": {
                "ticket": {"type": "string"},
                "classification": {"type": "string"}
            },
            "required": ["ticket", "classification"]}
    }},
    {"type": "function", "function": {
        "name": "sub_summarizer",
        "description": "Produit la synthèse finale de l'interaction. Appeler en dernier.",
        "parameters": {"type": "object",
            "properties": {
                "ticket": {"type": "string"},
                "classification": {"type": "string"},
                "response": {"type": "string", "description": "Réponse fournie au client"}
            },
            "required": ["ticket", "classification", "response"]}
    }},
]

SUBAGENTS_FN = {
    "sub_classifier": lambda ticket, **_: sub_classifier(ticket),
    "sub_specialist": lambda ticket, classification, **_: sub_specialist(ticket, json.loads(classification)),
    "sub_escalation": lambda ticket, classification, **_: sub_escalation(ticket, json.loads(classification)),
    "sub_summarizer": lambda ticket, classification, response, **_: sub_summarizer(ticket, json.loads(classification), response),
}


def run_hierarchical_pipeline(ticket: str) -> dict:
    """
    L'orchestrateur décide lui-même :
    - quels sous-agents appeler
    - dans quel ordre
    - si escalade nécessaire ou non
    """
    print(f"\n{'═'*60}")
    print(f"  PIPELINE HIÉRARCHIQUE — Support Client")
    print(f"{'═'*60}")
    print(f"  Ticket : {ticket[:80]}{'...' if len(ticket) > 80 else ''}")
    print(f"{'─'*60}")
    print("  [orchestrateur] analyse et planification...")

    messages = [
        {"role": "system", "content": """Tu es un orchestrateur de support client.
Tu as accès à 4 sous-agents : sub_classifier, sub_specialist, sub_escalation, sub_summarizer.

Règles de décision :
- Toujours commencer par sub_classifier
- Si needs_escalation=true ou priority=urgent → utiliser sub_escalation au lieu de sub_specialist
- Sinon → utiliser sub_specialist
- Toujours finir par sub_summarizer
- Passer les outputs d'un agent à l'autre en sérialisant en JSON string

Traite le ticket jusqu'à avoir la synthèse finale."""},
        {"role": "user", "content": f"Traite ce ticket support :\n\n{ticket}"}
    ]

    context = {}   # stocke les outputs intermédiaires

    for _ in range(10):   # max 10 appels LLM pour l'orchestrateur
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=SUBAGENTS_DECL,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        # Pas d'appel d'outil → l'orchestrateur a terminé
        if not msg.tool_calls:
            break

        # Exécuter les sous-agents demandés par l'orchestrateur
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            print(f"  [orchestrateur] → appel {fn_name}")
            result = SUBAGENTS_FN[fn_name](**fn_args)

            # Sérialiser le résultat pour le remettre à l'orchestrateur
            result_str = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
            context[fn_name] = result

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    print(f"{'═'*60}\n")

    return {
        "ticket":         ticket,
        "classification": context.get("sub_classifier", {}),
        "response":       context.get("sub_specialist") or context.get("sub_escalation", ""),
        "summary":        context.get("sub_summarizer", {}),
        "escalated":      "sub_escalation" in context,
    }


# ── CLI ──────────────────────────────────────────

SAMPLE_TICKETS = [
    "Je n'arrive pas à me connecter à mon compte depuis 2 jours, j'ai un message d'erreur 403.",
    "J'ai été débité deux fois pour ma commande #4521, remboursez-moi IMMÉDIATEMENT c'est scandaleux !",
    "My package was supposed to arrive last week but I still haven't received it. Order #7823.",
]

def main():
    print("\n🤖 Multi-Agent Hiérarchique — Support Client")
    print("   Orchestrateur → [Classifier / Specialist / Escalation] → Summarizer")
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
                result = run_hierarchical_pipeline(t)
                escalated = "⚠️ ESCALADÉ" if result["escalated"] else "✅ Standard"
                print(f"{escalated} | Réponse :\n{result['response']}\n")
                print(f"📋 Résumé : {json.dumps(result['summary'], ensure_ascii=False, indent=2)}\n")
            continue

        result = run_hierarchical_pipeline(ticket)
        escalated = "⚠️ ESCALADÉ" if result["escalated"] else "✅ Standard"
        print(f"\n{escalated} — Réponse au client :\n{result['response']}\n")
        print(f"📋 Résumé :\n{json.dumps(result['summary'], ensure_ascii=False, indent=2)}\n")

if __name__ == "__main__":
    main()
