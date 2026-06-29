import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Multi-Agent Hiérarchique — Support Client
Architecture : Manager → [Classifier, Specialist/Escalation, Summarizer]
Intégration LLMInsight : init() + patch() uniquement — le backend
calcule automatiquement tokens, coûts et métriques.
"""

import os, json, re
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
# ── LLMInsight ───────────────────────────────────────────────────────────
llminsight.init(
    project_id=llminsight_project_id,
    api_key=llminsight_api_key,
    base_url=llminsight_base_url,
)
llminsight.patch()
llminsight.add_guardrail("prompt_injection")
llminsight.add_guardrail("cost_limit", max_cost_usd=0.10)

# ── Client Azure OpenAI ──────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=llminsight_api_key,
    azure_endpoint=azure_endpoint,   # ex: https://mon-resource.openai.azure.com
    api_version=azure_api_version,
)
MODEL = azure_deployment  # nom du déploiement Azure

def parse_json(text: str, fallback: dict) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        try:    return json.loads(m.group()) if m else fallback
        except: return fallback


def _llm_call(system: str, user: str, iteration: int, tools=None) -> object:
    """Appel LLM bas niveau — capturé automatiquement par llminsight.patch()."""
    if hasattr(llminsight, "set_agent_context"):
        llminsight.set_agent_context(iteration=iteration)
    kwargs = dict(
        model=MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user",  "content":user}],
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


# ── Workers ──────────────────────────────────────────────────────────────

def worker_classifier(ticket: str, iteration: int) -> dict:
    print(f"    ├─ [Classifier Worker] iter {iteration}")
    resp = _llm_call(
        system="""Classifie ce ticket support. Retourne UNIQUEMENT un JSON :
{
  "category": "billing|technical|account|shipping|other",
  "priority": "low|medium|high|urgent",
  "language": "fr|en|ar|other",
  "needs_escalation": true
}""",
        user=ticket,
        iteration=iteration,
    )
    return parse_json(
        resp.choices[0].message.content.strip(),
        {"category":"other","priority":"medium","language":"fr","needs_escalation":False}
    )


def worker_specialist(ticket: str, clf: dict, iteration: int) -> str:
    print(f"    ├─ [Specialist Worker] iter {iteration}")
    roles = {"billing":"expert facturation","technical":"expert technique N2",
             "account":"expert comptes","shipping":"expert logistique","other":"agent généraliste"}
    role_label = roles.get(clf.get("category","other"), "agent support")
    lang_map = {"fr":"Réponds en français.","en":"Reply in English.","ar":"أجب باللغة العربية."}
    lang = lang_map.get(clf.get("language","fr"), "Réponds dans la langue du client.")
    resp = _llm_call(
        system=f"Tu es {role_label}. {lang} Sois professionnel et empathique.",
        user=f"Ticket: {ticket}\nContexte: {json.dumps(clf,ensure_ascii=False)}",
        iteration=iteration,
    )
    return resp.choices[0].message.content.strip()


def worker_escalation(ticket: str, clf: dict, iteration: int) -> str:
    print(f"    ├─ [Escalation Worker] iter {iteration} ⚠️")
    resp = _llm_call(
        system="""Tu es responsable support senior. Ticket prioritaire.
Rédige une réponse avec : accusé immédiat, délai précis, référent dédié (fictif).""",
        user=f"Ticket urgent: {ticket}\nPriorité: {clf.get('priority')}",
        iteration=iteration,
    )
    return resp.choices[0].message.content.strip()


def worker_summarizer(ticket: str, clf: dict, response: str,
                      escalated: bool, iteration: int) -> dict:
    print(f"    └─ [Summarizer Worker] iter {iteration}")
    resp = _llm_call(
        system="""Synthèse support. Retourne UNIQUEMENT un JSON :
{
  "ticket_id": "TKT-XXXX",
  "category": "...", "priority": "...",
  "issue": "1 phrase", "resolution": "1 phrase",
  "escalated": true,
  "follow_up_required": true,
  "estimated_resolution_time": "immédiat|24h|48h|72h"
}""",
        user=f"Ticket:\n{ticket}\n\nClassification:\n{json.dumps(clf,ensure_ascii=False)}\n\nRéponse:\n{response}",
        iteration=iteration,
    )
    return parse_json(
        resp.choices[0].message.content.strip(),
        {"issue":ticket[:80],"resolution":response[:80],"escalated":escalated}
    )


# ── Manager (orchestrateur) ───────────────────────────────────────────────

WORKER_TOOLS = [
    {"type":"function","function":{
        "name":"worker_classifier",
        "description":"Classifie le ticket : catégorie, priorité, langue, escalade",
        "parameters":{"type":"object","properties":{"ticket":{"type":"string"}},"required":["ticket"]}
    }},
    {"type":"function","function":{
        "name":"worker_specialist",
        "description":"Rédige la réponse standard selon la catégorie du ticket",
        "parameters":{"type":"object","properties":{
            "ticket":{"type":"string"},
            "classification":{"type":"string","description":"JSON de classification"}
        },"required":["ticket","classification"]}
    }},
    {"type":"function","function":{
        "name":"worker_escalation",
        "description":"Traitement prioritaire pour tickets urgent ou needs_escalation=true",
        "parameters":{"type":"object","properties":{
            "ticket":{"type":"string"},
            "classification":{"type":"string"}
        },"required":["ticket","classification"]}
    }},
    {"type":"function","function":{
        "name":"worker_summarizer",
        "description":"Synthèse finale de l'interaction. À appeler en dernier.",
        "parameters":{"type":"object","properties":{
            "ticket":{"type":"string"},
            "classification":{"type":"string"},
            "response":{"type":"string"}
        },"required":["ticket","classification","response"]}
    }},
]


def agent_manager(ticket: str) -> dict:
    """Manager Agent : orchestre les workers via function calling."""
    print(f"  [Manager Agent] Analyse et planification...")

    messages = [
        {"role":"system","content":"""Tu es un Manager Agent de support client.
Tu coordonnes des workers spécialisés via des appels de fonctions.

Règles strictes :
1. Toujours appeler worker_classifier en premier
2. Si needs_escalation=true ou priority=urgent → worker_escalation, sinon worker_specialist
3. Toujours finir par worker_summarizer
4. Passer les résultats JSON en string entre workers"""},
        {"role":"user","content":f"Traite ce ticket client :\n\n{ticket}"}
    ]

    context = {}
    iteration = 0

    for _ in range(10):
        if hasattr(llminsight, "set_agent_context"):
            llminsight.set_agent_context(iteration=iteration)

        resp = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=WORKER_TOOLS, tool_choice="auto",
        )

        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"  [Manager] iter {iteration} → {fn_name}")

            clf_obj = context.get("worker_classifier", {})

            if fn_name == "worker_classifier":
                result = worker_classifier(fn_args["ticket"], iteration)
                context["worker_classifier"] = result
                result_str = json.dumps(result, ensure_ascii=False)

            elif fn_name == "worker_specialist":
                clf = parse_json(fn_args.get("classification","{}"), clf_obj)
                result = worker_specialist(fn_args["ticket"], clf, iteration)
                context["worker_specialist"] = result
                result_str = result

            elif fn_name == "worker_escalation":
                clf = parse_json(fn_args.get("classification","{}"), clf_obj)
                result = worker_escalation(fn_args["ticket"], clf, iteration)
                context["worker_escalation"] = result
                result_str = result

            elif fn_name == "worker_summarizer":
                clf = parse_json(fn_args.get("classification","{}"), clf_obj)
                resp_text = fn_args.get("response",
                    context.get("worker_specialist") or context.get("worker_escalation",""))
                escalated = "worker_escalation" in context
                result = worker_summarizer(fn_args["ticket"], clf, resp_text, escalated, iteration)
                context["worker_summarizer"] = result
                result_str = json.dumps(result, ensure_ascii=False)
            else:
                result_str = ""

            messages.append({"role":"tool","tool_call_id":tc.id,"content":result_str})

        iteration += 1

    return context


# ── Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(ticket: str) -> dict:
    print(f"\n{'═'*60}")
    print(f"  PIPELINE HIÉRARCHIQUE — Support Client")
    print(f"  Ticket : {ticket[:70]}{'...' if len(ticket)>70 else ''}")
    print(f"{'─'*60}")

    context = agent_manager(ticket)

    escalated = "worker_escalation" in context
    response  = context.get("worker_specialist") or context.get("worker_escalation","")
    summary   = context.get("worker_summarizer", {})

    print(f"{'═'*60}\n")

    return {
        "ticket": ticket,
        "classification": context.get("worker_classifier", {}),
        "response": response,
        "summary": summary,
        "escalated": escalated,
    }


# ── CLI ──────────────────────────────────────────────────────────────────

DEMO = [
    "Je n'arrive pas à me connecter à mon compte depuis 2 jours, erreur 403.",
    "J'ai été débité deux fois pour ma commande #4521, remboursez-moi IMMÉDIATEMENT !",
    "My package was supposed to arrive last week. Order #7823.",
]

def main():
    print("\n🤖 Multi-Agent Hiérarchique — Support Client")
    print("   Manager → [Classifier, Specialist/Escalation, Summarizer]")
    print("   'demo' → tickets d'exemple | 'quitter' → arrêter\n")
    while True:
        try: ticket = input("📩 Ticket : ").strip()
        except (KeyboardInterrupt, EOFError): print("\nAu revoir !"); break
        if not ticket: continue
        if ticket.lower() in ("quitter","exit","quit","q"): print("Au revoir !"); break
        if ticket.lower() == "demo":
            for t in DEMO:
                r = run_pipeline(t)
                tag = "⚠️ ESCALADÉ" if r["escalated"] else "✅ Standard"
                print(f"{tag} | {r['response'][:120]}\n")
            continue
        r = run_pipeline(ticket)
        tag = "⚠️ ESCALADÉ" if r["escalated"] else "✅ Standard"
        print(f"\n{tag} — Réponse :\n{r['response']}\n")

if __name__ == "__main__":
    main()
