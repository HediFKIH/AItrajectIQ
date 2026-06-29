"""
Agent AI — Azure OpenAI + LLMInsight
Intégration : 2 lignes seulement (init + patch)
"""

import os
import math
import datetime
import json
from dotenv import load_dotenv
import wikipedia
from openai import AzureOpenAI
import llminsight

load_dotenv()  # Charger les variables d'environnement depuis un fichier .env
llminsight_api_key = os.getenv("API_KEY")
llminsight_project_id = os.getenv("LLMINSIGHT_PROJECT_ID") 
llminsight_base_url = os.getenv("LLMINSIGHT_BASE_URL") 
azure_endpoint = os.getenv("AZURE_ENDPOINT")   # ex: https://mon-resource.openai.azure.com
azure_api_version = os.getenv("API_VERSION", "2025-01-01-preview")
azure_deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")

# ── Intégration LLMInsight ───────────────────────


llminsight.init(
    project_id=llminsight_project_id,
    api_key=llminsight_api_key,
    base_url=llminsight_base_url,
)
llminsight.patch()   # ← intercepte OpenAI automatiquement

# ──────────────────────────────────────────────
# 2. Client Azure OpenAI
# ──────────────────────────────────────────────
client = AzureOpenAI(
    api_key=llminsight_api_key,
    azure_endpoint=azure_endpoint,   # ex: https://mon-resource.openai.azure.com
    api_version=azure_api_version,
)
MODEL = azure_deployment  # nom du déploiement Azure


# ── Outils ───────────────────────────────────────

def calculatrice(expression: str) -> str:
    try:
        env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        env["abs"] = abs
        return f"{expression} = {eval(expression, {'__builtins__': {}}, env)}"
    except Exception as e:
        return f"Erreur : {e}"

def recherche_wikipedia(query: str) -> str:
    try:
        wikipedia.set_lang("fr")
        return wikipedia.summary(query, sentences=4, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return "Sujet ambigu, précise parmi : " + ", ".join(e.options[:5])
    except Exception:
        try:
            wikipedia.set_lang("en")
            return "[EN] " + wikipedia.summary(query, sentences=4, auto_suggest=True)
        except Exception as e2:
            return f"Erreur Wikipedia : {e2}"

def horloge(fuseau: str = "local") -> str:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if fuseau == "utc":
        return f"UTC : {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    local = now_utc + datetime.timedelta(hours=1)
    return f"Heure locale (CET) : {local.strftime('%Y-%m-%d %H:%M:%S')}"

TOOLS_FN = {
    "calculatrice": calculatrice,
    "recherche_wikipedia": recherche_wikipedia,
    "horloge": horloge,
}

TOOLS_DECL = [
    {"type": "function", "function": {
        "name": "calculatrice",
        "description": "Évalue une expression mathématique Python (sqrt, log, **, %, etc.)",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}
    }},
    {"type": "function", "function": {
        "name": "recherche_wikipedia",
        "description": "Recherche des informations factuelles sur Wikipedia",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "horloge",
        "description": "Retourne la date et l'heure actuelle",
        "parameters": {"type": "object",
            "properties": {"fuseau": {"type": "string", "enum": ["local", "utc"]}}}
    }},
]

# ── Boucle agentique ─────────────────────────────

def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": (
            "Tu es un assistant intelligent et précis. "
            "Utilise les outils disponibles pour les calculs, "
            "informations factuelles et l'heure. "
            "Réponds toujours en français de façon concise."
        )},
        {"role": "user", "content": question}
    ]

    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DECL,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or "Pas de réponse."

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            print(f"  🔧 {fn_name}({fn_args})")
            result = TOOLS_FN[fn_name](**fn_args)
            print(f"  📋 {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Nombre maximum d'itérations atteint."

# ── CLI ──────────────────────────────────────────

def main():
    print("\n🤖 Agent — Azure GPT-4o-mini")
    print("   Outils : calculatrice · Wikipedia · horloge")
    print("   Tapez 'quitter' pour arrêter\n")
    print("─" * 60)

    while True:
        try:
            question = input("\n❓ Votre question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break
        if not question:
            continue
        if question.lower() in ("quitter", "exit", "quit", "q"):
            print("Au revoir !")
            break

        print("─" * 60)
        answer = run_agent(question)
        print(f"\n✅ {answer}\n")
        print("─" * 60)

if __name__ == "__main__":
    main()
