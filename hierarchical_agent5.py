"""
Système multi-agents hiérarchique avec LLMInsight observabilité.

Architecture :
    Manager Agent
    ├── Research Worker   — cherche et synthétise l'information
    ├── Analyst Worker    — analyse et tire des conclusions
    └── Writer Worker     — rédige la réponse finale

Usage :
    python hierarchical_agent.py
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
import llminsight

load_dotenv()
PROJECT_ID = os.getenv("LLMINSIGHT_PROJECT_ID")
API_KEY = os.getenv("API_KEY")
BASE = os.getenv("LLMINSIGHT_BASE_URL")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")   # ex: https://mon-resource.openai.azure.com
AZURE_API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
# ── Init LLMInsight ────────────────────────────────────────────────────────────
llminsight.init(
    project_id=PROJECT_ID,
    api_key=API_KEY,
    base_url=BASE,
)
llminsight.patch() 

# ── Azure OpenAI client ────────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,   # ex: https://mon-resource.openai.azure.com
    api_version=AZURE_API_VERSION,
)
MODEL = AZURE_DEPLOYMENT

# ══════════════════════════════════════════════════════════════════════════════
# WORKER AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def research_worker(task: str, context: dict) -> str:
    """Cherche et synthétise l'information pertinente."""
    with llminsight.trace("research_worker", {"task": task[:80]}):
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un agent de recherche expert. "
                    "Ton rôle est de rassembler et synthétiser l'information pertinente "
                    "sur le sujet donné. Sois factuel, exhaustif et structuré. "
                    "Réponds en français."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Tâche de recherche : {task}\n\n"
                    f"Contexte fourni par le manager : {json.dumps(context, ensure_ascii=False)}\n\n"
                    "Fournis une synthèse de recherche complète."
                ),
            },
        ]
        response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.3)
        return response.choices[0].message.content


def analyst_worker(task: str, research_output: str, context: dict) -> str:
    """Analyse les données et tire des conclusions."""
    with llminsight.trace("analyst_worker", {"task": task[:80]}):
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un agent analyste expert. "
                    "Ton rôle est d'analyser les informations fournies, d'identifier "
                    "les patterns, les risques, les opportunités et de formuler "
                    "des conclusions claires et actionnables. Réponds en français."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Tâche d'analyse : {task}\n\n"
                    f"Résultats de recherche :\n{research_output}\n\n"
                    f"Contexte : {json.dumps(context, ensure_ascii=False)}\n\n"
                    "Fournis une analyse approfondie avec conclusions et recommandations."
                ),
            },
        ]
        response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.4)
        return response.choices[0].message.content


def writer_worker(task: str, research_output: str, analysis_output: str, context: dict) -> str:
    """Rédige la réponse finale, claire et bien structurée."""
    with llminsight.trace("writer_worker", {"task": task[:80]}):
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un agent rédacteur expert. "
                    "Ton rôle est de transformer les recherches et analyses en une réponse "
                    "finale claire, bien structurée et adaptée au public cible. "
                    "La réponse doit être directement utilisable. Réponds en français."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Tâche de rédaction : {task}\n\n"
                    f"Recherche :\n{research_output}\n\n"
                    f"Analyse :\n{analysis_output}\n\n"
                    f"Contexte : {json.dumps(context, ensure_ascii=False)}\n\n"
                    "Rédige la réponse finale complète et professionnelle."
                ),
            },
        ]
        response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.5)
        return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# MANAGER AGENT — orchestre via function calling
# ══════════════════════════════════════════════════════════════════════════════

MANAGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_research",
            "description": "Délègue une tâche de recherche et synthèse d'information à l'agent Research Worker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Description précise de ce qui doit être recherché."},
                    "context": {"type": "object", "description": "Contexte additionnel utile pour la recherche."},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_analyst",
            "description": "Délègue une tâche d'analyse à l'agent Analyst Worker. Utiliser après la recherche.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Description précise de ce qui doit être analysé."},
                    "research_output": {"type": "string", "description": "Le résultat de la recherche à analyser."},
                    "context": {"type": "object", "description": "Contexte additionnel."},
                },
                "required": ["task", "research_output"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_writer",
            "description": "Délègue la rédaction de la réponse finale à l'agent Writer Worker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Description précise du document à rédiger."},
                    "research_output": {"type": "string", "description": "Résultats de recherche."},
                    "analysis_output": {"type": "string", "description": "Résultats d'analyse."},
                    "context": {"type": "object", "description": "Contexte additionnel."},
                },
                "required": ["task", "research_output", "analysis_output"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_response",
            "description": "Finalise et retourne la réponse complète à l'utilisateur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string", "description": "La réponse finale complète à présenter."},
                    "summary": {"type": "string", "description": "Résumé exécutif en 2-3 phrases."},
                },
                "required": ["final_answer", "summary"],
            },
        },
    },
]


def run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatche l'appel d'outil vers le bon worker."""
    with llminsight.tool_span(tool_name, tool_args):
        if tool_name == "delegate_to_research":
            result = research_worker(
                task=tool_args["task"],
                context=tool_args.get("context", {}),
            )
        elif tool_name == "delegate_to_analyst":
            result = analyst_worker(
                task=tool_args["task"],
                research_output=tool_args["research_output"],
                context=tool_args.get("context", {}),
            )
        elif tool_name == "delegate_to_writer":
            result = writer_worker(
                task=tool_args["task"],
                research_output=tool_args["research_output"],
                analysis_output=tool_args["analysis_output"],
                context=tool_args.get("context", {}),
            )
        elif tool_name == "finalize_response":
            result = json.dumps(tool_args, ensure_ascii=False)
        else:
            result = f"Outil inconnu : {tool_name}"
    return result


def manager_agent(user_query: str) -> dict:
    """
    Agent Manager : décompose la requête, orchestre les workers,
    agrège et retourne la réponse finale.
    """
    print(f"\n{'═'*60}")
    print(f"  MANAGER — Requête reçue")
    print(f"{'═'*60}")
    print(f"  {user_query}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un Manager Agent expert en orchestration multi-agents. "
                "Pour chaque requête utilisateur, tu dois :\n"
                "1. D'abord déléguer la RECHERCHE à delegate_to_research\n"
                "2. Ensuite déléguer l'ANALYSE à delegate_to_analyst (en passant le résultat de recherche)\n"
                "3. Ensuite déléguer la RÉDACTION à delegate_to_writer (en passant recherche + analyse)\n"
                "4. Enfin appeler finalize_response avec la réponse finale\n\n"
                "Ne saute aucune étape. Chaque worker a un rôle précis et complémentaire. "
                "Réponds en français."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    # Boucle agentique — le manager appelle les workers en séquence
    MAX_ITERATIONS = 10
    final_result = None

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=MANAGER_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # Pas d'appel d'outil → le manager a fini
        if not msg.tool_calls:
            print(f"\n  Manager — réponse textuelle finale :\n  {msg.content}")
            break

        # Traitement de chaque appel d'outil
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"\n  ► Délégation vers [{tool_name}]")
            if "task" in tool_args:
                print(f"    Tâche : {tool_args['task'][:100]}")

            result = run_tool(tool_name, tool_args)

            # Capture du résultat final
            if tool_name == "finalize_response":
                try:
                    final_result = json.loads(result)
                except json.JSONDecodeError:
                    final_result = {"final_answer": result, "summary": ""}

            # Injecter le résultat dans la conversation du manager
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

            print(f"    ✓ Résultat reçu ({len(result)} chars)")

    return final_result or {"final_answer": "Erreur : pas de réponse finale.", "summary": ""}


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Exemples de requêtes
    queries = [
        "Analyse les avantages et les risques de l'adoption de l'IA générative dans une PME industrielle.",
    ]

    for query in queries:
        with llminsight.trace("hierarchical_pipeline", {"query": query[:80]}):
            result = manager_agent(query)

        print(f"\n{'═'*60}")
        print("  RÉSULTAT FINAL")
        print(f"{'═'*60}")
        print(f"\n📋 RÉSUMÉ :\n{result.get('summary', '')}")
        print(f"\n📄 RÉPONSE COMPLÈTE :\n{result.get('final_answer', '')}")
        print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
