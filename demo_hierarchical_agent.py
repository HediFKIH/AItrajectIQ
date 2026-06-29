"""
Demo : Architecture agentique hiérarchique
  Manager Agent  →  Search Worker
                 →  Analysis Worker
                 →  Summary Worker
                 →  Validation Worker

Tester avec LLMInsight (sans modifier ce fichier) :
    python -m llminsight run demo_hierarchical_agent.py

Ou directement (sans instrumentation) :
    python demo_hierarchical_agent.py

Utilise un faux transport HTTP — pas besoin de clé API réelle.
LLMInsight intercepte les appels au niveau du SDK OpenAI et enregistre les spans.
"""

import time
import random
import json
import httpx
import openai


# ---------------------------------------------------------------------------
# Faux transport HTTP pour OpenAI SDK
# LLMInsight patche openai.Completions.create — le vrai SDK est appelé,
# mais le transport HTTP renvoie une fausse réponse sans toucher l'API.
# ---------------------------------------------------------------------------

_FAKE_RESPONSES = {
    "manager": [
        "Plan: 1. Rechercher les données → 2. Analyser les tendances → 3. Synthétiser → 4. Valider la cohérence",
        "Décomposition: collecte sources primaires, analyse comparative, rapport exécutif, contrôle qualité",
    ],
    "search": [
        "Source 1: Données Q1 2026 — croissance +12% vs Q1 2025 (confiance: 94%)\nSource 2: 3 documents référencés — mise à jour il y a 2h\nSource 3: Corrélation identifiée avec les indicateurs macro-économiques",
        "Document A: Risques opérationnels catalogués — 7 items critiques\nDocument B: Historique incidents — 2 occurrences similaires\nDocument C: Plan de mitigation existant à réviser",
    ],
    "analysis": [
        "Tendance principale: croissance stable +12% sur la période\nAnomalie: pic inhabituel J+3 — investigation recommandée\nCorrélation forte (r=0.87) entre les variables clés\nRecommandation: focus sur les 20% de facteurs à fort impact",
        "Risque majeur identifié: dépendance fournisseur unique (score: 8/10)\nRisque modéré: volatilité marché (score: 5/10)\nRisque faible: turnover équipe (score: 3/10)\nPriorité: diversification fournisseurs Q2 2026",
    ],
    "summary": [
        "## Rapport Exécutif — Analyse Q1 2026\n\nPerformance globale au-dessus des objectifs (+12%).\n1 anomalie détectée nécessitant un suivi.\n\nRecommandations:\n- Capitaliser sur les tendances positives\n- Investiguer le pic J+3\n- Réviser la stratégie pricing Q2",
        "## Rapport Risques Opérationnels\n\n3 risques identifiés, 1 critique (fournisseur unique).\n\nActions prioritaires:\n- Qualifier 2 fournisseurs alternatifs (délai: 6 semaines)\n- Mettre à jour le plan de continuité\n- Former l'équipe aux nouvelles procédures",
    ],
    "validation": [
        "Validation OK — Score qualité: 0.94 — Cohérence: BONNE — Aucune contradiction",
        "Validation OK — Score qualité: 0.91 — Cohérence: BONNE — 1 point de clarification mineur",
    ],
}

_call_counters = {}

def _get_fake_response(agent_type: str) -> str:
    pool = _FAKE_RESPONSES.get(agent_type, ["Réponse simulée."])
    idx = _call_counters.get(agent_type, 0) % len(pool)
    _call_counters[agent_type] = idx + 1
    return pool[idx]


class _FakeOpenAITransport(httpx.BaseTransport):
    """Transport HTTP fictif — renvoie de vraies réponses OpenAI structurées."""

    def __init__(self, agent_type: str, latency_range=(600, 3500)):
        self.agent_type = agent_type
        self.latency_range = latency_range

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ms = random.randint(*self.latency_range)
        time.sleep(ms / 1000)

        content = _get_fake_response(self.agent_type)
        tokens_in = random.randint(800, 6000)
        tokens_out = len(content) // 4

        body = json.dumps({
            "id": f"chatcmpl-{random.randint(100000,999999)}",
            "object": "chat.completion",
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
        }).encode("utf-8")

        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
        )


def _make_client(agent_type: str, latency_range=(600, 3500)) -> openai.OpenAI:
    """Crée un client OpenAI avec transport simulé pour l'agent donné."""
    return openai.OpenAI(
        api_key="demo-fake-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx.Client(transport=_FakeOpenAITransport(agent_type, latency_range)),
    )


# ---------------------------------------------------------------------------
# Agents workers (niveau 2)
# ---------------------------------------------------------------------------

def search_worker(query: str, context: dict) -> str:
    print(f"    [Search Worker] Recherche: '{query[:60]}'")
    client = _make_client("search", latency_range=(800, 2500))
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Tu es un agent de recherche. Trouve les informations les plus pertinentes."},
            {"role": "user", "content": f"Recherche: {query}\nDomaine: {context.get('domain', 'général')}"},
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content


def analysis_worker(data: str, context: dict) -> str:
    print(f"    [Analysis Worker] Analyse en cours...")
    client = _make_client("analysis", latency_range=(1500, 4000))
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Tu es un agent d'analyse. Identifie les tendances, anomalies et recommandations."},
            {"role": "user", "content": f"Analyse:\n{data}\nObjectif: {context.get('goal', 'analyse générale')}"},
        ],
        max_tokens=400,
    )
    return resp.choices[0].message.content


def summary_worker(analysis: str, search_results: str, context: dict) -> str:
    print(f"    [Summary Worker] Rédaction du rapport...")
    client = _make_client("summary", latency_range=(2000, 5000))
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Tu es un agent de synthèse. Produis un rapport exécutif clair et actionnable."},
            {"role": "user", "content": f"Recherche:\n{search_results}\n\nAnalyse:\n{analysis}\n\nObjectif: {context.get('goal', '')}"},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content


def validation_worker(report: str, context: dict) -> str:
    print(f"    [Validation Worker] Vérification de la cohérence...")
    client = _make_client("validation", latency_range=(400, 1200))
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Tu es un agent de validation. Vérifie la cohérence et la qualité du rapport."},
            {"role": "user", "content": f"Valide ce rapport:\n{report}"},
        ],
        max_tokens=150,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Manager Agent (niveau 1) — orchestre les workers
# ---------------------------------------------------------------------------

def manager_agent(task: str, context: dict) -> dict:
    """Agent manager qui décompose la tâche et orchestre les workers."""
    print(f"\n  [Manager] Tâche: '{task[:70]}'")

    client = _make_client("manager", latency_range=(300, 900))
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Tu es un manager d'agents. Décompose la tâche en étapes claires pour tes workers."},
            {"role": "user", "content": f"Tâche: {task}\nDomaine: {context.get('domain', 'général')}"},
        ],
        max_tokens=200,
    )
    plan = resp.choices[0].message.content
    print(f"  [Manager] Plan: {plan[:80].encode('ascii', 'replace').decode('ascii')}...")

    search_results = search_worker(task, context)
    analysis      = analysis_worker(search_results, context)
    report        = summary_worker(analysis, search_results, context)
    validation    = validation_worker(report, context)

    print(f"  [Manager] Tous les workers ont terminé.")
    return {
        "plan": plan,
        "search": search_results,
        "analysis": analysis,
        "report": report,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------

def run_research_pipeline(queries: list, domain: str = "Business Intelligence"):
    print(f"\n{'='*60}")
    print(f"  Pipeline Hiérarchique — {domain}")
    print(f"  {len(queries)} requêtes à traiter")
    print(f"{'='*60}")

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n[Iteration {i}/{len(queries)}]")
        context = {"domain": domain, "goal": f"Répondre à: {query}", "iteration": i}
        result = manager_agent(query, context)
        results.append({"query": query, "result": result})
        print(f"  → Iteration {i} terminée\n")

    print(f"{'='*60}")
    print(f"  Pipeline terminé — {len(results)} rapports produits")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    queries = [
        "Analyser les performances Q1 2026 et identifier les axes d'amélioration",
        "Évaluer les risques opérationnels du trimestre suivant",
    ]
    run_research_pipeline(queries, domain="Business Intelligence")
