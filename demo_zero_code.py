import os
import sys, io

from dotenv import load_dotenv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Demo ZERO-CODE : application agentique SANS aucun agent_span.
Le SDK detecte automatiquement la hierarchie via la pile d'appels Python.

Aucune modification du code de l'application n'est necessaire.
Seules les 3 lignes d'initialisation sont ajoutees.
"""

import time
import json
import random
from unittest.mock import MagicMock
load_dotenv()
PROJECT_ID = os.getenv("LLMINSIGHT_PROJECT_ID")
API_KEY = os.getenv("API_KEY")
BASE = os.getenv("LLMINSIGHT_BASE_URL")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")   # ex: https://mon-resource.openai.azure.com
AZURE_API_VERSION = os.getenv("API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
# ── 1. Mock du client OpenAI ──

def fake_create(self, *args, **kwargs):
    messages = kwargs.get("messages", [])
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    time.sleep(random.uniform(0.1, 0.3))

    if "manager" in system_msg.lower():
        answer = "Plan: recherche -> calcul -> redaction"
    elif "recherche" in system_msg.lower():
        answer = "Tesla Model 3: 42 000 EUR, autonomie 500km, concurrent BYD Seal 35 000 EUR"
    elif "calcul" in system_msg.lower():
        answer = "Cout total 5 ans: 52 300 EUR. Economie vs thermique: 18 700 EUR"
    elif "redaction" in system_msg.lower():
        answer = "La Tesla Model 3 est rentable sur 5 ans avec 18 700 EUR d'economie."
    else:
        answer = f"Reponse a: {user_msg[:50]}"

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = answer
    response.usage = MagicMock()
    response.usage.prompt_tokens = random.randint(200, 1000)
    response.usage.completion_tokens = random.randint(50, 300)
    response.model = kwargs.get("model", AZURE_DEPLOYMENT)
    return response

import openai.resources.chat.completions
openai.resources.chat.completions.Completions.create = fake_create

# ── 2. LLMInsight : SEULEMENT 3 LIGNES ──
import llminsight
llminsight.init(project_id=PROJECT_ID, api_key=API_KEY, base_url=BASE)
llminsight.patch()

# ── 3. L'application du client — AUCUNE reference a llminsight ──
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# Capture locale des spans
collected_spans = []
from llminsight import client as _cli
_orig_post = _cli.post_async
def _capture(endpoint, payload):
    if "/ingest/span" in endpoint:
        collected_spans.append(payload.copy())
    _orig_post(endpoint, payload)
_cli.post_async = _capture


def worker_recherche(question):
    """Simule un agent de recherche — AUCUN llminsight.agent_span ici."""
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un worker RECHERCHE."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


def worker_calcul(data):
    """Simule un agent de calcul — AUCUN llminsight.agent_span ici."""
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un worker CALCUL."},
            {"role": "user", "content": data},
        ],
    )
    return resp.choices[0].message.content


def worker_redaction(recherche, calcul):
    """Simule un agent de redaction — AUCUN llminsight.agent_span ici."""
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un worker REDACTION."},
            {"role": "user", "content": f"Recherche: {recherche}\nCalcul: {calcul}"},
        ],
    )
    return resp.choices[0].message.content


def manager(question):
    """Le manager delegue aux workers — AUCUN llminsight.agent_span ici."""
    plan = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Tu es un MANAGER."},
            {"role": "user", "content": question},
        ],
    )
    print(f"  [MANAGER] Plan: {plan.choices[0].message.content}")

    rech = worker_recherche(question)
    print(f"  [RECHERCHE] {rech[:60]}...")

    calc = worker_calcul(rech)
    print(f"  [CALCUL] {calc[:60]}...")

    redac = worker_redaction(rech, calc)
    print(f"  [REDACTION] {redac[:60]}...")

    return redac


# ── 4. Execution ──
print("=" * 70)
print("  DEMO ZERO-CODE : hierarchie detectee automatiquement")
print("  Aucun agent_span dans le code de l'application")
print("=" * 70)

question = "Cout total Tesla Model 3 sur 5 ans ?"
answer = manager(question)

print(f"\n  REPONSE: {answer}")
time.sleep(2)

# ── 5. Afficher le caller_chain de chaque span ──
print(f"\n{'='*70}")
print("  CALLER CHAINS DETECTEES AUTOMATIQUEMENT")
print(f"{'='*70}")

for i, s in enumerate(collected_spans):
    chain = s.get("caller_chain", [])
    model = s.get("model", "")
    tokens = f"{s.get('tokens_in', 0)}->{s.get('tokens_out', 0)} tok"
    print(f"\n  Span {i+1} [{model}] {tokens}")
    print(f"    caller_chain: {' -> '.join(chain) if chain else '(vide)'}")

# ── 6. Reconstruire l'arbre et l'afficher ──
print(f"\n{'='*70}")
print("  ARBRE RECONSTRUIT DEPUIS LES CALLER_CHAINS")
print(f"{'='*70}\n")


def build_tree_from_chains(spans):
    synth_agents = {}
    roots = []
    for s in spans:
        chain = s.get("caller_chain", [])
        parent_node = None
        for func_name in chain:
            if func_name not in synth_agents:
                synth_agents[func_name] = {
                    "span_type": "agent", "name": func_name,
                    "role": "auto-detected", "latency_ms": 0, "children": [],
                }
                if parent_node is None:
                    roots.append(synth_agents[func_name])
                else:
                    if synth_agents[func_name] not in parent_node["children"]:
                        parent_node["children"].append(synth_agents[func_name])
            parent_node = synth_agents[func_name]
        leaf = {**s, "children": []}
        if parent_node:
            parent_node["children"].append(leaf)
        else:
            roots.append(leaf)

    def _sum_latency(node):
        if node.get("role") == "auto-detected":
            total = 0
            for child in node.get("children", []):
                _sum_latency(child)
                total += child.get("latency_ms", 0)
            node["latency_ms"] = total
    for r in roots:
        _sum_latency(r)
    return roots


def print_tree(node, prefix="", is_last=True):
    connector = "+-  " if is_last else "|-- "
    stype = node.get("span_type", "?")
    name = node.get("name", "?")
    latency = node.get("latency_ms", 0)

    if stype == "agent":
        label = f"[AGENT] {name} ({latency}ms)"
    elif stype == "llm":
        model = node.get("model", "")
        tin = node.get("tokens_in", 0)
        tout = node.get("tokens_out", 0)
        label = f"[LLM] {model} {tin}->{tout} tok ({latency}ms)"
    else:
        label = f"[{stype}] {name} ({latency}ms)"

    print(f"{prefix}{connector}{label}")
    child_prefix = prefix + ("    " if is_last else "|   ")
    children = node.get("children", [])
    for i, child in enumerate(children):
        print_tree(child, child_prefix, i == len(children) - 1)


tree = build_tree_from_chains(collected_spans)
for i, root in enumerate(tree):
    print_tree(root, "  ", i == len(tree) - 1)

print(f"\n{'='*70}")
