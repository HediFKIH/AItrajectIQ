import os
import sys, io

from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Demo: Application agentique hiérarchique avec AItrajectIQ.

Architecture:
    Manager (décide quel worker appeler)
    ├── Worker Recherche (simule une recherche web + synthèse)
    ├── Worker Calcul (simule un calcul financier)
    └── Worker Rédaction (rédige la réponse finale)

Le manager reçoit une question, décide quel(s) worker(s) mobiliser,
délègue, puis synthétise la réponse finale.
"""

import time
import json
import random
load_dotenv()

PROJECT_ID = os.getenv("AITRAJECTIQ_PROJECT_ID")
BASE = os.getenv("AITRAJECTIQ_BASE_URL", "http://localhost:8000")
# The project key of AItrajectIQ (atq_...); Bedrock itself is mocked below.
AITRAJECTIQ_API_KEY = os.getenv("AITRAJECTIQ_API_KEY")
if not AITRAJECTIQ_API_KEY:
    raise SystemExit("AITRAJECTIQ_API_KEY manquant : crée une clé projet (atq_...) dans le dashboard, page « Projets et clés ».")
MODEL_ID = os.getenv("MODEL_ID")

# ── 2. Initialiser AItrajectIQ ──
import aitrajectiq

aitrajectiq.init(project_id=PROJECT_ID, api_key=AITRAJECTIQ_API_KEY, base_url=BASE)

aitrajectiq.patch()

# ── 1. Mock du client AWS Bedrock (Anthropic Claude) ──

call_counter = 0

def fake_bedrock_invoke(**kwargs):
    """Mock pour AWS Bedrock invoke_model — simule une réponse Claude."""
    global call_counter
    call_counter += 1

    body = json.loads(kwargs.get("body", "{}"))
    messages = body.get("messages", [])
    system = body.get("system", "")
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

    time.sleep(random.uniform(0.1, 0.4))

    if "manager" in system.lower():
        answer = json.dumps({
            "plan": "1) Recherche web, 2) Calcul financier, 3) Rédaction finale",
            "workers": ["recherche", "calcul", "redaction"],
        })
    elif "recherche" in system.lower():
        answer = "Résultats de recherche: Tesla Model 3 prix moyen 42 000€ en 2026. Autonomie 500km. Concurrent: BYD Seal à 35 000€."
    elif "calcul" in system.lower():
        answer = "Analyse financière: Coût total sur 5 ans = 52 300€ (achat 42k + assurance 4.5k + énergie 3.8k + entretien 2k). Économie vs thermique: 18 700€."
    elif "redaction" in system.lower() or "rédaction" in system.lower():
        answer = ("Synthèse: La Tesla Model 3 à 42 000€ est un choix rentable sur 5 ans "
                  "avec une économie de 18 700€ par rapport à un véhicule thermique équivalent. "
                  "Le concurrent principal est le BYD Seal à 35 000€.")
    else:
        answer = f"Réponse générique à: {user_msg[:50]}"

    tokens_in = random.randint(200, 1500)
    tokens_out = random.randint(50, 400)

    # Format de réponse Anthropic Claude sur Bedrock
    response_body = json.dumps({
        "id": f"msg_{random.randint(100000, 999999)}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
        "model": MODEL_ID,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        },
    })

    # Simule la réponse de l'API Bedrock
    class FakeStreamBody:
        def read(self):
            return response_body.encode("utf-8")

    # Retourne un dictionnaire simulant la réponse Bedrock
    return {"body": FakeStreamBody()}


# ── 3. Création du mock client AWS Bedrock ──
# On crée un mock client et on l'instrumente avec wrap_bedrock_client()
# pour que AItrajectIQ capture les spans LLM même avec la simulation

class MockBedrockClient:
    def invoke_model(self, **kwargs):
        return fake_bedrock_invoke(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

client = MockBedrockClient()
aitrajectiq.wrap_bedrock_client(client)

# Collecte des spans pour l'affichage local
collected_spans = []
from aitrajectiq import client as _cli
_orig_post_async = _cli.post_async


def _capturing_post_async(endpoint, payload):
    if "/ingest/span" in endpoint:
        collected_spans.append(payload.copy())
    _orig_post_async(endpoint, payload)


_cli.post_async = _capturing_post_async


def _call_bedrock(system_prompt: str, user_message: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Helper pour appeler Claude via AWS Bedrock et parser la réponse."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ],
    })

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def simulate_web_search(query: str) -> str:
    time.sleep(random.uniform(0.05, 0.15))
    return f"Résultats pour '{query}': 3 articles trouvés."


def agent_manager(question: str) -> str:
    """Agent manager: décide, délègue, synthétise."""

    iteration = 0

    with aitrajectiq.agent_span("manager", role="manager", metadata={"question": question}):

        print("\n  [MANAGER] Analyse de la question et planification...")
        aitrajectiq.set_agent_context(iteration=iteration)
        plan = _call_bedrock(
            system_prompt="Tu es un MANAGER. Analyse la question et planifie les workers à appeler.",
            user_message=question,
            max_tokens=512,
            temperature=0.3,
        )
        print(f"  [MANAGER] Plan: {plan[:80]}...")

        # Worker 1: Recherche
        with aitrajectiq.agent_span("worker-recherche", role="worker"):
            print("  [WORKER-RECHERCHE] Recherche en cours...")

            with aitrajectiq.tool_span("web_search", {"query": question}) as tool:
                search_results = simulate_web_search(question)
                tool.set_output(search_results)
            print(f"  [WORKER-RECHERCHE] Outil: {search_results[:60]}...")

            iteration += 1
            aitrajectiq.set_agent_context(iteration=iteration)
            recherche_result = _call_bedrock(
                system_prompt="Tu es un worker RECHERCHE. Synthétise les résultats.",
                user_message=f"Question: {question}\nRésultats: {search_results}",
                max_tokens=1024,
                temperature=0.5,
            )
            print(f"  [WORKER-RECHERCHE] Résultat: {recherche_result[:60]}...")

        # Worker 2: Calcul
        with aitrajectiq.agent_span("worker-calcul", role="worker"):
            print("  [WORKER-CALCUL] Analyse financière en cours...")
            iteration += 1
            aitrajectiq.set_agent_context(iteration=iteration)
            calcul_result = _call_bedrock(
                system_prompt="Tu es un worker CALCUL. Fais l'analyse financière.",
                user_message=f"Données: {recherche_result}",
                max_tokens=1024,
                temperature=0.2,
            )
            print(f"  [WORKER-CALCUL] Résultat: {calcul_result[:60]}...")

        # Worker 3: Rédaction
        with aitrajectiq.agent_span("worker-redaction", role="worker"):
            print("  [WORKER-REDACTION] Rédaction de la réponse finale...")
            iteration += 1
            aitrajectiq.set_agent_context(iteration=iteration)
            final_answer = _call_bedrock(
                system_prompt="Tu es un worker RÉDACTION. Rédige une réponse claire et structurée.",
                user_message=f"Recherche: {recherche_result}\nCalcul: {calcul_result}",
                max_tokens=1024,
                temperature=0.7,
            )
            print(f"  [WORKER-REDACTION] Réponse: {final_answer[:60]}...")

        # Synthèse manager
        print("  [MANAGER] Validation finale...")
        return final_answer


# ── 4. Exécution (tout dans UNE seule trace) ──
print("=" * 70)
print("  DEMO AGENT HIÉRARCHIQUE avec AItrajectIQ")
print("  Architecture: Manager -> [Recherche, Calcul, Redaction]")
print("  Provider: Anthropic Claude via AWS Bedrock")
print("=" * 70)

question = "Quel est le coût total de possession d'une Tesla Model 3 sur 5 ans ?"

with aitrajectiq.trace("agent-hierarchy-demo", {"user": "demo-user", "architecture": "hierarchical"}) as trace_id:
    print(f"\n  Trace ID: {trace_id}")

    # Test rapide de connexion (dans la trace pour éviter une auto-trace séparée)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Reply with exactly: API key is valid!"}
        ]
    })
    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        reply  = result["content"][0]["text"].strip()
        print(f"✅  Success!")
        print(f"    Model    : {MODEL_ID}")
        print(f"    Response : {reply}")
        print(f"    Tokens   : {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out")
    except Exception as e:
        print(f"❌  Erreur : {e}")

    answer = agent_manager(question)

print(f"\n{'='*70}")
print(f"  QUESTION: {question}")
print(f"  RÉPONSE:  {answer}")
print(f"{'='*70}")

# ── 5. Attendre que les spans soient envoyés ──
print("\n  Envoi des spans en cours...")
if not aitrajectiq.flush(timeout=10):
    print("  ⚠️  Certains spans n'ont pas pu être envoyés dans le délai.")

# ── 6. Construire et afficher le graphe ──
print(f"\n{'='*70}")
print("  GRAPHE DE LA TRACE (arbre hiérarchique)")
print(f"{'='*70}\n")


def build_tree(spans):
    """Construit l'arbre a partir des parent_span_id (mode manuel avec agent_span).
    Si aucune hierarchie manuelle n'est detectee, reconstruit l'arbre
    automatiquement a partir des caller_chain (mode zero-code).
    """
    by_id = {}
    roots = []
    has_manual_hierarchy = any(s.get("parent_span_id") for s in spans)

    if has_manual_hierarchy:
        for s in spans:
            sid = s.get("span_id")
            if sid:
                by_id[sid] = {**s, "children": []}
        for s in spans:
            sid = s.get("span_id")
            psid = s.get("parent_span_id")
            if sid and psid and psid in by_id:
                by_id[psid]["children"].append(by_id[sid])
            elif sid:
                roots.append(by_id[sid])
        return roots

    # Mode zero-code : reconstruire depuis caller_chain
    synth_agents = {}

    for s in spans:
        chain = s.get("caller_chain", [])
        if not chain:
            roots.append({**s, "children": []})
            continue

        parent_node = None
        for func_name in chain:
            if func_name not in synth_agents:
                synth_agents[func_name] = {
                    "span_type": "agent",
                    "name": func_name,
                    "agent_name": func_name,
                    "role": "auto-detected",
                    "span_id": f"auto-{func_name}",
                    "latency_ms": 0,
                    "children": [],
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

    # Calculer la latence des agents synthetiques (somme de leurs enfants)
    def _sum_latency(node):
        if node.get("span_type") == "agent" and node.get("role") == "auto-detected":
            total = 0
            for child in node.get("children", []):
                _sum_latency(child)
                total += child.get("latency_ms", 0)
            node["latency_ms"] = total

    for r in roots:
        _sum_latency(r)

    return roots


def print_tree(node, prefix="", is_last=True):
    """Affiche un nœud de l'arbre avec des caractères de dessin."""
    connector = "└── " if is_last else "├── "
    span_type = node.get("span_type", "?")
    name = node.get("name", "?")
    latency = node.get("latency_ms", 0)
    role = node.get("role", "")

    if span_type == "agent":
        icon = "[AGENT]"
        extra = f" (role={role})" if role else ""
    elif span_type == "llm":
        icon = "[LLM]"
        model = node.get("model", "")
        tin = node.get("tokens_in", 0)
        tout = node.get("tokens_out", 0)
        extra = f" [{model}] {tin}->{tout} tokens"
    elif span_type == "tool":
        icon = "[TOOL]"
        extra = f" input={json.dumps(node.get('tool_input', {}), ensure_ascii=False)[:50]}"
    else:
        icon = "[?]"
        extra = ""

    status_icon = "[OK]" if node.get("status") != "failed" and not node.get("error") else "[FAIL]"

    print(f"{prefix}{connector}{icon} {span_type}:{name}{extra}  ({latency}ms) {status_icon}")

    child_prefix = prefix + ("    " if is_last else "│   ")
    children = node.get("children", [])
    for i, child in enumerate(children):
        print_tree(child, child_prefix, i == len(children) - 1)


tree = build_tree(collected_spans)

print(f"  trace: agent-hierarchy-demo  (trace_id={trace_id})")
print(f"  │")
for i, root in enumerate(tree):
    print_tree(root, "  ", i == len(tree) - 1)

# ── 7. Statistiques ──
print(f"\n{'='*70}")
print("  STATISTIQUES")
print(f"{'='*70}")

agent_spans = [s for s in collected_spans if s.get("span_type") == "agent"]
llm_spans = [s for s in collected_spans if s.get("span_type") == "llm"]
tool_spans = [s for s in collected_spans if s.get("span_type") == "tool"]

print(f"  Spans total:    {len(collected_spans)}")
print(f"    - Agent:      {len(agent_spans)}")
print(f"    - LLM:        {len(llm_spans)}")
print(f"    - Tool:       {len(tool_spans)}")

total_tokens_in = sum(s.get("tokens_in", 0) for s in llm_spans)
total_tokens_out = sum(s.get("tokens_out", 0) for s in llm_spans)
print(f"  Tokens total:   {total_tokens_in} in / {total_tokens_out} out")

max_depth = 0


def measure_depth(spans, parent_id=None, depth=0):
    global max_depth
    max_depth = max(max_depth, depth)
    for s in spans:
        if s.get("parent_span_id") == parent_id and s.get("span_id"):
            measure_depth(spans, s["span_id"], depth + 1)


measure_depth(collected_spans)
print(f"  Profondeur max: {max_depth} niveaux")

for agent in agent_spans:
    agent_name = agent.get("agent_name", "?")
    agent_role = agent.get("role", "?")
    agent_latency = agent.get("latency_ms", 0)
    agent_sid = agent.get("span_id")
    child_llms = [s for s in llm_spans if s.get("parent_span_id") == agent_sid]
    child_tokens = sum(s.get("tokens_in", 0) + s.get("tokens_out", 0) for s in child_llms)
    print(f"\n  Agent '{agent_name}' (role={agent_role}):")
    print(f"    Latence:      {agent_latency}ms")
    print(f"    Appels LLM:   {len(child_llms)}")
    print(f"    Tokens:       {child_tokens}")

print(f"\n{'='*70}")

# ── 8. Generer le graphe HTML interactif ──
def _js_safe(text):
    """Echappe une string pour l'injecter dans du JavaScript."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")

def generate_html_graph(spans, trace_id_val, output_path="trace_graph.html"):
    nodes_list = []
    edges_list = []
    for s in spans:
        sid = s.get("span_id", "")
        stype = s.get("span_type", "?")
        name = s.get("name", "?")
        latency = s.get("latency_ms", 0)
        role = s.get("role", "")
        tokens_in = s.get("tokens_in", 0)
        tokens_out = s.get("tokens_out", 0)
        model = s.get("model", "")
        error = s.get("error", "")
        tool_input_str = ""
        if s.get("tool_input"):
            raw = json.dumps(s["tool_input"], ensure_ascii=False)
            tool_input_str = raw[:60].replace('"', "'")

        if stype == "agent":
            color = "#4CAF50" if role == "manager" else "#2196F3"
            shape = "box"
            label = f"{name}\n({role})\n{latency}ms"
        elif stype == "llm":
            color = "#FF9800"
            shape = "ellipse"
            label = f"{model}\n{tokens_in}->{tokens_out} tok\n{latency}ms"
        elif stype == "tool":
            color = "#9C27B0"
            shape = "diamond"
            label = f"{name}\n{latency}ms"
        else:
            color = "#607D8B"
            shape = "dot"
            label = name

        if error:
            color = "#F44336"

        title_parts = [f"Type: {stype}", f"Name: {name}", f"Span ID: {sid}", f"Latency: {latency}ms"]
        if role:
            title_parts.append(f"Role: {role}")
        if model:
            title_parts.append(f"Model: {model}")
        if tokens_in:
            title_parts.append(f"Tokens in: {tokens_in}")
        if tokens_out:
            title_parts.append(f"Tokens out: {tokens_out}")
        if tool_input_str:
            title_parts.append(f"Input: {tool_input_str}")
        if error:
            title_parts.append(f"Error: {error}")
        title = "\n".join(title_parts)

        nodes_list.append({
            "id": sid, "label": label, "color": color, "shape": shape,
            "title": title,
            "font": {"size": 12, "color": "#fff"},
            "borderWidth": 2, "shadow": True,
        })

        psid = s.get("parent_span_id")
        if psid:
            edges_list.append({
                "from": psid, "to": sid, "arrows": "to",
                "color": {"color": "#999"}, "width": 2,
            })

    n_agents = len([s for s in spans if s.get("span_type") == "agent"])
    n_llm = len([s for s in spans if s.get("span_type") == "llm"])
    n_tools = len([s for s in spans if s.get("span_type") == "tool"])
    total_tok = sum(s.get("tokens_in", 0) + s.get("tokens_out", 0) for s in spans)
    max_lat = max((s.get("latency_ms", 0) for s in spans if s.get("span_type") == "agent" and s.get("role") == "manager"), default=0)

    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>AItrajectIQ - Trace Graph</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }
        #header {
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 20px 30px; border-bottom: 2px solid #e94560;
        }
        #header h1 { font-size: 22px; color: #e94560; margin-bottom: 5px; }
        #header p { font-size: 13px; color: #aaa; }
        #graph { width: 100%%; height: calc(100vh - 180px); }
        #legend {
            display: flex; gap: 20px; padding: 12px 30px;
            background: #16213e; border-bottom: 1px solid #333;
        }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
        .legend-dot { width: 14px; height: 14px; border-radius: 3px; }
        #stats {
            position: absolute; bottom: 15px; right: 20px; background: rgba(22,33,62,0.95);
            border: 1px solid #444; border-radius: 8px; padding: 15px; font-size: 12px;
            min-width: 200px;
        }
        #stats h3 { color: #e94560; margin-bottom: 8px; font-size: 14px; }
        .row { display: flex; justify-content: space-between; margin: 3px 0; }
        .val { color: #4fc3f7; font-weight: bold; }
    </style>
</head>
<body>
    <div id="header">
        <h1>AItrajectIQ - Trace Hierarchy Graph</h1>
        <p>Trace: %s | Spans: %d</p>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#4CAF50"></div> Manager</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2196F3"></div> Worker</div>
        <div class="legend-item"><div class="legend-dot" style="background:#FF9800;border-radius:50%%"></div> LLM Call</div>
        <div class="legend-item"><div class="legend-dot" style="background:#9C27B0;transform:rotate(45deg)"></div> Tool</div>
        <div class="legend-item"><div class="legend-dot" style="background:#F44336"></div> Error</div>
    </div>
    <div id="graph"></div>
    <div id="stats">
        <h3>Stats</h3>
        <div class="row"><span>Agents</span><span class="val">%d</span></div>
        <div class="row"><span>LLM calls</span><span class="val">%d</span></div>
        <div class="row"><span>Tools</span><span class="val">%d</span></div>
        <div class="row"><span>Total tokens</span><span class="val">%d</span></div>
        <div class="row"><span>Total latency</span><span class="val">%dms</span></div>
    </div>
    <script>
        var nodes = new vis.DataSet(NODES_DATA);
        var edges = new vis.DataSet(EDGES_DATA);
        var container = document.getElementById('graph');
        var data = { nodes: nodes, edges: edges };
        var options = {
            layout: {
                hierarchical: {
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 120,
                    nodeSpacing: 180,
                }
            },
            physics: false,
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true,
            },
            nodes: { margin: 10 },
        };
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>""" % (trace_id_val, len(spans), n_agents, n_llm, n_tools, total_tok, max_lat)

    nodes_json = json.dumps(nodes_list, ensure_ascii=False)
    edges_json = json.dumps(edges_list, ensure_ascii=False)
    html_content = html_content.replace("NODES_DATA", nodes_json)
    html_content = html_content.replace("EDGES_DATA", edges_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n  Graphe HTML genere: {output_path}")
    return output_path


graph_path = generate_html_graph(collected_spans, trace_id, "trace_graph.html")
print(f"  Ouvrez {graph_path} dans un navigateur pour voir le graphe interactif.")
