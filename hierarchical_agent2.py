import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Multi-Agent Hiérarchique — Support Client
Architecture : Manager → [Classifier, Specialist/Escalation, Summarizer]
Fichier 100% autonome — aucune dépendance locale (pas de stats.py)
"""

import os, json, time, re
from dataclasses import dataclass
from typing import Optional
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

# ── Client Azure OpenAI ──────────────────────────
client = AzureOpenAI(
    api_key=llminsight_api_key,
    azure_endpoint=azure_endpoint,   # ex: https://mon-resource.openai.azure.com
    api_version=azure_api_version,
)
MODEL = azure_deployment  # nom du déploiement Azure
COST_IN  = 0.000150 / 1000
COST_OUT = 0.000600 / 1000


# ═══════════════════════════════════════════════════════════════════════
# Stats — intégrées directement (pas de fichier externe)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SpanRecord:
    agent_name: str
    role: str
    iteration: int
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0
    cost_usd: float = 0
    model: str = "gpt-4o-mini"
    output_format_ok: Optional[bool] = None

    @property
    def tokens_total(self): return self.tokens_in + self.tokens_out
    @property
    def token_efficiency(self):
        return round(self.tokens_out / self.tokens_in, 4) if self.tokens_in else 0
    @property
    def latency_class(self):
        if self.latency_ms < 1000: return "fast"
        if self.latency_ms < 3000: return "normal"
        return "slow"


class TraceStats:
    def __init__(self, trace_name: str, pattern: str):
        self.trace_name = trace_name
        self.pattern    = pattern.upper()
        self.spans: list[SpanRecord] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def record(self, agent_name, role, iteration, tokens_in=0, tokens_out=0,
               latency_ms=0, cost_usd=0, model="gpt-4o-mini", output_format_ok=None):
        sp = SpanRecord(agent_name=agent_name, role=role, iteration=iteration,
                        tokens_in=tokens_in, tokens_out=tokens_out,
                        latency_ms=latency_ms, cost_usd=cost_usd,
                        model=model, output_format_ok=output_format_ok)
        self.spans.append(sp)
        return sp

    def finish(self):
        self.end_time = time.time()

    @property
    def total_tokens_in(self):  return sum(s.tokens_in  for s in self.spans)
    @property
    def total_tokens_out(self): return sum(s.tokens_out for s in self.spans)
    @property
    def total_cost(self):       return sum(s.cost_usd   for s in self.spans)
    @property
    def total_latency_ms(self):
        return ((self.end_time or time.time()) - self.start_time) * 1000
    @property
    def n_llm_calls(self):  return len(self.spans)
    @property
    def n_agents(self):     return len(set(s.agent_name for s in self.spans))
    @property
    def n_iterations(self): return len(set(s.iteration for s in self.spans))

    def by_agent(self) -> dict:
        agents = {}
        for s in self.spans:
            if s.agent_name not in agents:
                agents[s.agent_name] = dict(calls=0, tokens_in=0, tokens_out=0,
                    latency_ms=0, cost_usd=0, role=s.role)
            a = agents[s.agent_name]
            a["calls"]      += 1
            a["tokens_in"]  += s.tokens_in
            a["tokens_out"] += s.tokens_out
            a["latency_ms"] += s.latency_ms
            a["cost_usd"]   += s.cost_usd
        return agents

    def by_iteration(self) -> dict:
        iters = {}
        for s in self.spans:
            i = s.iteration
            if i not in iters:
                iters[i] = dict(agents=[], tokens=0, latency_ms=0, cost_usd=0)
            iters[i]["agents"].append(s)
            iters[i]["tokens"]     += s.tokens_total
            iters[i]["latency_ms"] += s.latency_ms
            iters[i]["cost_usd"]   += s.cost_usd
        return iters

    def print_dashboard(self):
        W = 68
        agents = self.by_agent()
        iters  = self.by_iteration()
        COLORS = ["\033[94m","\033[91m","\033[93m","\033[95m","\033[92m","\033[96m"]
        RESET  = "\033[0m"
        cmap   = {n: COLORS[i % len(COLORS)] for i,n in enumerate(agents.keys())}

        print(f"\n{'═'*W}")
        print(f"  LLMInsight — Agent Trace   [{self.pattern}]")
        print(f"  {self.n_llm_calls} LLM calls · {self.n_agents} agents · "
              f"{self.n_iterations} iterations · "
              f"{self.total_tokens_in:,} in · {self.total_tokens_out:,} out · "
              f"${self.total_cost:.4f}")
        print(f"{'─'*W}")

        print(f"\n  {'Agent':<28} {'Calls':>5}  {'Tokens':>8}  {'Latency':>10}  {'Cost':>8}")
        print(f"  {'─'*28} {'─'*5}  {'─'*8}  {'─'*10}  {'─'*8}")
        for name, a in agents.items():
            c = cmap[name]
            tok = a["tokens_in"] + a["tokens_out"]
            role_tag = " [MGR]" if a["role"] == "manager" else ""
            print(f"  {c}■{RESET} {name:<26} {a['calls']:>5}  {tok:>8,}  "
                  f"{a['latency_ms']:>9,.0f}ms  \033[96m${a['cost_usd']:.4f}{RESET}{role_tag}")

        print(f"\n{'─'*W}  Agent Chain")
        for idx in sorted(iters.keys()):
            it = iters[idx]
            chain = " → ".join(f"{cmap.get(s.agent_name,'')}{s.agent_name}{RESET}" for s in it["agents"])
            print(f"  Iter {idx+1}  {chain}")

        print(f"\n{'─'*W}")
        for idx in sorted(iters.keys()):
            it = iters[idx]
            print(f"\n  \033[1mIteration {idx+1}\033[0m  \033[90m"
                  f"{len(it['agents'])} agents · {it['tokens']:,} tok · "
                  f"{it['latency_ms']:,.0f}ms · ${it['cost_usd']:.4f}\033[0m")
            for s in it["agents"]:
                c = cmap.get(s.agent_name,"")
                role_icon = "👔" if s.role == "manager" else "⚙️"
                fmt = f"  \033[92m✓ Format{RESET}" if s.output_format_ok else (
                      f"  \033[91m✗ Format{RESET}" if s.output_format_ok is not None else "")
                lat_c = "\033[92m" if s.latency_class=="fast" else (
                        "\033[93m" if s.latency_class=="normal" else "\033[91m")
                print(f"    {role_icon} {c}{s.agent_name:<26}{RESET}  "
                      f"{s.tokens_in:,}→{s.tokens_out:,} tok  "
                      f"{s.latency_ms:,.0f}ms  \033[96m${s.cost_usd:.4f}{RESET}"
                      f"{fmt}  {lat_c}[{s.latency_class}]{RESET}  "
                      f"\033[90meff:{s.token_efficiency:.4f}\033[0m")
        print(f"\n{'═'*W}")


# ═══════════════════════════════════════════════════════════════════════
# Helpers LLM
# ═══════════════════════════════════════════════════════════════════════

def parse_json(text: str, fallback: dict) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        try:    return json.loads(m.group()) if m else fallback
        except: return fallback


def _llm_call(system: str, user: str, agent_name: str, role: str,
              iteration: int, stats: TraceStats, tools=None) -> object:
    """Appel LLM bas niveau — capturé par llminsight.patch() + stats locales."""
    if hasattr(llminsight, "set_context"):
        llminsight.set_context(iteration=iteration)
    kwargs = dict(
        model=MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user",  "content":user}],
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.time() - t0) * 1000
    u = resp.usage
    tok_in  = u.prompt_tokens     if u else 0
    tok_out = u.completion_tokens if u else 0
    cost    = tok_in * COST_IN + tok_out * COST_OUT
    stats.record(agent_name=agent_name, role=role, iteration=iteration,
                 tokens_in=tok_in, tokens_out=tok_out,
                 latency_ms=latency_ms, cost_usd=cost, model=MODEL)
    return resp


# ═══════════════════════════════════════════════════════════════════════
# Workers
# ═══════════════════════════════════════════════════════════════════════

def worker_classifier(ticket: str, iteration: int, stats: TraceStats) -> dict:
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
        agent_name="Classifier Worker Agent", role="worker",
        iteration=iteration, stats=stats,
    )
    data = parse_json(
        resp.choices[0].message.content.strip(),
        {"category":"other","priority":"medium","language":"fr","needs_escalation":False}
    )
    stats.spans[-1].output_format_ok = all(k in data for k in ("category","priority","needs_escalation"))
    return data


def worker_specialist(ticket: str, clf: dict, iteration: int, stats: TraceStats) -> str:
    print(f"    ├─ [Specialist Worker] iter {iteration}")
    roles = {"billing":"expert facturation","technical":"expert technique N2",
             "account":"expert comptes","shipping":"expert logistique","other":"agent généraliste"}
    role_label = roles.get(clf.get("category","other"), "agent support")
    lang_map = {"fr":"Réponds en français.","en":"Reply in English.","ar":"أجب باللغة العربية."}
    lang = lang_map.get(clf.get("language","fr"), "Réponds dans la langue du client.")
    resp = _llm_call(
        system=f"Tu es {role_label}. {lang} Sois professionnel et empathique.",
        user=f"Ticket: {ticket}\nContexte: {json.dumps(clf,ensure_ascii=False)}",
        agent_name="Specialist Worker Agent", role="worker",
        iteration=iteration, stats=stats,
    )
    result = resp.choices[0].message.content.strip()
    stats.spans[-1].output_format_ok = len(result) > 20
    return result


def worker_escalation(ticket: str, clf: dict, iteration: int, stats: TraceStats) -> str:
    print(f"    ├─ [Escalation Worker] iter {iteration} ⚠️")
    resp = _llm_call(
        system="""Tu es responsable support senior. Ticket prioritaire.
Rédige une réponse avec : accusé immédiat, délai précis, référent dédié (fictif).""",
        user=f"Ticket urgent: {ticket}\nPriorité: {clf.get('priority')}",
        agent_name="Escalation Worker Agent", role="worker",
        iteration=iteration, stats=stats,
    )
    result = resp.choices[0].message.content.strip()
    stats.spans[-1].output_format_ok = len(result) > 20
    return result


def worker_summarizer(ticket: str, clf: dict, response: str,
                      escalated: bool, iteration: int, stats: TraceStats) -> dict:
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
        agent_name="Summary Worker Agent", role="worker",
        iteration=iteration, stats=stats,
    )
    data = parse_json(
        resp.choices[0].message.content.strip(),
        {"issue":ticket[:80],"resolution":response[:80],"escalated":escalated}
    )
    stats.spans[-1].output_format_ok = all(k in data for k in ("ticket_id","issue","resolution"))
    return data


# ═══════════════════════════════════════════════════════════════════════
# Manager (orchestrateur)
# ═══════════════════════════════════════════════════════════════════════

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

WORKER_FN = {
    "worker_classifier": worker_classifier,
    "worker_specialist": worker_specialist,
    "worker_escalation": worker_escalation,
    "worker_summarizer": worker_summarizer,
}


def agent_manager(ticket: str, stats: TraceStats) -> dict:
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
        if hasattr(llminsight, "set_context"):
            llminsight.set_context(iteration=iteration)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=WORKER_TOOLS, tool_choice="auto",
        )
        latency_ms = (time.time() - t0) * 1000
        u = resp.usage
        tok_in  = u.prompt_tokens     if u else 0
        tok_out = u.completion_tokens if u else 0
        cost    = tok_in * COST_IN + tok_out * COST_OUT
        stats.record(agent_name="Manager Agent", role="manager", iteration=iteration,
                     tokens_in=tok_in, tokens_out=tok_out,
                     latency_ms=latency_ms, cost_usd=cost, model=MODEL,
                     output_format_ok=True)

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
                result = worker_classifier(fn_args["ticket"], iteration, stats)
                context["worker_classifier"] = result
                result_str = json.dumps(result, ensure_ascii=False)

            elif fn_name == "worker_specialist":
                clf = parse_json(fn_args.get("classification","{}"), clf_obj)
                result = worker_specialist(fn_args["ticket"], clf, iteration, stats)
                context["worker_specialist"] = result
                result_str = result

            elif fn_name == "worker_escalation":
                clf = parse_json(fn_args.get("classification","{}"), clf_obj)
                result = worker_escalation(fn_args["ticket"], clf, iteration, stats)
                context["worker_escalation"] = result
                result_str = result

            elif fn_name == "worker_summarizer":
                clf = parse_json(fn_args.get("classification","{}"), clf_obj)
                resp_text = fn_args.get("response",
                    context.get("worker_specialist") or context.get("worker_escalation",""))
                escalated = "worker_escalation" in context
                result = worker_summarizer(fn_args["ticket"], clf, resp_text, escalated, iteration, stats)
                context["worker_summarizer"] = result
                result_str = json.dumps(result, ensure_ascii=False)
            else:
                result_str = ""

            messages.append({"role":"tool","tool_call_id":tc.id,"content":result_str})

        iteration += 1

    return context


# ═══════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(ticket: str) -> dict:
    stats = TraceStats("support-hierarchical", "HIERARCHICAL")

    print(f"\n{'═'*60}")
    print(f"  PIPELINE HIÉRARCHIQUE — Support Client")
    print(f"  Ticket : {ticket[:70]}{'...' if len(ticket)>70 else ''}")
    print(f"{'─'*60}")

    context = agent_manager(ticket, stats)
    stats.finish()

    escalated = "worker_escalation" in context
    response  = context.get("worker_specialist") or context.get("worker_escalation","")
    summary   = context.get("worker_summarizer", {})

    stats.print_dashboard()

    return {
        "ticket": ticket,
        "classification": context.get("worker_classifier", {}),
        "response": response,
        "summary": summary,
        "escalated": escalated,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

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
