"""
Recommendations computed from a project's own traffic:

    model        simple calls sent to a premium model, complex calls failing on a small one
    caching      a long system prompt resent on every call without prompt caching
    prompt       span metrics (hallucination, relevance, format...) below a healthy level,
                 long contexts for short answers, high temperatures
    agent        trace metrics (loops, tool errors, completion...) below a healthy level
    harness      what to change around the agent: hooks, ignored files, terminal, subagents,
                 MCP servers, permissions, tool outputs (services/harness.py)
    reliability  models whose calls fail often

Savings are recomputed from the recorded calls and the price table (utils/cost.py), over the
analysed period only. Models without a known price get no estimate.

Task complexity is estimated per LLM call from the context size, the answer length and the
tools used in its trace; see classify_complexity.

recommendations() covers a period of a project; trace_recommendations() a single trace, step
by step (which call could use a cheaper model, repeated tool calls, tool errors, context
growth, the trace's own metric verdicts).

Each recommendation carries a ready-to-copy `fix` when one exists (an instruction, a Cursor
rule, a code change), and both entry points return `prompts`: the fired instructions written
where the agent reads them (Cursor rule, CLAUDE.md, or the most used system prompt).
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from beanie import PydanticObjectId

from evaluation.actions import _arguments, is_agentic, tool_failed
from evaluation.runner import JUDGED_METRICS, judge_tag
from models.metric import MetricName
from models.span import Span, SpanType
from models.trace import Trace
from services import evaluations, harness, insights, pricing
from utils.cost import compute_cost, price_key, price_keys, price_of

MAX_CALLS = 20_000

SIMPLE, MEDIUM, COMPLEX = "simple", "moyenne", "complexe"
COMPLEXITIES = (SIMPLE, MEDIUM, COMPLEX)

SIMPLE_MAX_TOKENS_IN = 4_000
SIMPLE_MAX_TOKENS_OUT = 500
COMPLEX_MIN_TOKENS_IN = 30_000
COMPLEX_MIN_TOKENS_OUT = 2_000
COMPLEX_MIN_TOOL_CALLS = 5
AGENTIC_SOURCES = frozenset({"cursor", "claude-code"})

MIN_CALLS = 5
MIN_EVALUATED = 5
GOOD_QUALITY = 0.8
POOR_QUALITY = 0.6
UPGRADE_BAD_RATE = 0.25
CACHE_MIN_PROMPT_CHARS = 4_000      # about 1 000 tokens, OpenAI's minimum cacheable prefix
CACHE_MAX_CACHED_SHARE = 0.1
LONG_CONTEXT_MIN_TOKENS_IN = 8_000
LONG_CONTEXT_RATIO = 50
HIGH_TEMPERATURE = 1.0
FAILURE_MIN_CALLS = 10
FAILURE_RATE = 0.05

# Same-provider model one tier down, for calls that don't need the bigger one.
CHEAPER_MODEL = {
    "gpt-5.4": "gpt-5.4-mini",
    "gpt-4.1": "gpt-4.1-mini",
    "gpt-4o": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4o-mini",
    "gpt-4": "gpt-4o-mini",
    "o1": "o4-mini",
    "o3": "o4-mini",
    "claude-opus-4.7": "claude-sonnet-4.6",
    "claude-opus-4.6": "claude-sonnet-4.6",
    "claude-opus-4.5": "claude-sonnet-4.6",
    "claude-opus-4.1": "claude-sonnet-4.6",
    "claude-opus-4": "claude-sonnet-4.6",
    "claude-3-opus": "claude-sonnet-4.6",
    "claude-sonnet-4.6": "claude-haiku-4.5",
    "claude-sonnet-4.5": "claude-haiku-4.5",
    "claude-sonnet-4": "claude-haiku-4.5",
    "claude-3-7-sonnet": "claude-haiku-4.5",
    "claude-3-5-sonnet": "claude-haiku-4.5",
    "claude-3-sonnet": "claude-haiku-4.5",
    "gemini-2.5-pro": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-flash",
    "mistral-large": "mistral-small",
    "mistral-medium": "mistral-small",
    "command-r-plus": "command-r",
    "command-a": "command-r",
    "deepseek-reasoner": "deepseek-chat",
    "llama-3.3-70b": "llama-3.1-8b",
    "llama-3.1-70b": "llama-3.1-8b",
}

QUALITY_METRICS = frozenset(m.value for m in (
    MetricName.answer_relevance,
    MetricName.general_faithfulness,
    MetricName.contextual_faithfulness,
    MetricName.completeness,
    MetricName.output_format_compliance,
    MetricName.is_json,
))

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Call:
    trace_id: str
    model: str
    tokens_in: int
    tokens_out: int
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    failed: bool = False
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    quality: Optional[float] = None
    agentic: bool = False
    complexity: str = SIMPLE

    @property
    def key(self) -> Optional[str]:
        return price_key(self.model)

    def cost_with(self, model: str) -> float:
        return compute_cost(model, self.tokens_in, self.tokens_out, self.cache_read, self.cache_write)


@dataclass
class Recommendation:
    id: str
    category: str
    severity: str
    title: str
    detail: str
    action: str
    savings_usd: Optional[float] = None
    evidence: dict = field(default_factory=dict)
    examples: list = field(default_factory=list)
    spans: list = field(default_factory=list)       # steps of one trace: {span_id, step, label, reason}
    fix: Optional[dict] = None                      # text to copy: {kind, label, content, file}
    extra_fixes: list = field(default_factory=list) # more files of the same fix (a hook's script)


# ---------------------------------------------------------------------------
# Ready-to-copy fixes
# ---------------------------------------------------------------------------

CURSOR_RULE_FILE = ".cursor/rules/aitrajectiq.mdc"
CLAUDE_RULE_FILE = "CLAUDE.md"


def agent_target(sources) -> str:
    """Where an agent's instructions live: a Cursor rule, CLAUDE.md, or the system prompt of
    an agent built with the SDK. `sources` are the trace sources of the agent runs."""
    counts = Counter(s for s in sources if s in AGENTIC_SOURCES)
    return counts.most_common(1)[0][0] if counts else "sdk"


def instructions_fix(lines: list[str], target: str) -> Optional[dict]:
    """Instructions to add where the target agent reads them."""
    if not lines:
        return None
    bullets = "\n".join(f"- {line}" for line in lines)
    if target == "cursor":
        return {"kind": "rule", "label": f"Paste into {CURSOR_RULE_FILE}", "file": CURSOR_RULE_FILE,
                "content": f"---\ndescription: Instructions suggested by AItrajectIQ\nalwaysApply: true\n---\n\n{bullets}\n"}
    if target == "claude-code":
        return {"kind": "rule", "label": f"Add to {CLAUDE_RULE_FILE}", "file": CLAUDE_RULE_FILE,
                "content": f"## Instructions\n\n{bullets}\n"}
    return {"kind": "prompt", "label": "Add to the system prompt", "file": None, "content": bullets}


def _instructions_action(target: str) -> str:
    return {
        "cursor": "Add this rule to the project: Cursor reads it on every request.",
        "claude-code": "Add these instructions to CLAUDE.md: Claude Code reads them in every session.",
    }.get(target, "Add these instructions to the system prompt.")


def _code_fix(label: str, content: str) -> dict:
    return {"kind": "code", "label": label, "file": None, "content": content}


def _routing_code(small: str, big: str) -> str:
    return (
        "def choose_model(prompt: str, tools: list | None = None) -> str:\n"
        f"    # short request (about {SIMPLE_MAX_TOKENS_IN} tokens) with no tools\n"
        f"    if len(prompt) <= {SIMPLE_MAX_TOKENS_IN * 4} and not tools:\n"
        f"        return \"{small}\"\n"
        f"    return \"{big}\"\n"
    )


RETRY_CODE = (
    "from tenacity import retry, stop_after_attempt, wait_exponential\n\n"
    "@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))\n"
    "def call_model(**kwargs):\n"
    "    return client.chat.completions.create(**kwargs)\n"
)


# ---------------------------------------------------------------------------
# Cheaper model
# ---------------------------------------------------------------------------

_CLAUDE = re.compile(r"^claude-(opus|sonnet|haiku)-(\d+(?:\.\d+)?)$")
_GPT = re.compile(r"^gpt-\d+(?:\.\d+)?$")
_GEMINI_PRO = re.compile(r"^(gemini-\d+(?:\.\d+)?)-pro$")
_CLAUDE_TIER_BELOW = {"opus": "sonnet", "sonnet": "haiku"}


def _family_cheaper(key: str) -> Optional[str]:
    """For models CHEAPER_MODEL doesn't list (new releases priced from the web): the latest
    Claude of the tier below, the -mini of a GPT, the -flash of a Gemini Pro."""
    keys = price_keys()
    match = _CLAUDE.match(key)
    if match and match.group(1) in _CLAUDE_TIER_BELOW:
        tier = _CLAUDE_TIER_BELOW[match.group(1)]
        found = [(tuple(int(p) for p in m.group(2).split(".")), k)
                 for k in keys if (m := _CLAUDE.match(k)) and m.group(1) == tier]
        return max(found)[1] if found else None
    if _GPT.match(key):
        return f"{key}-mini" if f"{key}-mini" in keys else None
    match = _GEMINI_PRO.match(key)
    if match:
        return f"{match.group(1)}-flash" if f"{match.group(1)}-flash" in keys else None
    return None


def alternative_status(key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """The cheaper model of `key`, or None and why there is none."""
    if not key:
        return None, "Unknown price: no savings can be estimated."
    alt = CHEAPER_MODEL.get(key) or _family_cheaper(key)
    if not alt:
        return None, "No smaller model known from this provider."
    own, other = price_of(key), price_of(alt)
    if not other:
        return None, f"{alt} has no known price."
    if not own or other["in"] >= own["in"]:
        return None, f"{alt} is not cheaper."
    return alt, None


def cheaper_model(key: Optional[str]) -> Optional[str]:
    """Same-provider model one tier down, only when it is priced and cheaper."""
    return alternative_status(key)[0]


# ---------------------------------------------------------------------------
# Call classification
# ---------------------------------------------------------------------------

def classify_complexity(tokens_in: int, tokens_out: int, agentic: bool, tool_calls: int) -> str:
    """A call inside an agent run is never simple: its prompt is one step of a larger task."""
    if agentic:
        if tool_calls >= COMPLEX_MIN_TOOL_CALLS or tokens_in >= COMPLEX_MIN_TOKENS_IN:
            return COMPLEX
        return MEDIUM
    if tokens_in >= COMPLEX_MIN_TOKENS_IN or tokens_out >= COMPLEX_MIN_TOKENS_OUT:
        return COMPLEX
    if tokens_in <= SIMPLE_MAX_TOKENS_IN and tokens_out <= SIMPLE_MAX_TOKENS_OUT:
        return SIMPLE
    return MEDIUM


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(part.get("text") or part.get("content")) if isinstance(part, dict) else _text(part)
                         for part in content)
    return ""


def _numeric(value) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def quality_score(metrics: list, tag: Optional[str]) -> Optional[float]:
    """Average of the answer-quality metrics of a call, skipping verdicts of an older judge."""
    values = []
    for m in metrics or []:
        name = m.get("name")
        if name not in QUALITY_METRICS:
            continue
        if tag and name in JUDGED_METRICS and m.get("judge") != tag:
            continue
        value = _numeric(m.get("value"))
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def build_calls(span_docs: list, trace_docs: list, tag: Optional[str] = None) -> list[Call]:
    """LLM span documents (with `first_message`) and their traces, as classified calls."""
    traces = {str(t["_id"]): t for t in trace_docs}
    calls = []
    for doc in span_docs:
        tokens_in = doc.get("tokens_in") or 0
        tokens_out = doc.get("tokens_out") or 0
        failed = doc.get("status") in ("failed", "error") or bool(doc.get("error"))
        if not tokens_in and not tokens_out and not failed:
            continue
        trace = traces.get(str(doc["trace_id"]), {})
        tool_calls = trace.get("total_tool_spans") or 0
        agentic = tool_calls > 0 or (trace.get("metadata") or {}).get("source") in AGENTIC_SOURCES
        model = doc.get("model") or "unknown"
        cache_read = doc.get("cache_read_tokens") or 0
        cache_write = doc.get("cache_write_tokens") or 0
        first = doc.get("first_message")
        system = _text(first.get("content")) if isinstance(first, dict) and first.get("role") == "system" else ""
        cost = doc.get("cost_usd")
        calls.append(Call(
            trace_id=str(doc["trace_id"]),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_read=cache_read,
            cache_write=cache_write,
            cost=cost if cost is not None else compute_cost(model, tokens_in, tokens_out, cache_read, cache_write),
            failed=failed,
            temperature=doc.get("temperature"),
            system_prompt=system or None,
            quality=quality_score(doc.get("metrics"), tag),
            agentic=agentic,
            complexity=classify_complexity(tokens_in, tokens_out, agentic, tool_calls),
        ))
    return calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _usd(value: float) -> str:
    return f"${value:.4f}" if value < 1 else f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _display_name(calls: list[Call]) -> str:
    return Counter(c.model for c in calls).most_common(1)[0][0]


def _examples(calls: list[Call], limit: int = 3) -> list:
    seen, out = set(), []
    for c in sorted(calls, key=lambda c: c.cost, reverse=True):
        if c.trace_id not in seen:
            seen.add(c.trace_id)
            out.append({"trace_id": c.trace_id, "reason": None})
        if len(out) == limit:
            break
    return out


def _avg_quality(calls: list[Call]) -> tuple[Optional[float], int]:
    scores = [c.quality for c in calls if c.quality is not None]
    return (sum(scores) / len(scores), len(scores)) if scores else (None, 0)


def severity_from_savings(savings: float, total_cost: float) -> str:
    share = savings / total_cost if total_cost > 0 else 0.0
    return "high" if share >= 0.10 else "medium" if share >= 0.02 else "low"


# ---------------------------------------------------------------------------
# Rules on calls
# ---------------------------------------------------------------------------

def model_routing(calls: list[Call], total_cost: float) -> list[Recommendation]:
    """Simple calls on a model that has a cheaper sibling."""
    simple_by_key: dict = defaultdict(list)
    all_by_key: dict = defaultdict(list)
    alt_of: dict = {}
    for c in calls:
        if c.key not in alt_of:
            alt_of[c.key] = cheaper_model(c.key)
        if alt_of[c.key]:
            all_by_key[c.key].append(c)
            if c.complexity == SIMPLE and not c.failed:
                simple_by_key[c.key].append(c)

    recs = []
    for key, simple in simple_by_key.items():
        if len(simple) < MIN_CALLS:
            continue
        alt = alt_of[key]
        current = sum(c.cost for c in simple)
        alternative = sum(c.cost_with(alt) for c in simple)
        savings = current - alternative
        if savings <= 0:
            continue
        name = _display_name(all_by_key[key])
        alt_quality, alt_evaluated = _avg_quality([c for c in calls if c.key == alt and c.complexity == SIMPLE])
        own_quality, own_evaluated = _avg_quality(simple)

        detail = f"{len(simple)} of the {len(all_by_key[key])} calls to {name} are short requests with no tools."
        if alt_evaluated >= MIN_EVALUATED and alt_quality >= GOOD_QUALITY:
            detail += f" {alt} is already used in this project, with a quality of {_pct(alt_quality)}."
        recs.append(Recommendation(
            id=f"model-routing:{key}",
            category="model",
            severity=severity_from_savings(savings, total_cost),
            title=f"Move simple requests from {name} to {alt}",
            detail=detail,
            action=f"Send short requests to {alt} and keep {name} for long tasks.",
            fix=_code_fix("Routing example (Python)", _routing_code(alt, name)),
            savings_usd=round(savings, 6),
            evidence={
                "model": name,
                "alternative": alt,
                "simple_calls": len(simple),
                "model_calls": len(all_by_key[key]),
                "current_cost_usd": round(current, 6),
                "alternative_cost_usd": round(alternative, 6),
                "quality": round(own_quality, 3) if own_evaluated >= MIN_EVALUATED else None,
                "alternative_quality": round(alt_quality, 3) if alt_evaluated >= MIN_EVALUATED else None,
            },
            examples=_examples(simple),
        ))
    return recs


def _stronger_model(key: str) -> Optional[str]:
    """Cheapest model whose cheaper sibling is `key`."""
    candidates = [big for big, small in CHEAPER_MODEL.items() if small == key]
    return min(candidates, key=lambda k: price_of(k)["in"]) if candidates else None


def model_upgrade(calls: list[Call]) -> list[Recommendation]:
    """Complex calls that fail or score poorly on a small model."""
    small_models = set(CHEAPER_MODEL.values())
    complex_by_key: dict = defaultdict(list)
    for c in calls:
        if c.key in small_models and c.complexity == COMPLEX:
            complex_by_key[c.key].append(c)

    recs = []
    for key, group in complex_by_key.items():
        if len(group) < MIN_CALLS:
            continue
        bad = [c for c in group if c.failed or (c.quality is not None and c.quality < POOR_QUALITY)]
        rate = len(bad) / len(group)
        if rate < UPGRADE_BAD_RATE:
            continue
        name = _display_name(group)
        stronger = _stronger_model(key)
        recs.append(Recommendation(
            id=f"model-upgrade:{key}",
            category="model",
            severity="high" if rate >= 0.5 else "medium",
            title=f"{name} fails on complex tasks",
            detail=f"{len(bad)} of the {len(group)} complex tasks failed or were of poor quality ({_pct(rate)}).",
            action=f"Send complex tasks to {stronger or 'a more capable model'} and keep {name} for the rest.",
            fix=_code_fix("Routing example (Python)", _routing_code(name, stronger)) if stronger else None,
            evidence={"model": name, "alternative": stronger, "complex_calls": len(group), "bad_calls": len(bad),
                      "bad_rate": round(rate, 3)},
            examples=_examples(bad),
        ))
    return recs


def _caching_action(key: str) -> str:
    if key.startswith("claude"):
        return "Mark the system prompt with cache_control: the cached part costs 10% of the normal price."
    if key.startswith(("gpt", "o1", "o3", "o4")):
        return ("Put the fixed parts of the prompt first and the variable parts (date, user, documents) "
                "last: OpenAI then caches the beginning automatically.")
    if key.startswith("gemini"):
        return "Enable Gemini context caching for this system prompt."
    return "Enable the provider's prompt caching and keep the fixed parts of the prompt first."


def _caching_fix(key: str) -> Optional[dict]:
    if key.startswith("claude"):
        return _code_fix("Anthropic call (Python)", (
            "client.messages.create(\n"
            f"    model=\"{key}\",\n"
            "    system=[{\"type\": \"text\", \"text\": SYSTEM_PROMPT,\n"
            "             \"cache_control\": {\"type\": \"ephemeral\"}}],\n"
            "    messages=messages,\n"
            ")\n"
        ))
    return None


def prompt_caching(calls: list[Call], total_cost: float) -> list[Recommendation]:
    """A long system prompt resent on every call with almost nothing read from the cache."""
    groups: dict = defaultdict(list)
    for c in calls:
        if c.key and c.system_prompt and len(c.system_prompt) >= CACHE_MIN_PROMPT_CHARS:
            digest = hashlib.sha1(c.system_prompt.encode("utf-8", "ignore")).hexdigest()[:12]
            groups[(c.key, digest)].append(c)

    recs = []
    for (key, digest), group in groups.items():
        if len(group) < MIN_CALLS:
            continue
        tokens_in = sum(c.tokens_in for c in group)
        cached_share = sum(c.cache_read for c in group) / tokens_in if tokens_in else 0.0
        if cached_share >= CACHE_MAX_CACHED_SHARE:
            continue
        prompt = group[0].system_prompt
        prompt_tokens = len(prompt) // 4
        price = price_of(key)
        savings = (len(group) - 1) * prompt_tokens * (price["in"] - price["cache_read"]) / 1_000_000
        savings = min(savings, sum(c.cost for c in group))
        if savings <= 0:
            continue
        name = _display_name(group)
        recs.append(Recommendation(
            id=f"prompt-caching:{key}:{digest}",
            category="caching",
            severity=severity_from_savings(savings, total_cost),
            title=f"Cache the system prompt of {name}",
            detail=f"The same prompt of about {prompt_tokens} tokens is resent on each of the {len(group)} calls, without caching.",
            action=_caching_action(key),
            fix=_caching_fix(key),
            savings_usd=round(savings, 6),
            evidence={"model": name, "calls": len(group), "prompt_tokens_estimate": prompt_tokens,
                      "cached_share": round(cached_share, 3), "prompt_preview": prompt[:200]},
            examples=_examples(group),
        ))
    return recs


def long_context(calls: list[Call], total_cost: float) -> list[Recommendation]:
    """Large contexts for short answers outside agent runs (chat history, retrieval)."""
    groups: dict = defaultdict(list)
    for c in calls:
        if not c.agentic and c.tokens_in >= LONG_CONTEXT_MIN_TOKENS_IN \
                and c.tokens_in >= LONG_CONTEXT_RATIO * max(c.tokens_out, 1):
            groups[c.model].append(c)

    recs = []
    for model, group in groups.items():
        if len(group) < MIN_CALLS:
            continue
        cost = sum(c.cost for c in group)
        avg_in = sum(c.tokens_in for c in group) // len(group)
        avg_out = sum(c.tokens_out for c in group) // len(group)
        recs.append(Recommendation(
            id=f"long-context:{model}",
            category="prompt",
            severity="medium" if total_cost > 0 and cost / total_cost >= 0.10 else "low",
            title=f"Context too long for short answers ({model})",
            detail=f"{len(group)} calls send {avg_in} tokens on average for {avg_out} tokens of answer.",
            action="Summarize the conversation history and send fewer documents (lower top-k).",
            evidence={"model": model, "calls": len(group), "avg_tokens_in": avg_in, "avg_tokens_out": avg_out,
                      "cost_usd": round(cost, 6)},
            examples=_examples(group),
        ))
    return recs


def high_temperature(calls: list[Call]) -> list[Recommendation]:
    hot = [c for c in calls if c.temperature is not None and c.temperature > HIGH_TEMPERATURE]
    if len(hot) < MIN_CALLS:
        return []
    quality, evaluated = _avg_quality(hot)
    poor = evaluated >= MIN_EVALUATED and quality < GOOD_QUALITY
    detail = f"{len(hot)} calls have a temperature above {HIGH_TEMPERATURE}."
    if poor:
        detail += f" Their average quality is {_pct(quality)}."
    return [Recommendation(
        id="high-temperature",
        category="prompt",
        severity="medium" if poor else "low",
        title="Temperature too high",
        detail=detail,
        action="Use a temperature between 0 and 0.3, except for creative generation.",
        fix=_code_fix("Call parameter", "temperature=0.2\n"),
        evidence={"calls": len(hot), "quality": round(quality, 3) if evaluated >= MIN_EVALUATED else None},
        examples=_examples(hot),
    )]


def failing_models(calls: list[Call]) -> list[Recommendation]:
    groups: dict = defaultdict(list)
    for c in calls:
        groups[c.model].append(c)
    recs = []
    for model, group in groups.items():
        if len(group) < FAILURE_MIN_CALLS:
            continue
        failed = [c for c in group if c.failed]
        rate = len(failed) / len(group)
        if rate < FAILURE_RATE:
            continue
        recs.append(Recommendation(
            id=f"failing-model:{model}",
            category="reliability",
            severity="high" if rate >= 0.2 else "medium",
            title=f"{model} fails often",
            detail=f"{len(failed)} of the {len(group)} calls failed ({_pct(rate)}).",
            action="Retry automatically on 429 and 5xx errors, with a fallback model.",
            fix=_code_fix("Automatic retry (Python)", RETRY_CODE),
            evidence={"model": model, "calls": len(group), "failed_calls": len(failed), "failure_rate": round(rate, 3)},
            examples=_examples(failed),
        ))
    return recs


# ---------------------------------------------------------------------------
# Rules on evaluation metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricRule:
    metric: str
    level: str              # span | trace
    category: str           # prompt | agent
    higher_is_worse: bool
    threshold: float        # average beyond which the rule fires
    critical: float         # average beyond which it is high severity
    bad: dict               # $elemMatch condition of a poor verdict, for examples
    title: str
    finding: str            # {value} = average as a percentage, {count} = number of evaluations
    action: str
    snippet: Optional[str] = None   # instruction to add to the prompt or the agent's rules
    code: Optional[str] = None      # code change when the fix is not an instruction


_LOW = {"value": {"$lt": 0.5}}
_FALSE = {"value": False}

METRIC_RULES = (
    MetricRule(
        MetricName.hallucination.value, "span", "prompt", True, 0.2, 0.4, {"value.0": {"$exists": True}},
        "Answers with unsupported claims",
        "{value} of the answers contain at least one unsupported claim.",
        "Ask the model to answer only from the documents provided.",
        snippet=("Answer only from the context provided. If the information is not there, say that you don't "
                 "know."),
    ),
    MetricRule(
        MetricName.general_faithfulness.value, "span", "prompt", False, 0.7, 0.5, _LOW,
        "Answers not faithful to the sources",
        "Average faithfulness to the sources: {value}.",
        "Ask for the source of every claim.",
        snippet=("For every claim, quote the passage of the context that supports it. Keep what comes from the "
                 "context apart from what you infer."),
    ),
    MetricRule(
        MetricName.contextual_faithfulness.value, "span", "prompt", False, 0.7, 0.5, _LOW,
        "Answers that stray from the retrieved context",
        "Average faithfulness to the context: {value}.",
        "Wrap the retrieved documents in tags and allow only that content.",
        snippet="Answer only from the text between <context> and </context>.",
    ),
    MetricRule(
        MetricName.answer_relevance.value, "span", "prompt", False, 0.7, 0.5, _LOW,
        "Answers that miss the question",
        "Average answer relevance: {value}.",
        "Restate the question at the end of the prompt and remove contradictory instructions.",
        snippet="Answer the question directly in the first sentence, then give the useful details.",
    ),
    MetricRule(
        MetricName.completeness.value, "span", "prompt", False, 0.7, 0.5, _LOW,
        "Incomplete answers",
        "Average answer completeness: {value}.",
        "List the points to cover and raise max_tokens if answers are cut off.",
        snippet="Cover every point requested, then check that nothing is missing before you finish.",
    ),
    MetricRule(
        MetricName.output_format_compliance.value, "span", "prompt", False, 0.9, 0.7, _FALSE,
        "Output format not followed",
        "{value} of the answers follow the expected format.",
        "Give the exact format with an example, and use the provider's structured outputs.",
        snippet="Answer exactly in the requested format, with no text before or after.",
    ),
    MetricRule(
        MetricName.is_json.value, "span", "prompt", False, 0.9, 0.7, _FALSE,
        "Invalid JSON outputs",
        "{value} of the answers are valid JSON.",
        "Enable the provider's JSON mode.",
        snippet="Answer only with valid JSON, with no surrounding text.",
        code="response_format={\"type\": \"json_object\"}\n",
    ),
    MetricRule(
        MetricName.moderation.value, "span", "prompt", True, 0.3, 0.6, {"value": {"$gt": 0.5}},
        "Content flagged by moderation",
        "Average moderation score: {value}.",
        "Add safety rules to the system prompt and filter the answers.",
        snippet="Politely refuse hateful, violent, sexual or illegal requests, and never produce such content.",
    ),
    MetricRule(
        MetricName.context_precision.value, "span", "prompt", False, 0.6, 0.4, _LOW,
        "Retrieved documents of low relevance",
        "Average context precision: {value}.",
        "Retrieve fewer documents (lower top-k) or add a reranker.",
    ),
    MetricRule(
        MetricName.context_recall.value, "span", "prompt", False, 0.6, 0.4, _LOW,
        "Useful information missing from the context",
        "Average context recall: {value}.",
        "Retrieve more documents, revisit how they are chunked, or add keyword search.",
    ),
    MetricRule(
        MetricName.task_completion.value, "trace", "agent", False, 0.7, 0.5, _FALSE,
        "Tasks not completed by the agent",
        "{value} of the runs complete the requested task.",
        "Ask the agent to check the result before answering.",
        snippet=("Before answering, check that the requested result really exists (file changed, test "
                 "passing, question answered). Otherwise, keep going or explain what is blocking."),
    ),
    MetricRule(
        MetricName.tool_error_rate.value, "trace", "agent", True, 0.2, 0.4, {"value": {"$gt": 0.3}},
        "Too many failing tool calls",
        "Average tool error rate: {value}.",
        "Ask the agent to check parameters before the call and to read the errors.",
        snippet=("Check a tool's parameters before calling it; after an error, read the message and fix "
                 "the call."),
    ),
    MetricRule(
        MetricName.instruction_adherence.value, "trace", "agent", False, 0.7, 0.5, _LOW,
        "User instructions poorly followed",
        "Average instruction adherence: {value}.",
        "Remind the agent to do exactly what is asked, nothing more.",
        snippet=("Do exactly what is asked, nothing more: don't modify any file that wasn't requested and "
                 "respect the mode (read-only, plan). Reread the request before answering."),
    ),
    MetricRule(
        MetricName.claims_grounded.value, "trace", "agent", False, 0.8, 0.6, {"value": {"$lt": 0.7}},
        "Claims not proven by the actions",
        "{value} of the final claims are backed by an action.",
        "Ask the agent to claim only what it has verified.",
        snippet=("Only state a result if you have verified it (file read, command run, test executed). "
                 "Write \"not verified\" for the rest."),
    ),
)


def _fires(rule: MetricRule, avg: float) -> bool:
    return avg > rule.threshold if rule.higher_is_worse else avg < rule.threshold


def _critical(rule: MetricRule, avg: float) -> bool:
    return avg >= rule.critical if rule.higher_is_worse else avg <= rule.critical


def rule_target(rule: MetricRule, target: str) -> str:
    """Answer metrics are only computed outside agent runs: their fix goes in a system prompt."""
    return target if rule.category == "agent" else "sdk"


def rule_fix(rule: MetricRule, target: str) -> tuple[str, Optional[dict]]:
    """Action and ready-to-copy fix of a rule for the agent or app it applies to."""
    if rule.code:
        return rule.action, _code_fix("Call parameter", rule.code)
    if rule.snippet:
        where = rule_target(rule, target)
        return _instructions_action(where), instructions_fix([rule.snippet], where)
    return rule.action, None


def metric_recommendations(summary: dict, target: str = "sdk") -> list[tuple[MetricRule, Recommendation]]:
    """Rules fired by the metric averages of insights.metrics_summary."""
    out = []
    for rule in METRIC_RULES:
        stats = summary.get(f"{rule.level}_metrics", {}).get(rule.metric)
        if not stats or stats.get("avg") is None or (stats.get("evaluated_count") or 0) < MIN_EVALUATED:
            continue
        avg = stats["avg"]
        if not _fires(rule, avg):
            continue
        action, fix = rule_fix(rule, target)
        out.append((rule, Recommendation(
            id=f"metric:{rule.metric}",
            category=rule.category,
            severity="high" if _critical(rule, avg) else "medium",
            title=rule.title,
            detail=rule.finding.format(value=_pct(avg), count=stats["evaluated_count"]),
            action=action,
            evidence={"metric": rule.metric, "average": avg, "evaluated": stats["evaluated_count"],
                      "threshold": rule.threshold},
            fix=fix,
        )))
    return out


def sort_recommendations(recs: list[Recommendation]) -> list[Recommendation]:
    return sorted(recs, key=lambda r: (_SEVERITY_RANK[r.severity], -(r.savings_usd or 0.0)))


# ---------------------------------------------------------------------------
# Recommended prompt
# ---------------------------------------------------------------------------

PROMPT_MAX_CHARS = 20_000
_RULE_TITLES = {"cursor": "Recommended Cursor rule", "claude-code": "Recommended Claude Code instructions"}
_BASELINE_TITLES = {"cursor": "Starter Cursor rule", "claude-code": "Starter Claude Code instructions"}
BASELINE_METRICS = (
    MetricName.instruction_adherence.value,
    MetricName.claims_grounded.value,
    MetricName.task_completion.value,
    MetricName.tool_error_rate.value,
)


def _main_system_prompt(calls: list[Call]) -> Optional[tuple[str, str, int]]:
    """The system prompt that cost the most over the period: (prompt, model, calls)."""
    groups: dict = defaultdict(list)
    for c in calls:
        if c.system_prompt:
            groups[c.system_prompt].append(c)
    if not groups:
        return None
    prompt, group = max(groups.items(), key=lambda item: sum(c.cost for c in item[1]))
    return prompt, _display_name(group), len(group)


def recommended_prompts(rules: list[MetricRule], target: str, calls: list[Call],
                        baseline: bool = False) -> list[dict]:
    """The instructions of the fired rules, written where they belong: a complete Cursor rule
    or CLAUDE.md section for a coding agent, and the most used system prompt with the missing
    instructions appended. With `baseline`, a coding agent with no fired rule gets the rules
    that prevent its most common failures, marked baseline=True."""
    rules = [r for r in rules if r.snippet]
    agent_rules = [r for r in rules if r.category == "agent"]
    out = []
    if target in _RULE_TITLES and (agent_rules or baseline):
        is_baseline = not agent_rules
        chosen = agent_rules or [r for r in METRIC_RULES if r.metric in BASELINE_METRICS]
        fix = instructions_fix([r.snippet for r in chosen], target)
        out.append({
            "id": target, "target": target,
            "title": (_BASELINE_TITLES if is_baseline else _RULE_TITLES)[target], "file": fix["file"],
            "current": None, "content": fix["content"], "changes": [] if is_baseline else [r.title for r in agent_rules],
            "model": None, "calls": None, "source": "rules", "baseline": is_baseline,
        })
        rules = [r for r in rules if r.category != "agent"]

    main = _main_system_prompt(calls)
    if not main and not rules:
        return out
    current, model, count = main or (None, None, None)
    added = [r for r in rules if not current or r.snippet not in current]
    content = current or ""
    if added:
        bullets = "\n".join(f"- {r.snippet}" for r in added)
        content = f"{content.rstrip()}\n\n## Instructions\n{bullets}\n" if content else f"{bullets}\n"
    out.append({
        "id": "system", "target": "system_prompt",
        "title": f"Recommended system prompt ({model})" if model else "Instructions to add to the system prompt",
        "file": None,
        "current": current[:PROMPT_MAX_CHARS] if current else None,
        "content": content[:PROMPT_MAX_CHARS + 2_000],
        "changes": [r.title for r in added], "model": model, "calls": count, "source": "rules",
        "baseline": False,
    })
    return out


def prompt_findings(result: dict, limit: int = 20) -> list[str]:
    """Problems behind the prompt and agent recommendations, with the judge's reasons, for an
    AI rewrite of the recommended prompt."""
    out = []
    for r in result.get("recommendations", []):
        if r["category"] not in ("prompt", "agent"):
            continue
        out.append(f"{r['title']}: {r['detail']}")
        reasons = [e.get("reason") for e in r.get("examples") or []] + [s.get("reason") for s in r.get("spans") or []]
        out += [f"  example: {' '.join(reason.split())[:300]}" for reason in reasons if reason][:3]
    return out[:limit]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _bucket() -> dict:
    return {"calls": 0, "cost_usd": 0.0}


def complexity_breakdown(calls: list[Call]) -> tuple[dict, list]:
    """Calls and cost per complexity, overall and per model (most expensive models first)."""
    overall = {c: _bucket() for c in COMPLEXITIES}
    models: dict = {}
    for call in calls:
        alt, why_not = alternative_status(call.key) if call.model not in models else (None, None)
        row = models.setdefault(call.model, {
            "model": call.model,
            "price_known": call.key is not None,
            "alternative": alt,
            "no_alternative_reason": why_not,
            "calls": 0,
            "cost_usd": 0.0,
            **{c: _bucket() for c in COMPLEXITIES},
        })
        for bucket in (overall[call.complexity], row[call.complexity], row):
            bucket["calls"] += 1
            bucket["cost_usd"] += call.cost
    for bucket in overall.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    rows = sorted(models.values(), key=lambda r: r["cost_usd"], reverse=True)
    for row in rows:
        row["cost_usd"] = round(row["cost_usd"], 6)
        for c in COMPLEXITIES:
            row[c]["cost_usd"] = round(row[c]["cost_usd"], 6)
    return overall, rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _llm_call_docs(pid: PydanticObjectId, since: datetime) -> list:
    return await Span.aggregate([
        {"$match": {"project_id": pid, "span_type": SpanType.llm.value, "timestamp": {"$gte": since}}},
        {"$sort": {"timestamp": -1}},
        {"$limit": MAX_CALLS},
        {"$project": {
            "trace_id": 1, "model": 1, "tokens_in": 1, "tokens_out": 1, "cache_read_tokens": 1,
            "cache_write_tokens": 1, "cost_usd": 1, "status": 1, "error": 1, "temperature": 1, "metrics": 1,
            "first_message": {"$arrayElemAt": [{"$ifNull": ["$messages_in", []]}, 0]},
        }},
    ]).to_list()


async def _metric_examples(pid: PydanticObjectId, since: datetime, rule: MetricRule, limit: int = 3) -> list:
    """Latest items with a poor verdict on the rule's metric, with the judge's reason."""
    match = {"name": rule.metric, **rule.bad, **insights._current_judge(rule.metric)}
    if rule.level == "span":
        docs = await Span.find({"project_id": pid, "timestamp": {"$gte": since}, "metrics": {"$elemMatch": match}}) \
            .sort("-timestamp").limit(limit).to_list()
        trace_of = lambda d: str(d.trace_id)  # noqa: E731
    else:
        docs = await Trace.find({"project_id": pid, "started_at": {"$gte": since}, "metrics": {"$elemMatch": match}}) \
            .sort("-started_at").limit(limit).to_list()
        trace_of = lambda d: str(d.id)  # noqa: E731
    out = []
    for d in docs:
        metric = next((m for m in d.metrics if getattr(m.name, "value", m.name) == rule.metric), None)
        out.append({"trace_id": trace_of(d), "reason": metric.reason if metric else None})
    return out


async def recommendations(pid: PydanticObjectId, since: datetime) -> dict:
    span_docs = await _llm_call_docs(pid, since)
    trace_ids = list({d["trace_id"] for d in span_docs})
    trace_docs = await Trace.get_motor_collection().find(
        {"_id": {"$in": trace_ids}}, {"total_tool_spans": 1, "metadata.source": 1},
    ).to_list(None) if trace_ids else []
    calls = build_calls(span_docs, trace_docs, judge_tag())
    total_cost = sum(c.cost for c in calls)
    target = agent_target((t.get("metadata") or {}).get("source") for t in trace_docs)

    routing = model_routing(calls, total_cost)
    caching = prompt_caching(calls, total_cost)
    recs = [*routing, *model_upgrade(calls), *caching, *long_context(calls, total_cost),
            *high_temperature(calls), *failing_models(calls)]

    recs += [Recommendation(**f) for f in await harness.project_findings(pid, since, target)]

    summary = await insights.metrics_summary(pid, 0, since=since)
    fired = []
    for rule, rec in metric_recommendations(summary, target):
        rec.examples = await _metric_examples(pid, since, rule)
        recs.append(rec)
        fired.append(rule)

    overall, by_model = complexity_breakdown(calls)
    unpriced = Counter(c.model for c in calls if c.key is None)
    routing_usd = sum(r.savings_usd or 0 for r in routing)
    caching_usd = sum(r.savings_usd or 0 for r in caching)
    return {
        "analyzed_calls": len(calls),
        "truncated": len(span_docs) >= MAX_CALLS,
        "total_cost_usd": round(total_cost, 6),
        "savings": {
            "routing_usd": round(routing_usd, 6),
            "caching_usd": round(caching_usd, 6),
            "total_usd": combined_savings(total_cost, routing_usd, caching_usd),
        },
        "complexity": overall,
        "by_model": by_model[:20],
        "unpriced_models": [{"model": m, "calls": n} for m, n in unpriced.most_common(10)],
        "pricing": _pricing_status(),
        "agent_target": target,
        "prompts": recommended_prompts(fired, target, calls, baseline=True),
        "evaluation": {**await evaluations.status(pid, since), "min_evaluated": MIN_EVALUATED},
        "recommendations": [asdict(r) for r in sort_recommendations(recs)],
    }


def combined_savings(total: float, model_usd: float, caching_usd: float) -> float:
    """Model and caching savings together. Caching savings are priced on the current model:
    the share of the cost moved to a cheaper model saves proportionally less by caching."""
    if total <= 0:
        return 0.0
    model_usd = min(max(model_usd, 0.0), total)
    caching_usd = max(caching_usd, 0.0) * (1 - model_usd / total)
    return round(min(total, model_usd + caching_usd), 6)


def _pricing_status() -> dict:
    return {k: v for k, v in pricing.status().items() if k in ("source", "updated_at", "models")}


# ---------------------------------------------------------------------------
# One trace
# ---------------------------------------------------------------------------

RUN_SIMPLE_MAX_TOOLS = 3
RUN_SIMPLE_MAX_CONTEXT = 20_000
RUN_COMPLEX_MIN_TOOLS = 15
RUN_COMPLEX_MIN_CONTEXT = 100_000
TRACE_CACHE_MIN_CALLS = 3
TRACE_CACHE_MIN_PREFIX = 1_024      # OpenAI's minimum cacheable prefix
COST_CONCENTRATION = 0.5
CONTEXT_GROWTH_RATIO = 3
CONTEXT_GROWTH_MIN_TOKENS = 50_000
REPEATED_TOOL_CALLS = 3
TOOL_ERROR_RATE = 0.3
MAX_REFS = 10

_COMPLEXITY_RANK = {SIMPLE: 0, MEDIUM: 1, COMPLEX: 2}
RULES = {rule.metric: rule for rule in METRIC_RULES}
# Fired by the repeated identical tool calls of a trace, not by a metric average.
REPEATED_TOOL_RULE = MetricRule(
    MetricName.loop_detection.value, "trace", "agent", True, 0.15, 0.3, {"value": True},
    "The agent is looping",
    "{value} of the runs repeat the same actions.",
    "Ask the agent to change approach after two identical failures.",
    snippet=("Never retry an identical action that just failed: after two failures, change approach or "
             "ask for help."),
)
CONTEXT_SNIPPET = ("Read targeted excerpts rather than whole files, and hand broad searches to a "
                   "subagent.")


def _kind(span) -> str:
    return getattr(span.span_type, "value", span.span_type)


def _order(span) -> tuple:
    return (span.timestamp or datetime.min, span.step_index or 0)


def _cache_tokens(span) -> tuple[int, int]:
    usage = (span.metadata or {}).get("usage") or {}
    read = span.cache_read_tokens or usage.get("cache_read_tokens") or 0
    write = span.cache_write_tokens or usage.get("cache_write_tokens") or 0
    return int(read), int(write)


def _model_calls(span) -> int:
    """Claude Code reports the model calls of a turn as one span (metadata.model_calls)."""
    try:
        return max(1, int((span.metadata or {}).get("model_calls") or 1))
    except (TypeError, ValueError):
        return 1


def _context(span) -> int:
    """Input tokens of one model call: the average when the span groups several."""
    try:
        average = int((span.metadata or {}).get("avg_context_tokens") or 0)
    except (TypeError, ValueError):
        average = 0
    return average or (span.tokens_in or 0) // _model_calls(span)


def _llm_failed(span) -> bool:
    return (span.status or "").lower() in ("failed", "error") or bool(span.error)


def _label(span) -> str:
    return span.name or span.tool_name or span.model or _kind(span)


def _ref(span, reason: Optional[str] = None) -> dict:
    return {"span_id": str(span.id), "step": span.step_index, "label": _label(span),
            "reason": reason[:300] if reason else None}


def _metric_dicts(metrics) -> list:
    return [m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m) for m in metrics or []]


def _current(metric: dict, tag: Optional[str]) -> bool:
    return not (tag and metric.get("name") in JUDGED_METRICS and metric.get("judge") != tag)


def _metric_number(value) -> Optional[float]:
    if isinstance(value, list):
        return 1.0 if value else 0.0
    return _numeric(value)


def _metric_display(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return f"{len(value)} item(s)"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _tool_label(trace) -> str:
    source = (trace.metadata or {}).get("source")
    return {"cursor": "Cursor", "claude-code": "Claude Code"}.get(source, "the agent")


def run_complexity(tool_calls: int, max_context: int) -> str:
    """Size of an agent run: few tools on a small context is a short task (a question, a small fix)."""
    if tool_calls >= RUN_COMPLEX_MIN_TOOLS or max_context >= RUN_COMPLEX_MIN_CONTEXT:
        return COMPLEX
    if tool_calls <= RUN_SIMPLE_MAX_TOOLS and max_context <= RUN_SIMPLE_MAX_CONTEXT:
        return SIMPLE
    return MEDIUM


def _call_of(span, agentic: bool, tool_calls: int, tag: Optional[str]) -> Call:
    tokens_in, tokens_out = span.tokens_in or 0, span.tokens_out or 0
    model = span.model or "unknown"
    cache_read, cache_write = _cache_tokens(span)
    first = (span.messages_in or [None])[0]
    system = _text(first.get("content")) if isinstance(first, dict) and first.get("role") == "system" else ""
    cost = span.cost_usd
    return Call(
        trace_id=str(span.trace_id),
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read=cache_read,
        cache_write=cache_write,
        cost=cost if cost is not None else compute_cost(model, tokens_in, tokens_out, cache_read, cache_write),
        failed=_llm_failed(span),
        temperature=span.temperature,
        system_prompt=system or None,
        quality=quality_score(_metric_dicts(span.metrics), tag),
        agentic=agentic,
        complexity=classify_complexity(_context(span), tokens_out // _model_calls(span), agentic, tool_calls),
    )


def _int(n: int) -> str:
    return f"{n:,}"


_COMPLEXITY_LABEL = {SIMPLE: "Simple", MEDIUM: "Medium", COMPLEX: "Complex"}


def _not_simple_reason(call: Call, agentic: bool, tool_count: int, max_context: int) -> str:
    if agentic:
        return (f"Task too long for a smaller model: {tool_count} tool(s) and {_int(max_context)} context tokens "
                f"(short: at most {RUN_SIMPLE_MAX_TOOLS} tools and {_int(RUN_SIMPLE_MAX_CONTEXT)} tokens).")
    return (f"{_COMPLEXITY_LABEL[call.complexity]} request: {_int(call.tokens_in)} input tokens, "
            f"{_int(call.tokens_out)} output (simple: at most {_int(SIMPLE_MAX_TOKENS_IN)} and "
            f"{_int(SIMPLE_MAX_TOKENS_OUT)}, no tools).")


def _steps(llm: list, calls: list[Call], agentic: bool, run: str, total: float, notes: dict,
           tool_count: int = 0, max_context: int = 0) -> list[dict]:
    """One row per LLM call, with the cheaper model it could have used or why there is none.
    In an agent run every call depends on the run's size: they can all move down a tier only
    when the run is short."""
    steps, status_of = [], {}
    for span, call in zip(llm, calls):
        if call.key not in status_of:
            status_of[call.key] = alternative_status(call.key)
        alt, why_not = status_of[call.key]
        if call.failed:
            alt, why_not = None, "Failed call."
        elif alt and not (call.complexity == SIMPLE or (agentic and run == SIMPLE)):
            alt, why_not = None, _not_simple_reason(call, agentic, tool_count, max_context)
        alt_cost = call.cost_with(alt) if alt else None
        if alt_cost is not None and alt_cost >= call.cost:
            alt, alt_cost, why_not = None, None, f"{alt} would not cost less on this call."
        span_id = str(span.id)
        steps.append({
            "span_id": span_id,
            "step": span.step_index,
            "label": _label(span),
            "model": call.model,
            "price_known": call.key is not None,
            "complexity": call.complexity,
            "tokens_in": call.tokens_in,
            "tokens_out": call.tokens_out,
            "context_tokens": _context(span),
            "cache_read_tokens": call.cache_read,
            "cost_usd": round(call.cost, 6),
            "cost_share": round(call.cost / total, 3) if total > 0 else 0.0,
            "alternative": alt,
            "alternative_cost_usd": round(alt_cost, 6) if alt_cost is not None else None,
            "savings_usd": round(call.cost - alt_cost, 6) if alt_cost is not None else None,
            "no_alternative_reason": None if alt else why_not,
            "failed": call.failed,
        })
        if alt:
            notes[span_id].append(f"{alt} would have been enough: {_usd(alt_cost)} instead of {_usd(call.cost)}.")
        if call.failed:
            notes[span_id].append("Model call failed.")
        if call.temperature is not None and call.temperature > HIGH_TEMPERATURE:
            notes[span_id].append(f"Temperature of {call.temperature}: too high outside creative generation.")
        if not agentic and call.tokens_in >= LONG_CONTEXT_MIN_TOKENS_IN \
                and call.tokens_in >= LONG_CONTEXT_RATIO * max(call.tokens_out, 1):
            notes[span_id].append(f"{call.tokens_in} context tokens for an answer of {call.tokens_out}: "
                                  f"summarize the history or attach fewer documents.")
    return steps


def _trace_routing(trace, llm: list, calls: list[Call], steps: list[dict], agentic: bool,
                   tool_count: int, max_context: int, total: float) -> list[Recommendation]:
    groups: dict = defaultdict(list)
    for span, call, step in zip(llm, calls, steps):
        if step["alternative"]:
            groups[call.key].append((span, call, step))
    recs = []
    for key, group in groups.items():
        alt = group[0][2]["alternative"]
        current = sum(c.cost for _, c, _ in group)
        alternative = sum(s["alternative_cost_usd"] for _, _, s in group)
        savings = current - alternative
        name = _display_name([c for _, c, _ in group])
        fix = None
        if agentic:
            title = f"{alt} would have been enough for this task"
            detail = f"Short task: {tool_count} tool(s) and at most {max_context} context tokens."
            action = (f"For small tasks, pick {alt} in {_tool_label(trace)}; keep {name} for long or "
                      f"multi-file tasks.")
        else:
            title = f"Simple steps: {alt} is enough"
            detail = f"{len(group)} call(s) in this trace are short and use no tools."
            action = f"Send these requests to {alt} and keep {name} for complex steps."
            fix = _code_fix("Routing example (Python)", _routing_code(alt, name))
        recs.append(Recommendation(
            id=f"trace-model:{key}",
            category="model",
            severity=severity_from_savings(savings, total),
            title=title,
            detail=detail,
            action=action,
            savings_usd=round(savings, 6),
            evidence={"model": name, "alternative": alt, "calls": len(group),
                      "current_cost_usd": round(current, 6), "alternative_cost_usd": round(alternative, 6)},
            spans=[_ref(s) for s, _, _ in group][:MAX_REFS],
            fix=fix,
        ))
    return recs


def _trace_caching(llm: list, calls: list[Call], total: float, notes: dict) -> list[Recommendation]:
    """Several calls to one model resending their context with almost nothing read from the cache.
    The smallest context of the group is the part every call resent."""
    groups: dict = defaultdict(list)
    for span, call in zip(llm, calls):
        if not call.failed:
            groups[call.key or call.model].append((span, call))
    recs = []
    for key, group in groups.items():
        n = sum(_model_calls(s) for s, _ in group)
        tokens_in = sum(c.tokens_in for _, c in group)
        if n < TRACE_CACHE_MIN_CALLS or not tokens_in:
            continue
        share = sum(c.cache_read for _, c in group) / tokens_in
        prefix = min(_context(s) for s, _ in group)
        if share >= CACHE_MAX_CACHED_SHARE or prefix < TRACE_CACHE_MIN_PREFIX:
            continue
        price = price_of(group[0][1].key) if group[0][1].key else None
        savings = None
        if price:
            savings = min((n - 1) * prefix * (price["in"] - price["cache_read"]) / 1_000_000,
                          sum(c.cost for _, c in group))
        name = _display_name([c for _, c in group])
        for span, _ in group[1:]:
            notes[str(span.id)].append("Context resent without a cache read.")
        recs.append(Recommendation(
            id=f"trace-caching:{key}",
            category="caching",
            severity=severity_from_savings(savings, total) if savings else "low",
            title=f"Prompt cache unused ({name})",
            detail=f"{n} calls each resend at least {prefix} tokens, {_pct(share)} of them read from the cache.",
            action=_caching_action(key),
            fix=_caching_fix(key),
            savings_usd=round(savings, 6) if savings else None,
            evidence={"model": name, "calls": n, "prefix_tokens": prefix, "cached_share": round(share, 3)},
            spans=[_ref(s) for s, _ in group[1:]][:MAX_REFS],
        ))
    return recs


def _cost_concentration(llm: list, calls: list[Call], total: float, notes: dict) -> list[Recommendation]:
    if len(calls) < 3 or total <= 0:
        return []
    span, call = max(zip(llm, calls), key=lambda pair: pair[1].cost)
    share = call.cost / total
    if share < COST_CONCENTRATION:
        return []
    if _context(span) >= COMPLEX_MIN_TOKENS_IN:
        action = "Context too large: attach only the useful excerpts, or split the task."
    elif call.tokens_out >= COMPLEX_MIN_TOKENS_OUT:
        action = "Very long answer: ask for a more concise answer or limit max_tokens."
    else:
        action = "Check whether this step needs this model and all this context."
    notes[str(span.id)].append(f"{_pct(share)} of the trace's cost.")
    return [Recommendation(
        id="trace-cost-concentration",
        category="prompt",
        severity="medium" if share >= 0.75 else "low",
        title=f"One step accounts for {_pct(share)} of the cost",
        detail=f"Step {span.step_index} ({_label(span)}) cost {_usd(call.cost)} out of {_usd(total)}.",
        action=action,
        evidence={"cost_usd": round(call.cost, 6), "cost_share": round(share, 3),
                  "tokens_in": call.tokens_in, "tokens_out": call.tokens_out},
        spans=[_ref(span)],
    )]


def _context_growth(llm: list, agentic: bool, target: str, notes: dict) -> list[Recommendation]:
    contexts = [(s, _context(s)) for s in llm if _context(s) > 0]
    if not agentic or len(contexts) < 4:
        return []
    first = contexts[0][1]
    span, peak = max(contexts, key=lambda pair: pair[1])
    if peak < CONTEXT_GROWTH_MIN_TOKENS or peak < CONTEXT_GROWTH_RATIO * first:
        return []
    notes[str(span.id)].append("Largest context of the run.")
    return [Recommendation(
        id="trace-context-growth",
        category="agent",
        severity="medium",
        title="The context keeps growing during the run",
        detail=f"The context grows from {first} to {peak} tokens: every call pays for the whole history again.",
        action="Start a new conversation per task, and read excerpts rather than whole files.",
        evidence={"first_context_tokens": first, "peak_context_tokens": peak, "calls": len(contexts)},
        spans=[_ref(span)],
        fix=instructions_fix([CONTEXT_SNIPPET], target),
    )]


def _repeated_tools(tools: list, target: str, notes: dict) -> list[Recommendation]:
    groups: dict = defaultdict(list)
    for span in tools:
        name, args = _arguments(span)
        if args:
            groups[(name, json.dumps(args, sort_keys=True, default=str))].append(span)
    recs = []
    for (name, signature), group in groups.items():
        if len(group) < REPEATED_TOOL_CALLS:
            continue
        for span in group[1:]:
            notes[str(span.id)].append(f"Same call as step {group[0].step_index}.")
        preview = signature if len(signature) <= 160 else signature[:157] + "..."
        digest = hashlib.sha1(signature.encode("utf-8", "ignore")).hexdigest()[:8]
        action, fix = rule_fix(REPEATED_TOOL_RULE, target)
        recs.append(Recommendation(
            id=f"trace-repeated-tool:{name}:{digest}",
            category="agent",
            severity="high" if len(group) >= 5 else "medium",
            title=f"{name} called {len(group)} times with the same arguments",
            detail=f"The agent repeats the same action without new information ({preview}).",
            action=action,
            evidence={"tool": name, "calls": len(group)},
            spans=[_ref(s) for s in group][:MAX_REFS],
            fix=fix,
        ))
    return recs


def _tool_errors(trace, tools: list, target: str, notes: dict) -> list[Recommendation]:
    failed = [s for s in tools if tool_failed(s)]
    if not failed:
        return []
    unrecovered = []
    for span in failed:
        name = _arguments(span)[0]
        recovered = any(_arguments(t)[0] == name and not tool_failed(t) and _order(t) > _order(span) for t in tools)
        notes[str(span.id)].append("Tool error." if recovered else "Tool error, never succeeded afterwards.")
        if not recovered:
            unrecovered.append(span)
    rate = len(failed) / len(tools)
    trace_failed = getattr(trace.status, "value", trace.status) == "failed"
    severity = ("high" if rate >= TOOL_ERROR_RATE or (unrecovered and trace_failed)
                else "medium" if len(failed) >= 2 else "low")
    names = Counter(_arguments(s)[0] for s in failed)
    detail = (f"{len(failed)} of the {len(tools)} tool calls failed "
              f"({', '.join(f'{n} ×{c}' for n, c in names.most_common(5))}).")
    if unrecovered:
        detail += f" {len(unrecovered)} never succeeded afterwards."
    action, fix = rule_fix(RULES[MetricName.tool_error_rate.value], target)
    return [Recommendation(
        id="trace-tool-errors",
        category="agent",
        severity=severity,
        title=f"{len(failed)} failing tool call(s)",
        detail=detail,
        action=action,
        evidence={"tool_calls": len(tools), "failed": len(failed), "unrecovered": len(unrecovered),
                  "failure_rate": round(rate, 3)},
        spans=[_ref(s, s.tool_error or s.error) for s in failed][:MAX_REFS],
        fix=fix,
    )]


def _failed_calls(trace, llm: list, calls: list[Call]) -> list[Recommendation]:
    failed = [s for s, c in zip(llm, calls) if c.failed]
    if not failed:
        return []
    trace_failed = getattr(trace.status, "value", trace.status) == "failed"
    return [Recommendation(
        id="trace-llm-errors",
        category="reliability",
        severity="high" if len(failed) == len(calls) or trace_failed else "medium",
        title=f"{len(failed)} failed model call(s)",
        detail=f"{len(failed)} of the {len(calls)} LLM calls in this trace failed.",
        action="Retry automatically on 429 and 5xx errors, with a fallback model.",
        evidence={"calls": len(calls), "failed": len(failed)},
        spans=[_ref(s, s.error) for s in failed][:MAX_REFS],
        fix=_code_fix("Automatic retry (Python)", RETRY_CODE),
    )]


def _trace_metrics(trace, spans: list, tag: Optional[str], target: str, notes: dict) -> list[Recommendation]:
    """The METRIC_RULES applied to this trace's own verdicts instead of period averages."""
    trace_metrics = {m.get("name"): m for m in _metric_dicts(trace.metrics) if _current(m, tag)}
    span_metrics = [(s, {m.get("name"): m for m in _metric_dicts(s.metrics) if _current(m, tag)})
                    for s in spans if s.metrics]
    recs = []
    for rule in METRIC_RULES:
        action, fix = rule_fix(rule, target)
        if rule.level == "trace":
            metric = trace_metrics.get(rule.metric)
            value = _metric_number(metric.get("value")) if metric else None
            if value is None or not _fires(rule, value):
                continue
            recs.append(Recommendation(
                id=f"metric:{rule.metric}",
                category=rule.category,
                severity="high" if _critical(rule, value) else "medium",
                title=rule.title,
                detail=metric.get("reason") or f"Value {_metric_display(metric.get('value'))} against a threshold of {rule.threshold}.",
                action=action,
                evidence={"metric": rule.metric, "value": value, "threshold": rule.threshold},
                fix=fix,
            ))
            continue
        evaluated, bad = 0, []
        for span, metrics in span_metrics:
            metric = metrics.get(rule.metric)
            value = _metric_number(metric.get("value")) if metric else None
            if value is None:
                continue
            evaluated += 1
            if _fires(rule, value):
                bad.append((span, metric, value))
        if not bad:
            continue
        for span, _, _ in bad:
            notes[str(span.id)].append(f"{rule.title}.")
        recs.append(Recommendation(
            id=f"metric:{rule.metric}",
            category=rule.category,
            severity="high" if any(_critical(rule, v) for _, _, v in bad) else "medium",
            title=rule.title,
            detail=f"{len(bad)} of the {evaluated} evaluated answers in this trace are affected.",
            action=action,
            evidence={"metric": rule.metric, "evaluated": evaluated, "bad": len(bad), "threshold": rule.threshold},
            spans=[_ref(s, m.get("reason")) for s, m, _ in bad][:MAX_REFS],
            fix=fix,
        ))
    return recs


def _fired_rules(recs: list[Recommendation]) -> list[MetricRule]:
    """Metric rules behind a trace's recommendations, the repeated tools and tool errors included."""
    ids = {r.id for r in recs}
    if "trace-tool-errors" in ids:
        ids.add(f"metric:{MetricName.tool_error_rate.value}")
    rules = [rule for rule in METRIC_RULES if f"metric:{rule.metric}" in ids]
    if any(i.startswith("trace-repeated-tool:") for i in ids):
        rules.append(REPEATED_TOOL_RULE)
    return rules


def analyze_trace(trace, spans: list, tag: Optional[str] = None) -> dict:
    """Recommendations for one trace from its spans (Span documents, in any order)."""
    agentic = is_agentic(trace, spans)
    tools = sorted((s for s in spans if _kind(s) == "tool"), key=_order)
    llm = sorted((s for s in spans if _kind(s) == "llm" and (s.tokens_in or s.tokens_out or _llm_failed(s))),
                 key=_order)
    calls = [_call_of(s, agentic, len(tools), tag) for s in llm]
    total = sum(c.cost for c in calls)
    max_context = max((_context(s) for s in llm), default=0)
    if agentic:
        complexity = run_complexity(len(tools), max_context)
    else:
        complexity = max((c.complexity for c in calls), key=_COMPLEXITY_RANK.__getitem__, default=SIMPLE)

    target = agent_target([(trace.metadata or {}).get("source")]) if agentic else "sdk"
    notes: dict = defaultdict(list)
    steps = _steps(llm, calls, agentic, complexity, total, notes, len(tools), max_context)
    recs = [
        *_trace_routing(trace, llm, calls, steps, agentic, len(tools), max_context, total),
        *_trace_caching(llm, calls, total, notes),
        *_cost_concentration(llm, calls, total, notes),
        *_context_growth(llm, agentic, target, notes),
        *_repeated_tools(tools, target, notes),
        *_tool_errors(trace, tools, target, notes),
        *_failed_calls(trace, llm, calls),
        *_trace_metrics(trace, spans, tag, target, notes),
        *(Recommendation(**f) for f in (harness.analyze(trace, spans, target, notes) if agentic else [])),
    ]
    model_savings = sum(r.savings_usd or 0 for r in recs if r.category == "model")
    caching_savings = sum(r.savings_usd or 0 for r in recs if r.category == "caching")
    savings = combined_savings(total, model_savings, caching_savings)
    return {
        "trace_id": str(trace.id),
        "agentic": agentic,
        "complexity": complexity,
        "tool_calls": len(tools),
        "llm_calls": len(calls),
        "max_context_tokens": max_context,
        "cost_usd": round(total, 6),
        "savings": {"model_usd": round(model_savings, 6), "caching_usd": round(caching_savings, 6),
                    "total_usd": savings},
        "optimized_cost_usd": round(max(0.0, total - savings), 6),
        "steps": steps,
        "span_notes": dict(notes),
        "agent_target": target,
        "prompts": recommended_prompts(_fired_rules(recs), target, calls),
        "recommendations": [asdict(r) for r in sort_recommendations(recs)],
    }


async def trace_recommendations(trace: Trace) -> dict:
    spans = await Span.find(Span.trace_id == trace.id).sort("+step_index").to_list()
    return {**analyze_trace(trace, spans, judge_tag()), "pricing": _pricing_status(),
            "evaluation": evaluations.trace_status(trace)}
