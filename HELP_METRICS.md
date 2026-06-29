# Help: Définir correctement les métriques (MetricName / Metric.value)

Ce document explique comment **choisir**, **configurer** et surtout **définir les métriques** utilisées dans le projet (LLMInsight).  
Il décrit le **contrat de données** (`Metric`) et la **signification des `MetricName`** et de leurs `value`.

---

## 1) Notions de base

### 1.1 Span-level vs Trace-level
- **Span** (niveau étape) : une interaction de type `SpanType.llm` (LLM) ou `SpanType.tool` (outil).
  - Les métriques évaluées sur les spans sont regroupées dans `app/evaluation/runner.py` via `SPAN_EVALUATORS`.
- **Trace** (niveau exécution globale) : regroupe tous les spans d’un run.
  - Les métriques évaluées sur la trace sont regroupées dans `TRACE_EVALUATORS`.

### 1.2 Comment activer les métriques
Les métriques évaluées sont filtrées par `project.enabled_metrics` :
- `evaluate_span(span_id)` ne calcule que les métriques présentes dans `enabled_metrics` et supportées par `SPAN_EVALUATORS`
- `evaluate_trace(trace_id)` ne calcule que celles supportées par `TRACE_EVALUATORS`

> Donc définir une métrique “correctement” signifie : utiliser un **`MetricName` valide** et fournir un environnement de calcul cohérent (champs nécessaires dans `Span`/`Trace`).

---

## 2) Contrat officiel des métriques

### 2.1 Modèle `Metric`
Le type attendu est :

```ts
Metric = {
  name: MetricName,
  value: Any,
  reason?: string | null
}
```

Dans le code : `app/models/metric.py`

- `name` : une valeur de l’enum `MetricName`
- `value` : dépend du metric (float / bool / string / object / list…)
- `reason` : optionnel (`Optional[str]`)

### 2.2 `MetricName` (liste)
Les métriques connues dans le code sont celles de `MetricName` :

**Generic (souvent span-level)**
- `general_faithfulness`
- `answer_relevance`
- `completeness`
- `moderation`
- `hallucination`

**RAG / contexte**
- `context_precision`
- `context_recall`
- `contextual_faithfulness`

**Translation / éval traduction**
- `comet_score`

**Summarization**
- `summary_completeness`
- `conciseness`
- `compression_ratio`

**Code**
- `compilable`
- `code_relevance`

**Heuristiques (souvent “verbatim checks”)**
- `equals`
- `contains`
- `matches_regex`
- `is_json`
- `levenshtein`

**Agent span-level**
- `output_format_compliance`
- `token_efficiency`
- `latency_classification`

**Agent trace-level**
- `task_completion`
- `step_efficiency`
- `tool_error_rate`
- `loop_detection`
- `cost_efficiency`
- `action_consistency`
- `total_cost_summary`

---

## 3) Types de `value` attendus (guide rapide)

> Les types ci-dessous sont alignés sur les implémentations visibles dans `app/evaluation/runner.py` (pour les métriques évaluées automatiquement) et sur `app/proxy/metrics.py` (pour des métriques calculables via RAG/proxy).

### 3.1 Métriques span-level (principalement `runner.py`)
- **`general_faithfulness`**
  - `value`: `float` dans `[0.0, 1.0]`
  - sens: proportion de claims supportés par le contexte (ou validés par web search si activé)

- **`hallucination`**
  - `value`: 
    - mode context-only : `list[str]` = claims hallucinés
    - mode web : `list[str]` = claims contradits
  - sens: liste de claims non supportés / non supportés par la preuve

- **`answer_relevance`**
  - `value`: `float` dans `[0.0, 1.0]`
  - sens: pertinence de la réponse par rapport à la requête

- **`moderation`**
  - `value`: `float` dans `[0.0, 1.0]` (0 safe → 1 unsafe)

- **`is_json`**
  - `value`: `bool`

- **`output_format_compliance`**
  - `value`: `bool` (compliance de format demandé par le prompt)

- **`token_efficiency`**
  - `value`: `float` (ratio)
    - calcul : `tokens_out / tokens_in` (arrondi à 3 décimales)
  - note: la métrique peut retourner `0` si `tokens_in == 0`

- **`latency_classification`**
  - `value`: `string` parmi :
    - `fast` / `normal` / `slow` / `very_slow`

### 3.2 Métriques trace-level (principalement `runner.py`)
- **`task_completion`**
  - `value`: `bool` (la trace est jugée complétée)

- **`step_efficiency`**
  - `value`: `float` dans `[0.0, 1.0]` (ratio d’actions uniques / steps)

- **`tool_error_rate`**
  - `value`: `float` (taux d’erreurs outil, arrondi à 2 décimales)

- **`loop_detection`**
  - `value`: `bool`
  - sens: détecte une répétition d’actions (seuil interne: répétition >= 3)

- **`cost_efficiency`**
  - `value`: `float`
  - sens: coût moyen par step LLM dans la trace

- **`action_consistency`**
  - `value`: `float` dans `[0.0, 1.0]`

- **`total_cost_summary`**
  - `value`: `object` (dict) avec :
    - `llm_calls` : `int`
    - `tokens_in` : `int`
    - `tokens_out` : `int`
    - `total_cost_usd` : `float`
    - `duration_ms` : `int | null`

---

## 4) Comment “définir” les métriques correctement (prérequis de champs)

Pour que l’évaluation fonctionne, veillez à fournir les champs utilisés par le calcul :

### 4.1 Sur `Span` (pour span-level)
Recommandé selon la métrique :
- `messages_in` + `response` : nécessaires pour
  - `general_faithfulness`
  - `hallucination`
  - `answer_relevance`
  - `moderation`
  - `output_format_compliance`
- `response` :
  - `is_json` (parsing JSON)
- `tokens_in` / `tokens_out` :
  - `token_efficiency`
- `latency_ms` :
  - `latency_classification`
- `span_type` :
  - pour que `runner.evaluate_span()` traite uniquement `SpanType.llm`

### 4.2 Sur `Trace` (pour trace-level)
Recommandé :
- `trace.total_cost_usd` + `trace.duration_ms` :
  - `cost_efficiency`
  - `total_cost_summary`
- `spans` correctement renseignés :
  - `tool_error_rate` utilise `tool_error` sur spans de type outil
  - `step_efficiency`, `loop_detection`, `action_consistency` utilisent `response` sur spans LLM
- `trace` doit contenir `spans` avec `step_index` cohérent (tri)

---

## 5) Règles internes importantes (faithfulness vs hallucination)

Dans `app/evaluation/runner.py`, les métriques :
- `general_faithfulness`
- `hallucination`

peuvent utiliser :
1) **context-only** : comparaison stricte avec le contexte d’entrée (`messages_in`)
2) **web_search** (si activé via configuration globale) : extraction de claims + vérification par preuves web

> Donc “définir les métriques” inclut aussi définir **le mode** : si vous activez web search, les métriques changent de logique.

---

## 6) Exemples de JSON de métriques

### 6.1 Span-level (exemple)
```json
{
  "metrics": [
    {
      "name": "general_faithfulness",
      "value": 0.8,
      "reason": "3/4 claims verified ..."
    },
    {
      "name": "hallucination",
      "value": ["Claim inventée ...", "Tool output fabriqué ..."]
    },
    {
      "name": "token_efficiency",
      "value": 0.12,
      "reason": "120 out / 1000 in = good (0.12)"
    }
  ]
}
```

### 6.2 Trace-level (exemple)
```json
{
  "metrics": [
    { "name": "task_completion", "value": true, "reason": "..." },
    { "name": "step_efficiency", "value": 0.6, "reason": "3 unique actions / 5 steps" },
    {
      "name": "total_cost_summary",
      "value": {
        "llm_calls": 4,
        "tokens_in": 1200,
        "tokens_out": 800,
        "total_cost_usd": 0.123456,
        "duration_ms": 5320
      },
      "reason": "4 calls, 2000 tokens, $0.1235"
    }
  ]
}
```

---

## 7) Bonnes pratiques (pour éviter les valeurs inutilisables)

1. **Ne laissez pas `response` vide** pour les métriques basées sur du texte.
2. **Renseignez `tokens_in/tokens_out`** si vous voulez `token_efficiency`.
3. **Renseignez `latency_ms`** si vous voulez `latency_classification`.
4. Pour `hallucination` : attendez-vous à une `list[str]` (pas un score).
5. Pour `
