from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class MetricType(str, Enum):
    heuristic = "heuristic"
    generic = "generic"
    summarization_eval = "summarization_eval"
    translation_eval = "translation_eval"
    code_eval = "code_eval"
    RAG_eval = "RAG_eval"
    agent_span = "agent_span"
    agent_trace = "agent_trace"


class MetricName(str, Enum):
    general_faithfulness = "general_faithfulness"
    answer_relevance = "answer_relevance"
    completeness = "completeness"
    moderation = "moderation"
    hallucination = "hallucination"

    context_precision = "context_precision"
    context_recall = "context_recall"
    contextual_faithfulness = "contextual_faithfulness"

    comet_score = "comet_score"

    summary_completeness = "summary_completeness"
    conciseness = "conciseness"
    compression_ratio = "compression_ratio"

    compilable = "compilable"
    code_relevance = "code_relevance"

    equals = "equals"
    contains = "contains"
    matches_regex = "matches_regex"
    is_json = "is_json"
    levenshtein = "levenshtein"

    # --- Agent span-level ---
    output_format_compliance = "output_format_compliance"
    token_efficiency = "token_efficiency"
    latency_classification = "latency_classification"

    # --- Agent trace-level ---
    task_completion = "task_completion"
    step_efficiency = "step_efficiency"
    tool_error_rate = "tool_error_rate"
    loop_detection = "loop_detection"
    cost_efficiency = "cost_efficiency"
    action_consistency = "action_consistency"
    total_cost_summary = "total_cost_summary"



METRIC_TYPE_TO_NAMES = {
    MetricType.generic: [
        MetricName.general_faithfulness,
        MetricName.hallucination,
        MetricName.answer_relevance,
        MetricName.moderation,
        MetricName.completeness,
    ],
    MetricType.heuristic: [
        MetricName.equals,
        MetricName.contains,
        MetricName.is_json,
        MetricName.matches_regex,
        MetricName.levenshtein,
    ],
    MetricType.summarization_eval: [
        MetricName.summary_completeness,
        MetricName.conciseness,
        MetricName.compression_ratio,
    ],
    MetricType.translation_eval: [
        MetricName.comet_score,
    ],
    MetricType.code_eval: [
        MetricName.compilable,
        MetricName.code_relevance,
    ],
    MetricType.RAG_eval: [
        MetricName.context_precision,
        MetricName.context_recall,
        MetricName.contextual_faithfulness,
    ],
    MetricType.agent_span: [
        MetricName.is_json,
        MetricName.output_format_compliance,
        MetricName.token_efficiency,
        MetricName.latency_classification,
    ],
    MetricType.agent_trace: [
        MetricName.task_completion,
        MetricName.step_efficiency,
        MetricName.tool_error_rate,
        MetricName.loop_detection,
        MetricName.cost_efficiency,
        MetricName.action_consistency,
        MetricName.total_cost_summary,
    ],
}


class Metric(BaseModel):
    name: MetricName
    value: Any
    reason: Optional[str]=None