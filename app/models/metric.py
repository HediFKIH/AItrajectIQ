from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


class MetricType(str,Enum):
    heuristic = "heuristic"
    generic = "generic"
    summarization_eval = "summarization_eval"
    translation_eval = "translation_eval"
    code_eval = "code_eval"
    RAG_eval = "RAG_eval"

class MetricName(str, Enum):   
    general_faithfulness = "general_faithfulness"
    answer_relevance ="answer_relevance"
    completeness ="completeness"
    moderation = "moderation"
    hallucination ="hallucination"

    context_precision="context_precision"
    context_recall="context_recall"
    contextual_faithfulness="contextual_faithfulness"

    comet_score="comet_score"

    summary_completeness="summary_completeness"
    conciseness="conciseness"
    compression_ratio="compression_ratio"

    compilable = "compilable"
    code_relevance= "code_relevance"

    equals="equals"
    contains="contains"
    matches_regex="matches_regex"
    is_json="is_json"
    levenshtein="levenshtein"




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

}


class Metric(BaseModel):
    name: MetricName
    value: Any
    reason: Optional[str]=None