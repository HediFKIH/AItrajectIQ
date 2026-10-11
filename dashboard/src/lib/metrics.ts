import { formatCost, formatPercent } from "./format";
import type { MetricValue } from "./types";

export type MetricKind = "score" | "rate" | "ratio" | "currency" | "label" | "object";
export type Health = "good" | "warn" | "bad" | "neutral";

export interface MetricMeta {
  label: string;
  description: string;
  kind: MetricKind;
  /** Whether a higher average is better. */
  higherIsBetter: boolean;
  /** [warn, bad] limits on the average, read in the metric's direction. */
  thresholds?: [number, number];
}

const score = (label: string, description: string): MetricMeta => ({
  label,
  description,
  kind: "score",
  higherIsBetter: true,
  thresholds: [0.75, 0.5],
});

export const METRICS: Record<string, MetricMeta> = {
  general_faithfulness: score("Faithfulness", "Is the answer faithful to the prompt and the facts?"),
  answer_relevance: score("Relevance", "Does the answer address the question asked?"),
  completeness: score("Completeness", "Does the answer cover the whole request?"),
  context_precision: score("Context precision", "Share of the retrieved context that is useful."),
  context_recall: score("Context recall", "Share of the required information present in the context."),
  contextual_faithfulness: score("Context faithfulness", "Is the answer grounded in the provided context?"),
  summary_completeness: score("Summary completeness", "Does the summary include the key points?"),
  conciseness: score("Conciseness", "Does the answer avoid unnecessary content?"),
  code_relevance: score("Code relevance", "Does the code address the request?"),
  action_consistency: score("Action consistency", "Do the agent's actions follow logically toward the request?"),
  step_efficiency: score(
    "Non-duplicate actions",
    "Share of tool calls that don't repeat an action already done (same tool on the same file or the same command). A call with an unknown target is never counted as a duplicate.",
  ),
  claims_grounded: {
    label: "Grounded claims",
    description: "Share of claims in the final answer (file edited, tests passed…) backed by the executed actions.",
    kind: "score",
    higherIsBetter: true,
    thresholds: [0.9, 0.7],
  },
  instruction_adherence: score(
    "Instruction adherence",
    "Did the agent do what was asked, without unrequested changes or ignored instructions?",
  ),
  change_verification: {
    label: "Verified changes",
    description: "Share of traces that edit code and then run a test, a lint or a build.",
    kind: "rate",
    higherIsBetter: true,
    thresholds: [0.8, 0.5],
  },
  error_recovery: {
    label: "Error recovery",
    description: "Share of failed tool calls followed by a successful retry.",
    kind: "rate",
    higherIsBetter: true,
    thresholds: [0.8, 0.5],
  },
  comet_score: score("COMET score", "Translation quality."),
  hallucination: {
    label: "Hallucinations",
    description: "Share of answers containing at least one fabricated claim.",
    kind: "rate",
    higherIsBetter: false,
    thresholds: [0.1, 0.25],
  },
  moderation: {
    label: "Moderation",
    description: "Harmful content score (0 = clean).",
    kind: "score",
    higherIsBetter: false,
    thresholds: [0.2, 0.5],
  },
  task_completion: {
    label: "Task completion",
    description: "Share of traces where an LLM judge, given the request, the executed actions and the final answer, finds that the agent did what was asked.",
    kind: "rate",
    higherIsBetter: true,
    thresholds: [0.8, 0.6],
  },
  tool_error_rate: {
    label: "Tool errors",
    description: "Share of tool calls that errored.",
    kind: "rate",
    higherIsBetter: false,
    thresholds: [0.05, 0.15],
  },
  loop_detection: {
    label: "Loops detected",
    description: "Share of traces where the agent repeats the same action (same tool, same target) at least 3 times in a row.",
    kind: "rate",
    higherIsBetter: false,
    thresholds: [0.05, 0.15],
  },
  output_format_compliance: {
    label: "Format compliance",
    description: "Share of answers in the expected format.",
    kind: "rate",
    higherIsBetter: true,
    thresholds: [0.95, 0.8],
  },
  is_json: { label: "Valid JSON", description: "Share of answers that are valid JSON.", kind: "rate", higherIsBetter: true, thresholds: [0.95, 0.8] },
  compilable: { label: "Compilable code", description: "Share of code that compiles.", kind: "rate", higherIsBetter: true, thresholds: [0.9, 0.7] },
  equals: { label: "Exact match", description: "Share of answers equal to the expected output.", kind: "rate", higherIsBetter: true },
  contains: { label: "Contains", description: "Share of answers containing the expected output.", kind: "rate", higherIsBetter: true },
  matches_regex: { label: "Regex", description: "Share of answers that match the pattern.", kind: "rate", higherIsBetter: true },
  token_efficiency: {
    label: "Output / input ratio",
    description: "Tokens generated per token sent.",
    kind: "ratio",
    higherIsBetter: true,
  },
  compression_ratio: { label: "Compression ratio", description: "Summary length / source text length.", kind: "ratio", higherIsBetter: false },
  levenshtein: { label: "Levenshtein distance", description: "Distance from the expected answer.", kind: "ratio", higherIsBetter: false },
  cost_efficiency: {
    label: "Cost per step",
    description: "Average cost of a productive LLM call.",
    kind: "currency",
    higherIsBetter: false,
  },
  latency_classification: {
    label: "Latency class",
    description: "fast / normal / slow / very_slow per call.",
    kind: "label",
    higherIsBetter: true,
  },
  total_cost_summary: {
    label: "Cost summary",
    description: "Tokens / cost / duration summary per trace.",
    kind: "object",
    higherIsBetter: false,
  },
};

/** Trace-level metrics in the order a user reads them: was it done, can the answer be
 * trusted, then how the agent worked. */
export const TRACE_METRIC_ORDER = ["task_completion", "claims_grounded", "instruction_adherence", "tool_error_rate"];

/** No longer computed on traces; older traces may still hold them. */
export const HIDDEN_TRACE_METRICS = new Set([
  "total_cost_summary",
  "cost_efficiency",
  "action_consistency",
  "loop_detection",
  "step_efficiency",
  "change_verification",
  "error_recovery",
]);

/** Metrics that need the agent's final answer. */
export const ANSWER_METRICS = ["task_completion", "claims_grounded", "instruction_adherence"];

export function metricMeta(name: string): MetricMeta {
  return (
    METRICS[name] ?? {
      label: name.replace(/_/g, " "),
      description: "",
      kind: "score",
      higherIsBetter: true,
    }
  );
}

export function isChartable(name: string): boolean {
  const kind = metricMeta(name).kind;
  return kind !== "label" && kind !== "object";
}

export function formatMetricAverage(name: string, avg: number | null | undefined): string {
  if (avg === null || avg === undefined) return "—";
  const { kind } = metricMeta(name);
  if (kind === "currency") return formatCost(avg);
  if (kind === "ratio") return avg.toLocaleString("en-US", { maximumFractionDigits: 3 });
  if (kind === "score" && avg > 1) return avg.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return formatPercent(avg, 1);
}

export function metricHealth(name: string, avg: number | null | undefined): Health {
  const meta = metricMeta(name);
  if (avg === null || avg === undefined || !meta.thresholds) return "neutral";
  const [warn, bad] = meta.thresholds;
  if (meta.higherIsBetter) return avg >= warn ? "good" : avg >= bad ? "warn" : "bad";
  return avg <= warn ? "good" : avg <= bad ? "warn" : "bad";
}

/** A result as a number to average, like the backend: booleans 0/1, a list 1 when non-empty. */
export function numericMetricValue(value: MetricValue): number | null {
  if (Array.isArray(value)) return value.length > 0 ? 1 : 0;
  if (typeof value === "boolean") return value ? 1 : 0;
  return typeof value === "number" ? value : null;
}

/** Health of one evaluation result (span or trace level). */
export function valueHealth(name: string, value: MetricValue): Health {
  if (Array.isArray(value)) return metricHealth(name, value.length > 0 ? 1 : 0);
  if (typeof value === "boolean") return metricHealth(name, value ? 1 : 0);
  if (typeof value === "number") return metricHealth(name, value);
  if (name === "latency_classification" && typeof value === "string") {
    return value === "fast" || value === "normal" ? "good" : value === "slow" ? "warn" : "bad";
  }
  return "neutral";
}

export function formatMetricValue(name: string, value: MetricValue): string {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.length === 0 ? "none" : `${value.length}`;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    const { kind } = metricMeta(name);
    if (kind === "currency") return formatCost(value);
    if (kind === "ratio" || value > 1) return value.toLocaleString("en-US", { maximumFractionDigits: 3 });
    return formatPercent(value, 0);
  }
  if (typeof value === "string") return value;
  return "object";
}
