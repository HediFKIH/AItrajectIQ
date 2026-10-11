import { Bot, DatabaseZap, MessageSquareText, ShieldAlert, Shuffle, Wrench, type LucideIcon } from "lucide-react";
import type { Tone } from "@/components/ui/badge";
import { formatCost, formatNumber, formatPercent } from "@/lib/format";
import type { Complexity, Recommendation, RecommendationCategory, RecommendationSeverity } from "@/lib/types";

export const COMPLEXITIES: { key: Complexity; label: string; tone: Tone }[] = [
  { key: "simple", label: "Simple", tone: "success" },
  { key: "moyenne", label: "Medium", tone: "info" },
  { key: "complexe", label: "Complex", tone: "warning" },
];

export const COMPLEXITY = Object.fromEntries(COMPLEXITIES.map((c) => [c.key, c])) as Record<
  Complexity,
  (typeof COMPLEXITIES)[number]
>;

export const SEVERITY: Record<RecommendationSeverity, { label: string; tone: Tone }> = {
  high: { label: "High priority", tone: "danger" },
  medium: { label: "Medium priority", tone: "warning" },
  low: { label: "Low priority", tone: "neutral" },
};

export const CATEGORY: Record<RecommendationCategory, { label: string; icon: LucideIcon }> = {
  harness: { label: "Harness", icon: Wrench },
  model: { label: "Model", icon: Shuffle },
  caching: { label: "Caching", icon: DatabaseZap },
  prompt: { label: "Prompt", icon: MessageSquareText },
  agent: { label: "Agent", icon: Bot },
  reliability: { label: "Reliability", icon: ShieldAlert },
};

export const EVIDENCE_LABEL: Record<string, string> = {
  model: "Model",
  alternative: "Alternative",
  simple_calls: "Simple calls",
  complex_calls: "Complex calls",
  model_calls: "Model calls",
  calls: "Calls",
  bad_calls: "Low-quality calls",
  bad_rate: "Low-quality rate",
  failed_calls: "Failed calls",
  failure_rate: "Failure rate",
  current_cost_usd: "Current cost",
  alternative_cost_usd: "Cost with alternative",
  cost_usd: "Cost",
  cost_share: "Share of cost",
  quality: "Average quality",
  alternative_quality: "Alternative quality",
  prompt_tokens_estimate: "System prompt tokens (estimated)",
  prefix_tokens: "Tokens resent on every call",
  cached_share: "Share read from cache",
  avg_tokens_in: "Avg. input tokens",
  avg_tokens_out: "Avg. output tokens",
  tokens_in: "Input tokens",
  tokens_out: "Output tokens",
  first_context_tokens: "Context at first call",
  peak_context_tokens: "Peak context",
  tool: "Tool",
  tool_calls: "Tool calls",
  failed: "Failed",
  unrecovered: "Never succeeded afterwards",
  bad: "Affected",
  metric: "Metric",
  average: "Average",
  value: "Value",
  evaluated: "Evaluations",
  threshold: "Threshold",
  runs: "Runs analyzed",
  runs_with_edits: "Runs with edits",
  unverified_runs: "Unverified runs",
  check: "Check",
  failed_commands: "Failed commands",
  windows: "Windows",
  commands: "Commands",
  outputs: "Long outputs",
  tokens_estimate: "Estimated tokens",
  reads: "Reads",
  blocked: "Blocked commands",
  failed_edits: "Failed edits",
  central_files: "Central files",
};

export const isHarness = (rec: Recommendation) => rec.category === "harness";

export function formatEvidence(key: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (key.endsWith("_usd")) return formatCost(value);
    if (key.endsWith("_rate") || key.endsWith("_share") || key.includes("quality")) return formatPercent(value, 0);
    if (key === "average" || key === "threshold" || key === "value") return formatPercent(value, 0);
    return formatNumber(value);
  }
  return String(value);
}

/** The headline number of a recommendation: the money saved, or the metric against its threshold. */
export function recImpact(rec: Recommendation): { value: string; hint: string; tone: "success" | "warning" } | null {
  if (rec.savings_usd && rec.savings_usd > 0) {
    const current = rec.evidence.current_cost_usd;
    return {
      value: `−${formatCost(rec.savings_usd)}`,
      hint: typeof current === "number" && current > 0 ? `${formatPercent(rec.savings_usd / current, 0)} cheaper` : "estimated savings",
      tone: "success",
    };
  }
  const { average, threshold } = rec.evidence;
  if (typeof average === "number" && typeof threshold === "number") {
    return { value: formatPercent(average, 0), hint: `threshold ${formatPercent(threshold, 0)}`, tone: "warning" };
  }
  return null;
}
