import { useState } from "react";
import { ArrowRight, ChevronDown, Lightbulb } from "lucide-react";
import { TraceEvaluationNotice } from "@/components/recommendations/evaluation-banner";
import { FixFiles } from "@/components/recommendations/fix-block";
import { PromptPanel } from "@/components/recommendations/prompt-card";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { formatCost, formatNumber } from "@/lib/format";
import { CATEGORY, COMPLEXITY, SEVERITY } from "@/lib/recommendations";
import type { Recommendation, TraceRecommendations } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Distinct reasons, most frequent first, a model name in front when the trace uses several. */
function noAlternativeReasons(data: TraceRecommendations): [string, number][] {
  const models = new Set(data.steps.map((s) => s.model));
  const counts = new Map<string, number>();
  for (const s of data.steps) {
    if (!s.no_alternative_reason) continue;
    const reason = models.size > 1 ? `${s.model}: ${s.no_alternative_reason}` : s.no_alternative_reason;
    counts.set(reason, (counts.get(reason) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "success" }) {
  return (
    <div className="bg-muted/50 rounded-lg px-3 py-2">
      <div className="text-muted-foreground text-[11px]">{label}</div>
      <div className={cn("text-sm font-semibold tabular-nums", tone === "success" && "text-success")}>{value}</div>
    </div>
  );
}

function Item({
  rec,
  open,
  onToggle,
  onSelectSpan,
}: {
  rec: Recommendation;
  open: boolean;
  onToggle: () => void;
  onSelectSpan: (id: string) => void;
}) {
  const Icon = CATEGORY[rec.category].icon;
  const severity = SEVERITY[rec.severity];
  return (
    <div className="rounded-lg border">
      <button type="button" onClick={onToggle} className="flex w-full items-start gap-2.5 px-3 py-2 text-left">
        <Icon className="text-muted-foreground mt-0.5 size-4 shrink-0" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">{rec.title}</div>
          {rec.severity === "high" && (
            <div className="mt-1">
              <Badge tone={severity.tone}>Priority</Badge>
            </div>
          )}
        </div>
        {rec.savings_usd ? (
          <span className="text-success shrink-0 text-sm font-semibold tabular-nums">−{formatCost(rec.savings_usd)}</span>
        ) : null}
        <ChevronDown className={cn("text-muted-foreground mt-0.5 size-4 shrink-0 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="space-y-2 border-t px-3 py-2.5 text-xs">
          <p className="text-muted-foreground">{rec.detail}</p>
          <div className="flex gap-2">
            <ArrowRight className="text-primary mt-0.5 size-3.5 shrink-0" />
            <span>{rec.action}</span>
          </div>
          <FixFiles rec={rec} maxHeight={180} />
          {rec.spans && rec.spans.length > 0 && (
            <ul className="space-y-1">
              {rec.spans.map((ref) => (
                <li key={ref.span_id}>
                  <button
                    type="button"
                    onClick={() => onSelectSpan(ref.span_id)}
                    className="hover:bg-muted flex w-full gap-2 rounded px-1 py-0.5 text-left"
                    title="Show this step"
                  >
                    <span className="text-primary shrink-0 font-mono">#{ref.step}</span>
                    <span className="min-w-0">
                      <span className="block truncate font-medium">{ref.label}</span>
                      {ref.reason && <span className="text-muted-foreground line-clamp-2">{ref.reason}</span>}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export function TraceRecommendationsCard({
  data,
  loading,
  onSelectSpan,
}: {
  data?: TraceRecommendations;
  loading: boolean;
  onSelectSpan: (id: string) => void;
}) {
  const [openId, setOpenId] = useState<string | null | undefined>(undefined);

  if (!data) {
    return loading ? (
      <Card>
        <CardHeader title="Recommendations" />
        <div className="px-5 pb-5">
          <Skeleton className="h-24" />
        </div>
      </Card>
    ) : null;
  }

  const savings = data.savings.total_usd ?? data.savings.model_usd + data.savings.caching_usd;
  const complexity = COMPLEXITY[data.complexity];
  const recs = data.recommendations;
  const whyNoCheaper = data.steps.some((s) => s.alternative) ? [] : noAlternativeReasons(data);
  // The first recommendation is open until the user picks another one.
  const open = openId === undefined ? recs[0]?.id ?? null : openId;

  return (
    <Card>
      <CardHeader
        title="Recommendations"
        description={`${data.agentic ? "Agent run" : "LLM calls"} · ${formatNumber(data.tool_calls)} tool call(s) · max context ${formatNumber(data.max_context_tokens)} tokens`}
        action={<Badge tone={complexity.tone}>Complexity: {complexity.label.toLowerCase()}</Badge>}
      />
      <div className="space-y-3 px-5 pb-5">
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Cost" value={formatCost(data.cost_usd)} />
          <Stat label="Optimized cost" value={formatCost(data.optimized_cost_usd)} />
          <Stat label="Savings" value={savings > 0 ? `−${formatCost(savings)}` : "—"} tone={savings > 0 ? "success" : undefined} />
        </div>
        {data.evaluation && <TraceEvaluationNotice status={data.evaluation} traceId={data.trace_id} />}
        {recs.length === 0 ? (
          <div className="text-muted-foreground flex items-center gap-2 text-xs">
            <Lightbulb className="size-4" />
            {data.evaluation && !data.evaluation.evaluated
              ? "No cost issues found. Quality will be analyzed after evaluation."
              : "Nothing to improve found on this trace."}
          </div>
        ) : (
          <div className="space-y-2">
            {recs.map((rec) => (
              <Item
                key={rec.id}
                rec={rec}
                open={open === rec.id}
                onToggle={() => setOpenId(open === rec.id ? null : rec.id)}
                onSelectSpan={onSelectSpan}
              />
            ))}
          </div>
        )}
        {(data.prompts ?? []).map((p) => (
          <div key={p.id} className="rounded-lg border p-3">
            <PromptPanel prompt={p} traceId={data.trace_id} />
          </div>
        ))}
        {whyNoCheaper.length > 0 && (
          <div className="text-muted-foreground space-y-0.5 text-[11px]">
            <p className="font-medium">Why no cheaper model is suggested:</p>
            <ul className="list-disc space-y-0.5 pl-4">
              {whyNoCheaper.map(([reason, count]) => (
                <li key={reason}>
                  {reason}
                  {count > 1 && ` (${count} calls)`}
                </li>
              ))}
            </ul>
          </div>
        )}
        {data.steps.some((s) => !s.price_known) && (
          <p className="text-muted-foreground text-[11px]">
            Some models in this trace have unknown pricing: their cost is estimated at the default price and no
            savings are calculated for them.
          </p>
        )}
      </div>
    </Card>
  );
}
