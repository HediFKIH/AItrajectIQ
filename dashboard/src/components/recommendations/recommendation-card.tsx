import { useState } from "react";
import { Link } from "react-router";
import { ArrowRight, ChevronDown } from "lucide-react";
import { FixFiles } from "@/components/recommendations/fix-block";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { CATEGORY, EVIDENCE_LABEL, formatEvidence, recImpact } from "@/lib/recommendations";
import type { Recommendation } from "@/lib/types";
import { cn } from "@/lib/utils";

/** One recommendation: the problem, its impact, what to do and the text to copy. Figures and
 * examples stay folded under "Details". With `traceId`, the steps of that trace it concerns. */
export function RecommendationCard({ rec, traceId }: { rec: Recommendation; traceId?: string }) {
  const [open, setOpen] = useState(false);
  const Icon = CATEGORY[rec.category].icon;
  const impact = recImpact(rec);
  const evidence = Object.fromEntries(Object.entries(rec.evidence).filter(([k]) => k !== "prompt_preview"));
  const examples = rec.examples ?? [];
  const spans = traceId ? rec.spans ?? [] : [];
  const hasDetails = Object.keys(evidence).length > 0 || examples.length > 0;

  return (
    <Card className="space-y-3 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="bg-accent text-accent-foreground flex size-8 shrink-0 items-center justify-center rounded-lg">
            <Icon className="size-4" />
          </div>
          <div className="min-w-0">
            <h3 className="flex flex-wrap items-center gap-2 text-sm font-semibold tracking-tight">
              {rec.title}
              {rec.severity === "high" && <Badge tone="danger">Priority</Badge>}
            </h3>
            <p className="text-muted-foreground mt-0.5 text-xs">{rec.detail}</p>
          </div>
        </div>
        {impact && (
          <div className="shrink-0 text-right">
            <div className={cn("text-lg font-semibold tabular-nums", impact.tone === "success" ? "text-success" : "text-warning")}>
              {impact.value}
            </div>
            <div className="text-muted-foreground text-[11px]">{impact.hint}</div>
          </div>
        )}
      </div>

      <div className="flex gap-2 text-sm">
        <ArrowRight className="text-primary mt-0.5 size-4 shrink-0" />
        <span>{rec.action}</span>
      </div>

      <FixFiles rec={rec} />

      {spans.length > 0 && (
        <ul className="space-y-1 text-xs">
          {spans.map((ref) => (
            <li key={ref.span_id}>
              <Link
                to={`/traces/${traceId}?span=${ref.span_id}`}
                className="hover:bg-muted flex gap-2 rounded px-1 py-0.5"
                title="Show this step in the trace"
              >
                <span className="text-primary shrink-0 font-mono">#{ref.step}</span>
                <span className="min-w-0">
                  <span className="block truncate font-medium">{ref.label}</span>
                  {ref.reason && <span className="text-muted-foreground line-clamp-2">{ref.reason}</span>}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}

      {hasDetails && (
        <div>
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs"
          >
            Details
            <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
          </button>
          {open && (
            <div className="mt-2 space-y-3">
              {Object.keys(evidence).length > 0 && (
                <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
                  {Object.entries(evidence).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-3 border-b border-dashed py-1">
                      <dt className="text-muted-foreground">{EVIDENCE_LABEL[k] ?? k}</dt>
                      <dd className="truncate font-medium tabular-nums" title={String(v ?? "")}>
                        {formatEvidence(k, v)}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
              {examples.length > 0 && (
                <ul className="space-y-1 text-xs">
                  {examples.map((ex) => (
                    <li key={ex.trace_id} className="flex gap-2">
                      <Link to={`/traces/${ex.trace_id}`} className="text-primary shrink-0 font-mono hover:underline">
                        {ex.trace_id.slice(-8)}
                      </Link>
                      {ex.reason && (
                        <span className="text-muted-foreground line-clamp-2" title={ex.reason}>
                          {ex.reason}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
