import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { ChevronDown, Lightbulb, Wrench } from "lucide-react";
import { KpiCard } from "@/components/kpi-card";
import { RANGE_LABEL } from "@/components/layout/range-picker";
import { EvaluationBanner, TraceEvaluationNotice } from "@/components/recommendations/evaluation-banner";
import { RecommendedPromptCard } from "@/components/recommendations/prompt-card";
import { RecommendationCard } from "@/components/recommendations/recommendation-card";
import { StatusBadge } from "@/components/status-badge";
import { Badge } from "@/components/ui/badge";
import { Button, buttonClass } from "@/components/ui/button";
import { Card, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState } from "@/components/ui/empty-state";
import { Segmented } from "@/components/ui/segmented";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TBody, Td, Th, THead, Tr } from "@/components/ui/table";
import { useRecommendations, useTrace, useTraceRecommendations } from "@/hooks/queries";
import { formatCost, formatDateTime, formatNumber, formatPercent } from "@/lib/format";
import { CATEGORY, COMPLEXITIES, COMPLEXITY, isHarness } from "@/lib/recommendations";
import type {
  ComplexityBucket,
  Recommendation,
  RecommendationCategory,
  Recommendations,
  TraceRecommendations,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { useRange } from "@/store/session";

const TOP = 5;

function ShareBar({ share }: { share: number }) {
  return (
    <div className="bg-muted h-1.5 w-full overflow-hidden rounded-full">
      <div className="bg-primary h-full rounded-full" style={{ width: `${Math.min(Math.max(share * 100, 0), 100)}%` }} />
    </div>
  );
}

function BucketCell({ bucket, total }: { bucket: ComplexityBucket; total: number }) {
  if (!bucket.calls) return <span className="text-muted-foreground">—</span>;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-end gap-2 tabular-nums">
        <span className="font-medium">{formatNumber(bucket.calls)}</span>
        <span className="text-muted-foreground text-xs">{formatCost(bucket.cost_usd)}</span>
      </div>
      <ShareBar share={total ? bucket.calls / total : 0} />
    </div>
  );
}

/** Complexity of the calls per model and pricing provenance, folded by default. */
function DetailedAnalysis({ data }: { data: Recommendations }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-sm font-medium"
      >
        Detailed analysis by model
        <ChevronDown className={cn("size-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <>
          <Card className="overflow-hidden">
            <CardHeader
              title="Request complexity by model"
              description="Simple: short request (≤ 4,000 tokens), short response (≤ 500 tokens), no tools. Complex: very long context, long response or many tools."
            />
            <Table>
              <THead>
                <Tr>
                  <Th>Model</Th>
                  {COMPLEXITIES.map((c) => (
                    <Th key={c.key} className="w-40 text-right">
                      <Badge tone={c.tone}>{c.label}</Badge>
                    </Th>
                  ))}
                  <Th className="text-right">Cost</Th>
                  <Th>Cheaper option</Th>
                </Tr>
              </THead>
              <TBody>
                {data.by_model.map((row) => (
                  <Tr key={row.model}>
                    <Td className="max-w-56">
                      <span className="block truncate font-medium" title={row.model}>
                        {row.model}
                      </span>
                      {!row.price_known && <span className="text-warning text-[11px]">unknown pricing</span>}
                    </Td>
                    {COMPLEXITIES.map((c) => (
                      <Td key={c.key}>
                        <BucketCell bucket={row[c.key]} total={row.calls} />
                      </Td>
                    ))}
                    <Td className="text-right font-medium tabular-nums">{formatCost(row.cost_usd)}</Td>
                    <Td className="text-muted-foreground max-w-56">
                      {row.alternative ?? (
                        <span className="block text-[11px] leading-snug">{row.no_alternative_reason ?? "—"}</span>
                      )}
                    </Td>
                  </Tr>
                ))}
              </TBody>
            </Table>
          </Card>
          <div className="text-muted-foreground space-y-1 text-xs">
            {data.unpriced_models.length > 0 && (
              <p>
                Unknown pricing, excluded from savings calculations:{" "}
                {data.unpriced_models.map((m) => `${m.model} (${formatNumber(m.calls)})`).join(", ")}.
              </p>
            )}
            {data.pricing && (
              <p>
                {data.pricing.source === "web" && data.pricing.updated_at
                  ? `Pricing fetched from the web (LiteLLM, OpenRouter) on ${new Date(data.pricing.updated_at).toLocaleString("en-US")}.`
                  : "Built-in AItrajectIQ pricing: fetching prices from the web hasn't succeeded yet."}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}

type Filter = "all" | RecommendationCategory;

/** What to change around the agent rather than in its prompt, read from the tool calls of its runs.
 * Shown for agent runs even when empty, so a healthy harness reads as checked. */
function HarnessSection({ recs, agent, traceId }: { recs: Recommendation[]; agent: boolean; traceId?: string }) {
  if (recs.length === 0 && !agent) return null;
  return (
    <section className="space-y-3">
      <div>
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <Wrench className="text-primary size-4" />
          Agent harness
        </h2>
        <p className="text-muted-foreground text-xs">
          What to change around the agent, not in its prompt: verification hooks, ignored files, terminal,
          subagents, MCP servers, permissions. Each idea comes from the tool calls
          {traceId ? " of this trace" : " of your runs"} and lists the files to create.
        </p>
      </div>
      {recs.length === 0 ? (
        <Card>
          <EmptyState
            icon={Wrench}
            title="Nothing to change in the harness"
            description="Checks run after edits, with no blocked commands, no generated files read and no failing MCP servers."
          />
        </Card>
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {recs.map((rec) => (
            <RecommendationCard key={rec.id} rec={rec} traceId={traceId} />
          ))}
        </div>
      )}
    </section>
  );
}

/** Cost, cheaper model or why there is none, for each LLM call of the trace. */
function TraceSteps({ data }: { data: TraceRecommendations }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader
        title="LLM calls in this trace"
        description={`${data.agentic ? "Agent run" : "LLM calls"} · ${formatNumber(data.tool_calls)} tool call(s) · max context ${formatNumber(data.max_context_tokens)} tokens`}
      />
      <Table>
        <THead>
          <Tr>
            <Th className="w-14">Step</Th>
            <Th>Model</Th>
            <Th>Complexity</Th>
            <Th className="text-right">Cost</Th>
            <Th>Cheaper option</Th>
          </Tr>
        </THead>
        <TBody>
          {data.steps.map((s) => (
            <Tr key={s.span_id}>
              <Td>
                <Link to={`/traces/${data.trace_id}?span=${s.span_id}`} className="text-primary font-mono hover:underline">
                  #{s.step}
                </Link>
              </Td>
              <Td className="max-w-56">
                <span className="block truncate font-medium" title={s.model}>
                  {s.model}
                </span>
                {!s.price_known && <span className="text-warning text-[11px]">unknown pricing</span>}
              </Td>
              <Td>
                <Badge tone={COMPLEXITY[s.complexity].tone}>{COMPLEXITY[s.complexity].label}</Badge>
              </Td>
              <Td className="text-right tabular-nums">
                <span className="font-medium">{formatCost(s.cost_usd)}</span>
                {s.cost_share > 0 && <span className="text-muted-foreground block text-[11px]">{formatPercent(s.cost_share, 0)}</span>}
              </Td>
              <Td className="max-w-72">
                {s.alternative && s.savings_usd ? (
                  <span className="text-success font-medium">
                    {s.alternative}: −{formatCost(s.savings_usd)}
                  </span>
                ) : (
                  <span className="text-muted-foreground block text-[11px] leading-snug">{s.no_alternative_reason ?? "—"}</span>
                )}
              </Td>
            </Tr>
          ))}
        </TBody>
      </Table>
    </Card>
  );
}

/** Recommendations of the trace the user came from (/recommendations?trace=<id>). */
function TraceRecommendationsView({ traceId }: { traceId: string }) {
  const trace = useTrace(traceId);
  const query = useTraceRecommendations(traceId);
  const data = query.data;

  if (query.error) return <ErrorState error={query.error} />;

  const t = trace.data?.trace;
  const recs = data?.recommendations ?? [];
  const harnessRecs = recs.filter(isHarness);
  const otherRecs = recs.filter((r) => !isHarness(r));
  const high = recs.filter((r) => r.severity === "high").length;
  const savings = data ? data.savings.total_usd ?? data.savings.model_usd + data.savings.caching_usd : 0;
  const loading = query.isLoading || !data;

  return (
    <div className="space-y-6">
      <Card className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="min-w-0">
          <p className="text-muted-foreground text-xs">Trace recommendations</p>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="truncate text-lg font-semibold tracking-tight">{t ? t.name || "Untitled trace" : "…"}</h2>
            {t && <StatusBadge status={t.status} />}
            {t && <span className="text-muted-foreground text-xs">{formatDateTime(t.started_at)}</span>}
          </div>
        </div>
        <div className="flex gap-2">
          <Link to={`/traces/${traceId}`} className={buttonClass("outline", "sm")}>
            Open trace
          </Link>
          <Link to="/recommendations" className={buttonClass("ghost", "sm")}>
            All traces
          </Link>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <KpiCard
          label="Potential savings"
          value={formatCost(savings)}
          hint={data?.cost_usd ? `${formatPercent(savings / data.cost_usd, 0)} of cost` : undefined}
          color="var(--chart-2)"
          loading={loading}
        />
        <KpiCard
          label="Trace cost"
          value={formatCost(data?.cost_usd)}
          hint={data ? `${formatNumber(data.llm_calls)} LLM call(s) · ${formatNumber(data.tool_calls)} tool call(s)` : undefined}
          loading={loading}
        />
        <KpiCard
          label="Issues to fix"
          value={formatNumber(recs.length)}
          hint={high ? `${high} high priority` : undefined}
          color="var(--chart-4)"
          loading={loading}
        />
      </div>

      {data?.evaluation && <TraceEvaluationNotice status={data.evaluation} traceId={traceId} />}

      {data && <HarnessSection recs={harnessRecs} agent={data.agentic} traceId={traceId} />}

      {(data?.prompts ?? []).length > 0 && (
        <section className="space-y-3">
          <div>
            <h2 className="text-sm font-semibold tracking-tight">Recommended prompt</h2>
            <p className="text-muted-foreground text-xs">Instructions that fix the issues in this trace, ready to paste.</p>
          </div>
          <div className="grid gap-3 xl:grid-cols-2">
            {data!.prompts!.map((p) => (
              <RecommendedPromptCard key={`${traceId}-${p.id}`} prompt={p} traceId={traceId} />
            ))}
          </div>
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-tight">Top priorities</h2>
        {loading ? (
          <div className="space-y-3">
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
          </div>
        ) : otherRecs.length === 0 ? (
          <Card>
            <EmptyState
              icon={Lightbulb}
              title="Nothing to report"
              description={
                data?.evaluation && !data.evaluation.evaluated
                  ? "No cost issues found. Quality will be analyzed once the trace is evaluated."
                  : "No cost, prompt or quality issues found on this trace."
              }
            />
          </Card>
        ) : (
          <div className="grid gap-3 xl:grid-cols-2">
            {otherRecs.map((rec) => (
              <RecommendationCard key={rec.id} rec={rec} traceId={traceId} />
            ))}
          </div>
        )}
      </section>

      {data && data.steps.length > 0 && <TraceSteps data={data} />}
    </div>
  );
}

export function RecommendationsPage() {
  const [params] = useSearchParams();
  const traceId = params.get("trace");
  return traceId ? <TraceRecommendationsView key={traceId} traceId={traceId} /> : <ProjectRecommendations />;
}

function ProjectRecommendations() {
  const range = useRange();
  const query = useRecommendations(range);
  const [filter, setFilter] = useState<Filter>("all");
  const [showAll, setShowAll] = useState(false);

  const data = query.data;
  const recs = useMemo(() => data?.recommendations ?? [], [data]);
  const harnessRecs = useMemo(() => recs.filter(isHarness), [recs]);
  const otherRecs = useMemo(() => recs.filter((r) => !isHarness(r)), [recs]);
  const filterOptions = useMemo(() => {
    const present = new Set(otherRecs.map((r) => r.category));
    return [
      { value: "all" as Filter, label: `All (${otherRecs.length})` },
      ...(Object.keys(CATEGORY) as RecommendationCategory[])
        .filter((c) => present.has(c))
        .map((c) => ({ value: c as Filter, label: CATEGORY[c].label })),
    ];
  }, [otherRecs]);
  const active: Filter = filterOptions.some((o) => o.value === filter) ? filter : "all";
  const filtered = active === "all" ? otherRecs : otherRecs.filter((r) => r.category === active);
  const visible = showAll ? filtered : filtered.slice(0, TOP);

  if (query.error) return <ErrorState error={query.error} />;

  const loading = query.isLoading;
  const totalCalls = data?.analyzed_calls ?? 0;
  const totalSavings = data?.savings.total_usd ?? (data?.savings.routing_usd ?? 0) + (data?.savings.caching_usd ?? 0);
  const high = recs.filter((r) => r.severity === "high").length;
  const prompts = data?.prompts ?? [];
  const baselineOnly = prompts.length > 0 && prompts.every((p) => p.baseline);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <KpiCard
          label="Potential savings"
          value={formatCost(totalSavings)}
          hint={data?.total_cost_usd ? `${formatPercent(totalSavings / data.total_cost_usd, 0)} of cost` : undefined}
          color="var(--chart-2)"
          loading={loading}
        />
        <KpiCard
          label="Analyzed cost"
          value={formatCost(data?.total_cost_usd)}
          hint={data ? `${formatNumber(totalCalls)} calls in ${RANGE_LABEL[range]}` : undefined}
          loading={loading}
        />
        <KpiCard
          label="Issues to fix"
          value={formatNumber(recs.length)}
          hint={high ? `${high} high priority` : undefined}
          color="var(--chart-4)"
          loading={loading}
        />
      </div>

      {!loading && data && totalCalls === 0 ? (
        <Card>
          <EmptyState
            icon={Lightbulb}
            title={`No LLM calls in ${RANGE_LABEL[range]}`}
            description="Recommendations are computed from traced calls: cost, prompts and evaluations."
          />
        </Card>
      ) : (
        <>
          {data?.evaluation && <EvaluationBanner status={data.evaluation} range={range} />}

          {data && (
            <HarnessSection
              recs={harnessRecs}
              agent={data.agent_target === "cursor" || data.agent_target === "claude-code"}
            />
          )}

          {prompts.length > 0 && (
            <section className="space-y-3">
              <div>
                <h2 className="text-sm font-semibold tracking-tight">Recommended prompt</h2>
                <p className="text-muted-foreground text-xs">
                  {baselineOnly
                    ? "No issues measured yet: start from these rules, they prevent the most common agent mistakes."
                    : "Instructions that fix the measured issues, ready to paste."}
                </p>
              </div>
              <div className="grid gap-3 xl:grid-cols-2">
                {prompts.map((p) => (
                  <RecommendedPromptCard key={`${range}-${p.id}`} prompt={p} range={range} />
                ))}
              </div>
            </section>
          )}

          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-sm font-semibold tracking-tight">Top priorities</h2>
              {otherRecs.length > TOP && <Segmented value={active} options={filterOptions} onChange={setFilter} />}
            </div>
            {loading ? (
              <div className="space-y-3">
                <Skeleton className="h-32" />
                <Skeleton className="h-32" />
              </div>
            ) : filtered.length === 0 ? (
              <Card>
                <EmptyState
                  icon={Lightbulb}
                  title="Nothing to report"
                  description={
                    data?.evaluation?.judge && data.evaluation.evaluated >= (data.evaluation.min_evaluated ?? 5)
                      ? "No oversized model, no repeated prompt without caching, and every metric is above its threshold."
                      : "No oversized model and no repeated prompt without caching. Quality will be analyzed once traces are evaluated (see above)."
                  }
                />
              </Card>
            ) : (
              <>
                <div className="grid gap-3 xl:grid-cols-2">
                  {visible.map((rec) => (
                    <RecommendationCard key={rec.id} rec={rec} />
                  ))}
                </div>
                {filtered.length > TOP && (
                  <Button variant="outline" size="sm" onClick={() => setShowAll(!showAll)}>
                    {showAll ? "Show less" : `Show ${filtered.length - TOP} more`}
                  </Button>
                )}
              </>
            )}
          </section>

          {data && data.by_model.length > 0 && <DetailedAnalysis data={data} />}
        </>
      )}
    </div>
  );
}
