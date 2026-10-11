import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { Gauge } from "lucide-react";
import { Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_PROPS, ChartCard, ChartTooltip, Legend } from "@/components/charts";
import { RANGE_LABEL } from "@/components/layout/range-picker";
import { SpanTypeIcon } from "@/components/span-type";
import { StatusBadge } from "@/components/status-badge";
import { MetricList } from "@/components/trace/span-panel";
import { Badge } from "@/components/ui/badge";
import { buttonClass } from "@/components/ui/button";
import { Card, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState } from "@/components/ui/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TBody, Td, Th, THead, Tr } from "@/components/ui/table";
import {
  useEvaluatedTraces,
  useMetricSeries,
  useMetricsSummary,
  useProjectInfo,
  useTrace,
  useWorstSpans,
} from "@/hooks/queries";
import {
  axisTimeFormatter,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatRelative,
  tooltipTimeFormatter,
} from "@/lib/format";
import {
  formatMetricAverage,
  formatMetricValue,
  HIDDEN_TRACE_METRICS,
  isChartable,
  metricHealth,
  metricMeta,
  numericMetricValue,
  TRACE_METRIC_ORDER,
  valueHealth,
  type Health,
} from "@/lib/metrics";
import type { MetricAggregate, MetricResult, SpanNode, TraceTree } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useRange } from "@/store/session";

const HEALTH_TEXT: Record<Health, string> = {
  good: "text-success",
  warn: "text-warning",
  bad: "text-destructive",
  neutral: "text-foreground",
};
const HEALTH_BAR: Record<Health, string> = {
  good: "bg-success",
  warn: "bg-warning",
  bad: "bg-destructive",
  neutral: "bg-primary",
};
const HEALTH_TONE = { good: "success", warn: "warning", bad: "danger", neutral: "neutral" } as const;

interface MetricEntry {
  name: string;
  level: "span" | "trace";
  agg: MetricAggregate;
}

function MetricCard({ entry, selected, onSelect }: { entry: MetricEntry; selected: boolean; onSelect: () => void }) {
  const meta = metricMeta(entry.name);
  const health = metricHealth(entry.name, entry.agg.avg);
  const showBar = (meta.kind === "score" || meta.kind === "rate") && entry.agg.avg != null && entry.agg.avg <= 1;
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "bg-card rounded-xl border p-4 text-left shadow-xs transition-colors",
        selected ? "border-primary ring-primary/25 ring-2" : "hover:border-foreground/20",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-medium">{meta.label}</span>
        <span className="text-muted-foreground text-[11px] whitespace-nowrap">{formatNumber(entry.agg.evaluated_count)} evals</span>
      </div>
      <div className={cn("mt-1.5 text-2xl font-semibold tracking-tight tabular-nums", HEALTH_TEXT[health])}>
        {formatMetricAverage(entry.name, entry.agg.avg)}
      </div>
      {showBar && (
        <div className="bg-muted mt-2 h-1.5 overflow-hidden rounded-full">
          <div className={cn("h-full rounded-full", HEALTH_BAR[health])} style={{ width: `${entry.agg.avg! * 100}%` }} />
        </div>
      )}
      {meta.description && <p className="text-muted-foreground mt-2 line-clamp-2 text-xs">{meta.description}</p>}
    </button>
  );
}

function MetricGrid({
  title,
  entries,
  selected,
  onSelect,
}: {
  title: string;
  entries: MetricEntry[];
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  if (!entries.length) return null;
  return (
    <section className="space-y-2">
      <h3 className="text-muted-foreground text-xs font-semibold tracking-wide uppercase">{title}</h3>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {entries.map((e) => (
          <MetricCard key={e.name} entry={e} selected={e.name === selected} onSelect={() => onSelect(e.name)} />
        ))}
      </div>
    </section>
  );
}

function MetricDetail({ entry }: { entry: MetricEntry }) {
  const range = useRange();
  const navigate = useNavigate();
  const chartable = isChartable(entry.name);
  const series = useMetricSeries(chartable ? entry.name : null, range);
  const worst = useWorstSpans(entry.level === "span" ? entry.name : null, range);
  const xFmt = useMemo(() => axisTimeFormatter(range), [range]);
  const tipFmt = useMemo(() => tooltipTimeFormatter(range), [range]);
  const meta = metricMeta(entry.name);
  const bounded = (meta.kind === "score" || meta.kind === "rate") && (entry.agg.avg ?? 0) <= 1;

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {chartable && (
        <ChartCard
          title={`Trend · ${meta.label}`}
          description={`Average per interval over ${RANGE_LABEL[range]}`}
          loading={series.isLoading}
          height={260}
          action={
            <Legend
              items={[
                { label: "Average", color: "var(--chart-1)" },
                { label: "Evaluations", color: "var(--muted-foreground)" },
              ]}
            />
          }
        >
          <ResponsiveContainer>
            <ComposedChart data={series.data?.points ?? []} margin={{ top: 4, right: 4, bottom: 0, left: -4 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="t" tickFormatter={xFmt} {...AXIS_PROPS} />
              <YAxis
                yAxisId="avg"
                domain={bounded ? [0, 1] : ["auto", "auto"]}
                tickFormatter={(v) => formatMetricAverage(entry.name, v)}
                {...AXIS_PROPS}
                width={52}
              />
              <YAxis yAxisId="count" orientation="right" allowDecimals={false} hide />
              <Tooltip
                cursor={{ fill: "var(--muted)", opacity: 0.5 }}
                content={
                  <ChartTooltip
                    labelFormatter={tipFmt}
                    format={(v, key) => (key === "count" ? formatNumber(v) : formatMetricAverage(entry.name, v))}
                  />
                }
              />
              <Bar yAxisId="count" dataKey="count" name="Evaluations" fill="var(--muted-foreground)" fillOpacity={0.18} />
              <Line
                yAxisId="avg"
                dataKey="avg"
                name="Average"
                type="monotone"
                stroke="var(--chart-1)"
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls
              />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {entry.level === "span" ? (
        <Card className={cn("overflow-hidden", !chartable && "xl:col-span-2")}>
          <CardHeader
            title="Responses to review"
            description={`Latest spans below the threshold for "${meta.label}"`}
          />
          {worst.isLoading ? (
            <div className="p-5">
              <Skeleton className="h-40" />
            </div>
          ) : worst.error ? (
            <div className="p-5">
              <ErrorState error={worst.error} />
            </div>
          ) : !worst.data?.spans.length ? (
            <p className="text-muted-foreground px-5 pb-6 text-sm">No spans below the threshold in this period.</p>
          ) : (
            <Table>
              <THead>
                <Tr>
                  <Th>Span</Th>
                  <Th>Value</Th>
                  <Th className="text-right">Latency</Th>
                  <Th className="text-right">When</Th>
                </Tr>
              </THead>
              <TBody>
                {worst.data.spans.map((s) => {
                  const value = s.metrics[entry.name] ?? null;
                  return (
                    <Tr
                      key={s.span_id}
                      className="hover:bg-muted/50 cursor-pointer"
                      onClick={() => navigate(`/traces/${s.trace_id}?span=${s.span_id}`)}
                    >
                      <Td className="max-w-72">
                        <div className="flex items-center gap-2">
                          <SpanTypeIcon type={s.span_type} />
                          <div className="min-w-0">
                            <div className="truncate font-medium">{s.name ?? s.model ?? s.span_type}</div>
                            {s.response_preview && (
                              <div className="text-muted-foreground truncate text-xs" title={s.response_preview}>
                                {s.response_preview}
                              </div>
                            )}
                          </div>
                        </div>
                      </Td>
                      <Td>
                        <Badge tone={HEALTH_TONE[valueHealth(entry.name, value)]}>{formatMetricValue(entry.name, value)}</Badge>
                      </Td>
                      <Td className="text-right tabular-nums">{formatDuration(s.latency_ms)}</Td>
                      <Td className="text-muted-foreground text-right text-xs whitespace-nowrap">{formatRelative(s.timestamp)}</Td>
                    </Tr>
                  );
                })}
              </TBody>
            </Table>
          )}
        </Card>
      ) : (
        <EvaluatedTracesCard metric={entry.name} />
      )}
    </div>
  );
}

const HEALTH_ORDER: Record<Health, number> = { bad: 0, warn: 1, neutral: 2, good: 3 };

/** Verdict of each trace on a trace-level metric with the judge's reason, failures first. */
function EvaluatedTracesCard({ metric }: { metric: string }) {
  const range = useRange();
  const navigate = useNavigate();
  const evaluated = useEvaluatedTraces(metric, range);
  const meta = metricMeta(metric);
  const rows = useMemo(
    () =>
      (evaluated.data?.traces ?? [])
        .map((t, i) => ({ ...t, health: valueHealth(metric, t.value), i }))
        .sort((a, b) => HEALTH_ORDER[a.health] - HEALTH_ORDER[b.health] || a.i - b.i),
    [evaluated.data, metric],
  );
  const shown = evaluated.data?.traces.length ?? 0;

  return (
    <Card className="overflow-hidden xl:col-span-2">
      <CardHeader
        title="Evaluated traces"
        description={
          evaluated.data && evaluated.data.count > shown
            ? `"${meta.label}" · the ${shown} most recent of ${evaluated.data.count}, failures first`
            : `Verdict of each trace for "${meta.label}", failures first`
        }
      />
      {evaluated.isLoading ? (
        <div className="p-5">
          <Skeleton className="h-40" />
        </div>
      ) : evaluated.error ? (
        <div className="p-5">
          <ErrorState error={evaluated.error} />
        </div>
      ) : !rows.length ? (
        <p className="text-muted-foreground px-5 pb-6 text-sm">No traces evaluated by the current judge in this period.</p>
      ) : (
        <Table>
          <THead>
            <Tr>
              <Th>Trace</Th>
              <Th>Verdict</Th>
              <Th>Judge's reason</Th>
              <Th className="text-right">When</Th>
            </Tr>
          </THead>
          <TBody>
            {rows.map((t) => (
              <Tr key={t.trace_id} className="hover:bg-muted/50 cursor-pointer" onClick={() => navigate(`/traces/${t.trace_id}`)}>
                <Td className="max-w-64">
                  <div className="truncate font-medium" title={t.name ?? t.trace_id}>
                    {t.name ?? t.trace_id}
                  </div>
                  <div className="mt-0.5">
                    <StatusBadge status={t.status} />
                  </div>
                </Td>
                <Td>
                  <Badge tone={HEALTH_TONE[t.health]}>{formatMetricValue(metric, t.value)}</Badge>
                </Td>
                <Td className="text-muted-foreground max-w-xl text-xs">
                  <p className="line-clamp-3" title={t.reason ?? undefined}>
                    {t.reason ?? "—"}
                  </p>
                </Td>
                <Td className="text-muted-foreground text-right text-xs whitespace-nowrap">{formatRelative(t.started_at)}</Td>
              </Tr>
            ))}
          </TBody>
        </Table>
      )}
    </Card>
  );
}

/** Why LLM-judged metrics may be missing or stale: judge not configured or failing. */
function JudgeBanner() {
  const judge = useProjectInfo().data?.judge;
  if (!judge) return null;
  const failing = judge.last_error_at && (!judge.last_ok_at || judge.last_error_at > judge.last_ok_at);
  if (judge.configured && !failing) return null;
  return (
    <Card className="border-destructive/40 bg-destructive/5 p-4 text-sm">
      <p className="text-destructive font-medium">
        {judge.configured ? `The ${judge.model} judge is failing` : "No judge configured on the server"}
        {judge.last_error_at && <span className="font-normal"> · {formatRelative(judge.last_error_at)}</span>}
      </p>
      <p className="text-muted-foreground mt-1 text-xs break-words">
        {judge.last_error ??
          "Set EVAL_LLM_PROVIDER, EVAL_LLM_MODEL and EVAL_LLM_API_KEY in .env, then restart the backend."}{" "}
        LLM-judged metrics (task completion, faithfulness…) won't update while the error persists.
      </p>
    </Card>
  );
}

const byLabel = (a: MetricEntry, b: MetricEntry) => metricMeta(a.name).label.localeCompare(metricMeta(b.name).label, "en");

const traceRank = (name: string) => {
  const i = TRACE_METRIC_ORDER.indexOf(name);
  return i === -1 ? TRACE_METRIC_ORDER.length : i;
};

/** Trace metrics still computed, in reading order. */
const visibleTraceEntries = (entries: MetricEntry[]) =>
  entries.filter((e) => !HIDDEN_TRACE_METRICS.has(e.name)).sort((a, b) => traceRank(a.name) - traceRank(b.name));

/** Average per metric name of the given results (one trace's spans, or the trace itself). */
function aggregate(results: MetricResult[], level: MetricEntry["level"]): MetricEntry[] {
  const byName = new Map<string, { sum: number; n: number; count: number }>();
  for (const m of results) {
    if (metricMeta(m.name).kind === "object") continue;
    const acc = byName.get(m.name) ?? { sum: 0, n: 0, count: 0 };
    const value = numericMetricValue(m.value);
    acc.count += 1;
    if (value !== null) {
      acc.sum += value;
      acc.n += 1;
    }
    byName.set(m.name, acc);
  }
  return [...byName]
    .map(([name, acc]) => ({ name, level, agg: { avg: acc.n ? acc.sum / acc.n : null, evaluated_count: acc.count } }))
    .sort(byLabel);
}

/** Results of one metric on a single trace: its verdict, or every evaluated span of the trace. */
function TraceMetricDetail({ entry, trace }: { entry: MetricEntry; trace: TraceTree }) {
  const navigate = useNavigate();
  const meta = metricMeta(entry.name);
  if (entry.level === "trace") {
    return (
      <Card>
        <CardHeader title={`Verdict · ${meta.label}`} description="Judge's result for this trace" />
        <div className="px-5 pb-5">
          <MetricList metrics={trace.metrics.filter((m) => m.name === entry.name)} />
        </div>
      </Card>
    );
  }
  const rows = trace.spans
    .map((s) => ({ span: s, metric: s.metrics?.find((m) => m.name === entry.name) }))
    .filter((r): r is { span: SpanNode; metric: MetricResult } => !!r.metric)
    .map((r) => ({ ...r, health: valueHealth(entry.name, r.metric.value) }))
    .sort((a, b) => HEALTH_ORDER[a.health] - HEALTH_ORDER[b.health] || a.span.step_index - b.span.step_index);
  return (
    <Card className="overflow-hidden">
      <CardHeader title={`Evaluated spans · ${meta.label}`} description="Worst first" />
      <Table>
        <THead>
          <Tr>
            <Th>Span</Th>
            <Th>Value</Th>
            <Th>Reason</Th>
            <Th className="text-right">Latency</Th>
          </Tr>
        </THead>
        <TBody>
          {rows.map(({ span, metric, health }) => (
            <Tr
              key={span.id}
              className="hover:bg-muted/50 cursor-pointer"
              onClick={() => navigate(`/traces/${trace.trace.id}?span=${span.id}`)}
            >
              <Td className="max-w-64">
                <div className="flex items-center gap-2">
                  <SpanTypeIcon type={span.span_type} />
                  <span className="truncate font-medium">{span.name ?? span.span_type}</span>
                </div>
              </Td>
              <Td>
                <Badge tone={HEALTH_TONE[health]}>{formatMetricValue(entry.name, metric.value)}</Badge>
              </Td>
              <Td className="text-muted-foreground max-w-xl text-xs">
                <p className="line-clamp-2" title={metric.reason ?? undefined}>
                  {metric.reason ?? "—"}
                </p>
              </Td>
              <Td className="text-right tabular-nums">{formatDuration(span.latency_ms)}</Td>
            </Tr>
          ))}
        </TBody>
      </Table>
    </Card>
  );
}

/** Metrics of the trace the user came from (/metrics?trace=<id>). */
function TraceMetrics({ traceId }: { traceId: string }) {
  const { data, error, isLoading } = useTrace(traceId);
  const [picked, setPicked] = useState<string | null>(null);
  const { spanEntries, traceEntries } = useMemo(
    () => ({
      spanEntries: aggregate(data?.spans.flatMap((s) => s.metrics ?? []) ?? [], "span"),
      traceEntries: visibleTraceEntries(aggregate(data?.metrics ?? [], "trace")),
    }),
    [data],
  );

  if (error) return <ErrorState error={error} />;
  if (isLoading || !data) return <Skeleton className="h-64" />;

  const all = [...traceEntries, ...spanEntries];
  const selected = all.find((e) => e.name === picked) ?? all[0] ?? null;
  const t = data.trace;
  return (
    <div className="space-y-6">
      <Card className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="min-w-0">
          <p className="text-muted-foreground text-xs">Trace metrics</p>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="truncate text-lg font-semibold tracking-tight">{t.name || "Untitled trace"}</h2>
            <StatusBadge status={t.status} />
            <span className="text-muted-foreground text-xs">{formatDateTime(t.started_at)}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Link to={`/traces/${traceId}`} className={buttonClass("outline", "sm")}>
            Open trace
          </Link>
          <Link to="/metrics" className={buttonClass("ghost", "sm")}>
            All traces
          </Link>
        </div>
      </Card>
      {!all.length ? (
        <Card>
          <EmptyState icon={Gauge} title="This trace hasn't been evaluated yet" description="Metrics are computed when the trace ends." />
        </Card>
      ) : (
        <>
          <MetricGrid title="Agent behavior (trace)" entries={traceEntries} selected={selected?.name ?? null} onSelect={setPicked} />
          <MetricGrid title="LLM calls in this trace" entries={spanEntries} selected={selected?.name ?? null} onSelect={setPicked} />
          {selected && <TraceMetricDetail key={selected.name} entry={selected} trace={data} />}
        </>
      )}
    </div>
  );
}

export function MetricsPage() {
  const [params] = useSearchParams();
  const traceId = params.get("trace");
  return traceId ? <TraceMetrics key={traceId} traceId={traceId} /> : <ProjectMetrics />;
}

function ProjectMetrics() {
  const range = useRange();
  const summary = useMetricsSummary(range);
  const [picked, setPicked] = useState<string | null>(null);

  const { spanEntries, traceEntries, pending } = useMemo(() => {
    const data = summary.data;
    const toEntries = (rec: Record<string, MetricAggregate> | undefined, level: MetricEntry["level"]) =>
      Object.entries(rec ?? {})
        .filter(([name, agg]) => metricMeta(name).kind !== "object" && agg.evaluated_count > 0)
        .map(([name, agg]) => ({ name, level, agg }))
        .sort(byLabel);
    const spanEntries = toEntries(data?.span_metrics, "span");
    const traceEntries = visibleTraceEntries(toEntries(data?.trace_metrics, "trace"));
    const seen = new Set([...spanEntries, ...traceEntries].map((e) => e.name));
    const pending = (data?.enabled_metrics ?? []).filter(
      (m) => !seen.has(m) && !HIDDEN_TRACE_METRICS.has(m) && metricMeta(m).kind !== "object",
    );
    return { spanEntries, traceEntries, pending };
  }, [summary.data]);

  const all = [...spanEntries, ...traceEntries];
  const selected = all.find((e) => e.name === picked) ?? all.find((e) => isChartable(e.name)) ?? all[0] ?? null;

  if (summary.error) return <ErrorState error={summary.error} />;

  if (summary.isLoading) {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 8 }, (_, i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
    );
  }

  if (!all.length) {
    return (
      <div className="space-y-4">
      <JudgeBanner />
      <Card>
        <EmptyState
          icon={Gauge}
          title={`No evaluations in ${RANGE_LABEL[range]}`}
          description={
            <>
              Metrics are computed by the evaluation LLM at the end of each trace. Check that the server has an
              evaluation LLM configured and that metrics are enabled on the project
              {pending.length > 0 && <> (enabled: {pending.map((m) => metricMeta(m).label).join(", ")})</>}.
            </>
          }
        />
      </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <JudgeBanner />
      <MetricGrid title="Agent behavior (traces)" entries={traceEntries} selected={selected?.name ?? null} onSelect={setPicked} />
      <MetricGrid
        title="Response quality · chatbot and Q&A (LLM spans)"
        entries={spanEntries}
        selected={selected?.name ?? null}
        onSelect={setPicked}
      />
      {pending.length > 0 && (
        <p className="text-muted-foreground text-xs">
          Enabled, no results in this period: {pending.map((m) => metricMeta(m).label).join(", ")}.
        </p>
      )}
      {selected && <MetricDetail key={selected.name} entry={selected} />}
    </div>
  );
}
