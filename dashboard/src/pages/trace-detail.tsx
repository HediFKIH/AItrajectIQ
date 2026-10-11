import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { ArrowLeft, ChevronsDownUp, ChevronsUpDown, SearchX } from "lucide-react";
import { FollowLiveToggle } from "@/components/follow-live";
import { CopyButton } from "@/components/json-view";
import { ModeBadges, SourceBadge, StatusBadge } from "@/components/status-badge";
import { GraphView } from "@/components/trace/graph-view";
import { TraceRecommendationsCard } from "@/components/trace/recommendations-card";
import { MetricList, SpanPanel } from "@/components/trace/span-panel";
import { buildRows } from "@/components/trace/tree";
import { Waterfall } from "@/components/trace/waterfall";
import { Badge } from "@/components/ui/badge";
import { Button, buttonClass } from "@/components/ui/button";
import { Card, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState } from "@/components/ui/empty-state";
import { Segmented } from "@/components/ui/segmented";
import { Skeleton } from "@/components/ui/skeleton";
import { useTrace, useTraceRecommendations } from "@/hooks/queries";
import { ApiError } from "@/lib/api";
import { formatCost, formatDateTime, formatDuration, formatNumber } from "@/lib/format";
import {
  ANSWER_METRICS,
  formatMetricValue,
  HIDDEN_TRACE_METRICS,
  metricMeta,
  TRACE_METRIC_ORDER,
  valueHealth,
} from "@/lib/metrics";
import type { MetricResult } from "@/lib/types";
import { cn } from "@/lib/utils";

const HEALTH_TONE = { good: "success", warn: "warning", bad: "danger", neutral: "neutral" } as const;

type View = "graph" | "timeline";

function HeaderStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card rounded-lg border px-3 py-2">
      <div className="text-muted-foreground text-[11px]">{label}</div>
      <div className="text-sm font-semibold tabular-nums">{value}</div>
    </div>
  );
}

const METRIC_ID_PREFIX = "trace-metric-";

function TraceMetricChips({ metrics, onPick }: { metrics: MetricResult[]; onPick: (name: string) => void }) {
  if (!metrics.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {metrics.map((m) => (
        <button
          key={m.name}
          type="button"
          onClick={() => onPick(m.name)}
          title={m.reason ? `${m.reason}\n\nClick to see details` : "Click to see details"}
          className={cn("rounded-full transition-opacity hover:opacity-80", m.stale && "opacity-50")}
        >
          <Badge tone={HEALTH_TONE[valueHealth(m.name, m.value)]} className="cursor-pointer py-1">
            {metricMeta(m.name).label}: {formatMetricValue(m.name, m.value)}
            {m.stale && " · previous judge"}
          </Badge>
        </button>
      ))}
    </div>
  );
}

/** Span ids that appeared since the previous data (none on first load). */
function useFreshIds(ids: string[] | undefined): Set<string> {
  const known = useRef<Set<string> | null>(null);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  useEffect(() => {
    if (!ids) return;
    if (known.current) {
      const added = ids.filter((id) => !known.current!.has(id));
      if (added.length) setFresh(new Set(added));
    }
    known.current = new Set(ids);
  }, [ids]);
  return fresh;
}

export function TraceDetailPage() {
  const { traceId = "" } = useParams();
  // Remount per trace so selection, collapsed rows and new-span highlights start fresh.
  return <TraceDetail key={traceId} traceId={traceId} />;
}

function TraceDetail({ traceId }: { traceId: string }) {
  const [params, setParams] = useSearchParams();
  const { data, error, isLoading } = useTrace(traceId);
  const advice = useTraceRecommendations(traceId);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [pickedMetric, setPickedMetric] = useState<string | null>(null);
  const traceMetrics = useMemo(() => {
    const rank = (name: string) => {
      const i = TRACE_METRIC_ORDER.indexOf(name);
      return i === -1 ? TRACE_METRIC_ORDER.length : i;
    };
    return (data?.metrics ?? []).filter((m) => !HIDDEN_TRACE_METRICS.has(m.name)).sort((a, b) => rank(a.name) - rank(b.name));
  }, [data?.metrics]);
  const pickMetric = (name: string) => {
    setPickedMetric(name);
    document.getElementById(`${METRIC_ID_PREFIX}${name}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  const spans = data?.spans;
  const rows = useMemo(() => buildRows(spans ?? [], collapsed), [spans, collapsed]);
  const spanIds = useMemo(() => spans?.map((s) => s.id), [spans]);
  const freshIds = useFreshIds(spanIds);

  const selectedId = params.get("span") ?? rows[0]?.span.id ?? null;
  const selected = spans?.find((s) => s.id === selectedId) ?? null;
  const view: View = params.get("view") === "timeline" ? "timeline" : "graph";
  const updateParam = (key: string, value: string) =>
    setParams(
      (p) => {
        const next = new URLSearchParams(p);
        next.set(key, value);
        return next;
      },
      { replace: true },
    );
  const select = (id: string) => updateParam("span", id);
  const setView = (v: View) => updateParam("view", v);
  const toggle = (id: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const parentIds = useMemo(
    () => (spans ?? []).filter((s) => s.sdk_span_id && spans!.some((c) => c.parent_span_id === s.sdk_span_id)).map((s) => s.id),
    [spans],
  );

  if (error) {
    if (error instanceof ApiError && error.status === 404) {
      return (
        <EmptyState
          icon={SearchX}
          title="Trace not found"
          description="It doesn't exist or belongs to another project."
          action={
            <Link to="/traces" className={buttonClass("outline")}>
              Back to traces
            </Link>
          }
        />
      );
    }
    return <ErrorState error={error} />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-16 w-full" />
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_420px]">
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      </div>
    );
  }

  const t = data.trace;
  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="flex items-center justify-between gap-2">
          <Link to="/traces" className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs">
            <ArrowLeft className="size-3.5" />
            Traces
          </Link>
          <FollowLiveToggle />
        </div>
        <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="truncate text-xl font-semibold tracking-tight">{t.name || "Untitled trace"}</h2>
              <StatusBadge status={t.status} />
              <SourceBadge source={t.source} metadata={t.metadata} />
              <ModeBadges metadata={t.metadata} />
            </div>
            <div className="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span className="flex items-center gap-1 font-mono">
                {t.id}
                <CopyButton text={t.id} />
              </span>
              {t.otel_trace_id && (
                <span className="flex items-center gap-1 font-mono">
                  OTel {t.otel_trace_id}
                  <CopyButton text={t.otel_trace_id} />
                </span>
              )}
              <span>{formatDateTime(t.started_at)}</span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <HeaderStat label="Duration" value={t.status === "running" ? "running…" : formatDuration(t.duration_ms ?? data.total_ms)} />
            <HeaderStat label="Cost" value={formatCost(t.total_cost_usd)} />
            <HeaderStat label="Tokens" value={`${formatNumber(t.total_tokens_in)} → ${formatNumber(t.total_tokens_out)}`} />
            <HeaderStat label="Spans" value={`${t.total_spans} · ${t.total_llm_spans} LLM · ${t.total_tool_spans} tools`} />
          </div>
        </div>
        <TraceMetricChips metrics={traceMetrics} onPick={pickMetric} />
        {t.status !== "running" &&
          t.metadata?.source === "claude-code" &&
          traceMetrics.length > 0 &&
          !traceMetrics.some((m) => ANSWER_METRICS.includes(m.name)) && (
            <p className="text-muted-foreground text-xs">
              Task completion, Grounded claims and Instruction adherence need the final answer, which Claude Code
              telemetry doesn't send.{" "}
              <Link to="/connect" className="text-primary hover:underline">
                Install the Claude Code hook
              </Link>{" "}
              to get them.
            </p>
          )}
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_420px]">
        <Card className="overflow-hidden">
          <CardHeader
            title="Execution"
            description={`${data.spans.length} step${data.spans.length !== 1 ? "s" : ""} · ${formatDuration(data.total_ms)}`}
            action={
              <div className="flex items-center gap-2">
                {view === "timeline" && parentIds.length > 0 && (
                  <div className="flex gap-1">
                    <Button variant="ghost" size="sm" onClick={() => setCollapsed(new Set())} title="Expand all">
                      <ChevronsUpDown />
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => setCollapsed(new Set(parentIds))} title="Collapse all">
                      <ChevronsDownUp />
                    </Button>
                  </div>
                )}
                <Segmented
                  value={view}
                  onChange={setView}
                  options={[
                    { value: "graph", label: "Graph" },
                    { value: "timeline", label: "Timeline" },
                  ]}
                />
              </div>
            }
          />
          {rows.length && view === "graph" ? (
            <div className="border-t">
              <GraphView
                spans={data.spans}
                rootLabel={t.name || "Trace"}
                running={t.status === "running"}
                freshIds={freshIds}
                selectedId={selectedId}
                onSelect={select}
              />
            </div>
          ) : rows.length ? (
            <Waterfall
              rows={rows}
              totalMs={data.total_ms}
              selectedId={selectedId}
              collapsed={collapsed}
              freshIds={freshIds}
              onSelect={select}
              onToggle={toggle}
            />
          ) : (
            <div className="text-muted-foreground px-5 py-10 text-center text-sm">
              {t.status === "running" ? "Waiting for the first steps…" : "No steps recorded."}
            </div>
          )}
        </Card>

        <div className="space-y-4 lg:sticky lg:top-4">
          <TraceRecommendationsCard data={advice.data} loading={advice.isLoading} onSelectSpan={select} />
          {traceMetrics.length > 0 && (
            <Card>
              <CardHeader title="Trace evaluation" description="Judge verdict and reasoning for each metric" />
              <div className="px-5 pb-5">
                <MetricList metrics={traceMetrics} idPrefix={METRIC_ID_PREFIX} highlighted={pickedMetric} />
              </div>
            </Card>
          )}
          {selected ? (
            <SpanPanel
              span={selected}
              step={advice.data?.steps.find((s) => s.span_id === selected.id)}
              notes={advice.data?.span_notes[selected.id]}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}
