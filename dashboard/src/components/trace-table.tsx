import { useNavigate } from "react-router";
import { ModeBadges, SourceBadge, StatusBadge } from "@/components/status-badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TBody, Td, Th, THead, Tr } from "@/components/ui/table";
import { formatCompact, formatCost, formatDateTime, formatDuration, formatRelative } from "@/lib/format";
import { formatMetricValue, metricMeta, valueHealth } from "@/lib/metrics";
import type { TraceRow } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useTouched } from "@/store/live";

const HEALTH_DOT = { good: "bg-success", warn: "bg-warning", bad: "bg-destructive", neutral: "bg-muted-foreground/40" };

function MetricDots({ metrics }: { metrics?: TraceRow["metrics"] }) {
  const entries = Object.entries(metrics ?? {}).filter(([name]) => name !== "total_cost_summary");
  if (!entries.length) return <span className="text-muted-foreground">—</span>;
  return (
    <div className="flex items-center gap-1">
      {entries.map(([name, value]) => (
        <span
          key={name}
          title={`${metricMeta(name).label}: ${formatMetricValue(name, value)}`}
          className={cn("size-2 rounded-full", HEALTH_DOT[valueHealth(name, value)])}
        />
      ))}
    </div>
  );
}

export function TraceTable({
  traces,
  loading,
  compact,
  emptyLabel = "No traces in this period.",
}: {
  traces?: TraceRow[];
  loading?: boolean;
  compact?: boolean;
  emptyLabel?: string;
}) {
  const navigate = useNavigate();
  const touched = useTouched();

  if (loading && !traces) {
    return (
      <div className="space-y-2 p-4">
        {Array.from({ length: compact ? 5 : 8 }).map((_, i) => (
          <Skeleton key={i} className="h-9 w-full" />
        ))}
      </div>
    );
  }
  if (!traces?.length) return <div className="text-muted-foreground px-5 py-10 text-center text-sm">{emptyLabel}</div>;

  return (
    <Table>
      <THead>
        <Tr className="hover:bg-transparent">
          <Th>Status</Th>
          <Th>Name</Th>
          {!compact && <Th>Source</Th>}
          <Th className="text-right">Spans</Th>
          {!compact && <Th className="text-right">Tokens</Th>}
          <Th className="text-right">Cost</Th>
          <Th className="text-right">Duration</Th>
          {!compact && <Th>Evaluation</Th>}
          <Th className="text-right">Started</Th>
        </Tr>
      </THead>
      <TBody>
        {traces.map((t) => (
          <Tr
            key={t.id}
            onClick={() => navigate(`/traces/${t.id}`)}
            className={cn("hover:bg-muted/60 cursor-pointer", touched[t.id] && "animate-flash")}
          >
            <Td>
              <StatusBadge status={t.status} />
            </Td>
            <Td className="max-w-[280px]">
              <div className="truncate font-medium">{t.name || "Untitled trace"}</div>
              {!compact && <div className="text-muted-foreground truncate font-mono text-[11px]">{t.otel_trace_id || t.id}</div>}
            </Td>
            {!compact && (
              <Td>
                <div className="flex flex-wrap items-center gap-1">
                  <SourceBadge source={t.source} metadata={t.metadata} />
                  <ModeBadges metadata={t.metadata} />
                </div>
              </Td>
            )}
            <Td className="text-right tabular-nums">
              <span title={`${t.total_llm_spans} LLM · ${t.total_tool_spans} tools`}>{t.total_spans}</span>
            </Td>
            {!compact && (
              <Td className="text-muted-foreground text-right tabular-nums">
                {formatCompact(t.total_tokens_in)} → {formatCompact(t.total_tokens_out)}
              </Td>
            )}
            <Td className="text-right font-medium tabular-nums">{formatCost(t.total_cost_usd)}</Td>
            <Td className="text-right tabular-nums">{t.status === "running" ? "…" : formatDuration(t.duration_ms)}</Td>
            {!compact && (
              <Td>
                <MetricDots metrics={t.metrics} />
              </Td>
            )}
            <Td className="text-muted-foreground text-right whitespace-nowrap" title={formatDateTime(t.started_at)}>
              {formatRelative(t.started_at)}
            </Td>
          </Tr>
        ))}
      </TBody>
    </Table>
  );
}
