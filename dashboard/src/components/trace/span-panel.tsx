import type { ReactNode } from "react";
import { AlertTriangle, Lightbulb } from "lucide-react";
import { CodeBlock, CopyButton } from "@/components/json-view";
import { SPAN_TYPES, SpanTypeIcon } from "@/components/span-type";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { formatCost, formatDuration, formatNumber } from "@/lib/format";
import { formatMetricValue, metricMeta, valueHealth } from "@/lib/metrics";
import { COMPLEXITY } from "@/lib/recommendations";
import type { Message, MetricResult, SpanNode, TraceStep } from "@/lib/types";
import { cn } from "@/lib/utils";
import { spanHasError, spanLabel } from "./tree";

const HEALTH_TONE = { good: "success", warn: "warning", bad: "danger", neutral: "neutral" } as const;
const ROLE_TONE: Record<string, "primary" | "info" | "neutral" | "warning"> = {
  system: "warning",
  user: "info",
  assistant: "primary",
  tool: "neutral",
};

function Section({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-muted-foreground text-[11px] font-semibold tracking-wide uppercase">{title}</h4>
        {action}
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-muted-foreground text-[11px]">{label}</div>
      <div className="truncate text-sm font-medium tabular-nums">{value}</div>
    </div>
  );
}

export function MetricList({
  metrics,
  idPrefix,
  highlighted,
}: {
  metrics: MetricResult[];
  /** Gives each item the DOM id `${idPrefix}${name}` so it can be scrolled to. */
  idPrefix?: string;
  highlighted?: string | null;
}) {
  if (!metrics.length) return <p className="text-muted-foreground text-xs">Not evaluated yet.</p>;
  return (
    <div className="space-y-2">
      {metrics.map((m) => {
        const health = valueHealth(m.name, m.value);
        return (
          <div
            key={m.name}
            id={idPrefix ? `${idPrefix}${m.name}` : undefined}
            className={cn(
              "scroll-mt-4 rounded-lg border px-3 py-2 transition-colors",
              highlighted === m.name && "border-primary ring-primary/25 ring-2",
              m.stale && "opacity-60",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium">{metricMeta(m.name).label}</span>
              <Badge tone={HEALTH_TONE[health]}>{formatMetricValue(m.name, m.value)}</Badge>
            </div>
            {m.stale && (
              <p className="text-warning mt-1 text-[11px]">
                Verdict from a previous judge{m.judge ? ` (${m.judge})` : ""}: excluded from averages, run a re-evaluation.
              </p>
            )}
            {Array.isArray(m.value) && m.value.length > 0 && (
              <ul className="text-muted-foreground mt-1.5 list-disc space-y-0.5 pl-4 text-xs">
                {m.value.map((claim, i) => (
                  <li key={i}>{typeof claim === "string" ? claim : JSON.stringify(claim)}</li>
                ))}
              </ul>
            )}
            {m.reason && <p className="text-muted-foreground mt-1 text-xs">{m.reason}</p>}
          </div>
        );
      })}
    </div>
  );
}

function Messages({ messages }: { messages: Message[] }) {
  return (
    <div className="space-y-2">
      {messages.map((m, i) => (
        <div key={i} className="space-y-1">
          <Badge tone={ROLE_TONE[m.role ?? ""] ?? "neutral"}>{m.role ?? "message"}</Badge>
          <CodeBlock value={m.content ?? JSON.stringify(m)} maxHeight={220} />
        </div>
      ))}
    </div>
  );
}

function Advice({ step, notes }: { step?: TraceStep; notes?: string[] }) {
  if (!step && !notes?.length) return null;
  return (
    <Section title="Recommendations">
      <div className="bg-accent/40 space-y-2 rounded-lg p-3 text-xs">
        {step && (
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={COMPLEXITY[step.complexity].tone}>{COMPLEXITY[step.complexity].label} task</Badge>
            {step.cost_share > 0 && <Badge>{Math.round(step.cost_share * 100)}% of trace cost</Badge>}
            {step.alternative && step.savings_usd ? (
              <Badge tone="success">
                {step.alternative}: −{formatCost(step.savings_usd)}
              </Badge>
            ) : null}
          </div>
        )}
        {step?.no_alternative_reason && (
          <p className="text-muted-foreground">No cheaper model: {step.no_alternative_reason}</p>
        )}
        {notes && notes.length > 0 && (
          <ul className="space-y-1">
            {notes.map((note, i) => (
              <li key={i} className="flex gap-1.5">
                <Lightbulb className="text-primary mt-0.5 size-3.5 shrink-0" />
                <span>{note}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Section>
  );
}

export function SpanPanel({ span, step, notes }: { span: SpanNode; step?: TraceStep; notes?: string[] }) {
  const meta = SPAN_TYPES[span.span_type] ?? SPAN_TYPES.chain;
  const error = span.error || span.tool_error;
  const isModel = span.span_type === "llm" || span.span_type === "embedding";
  const isTool = span.span_type === "tool" || span.span_type === "retriever";
  const usage = (span.metadata?.usage ?? {}) as Record<string, unknown>;
  const cacheRead = Number(usage.cache_read_tokens ?? 0);
  const cacheWrite = Number(usage.cache_write_tokens ?? 0);
  const freshInput = Math.max(0, (span.tokens_in ?? 0) - cacheRead - cacheWrite);
  const modelCalls = Number(span.metadata?.model_calls ?? 0);
  const avgContext = Number(span.metadata?.avg_context_tokens ?? 0);

  return (
    <Card className="overflow-hidden">
      <div className="flex items-start gap-3 border-b px-4 py-3">
        <SpanTypeIcon type={span.span_type} className="mt-0.5 size-7" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold">{spanLabel(span)}</div>
          <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
            {meta.label}
            {span.sdk_span_id && (
              <>
                <span>·</span>
                <span className="truncate font-mono">{span.sdk_span_id}</span>
                <CopyButton text={span.sdk_span_id} />
              </>
            )}
          </div>
        </div>
        {spanHasError(span) ? (
          <Badge tone="danger">Error</Badge>
        ) : span.status === "running" ? (
          <Badge tone="neutral">Running</Badge>
        ) : (
          <Badge tone="success">OK</Badge>
        )}
      </div>

      <div className="space-y-5 p-4">
        <div className="grid grid-cols-3 gap-3">
          <Stat label="Duration" value={formatDuration(span.duration_ms)} />
          <Stat label="Start" value={`+${formatDuration(span.start_offset_ms)}`} />
          {isModel && <Stat label="Cost" value={formatCost(span.cost_usd)} />}
          {isModel && <Stat label="Model" value={<span title={span.model ?? ""}>{span.model ?? "—"}</span>} />}
          {isModel && <Stat label="Tokens" value={`${formatNumber(span.tokens_in)} → ${formatNumber(span.tokens_out)}`} />}
          {isModel && cacheRead + cacheWrite > 0 && <Stat label="Fresh input" value={formatNumber(freshInput)} />}
          {isModel && cacheRead > 0 && <Stat label="Cache read" value={formatNumber(cacheRead)} />}
          {isModel && cacheWrite > 0 && <Stat label="Cache write" value={formatNumber(cacheWrite)} />}
          {isModel && modelCalls > 0 && <Stat label="Model calls" value={formatNumber(modelCalls)} />}
          {isModel && avgContext > 0 && (
            <Stat
              label="Avg. context"
              value={<span title="Input tokens (including cache) per model call">{formatNumber(avgContext)}</span>}
            />
          )}
          {isModel && span.metadata?.price === "default" && (
            <Stat label="Price" value={<span title="Model missing from the price table: default pricing">default</span>} />
          )}
          {isModel && span.temperature != null && <Stat label="Temperature" value={span.temperature} />}
          {isTool && <Stat label="Tool" value={span.tool_name ?? "—"} />}
          {span.agent_name && <Stat label="Agent" value={span.agent_name} />}
          {span.provider && <Stat label="Provider" value={span.provider} />}
        </div>

        <Advice step={step} notes={notes} />

        {error && (
          <div className="border-destructive/30 bg-destructive/5 text-destructive flex gap-2 rounded-lg border p-3 text-xs">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <pre className="font-mono break-words whitespace-pre-wrap">{error}</pre>
          </div>
        )}

        {isModel && span.messages_in && span.messages_in.length > 0 && (
          <Section title={`Input · ${span.messages_in.length} message${span.messages_in.length > 1 ? "s" : ""}`}>
            <Messages messages={span.messages_in} />
          </Section>
        )}
        {isModel && span.response && (
          <Section title="Response">
            <CodeBlock value={span.response} />
          </Section>
        )}
        {isTool && span.tool_input && (
          <Section title="Tool input">
            <CodeBlock value={span.tool_input} />
          </Section>
        )}
        {isTool && span.tool_output && (
          <Section title="Tool output">
            <CodeBlock value={span.tool_output} />
          </Section>
        )}

        {(span.metrics.length > 0 || isModel) && (
          <Section title="Evaluation">
            <MetricList metrics={span.metrics} />
          </Section>
        )}

        {span.metadata && Object.keys(span.metadata).length > 0 && (
          <Section title="Metadata">
            <CodeBlock value={span.metadata} maxHeight={240} />
          </Section>
        )}

        {span.caller_chain && span.caller_chain.length > 0 && (
          <Section title="Call stack">
            <div className="flex flex-wrap items-center gap-1 text-xs">
              {span.caller_chain.map((f, i) => (
                <span key={i} className={cn("bg-muted rounded px-1.5 py-0.5 font-mono", i === span.caller_chain!.length - 1 && "bg-accent text-accent-foreground")}>
                  {f}
                </span>
              ))}
            </div>
          </Section>
        )}
      </div>
    </Card>
  );
}
