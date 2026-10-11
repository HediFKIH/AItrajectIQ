import { CheckCircle2, Clock, Loader2, XCircle } from "lucide-react";
import { Badge, type Tone } from "@/components/ui/badge";
import type { TraceStatus } from "@/lib/types";

const STATUS: Record<TraceStatus, { label: string; tone: Tone; icon: typeof CheckCircle2 }> = {
  running: { label: "Running", tone: "info", icon: Loader2 },
  completed: { label: "Completed", tone: "success", icon: CheckCircle2 },
  failed: { label: "Failed", tone: "danger", icon: XCircle },
  timeout: { label: "Timeout", tone: "warning", icon: Clock },
};

export function StatusBadge({ status }: { status: TraceStatus }) {
  const s = STATUS[status] ?? STATUS.completed;
  const Icon = s.icon;
  return (
    <Badge tone={s.tone}>
      <Icon className={status === "running" ? "size-3 animate-spin" : "size-3"} />
      {s.label}
    </Badge>
  );
}

/** Ingestion path, refined by the tool that produced the trace when its metadata says so. */
export function SourceBadge({ source, metadata }: { source: string; metadata?: Record<string, unknown> | null }) {
  const origin = metadata?.source;
  if (origin === "cursor") return <Badge tone="primary">Cursor</Badge>;
  if (origin === "claude-code") return <Badge tone="primary">Claude Code</Badge>;
  return <Badge tone={source === "otel" ? "primary" : "neutral"}>{source === "otel" ? "OTel" : "SDK"}</Badge>;
}

const MODE_LABEL: Record<string, string> = { agent: "Agent", plan: "Plan", ask: "Ask", manual: "Manual", debug: "Debug" };

/** Agent mode (Cursor: agent / plan / ask…) and model of a trace, when recorded. */
export function ModeBadges({ metadata }: { metadata?: Record<string, unknown> | null }) {
  const mode = typeof metadata?.mode === "string" ? metadata.mode : null;
  const model = typeof metadata?.model === "string" ? metadata.model : null;
  if (!mode && !model) return null;
  return (
    <>
      {mode && <Badge tone="info">Mode {MODE_LABEL[mode] ?? mode}</Badge>}
      {model && (
        <Badge tone="neutral" title="Model">
          {model}
        </Badge>
      )}
    </>
  );
}
