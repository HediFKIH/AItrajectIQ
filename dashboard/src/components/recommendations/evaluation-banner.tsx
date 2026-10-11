import type { ReactNode } from "react";
import { Gauge, Loader2, PlayCircle, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRunEvaluations } from "@/hooks/queries";
import { formatNumber } from "@/lib/format";
import type { EvaluationStatus, RangeKey, TraceEvaluationStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const JUDGE_SETUP = "Add EVAL_LLM_API_KEY and EVAL_LLM_MODEL to the backend .env file, then restart the backend.";

function judgeModel(tag: string | null): string {
  return tag ? tag.split("@")[0] : "";
}

function Banner({
  icon: Icon,
  tone,
  title,
  children,
  action,
  spin,
}: {
  icon: typeof Gauge;
  tone: "warning" | "info" | "muted";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  spin?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3",
        tone === "warning" && "border-warning/30 bg-warning/5",
        tone === "info" && "border-chart-2/25 bg-chart-2/5",
        tone === "muted" && "bg-muted/40",
      )}
    >
      <Icon
        className={cn(
          "size-5 shrink-0",
          spin && "animate-spin",
          tone === "warning" ? "text-warning" : tone === "info" ? "text-chart-2" : "text-muted-foreground",
        )}
      />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium">{title}</p>
        {children && <div className="text-muted-foreground mt-0.5 text-xs">{children}</div>}
      </div>
      {action}
    </div>
  );
}

/** Why quality recommendations are (not yet) there, with a button to evaluate the waiting traces. */
export function EvaluationBanner({ status, range }: { status: EvaluationStatus; range: RangeKey }) {
  const run = useRunEvaluations();
  const min = status.min_evaluated ?? 5;
  const error = run.error ? <p className="text-destructive mt-1">{(run.error as Error).message}</p> : null;

  if (!status.judge) {
    return (
      <Banner icon={ShieldAlert} tone="warning" title="Quality evaluation disabled">
        Only cost recommendations are shown: no evaluation model is configured. {JUDGE_SETUP}
      </Banner>
    );
  }
  if (status.running > 0) {
    return (
      <Banner icon={Loader2} spin tone="info" title={`Evaluating ${formatNumber(status.running)} trace(s)`}>
        Metrics and recommendations update automatically.
      </Banner>
    );
  }
  if (status.pending > 0) {
    const short = status.evaluated < min;
    return (
      <Banner
        icon={Gauge}
        tone={short ? "warning" : "info"}
        title={`${formatNumber(status.pending)} trace(s) not yet evaluated by ${judgeModel(status.judge)}`}
        action={
          <Button size="sm" disabled={run.isPending} onClick={() => run.mutate({ range })}>
            {run.isPending ? <Loader2 className="animate-spin" /> : <PlayCircle />}
            Evaluate now
          </Button>
        }
      >
        {short
          ? `Quality recommendations need at least ${min} evaluated traces (${formatNumber(status.evaluated)}/${min} so far).`
          : "Quality recommendations only take already evaluated traces into account."}{" "}
        Up to 30 traces per click, most recent first.
        {error}
      </Banner>
    );
  }
  if (status.ended_traces > 0 && status.evaluated < min) {
    return (
      <Banner icon={Gauge} tone="muted" title={`${formatNumber(status.evaluated)}/${min} traces evaluated`}>
        Quality recommendations will appear once {min} traces are evaluated in this period.
      </Banner>
    );
  }
  if (status.evaluated > 0) {
    return (
      <p className="text-muted-foreground text-xs">
        Quality measured on {formatNumber(status.evaluated)} trace(s) by {judgeModel(status.judge)}.
      </p>
    );
  }
  return null;
}

/** Evaluation state of one trace, with a button when the current judge hasn't seen it. */
export function TraceEvaluationNotice({ status, traceId }: { status: TraceEvaluationStatus; traceId: string }) {
  const run = useRunEvaluations();
  if (!status.judge) {
    return (
      <Banner icon={ShieldAlert} tone="warning" title="Quality not evaluated">
        No evaluation model is configured. {JUDGE_SETUP}
      </Banner>
    );
  }
  if (status.running) {
    return <Banner icon={Loader2} spin tone="info" title="Evaluating…" />;
  }
  if (status.evaluated) return null;
  if (status.trace_running) {
    return <Banner icon={Gauge} tone="muted" title="This trace is still running: it will be evaluated when it ends." />;
  }
  return (
    <Banner
      icon={Gauge}
      tone="info"
      title="This trace hasn't been evaluated by the current judge yet"
      action={
        <Button size="sm" disabled={run.isPending} onClick={() => run.mutate({ trace_id: traceId })}>
          {run.isPending ? <Loader2 className="animate-spin" /> : <PlayCircle />}
          Evaluate this trace
        </Button>
      }
    >
      Quality recommendations (instructions, claims, checks) appear after evaluation.
      {run.error && <p className="text-destructive mt-1">{(run.error as Error).message}</p>}
    </Banner>
  );
}
