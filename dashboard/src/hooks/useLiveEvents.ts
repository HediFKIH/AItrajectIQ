import { useEffect } from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { connectStream, type StreamEvent } from "@/lib/live";
import { live } from "@/store/live";
import type { SavedProject } from "@/store/session";

const FLUSH_MS = 1200;

/** Query groups refreshed by each event type (third element of the query key). */
const AFFECTED: Record<string, string[]> = {
  "trace.started": ["overview", "timeseries", "traces"],
  "trace.reopened": ["overview", "traces"],
  "trace.completed": ["overview", "timeseries", "traces", "costs", "cost-series", "recommendations"],
  "metrics.updated": ["traces", "metrics", "metric-series", "worst", "evaluated", "recommendations"],
};

/** Groups a span event changes: counters only for new spans (count 0 = an update of a
 * known span), costs only when the span is priced or unknown (OTel batches send no detail). */
function spanGroups(data: Record<string, unknown>): string[] {
  const groups = ["traces"];
  if (Number(data.count ?? 1) > 0) groups.push("overview", "timeseries");
  const span = data.span as { cost_usd?: number | null } | undefined;
  if (!span || (span.cost_usd ?? 0) > 0) groups.push("overview", "costs", "cost-series");
  return groups;
}

/** Batches invalidations so a burst of spans triggers one refetch per query. */
function createInvalidator(client: QueryClient, pid: string) {
  const groups = new Set<string>();
  const traces = new Set<string>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  let all = false;

  const flush = () => {
    timer = null;
    if (all) {
      client.invalidateQueries({ queryKey: ["p", pid] });
    } else {
      client.invalidateQueries({
        predicate: (q) => {
          const [scope, project, group, id] = q.queryKey as string[];
          if (scope !== "p" || project !== pid) return false;
          return groups.has(group) || (group === "trace" && traces.has(id));
        },
      });
    }
    groups.clear();
    traces.clear();
    all = false;
  };

  const schedule = () => {
    if (!timer) timer = setTimeout(flush, FLUSH_MS);
  };

  return {
    event(type: string, traceId?: string, affected: string[] = AFFECTED[type] ?? []) {
      affected.forEach((g) => groups.add(g));
      if (traceId) traces.add(traceId);
      schedule();
    },
    everything() {
      all = true;
      schedule();
    },
    cancel() {
      if (timer) clearTimeout(timer);
    },
  };
}

export function useLiveEvents(project: SavedProject | null) {
  const client = useQueryClient();

  useEffect(() => {
    if (!project) return;
    const controller = new AbortController();
    const invalidator = createInvalidator(client, project.id);
    let connectedOnce = false;
    live.reset();
    live.setStatus("connecting");

    const handle = ({ type, data }: StreamEvent) => {
      const traceId = typeof data.trace_id === "string" ? data.trace_id : undefined;
      switch (type) {
        case "ready":
          return;
        case "resync":
          invalidator.everything();
          return;
        case "trace.started": {
          const trace = data.trace as { id: string; name?: string | null } | undefined;
          if (trace) live.push({ traceId: trace.id, kind: "started", name: trace.name, at: Date.now() });
          invalidator.event(type, trace?.id);
          return;
        }
        case "trace.completed":
          if (traceId) live.push({ traceId, kind: "completed", status: String(data.status ?? ""), at: Date.now() });
          invalidator.event(type, traceId);
          return;
        case "span.created":
          if (traceId) live.push({ traceId, kind: "span", at: Date.now() });
          invalidator.event(type, traceId, spanGroups(data));
          return;
        default:
          invalidator.event(type, traceId);
      }
    };

    connectStream(
      project.key,
      {
        onOpen: () => {
          live.setStatus("live");
          // Catch up on anything that happened while disconnected.
          if (connectedOnce) invalidator.everything();
          connectedOnce = true;
        },
        onEvent: handle,
        onDisconnect: () => live.setStatus("offline"),
        onUnauthorized: () => live.setStatus("revoked"),
      },
      controller.signal,
    ).catch(() => live.setStatus("offline"));

    return () => {
      controller.abort();
      invalidator.cancel();
    };
  }, [client, project]);
}
