import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet, apiPost } from "@/lib/api";
import type {
  CostSeries,
  Costs,
  EvaluatedTraces,
  MetricSeries,
  MetricsSummary,
  Overview,
  ProblemSpans,
  ProjectInfo,
  RangeKey,
  RecommendedPrompt,
  Recommendations,
  TraceRecommendations,
  Timeseries,
  TraceFilters,
  TraceList,
  TraceTree,
} from "@/lib/types";
import { useCurrentProject } from "@/store/session";

/** Every query key starts with ["p", projectId] so live events can target one project. */
export const qk = {
  project: (pid: string) => ["p", pid, "project"] as const,
  overview: (pid: string, range: RangeKey) => ["p", pid, "overview", range] as const,
  timeseries: (pid: string, range: RangeKey) => ["p", pid, "timeseries", range] as const,
  traces: (pid: string, range: RangeKey, f: TraceFilters) => ["p", pid, "traces", range, f] as const,
  trace: (pid: string, id: string) => ["p", pid, "trace", id] as const,
  /** Under the trace key, so live events on the trace refresh it too. */
  traceRecommendations: (pid: string, id: string) => ["p", pid, "trace", id, "recommendations"] as const,
  costs: (pid: string, range: RangeKey) => ["p", pid, "costs", range] as const,
  costSeries: (pid: string, range: RangeKey) => ["p", pid, "cost-series", range] as const,
  metrics: (pid: string, range: RangeKey) => ["p", pid, "metrics", range] as const,
  metricSeries: (pid: string, name: string, range: RangeKey) => ["p", pid, "metric-series", name, range] as const,
  worst: (pid: string, name: string, range: RangeKey) => ["p", pid, "worst", name, range] as const,
  evaluated: (pid: string, name: string, range: RangeKey) => ["p", pid, "evaluated", name, range] as const,
  recommendations: (pid: string, range: RangeKey) => ["p", pid, "recommendations", range] as const,
};

function useKey() {
  const project = useCurrentProject();
  return { pid: project?.id ?? "", key: project?.key ?? "" };
}

export function useProjectInfo() {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.project(pid),
    queryFn: ({ signal }) => apiGet<ProjectInfo>("/api/v1/project", key, undefined, signal),
    enabled: !!key,
    staleTime: 60_000,
  });
}

export function useOverview(range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.overview(pid, range),
    queryFn: ({ signal }) => apiGet<Overview>("/api/v1/overview", key, { range }, signal),
    enabled: !!key,
    placeholderData: keepPreviousData,
  });
}

export function useTimeseries(range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.timeseries(pid, range),
    queryFn: ({ signal }) => apiGet<Timeseries>("/api/v1/timeseries", key, { range }, signal),
    enabled: !!key,
    placeholderData: keepPreviousData,
  });
}

export function useTraces(range: RangeKey, filters: TraceFilters) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.traces(pid, range, filters),
    queryFn: ({ signal }) =>
      apiGet<TraceList>(
        "/api/v1/traces",
        key,
        {
          range,
          status: filters.status,
          source: filters.source,
          q: filters.q,
          page: filters.page,
          page_size: filters.pageSize,
        },
        signal,
      ),
    enabled: !!key,
    placeholderData: keepPreviousData,
  });
}

export function useTrace(traceId: string) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.trace(pid, traceId),
    queryFn: ({ signal }) => apiGet<TraceTree>(`/api/v1/traces/${encodeURIComponent(traceId)}`, key, undefined, signal),
    enabled: !!key && !!traceId,
    // Safety net while a run is in progress, in case a live event is missed.
    refetchInterval: (query) => (query.state.data?.trace.status === "running" ? 5000 : false),
  });
}

export function useCosts(range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.costs(pid, range),
    queryFn: ({ signal }) => apiGet<Costs>("/api/v1/costs", key, { range }, signal),
    enabled: !!key,
    placeholderData: keepPreviousData,
  });
}

export function useCostSeries(range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.costSeries(pid, range),
    queryFn: ({ signal }) => apiGet<CostSeries>("/api/v1/costs/timeseries", key, { range }, signal),
    enabled: !!key,
    placeholderData: keepPreviousData,
  });
}

export function useMetricsSummary(range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.metrics(pid, range),
    queryFn: ({ signal }) => apiGet<MetricsSummary>("/api/v1/metrics/summary", key, { range }, signal),
    enabled: !!key,
    placeholderData: keepPreviousData,
  });
}

export function useMetricSeries(name: string | null, range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.metricSeries(pid, name ?? "", range),
    queryFn: ({ signal }) => apiGet<MetricSeries>("/api/v1/metrics/timeseries", key, { name, range }, signal),
    enabled: !!key && !!name,
    placeholderData: keepPreviousData,
  });
}

export function useWorstSpans(name: string | null, range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.worst(pid, name ?? "", range),
    queryFn: ({ signal }) => apiGet<ProblemSpans>("/api/v1/metrics/worst", key, { name, range, limit: 10 }, signal),
    enabled: !!key && !!name,
  });
}

export function useEvaluatedTraces(name: string | null, range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.evaluated(pid, name ?? "", range),
    queryFn: ({ signal }) => apiGet<EvaluatedTraces>("/api/v1/metrics/traces", key, { name, range, limit: 50 }, signal),
    enabled: !!key && !!name,
  });
}

export function useTraceRecommendations(traceId: string) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.traceRecommendations(pid, traceId),
    queryFn: ({ signal }) =>
      apiGet<TraceRecommendations>(`/api/v1/traces/${encodeURIComponent(traceId)}/recommendations`, key, undefined, signal),
    enabled: !!key && !!traceId,
    placeholderData: keepPreviousData,
    refetchInterval: (query) => (query.state.data?.evaluation?.running ? 4000 : false),
  });
}

/** AI rewrite of a recommended prompt, on demand: it calls the evaluation model. */
export function useRewritePrompt() {
  const { key } = useKey();
  return useMutation({
    mutationFn: (body: { prompt_id: string; range?: RangeKey; trace_id?: string }) =>
      apiPost<RecommendedPrompt>("/api/v1/recommendations/prompt", key, body),
  });
}

/** Evaluates one trace, or the period's traces without a verdict of the current judge.
 * Verdicts arrive as live events; the refetch shows the evaluations in progress. */
export function useRunEvaluations() {
  const { pid, key } = useKey();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: { range?: RangeKey; trace_id?: string }) =>
      apiPost<{ started: number }>("/api/v1/evaluations/run", key, body),
    onSuccess: (_, body) =>
      client.invalidateQueries({
        queryKey: body.trace_id ? qk.traceRecommendations(pid, body.trace_id) : ["p", pid, "recommendations"],
      }),
  });
}

export function useRecommendations(range: RangeKey) {
  const { pid, key } = useKey();
  return useQuery({
    queryKey: qk.recommendations(pid, range),
    queryFn: ({ signal }) => apiGet<Recommendations>("/api/v1/recommendations", key, { range }, signal),
    enabled: !!key,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    refetchInterval: (query) => (query.state.data?.evaluation?.running ? 5000 : false),
  });
}
