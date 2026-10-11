export type RangeKey = "1h" | "24h" | "7d" | "30d";
export type TraceStatus = "running" | "completed" | "failed" | "timeout";
export type SpanType = "llm" | "tool" | "agent" | "chain" | "retriever" | "embedding";
export type MetricValue = number | boolean | string | unknown[] | Record<string, unknown> | null;

export interface ProjectInfo {
  id: string;
  name: string;
  type: string;
  task: string | null;
  enabled_metrics: string[];
  key: { name: string; prefix: string; scopes: string[] };
  judge?: JudgeStatus;
}

/** Evaluation LLM of the server and the outcome of its latest calls. */
export interface JudgeStatus {
  configured: boolean;
  provider: string | null;
  model: string | null;
  tag: string | null;
  last_ok_at: string | null;
  last_error: string | null;
  last_error_at: string | null;
}

export interface Totals {
  traces: number;
  failed_traces: number;
  running_traces: number;
  error_rate: number;
  error_spans: number;
  llm_calls: number;
  tool_calls: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  avg_duration_ms: number | null;
}

export interface Overview {
  range: RangeKey;
  current: Totals;
  previous: Totals;
  change: Record<"traces" | "cost_usd" | "llm_calls" | "error_rate" | "latency_p95_ms" | "tokens", number | null>;
}

export interface TimePoint {
  t: string;
  traces: number;
  failed: number;
  errors: number;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  llm_calls: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
}

export interface Timeseries {
  range: RangeKey;
  bucket_ms: number;
  points: TimePoint[];
}

export interface TraceRow {
  id: string;
  name: string | null;
  status: TraceStatus;
  source: string;
  otel_trace_id: string | null;
  total_spans: number;
  total_llm_spans: number;
  total_tool_spans: number;
  total_tokens_in: number;
  total_tokens_out: number;
  total_cost_usd: number;
  duration_ms: number | null;
  started_at: string | null;
  ended_at: string | null;
  has_metrics: boolean;
  metrics?: Record<string, MetricValue>;
  metadata?: Record<string, unknown> | null;
}

export interface TraceList {
  total: number;
  page: number;
  page_size: number;
  traces: TraceRow[];
}

export interface MetricResult {
  name: string;
  value: MetricValue;
  reason?: string | null;
  judge?: string | null;
  /** LLM verdict from an older judge: left out of the averages until re-evaluated. */
  stale?: boolean;
  [key: string]: unknown;
}

export interface Message {
  role?: string;
  content?: string | null;
  [key: string]: unknown;
}

export interface SpanNode {
  id: string;
  step_index: number;
  span_type: SpanType;
  name: string | null;
  status: string | null;
  error: string | null;
  source: string;
  sdk_span_id: string | null;
  parent_span_id: string | null;
  caller_chain: string[] | null;
  latency_ms: number | null;
  timestamp: string | null;
  metrics: MetricResult[];
  depth: number;
  start_offset_ms: number;
  duration_ms: number;
  agent_name?: string | null;
  role?: string | null;
  metadata?: Record<string, unknown> | null;
  provider?: string | null;
  model?: string | null;
  tokens_in?: number | null;
  tokens_out?: number | null;
  cost_usd?: number | null;
  temperature?: number | null;
  response?: string | null;
  messages_in?: Message[] | null;
  tool_name?: string | null;
  tool_input?: Record<string, unknown> | null;
  tool_output?: string | null;
  tool_error?: string | null;
}

export interface TraceTree {
  trace: TraceRow;
  metrics: MetricResult[];
  spans: SpanNode[];
  origin: string | null;
  total_ms: number;
}

export interface CostItem {
  cost_usd: number;
  calls: number;
  tokens_in: number;
  tokens_out: number;
}

export interface Costs {
  range: RangeKey;
  total_cost_usd: number;
  previous_cost_usd: number;
  change: number | null;
  avg_cost_per_trace: number;
  by_model: (CostItem & { model: string })[];
  by_agent: (CostItem & { agent: string })[];
  top_traces: TraceRow[];
}

export interface CostSeries {
  range: RangeKey;
  series: string[];
  points: ({ t: string } & Record<string, number | string>)[];
}

export interface MetricAggregate {
  avg: number | null;
  evaluated_count: number;
}

export interface MetricsSummary {
  range: RangeKey;
  period_days: number;
  enabled_metrics: string[];
  span_metrics: Record<string, MetricAggregate>;
  trace_metrics: Record<string, MetricAggregate>;
}

export interface MetricSeries {
  range: RangeKey;
  metric: string;
  points: { t: string; avg: number | null; count: number }[];
}

export interface ProblemSpan {
  span_id: string;
  trace_id: string;
  span_type: SpanType;
  name: string | null;
  model: string | null;
  latency_ms: number | null;
  error: string | null;
  timestamp: string | null;
  metrics: Record<string, MetricValue>;
  response_preview: string | null;
}

export interface ProblemSpans {
  count: number;
  spans: ProblemSpan[];
}

/** Tool that produced a trace (Cursor, Claude Code) or its ingestion path (SDK, OTel). */
export type TraceOrigin = "cursor" | "claude-code" | "sdk" | "otel";

export interface EvaluatedTrace {
  trace_id: string;
  name: string | null;
  status: TraceStatus;
  origin: TraceOrigin;
  started_at: string;
  value: MetricValue;
  reason: string | null;
}

export interface EvaluatedTraces {
  metric: string;
  count: number;
  traces: EvaluatedTrace[];
}

export type Complexity = "simple" | "moyenne" | "complexe";
export type RecommendationCategory = "harness" | "model" | "caching" | "prompt" | "agent" | "reliability";
export type RecommendationSeverity = "high" | "medium" | "low";

export interface ComplexityBucket {
  calls: number;
  cost_usd: number;
}

export interface ModelComplexityRow extends Record<Complexity, ComplexityBucket> {
  model: string;
  price_known: boolean;
  alternative: string | null;
  no_alternative_reason?: string | null;
  calls: number;
  cost_usd: number;
}

export interface Recommendation {
  id: string;
  category: RecommendationCategory;
  severity: RecommendationSeverity;
  title: string;
  detail: string;
  action: string;
  savings_usd: number | null;
  evidence: Record<string, unknown>;
  examples: { trace_id: string; reason: string | null }[];
  spans?: SpanRef[];
  fix?: Fix | null;
  /** More files of the same fix, e.g. the script a hook configuration runs. */
  extra_fixes?: Fix[];
}

/** Text to copy: an instruction for a prompt, a Cursor rule / CLAUDE.md section, code, a
 * configuration to merge (hooks.json, settings.json, .cursorignore) or a file to create. */
export interface Fix {
  kind: "prompt" | "rule" | "code" | "config" | "file";
  label: string;
  content: string;
  file: string | null;
}

export type AgentTarget = "cursor" | "claude-code" | "sdk";

export interface RecommendedPrompt {
  id: string;
  target: "cursor" | "claude-code" | "system_prompt";
  title: string;
  file: string | null;
  current: string | null;
  content: string;
  changes: string[];
  model: string | null;
  calls: number | null;
  source: "rules" | "ai";
  /** Starting rules for a coding agent with no measured problem yet. */
  baseline?: boolean;
}

/** Judge coverage of the period: traces with a verdict of the current judge, and waiting ones. */
export interface EvaluationStatus {
  judge: string | null;
  ended_traces: number;
  evaluated: number;
  pending: number;
  running: number;
  /** Evaluations a metric needs before it can fire a recommendation. */
  min_evaluated?: number;
}

export interface TraceEvaluationStatus {
  judge: string | null;
  evaluated: boolean;
  running: boolean;
  trace_running: boolean;
}

export interface SpanRef {
  span_id: string;
  step: number;
  label: string;
  reason: string | null;
}

export interface TraceStep {
  span_id: string;
  step: number;
  label: string;
  model: string;
  price_known: boolean;
  complexity: Complexity;
  tokens_in: number;
  tokens_out: number;
  context_tokens: number;
  cache_read_tokens: number;
  cost_usd: number;
  cost_share: number;
  alternative: string | null;
  alternative_cost_usd: number | null;
  savings_usd: number | null;
  /** Why no cheaper model is proposed for this call. */
  no_alternative_reason?: string | null;
  failed: boolean;
}

export interface TraceRecommendations {
  trace_id: string;
  agentic: boolean;
  complexity: Complexity;
  tool_calls: number;
  llm_calls: number;
  max_context_tokens: number;
  cost_usd: number;
  savings: { model_usd: number; caching_usd: number; total_usd?: number };
  optimized_cost_usd: number;
  steps: TraceStep[];
  span_notes: Record<string, string[]>;
  agent_target?: AgentTarget;
  prompts?: RecommendedPrompt[];
  recommendations: Recommendation[];
  pricing?: { source: "web" | "builtin"; updated_at: string | null; models: number };
  evaluation?: TraceEvaluationStatus;
}

export interface Recommendations {
  range: RangeKey;
  analyzed_calls: number;
  truncated: boolean;
  total_cost_usd: number;
  savings: { routing_usd: number; caching_usd: number; total_usd?: number };
  complexity: Record<Complexity, ComplexityBucket>;
  by_model: ModelComplexityRow[];
  unpriced_models: { model: string; calls: number }[];
  pricing?: { source: "web" | "builtin"; updated_at: string | null; models: number };
  agent_target?: AgentTarget;
  prompts?: RecommendedPrompt[];
  evaluation?: EvaluationStatus;
  recommendations: Recommendation[];
}

export interface TraceFilters {
  status?: TraceStatus | "";
  source?: TraceOrigin | "";
  q?: string;
  page: number;
  pageSize: number;
}
