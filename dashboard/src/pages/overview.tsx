import { useMemo } from "react";
import { Link } from "react-router";
import { Activity, ArrowRight } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_PROPS, ChartCard, ChartTooltip, Legend } from "@/components/charts";
import { KpiCard } from "@/components/kpi-card";
import { RANGE_LABEL } from "@/components/layout/range-picker";
import { TraceTable } from "@/components/trace-table";
import { buttonClass } from "@/components/ui/button";
import { Card, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState } from "@/components/ui/empty-state";
import { useOverview, useTimeseries, useTraces } from "@/hooks/queries";
import {
  axisTimeFormatter,
  formatCompact,
  formatCost,
  formatDuration,
  formatNumber,
  formatPercent,
  tooltipTimeFormatter,
} from "@/lib/format";
import { useRange } from "@/store/session";

const RECENT_FILTERS = { page: 1, pageSize: 8 };

export function OverviewPage() {
  const range = useRange();
  const overview = useOverview(range);
  const series = useTimeseries(range);
  const recent = useTraces(range, RECENT_FILTERS);

  const points = useMemo(
    () => (series.data?.points ?? []).map((p) => ({ ...p, succeeded: p.traces - p.failed })),
    [series.data],
  );
  const xFmt = useMemo(() => axisTimeFormatter(range), [range]);
  const tipFmt = useMemo(() => tooltipTimeFormatter(range), [range]);
  const cur = overview.data?.current;
  const change = overview.data?.change;
  const loading = overview.isLoading;

  if (overview.error) return <ErrorState error={overview.error} />;

  const noData = !loading && cur && cur.traces === 0;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-5">
        <KpiCard
          label="Traces"
          value={formatNumber(cur?.traces)}
          change={change?.traces}
          hint={cur ? `${formatNumber(cur.llm_calls)} LLM calls` : undefined}
          spark={points.map((p) => p.traces)}
          loading={loading}
        />
        <KpiCard
          label="Cost"
          value={formatCost(cur?.cost_usd, { compact: true })}
          change={change?.cost_usd}
          higherIsBetter={false}
          hint={cur?.traces ? `${formatCost(cur.cost_usd / cur.traces)} / trace` : undefined}
          spark={points.map((p) => p.cost_usd)}
          color="var(--chart-3)"
          loading={loading}
        />
        <KpiCard
          label="Tokens"
          value={formatCompact((cur?.tokens_in ?? 0) + (cur?.tokens_out ?? 0))}
          change={change?.tokens}
          higherIsBetter={false}
          hint={cur ? `${formatCompact(cur.tokens_in)} in · ${formatCompact(cur.tokens_out)} out` : undefined}
          spark={points.map((p) => p.tokens_in + p.tokens_out)}
          color="var(--chart-2)"
          loading={loading}
        />
        <KpiCard
          label="p95 latency"
          value={formatDuration(cur?.latency_p95_ms)}
          change={change?.latency_p95_ms}
          higherIsBetter={false}
          hint={cur?.latency_p50_ms != null ? `p50 ${formatDuration(cur.latency_p50_ms)}` : undefined}
          spark={points.map((p) => p.latency_p95_ms ?? 0)}
          color="var(--chart-4)"
          loading={loading}
        />
        <KpiCard
          label="Failure rate"
          value={formatPercent(cur?.error_rate)}
          change={change?.error_rate}
          higherIsBetter={false}
          hint={cur ? `${formatNumber(cur.error_spans)} error spans` : undefined}
          spark={points.map((p) => p.failed + p.errors)}
          color="var(--chart-5)"
          loading={loading}
        />
      </div>

      {noData ? (
        <Card>
          <EmptyState
            icon={Activity}
            title={`No traces in ${RANGE_LABEL[range]}`}
            description="Connect Cursor, Claude Code or your own app: every run shows up here live, with its cost and evaluation."
            action={
              <Link to="/connect" className={buttonClass("default", "sm")}>
                Connect an agent
                <ArrowRight />
              </Link>
            }
          />
        </Card>
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard
              title="Activity"
              description="Traces, failures and error spans"
              loading={series.isLoading}
              action={
                <Legend
                  items={[
                    { label: "Succeeded", color: "var(--chart-1)" },
                    { label: "Failed", color: "var(--chart-5)" },
                    { label: "Error spans", color: "var(--chart-4)" },
                  ]}
                />
              }
            >
              <ResponsiveContainer>
                <ComposedChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: -12 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="t" tickFormatter={xFmt} {...AXIS_PROPS} />
                  <YAxis allowDecimals={false} {...AXIS_PROPS} width={40} />
                  <Tooltip
                    cursor={{ fill: "var(--muted)", opacity: 0.5 }}
                    content={<ChartTooltip labelFormatter={tipFmt} format={(v) => formatNumber(v)} />}
                  />
                  <Bar dataKey="succeeded" name="Succeeded" stackId="a" fill="var(--chart-1)" />
                  <Bar dataKey="failed" name="Failed" stackId="a" fill="var(--chart-5)" radius={[3, 3, 0, 0]} />
                  <Line dataKey="errors" name="Error spans" type="monotone" stroke="var(--chart-4)" strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Cost" description={`LLM spend over ${RANGE_LABEL[range]}`} loading={series.isLoading}>
              <ResponsiveContainer>
                <AreaChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: -4 }}>
                  <defs>
                    <linearGradient id="cost-fill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--chart-3)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="var(--chart-3)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="t" tickFormatter={xFmt} {...AXIS_PROPS} />
                  <YAxis tickFormatter={(v) => formatCost(v, { compact: true })} {...AXIS_PROPS} width={56} />
                  <Tooltip content={<ChartTooltip labelFormatter={tipFmt} format={(v) => formatCost(v)} />} />
                  <Area dataKey="cost_usd" name="Cost" type="monotone" stroke="var(--chart-3)" strokeWidth={2} fill="url(#cost-fill)" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Tokens"
              description="Sent and generated"
              loading={series.isLoading}
              action={
                <Legend
                  items={[
                    { label: "Input", color: "var(--chart-2)" },
                    { label: "Output", color: "var(--chart-1)" },
                  ]}
                />
              }
            >
              <ResponsiveContainer>
                <AreaChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: -8 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="t" tickFormatter={xFmt} {...AXIS_PROPS} />
                  <YAxis tickFormatter={(v) => formatCompact(v)} {...AXIS_PROPS} width={44} />
                  <Tooltip content={<ChartTooltip labelFormatter={tipFmt} format={(v) => formatNumber(v)} />} />
                  <Area dataKey="tokens_in" name="Input" stackId="t" type="monotone" stroke="var(--chart-2)" fill="var(--chart-2)" fillOpacity={0.2} />
                  <Area dataKey="tokens_out" name="Output" stackId="t" type="monotone" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.25} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard
              title="Trace latency"
              description="Median and 95th percentile"
              loading={series.isLoading}
              action={
                <Legend
                  items={[
                    { label: "p50", color: "var(--chart-2)" },
                    { label: "p95", color: "var(--chart-4)" },
                  ]}
                />
              }
            >
              <ResponsiveContainer>
                <LineChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: -4 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="t" tickFormatter={xFmt} {...AXIS_PROPS} />
                  <YAxis tickFormatter={(v) => formatDuration(v)} {...AXIS_PROPS} width={56} />
                  <Tooltip content={<ChartTooltip labelFormatter={tipFmt} format={(v) => formatDuration(v)} />} />
                  <Line dataKey="latency_p50_ms" name="p50" type="monotone" stroke="var(--chart-2)" strokeWidth={2} dot={false} connectNulls />
                  <Line dataKey="latency_p95_ms" name="p95" type="monotone" stroke="var(--chart-4)" strokeWidth={2} dot={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <Card>
            <CardHeader
              title="Latest traces"
              description="Updated live"
              action={
                <Link to="/traces" className={buttonClass("ghost", "sm")}>
                  All traces
                  <ArrowRight />
                </Link>
              }
            />
            <TraceTable traces={recent.data?.traces} loading={recent.isLoading} compact />
          </Card>
        </>
      )}
    </div>
  );
}
