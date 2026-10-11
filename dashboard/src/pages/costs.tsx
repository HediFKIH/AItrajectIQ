import { useMemo } from "react";
import { Wallet } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_PROPS, ChartCard, ChartTooltip, Legend, SERIES_COLORS } from "@/components/charts";
import { KpiCard } from "@/components/kpi-card";
import { RANGE_LABEL } from "@/components/layout/range-picker";
import { TraceTable } from "@/components/trace-table";
import { Card, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState } from "@/components/ui/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TBody, Td, Th, THead, Tr } from "@/components/ui/table";
import { useCosts, useCostSeries } from "@/hooks/queries";
import { axisTimeFormatter, formatCompact, formatCost, formatNumber, formatPercent, tooltipTimeFormatter } from "@/lib/format";
import type { CostItem } from "@/lib/types";
import { useRange } from "@/store/session";

const seriesLabel = (name: string) => (name === "other" ? "Other models" : name);

function ShareBar({ share, color }: { share: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="bg-muted h-1.5 w-24 overflow-hidden rounded-full">
        <div className="h-full rounded-full" style={{ width: `${Math.max(share * 100, 1)}%`, background: color }} />
      </div>
      <span className="text-muted-foreground w-12 text-right text-xs tabular-nums">{formatPercent(share, 1)}</span>
    </div>
  );
}

function CostTable({
  rows,
  label,
  total,
  colorOf,
}: {
  rows: (CostItem & { name: string })[];
  label: string;
  total: number;
  colorOf: (name: string, index: number) => string;
}) {
  return (
    <Table>
      <THead>
        <Tr>
          <Th>{label}</Th>
          <Th className="text-right">Calls</Th>
          <Th className="text-right">Tokens</Th>
          <Th className="text-right">Cost</Th>
          <Th>Share</Th>
        </Tr>
      </THead>
      <TBody>
        {rows.map((r, i) => (
          <Tr key={r.name}>
            <Td className="max-w-56">
              <span className="flex items-center gap-2">
                <span className="size-2 shrink-0 rounded-full" style={{ background: colorOf(r.name, i) }} />
                <span className="truncate font-medium" title={r.name}>
                  {r.name}
                </span>
              </span>
            </Td>
            <Td className="text-right tabular-nums">{formatNumber(r.calls)}</Td>
            <Td className="text-muted-foreground text-right tabular-nums">
              {formatCompact(r.tokens_in)} / {formatCompact(r.tokens_out)}
            </Td>
            <Td className="text-right font-medium tabular-nums">{formatCost(r.cost_usd)}</Td>
            <Td>
              <ShareBar share={total > 0 ? r.cost_usd / total : 0} color={colorOf(r.name, i)} />
            </Td>
          </Tr>
        ))}
      </TBody>
    </Table>
  );
}

export function CostsPage() {
  const range = useRange();
  const costs = useCosts(range);
  const series = useCostSeries(range);
  const xFmt = useMemo(() => axisTimeFormatter(range), [range]);
  const tipFmt = useMemo(() => tooltipTimeFormatter(range), [range]);

  const seriesNames = useMemo(() => series.data?.series ?? [], [series.data]);
  const modelColor = useMemo(() => {
    const map = new Map(seriesNames.map((name, i) => [name, SERIES_COLORS[i % SERIES_COLORS.length]]));
    return (name: string, i: number) => map.get(name) ?? SERIES_COLORS[i % SERIES_COLORS.length];
  }, [seriesNames]);

  if (costs.error) return <ErrorState error={costs.error} />;

  const c = costs.data;
  const loading = costs.isLoading;
  const byModel = (c?.by_model ?? []).map((m) => ({ ...m, name: m.model }));
  const byAgent = (c?.by_agent ?? []).map((a) => ({ ...a, name: a.agent }));
  const top = byModel[0];
  const totalCalls = byModel.reduce((s, m) => s + m.calls, 0);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <KpiCard
          label={`Spend over ${RANGE_LABEL[range]}`}
          value={formatCost(c?.total_cost_usd)}
          change={c?.change}
          higherIsBetter={false}
          hint={c ? `${formatCost(c.previous_cost_usd)} in the previous period` : undefined}
          color="var(--chart-3)"
          loading={loading}
        />
        <KpiCard
          label="Average cost per trace"
          value={formatCost(c?.avg_cost_per_trace)}
          hint={totalCalls ? `${formatNumber(totalCalls)} LLM calls` : undefined}
          loading={loading}
        />
        <KpiCard
          label="Most expensive model"
          value={<span className="block truncate text-xl">{top?.model ?? "—"}</span>}
          hint={top && c?.total_cost_usd ? `${formatPercent(top.cost_usd / c.total_cost_usd, 0)} of spend` : undefined}
          loading={loading}
        />
      </div>

      {!loading && c && c.total_cost_usd === 0 && byModel.length === 0 ? (
        <Card>
          <EmptyState
            icon={Wallet}
            title={`No spend in ${RANGE_LABEL[range]}`}
            description="Costs are computed from the tokens of each LLM call and the model's pricing."
          />
        </Card>
      ) : (
        <>
          <ChartCard
            title="Spend by model"
            description="Cumulative cost per interval"
            loading={series.isLoading}
            height={280}
            action={<Legend items={seriesNames.map((name, i) => ({ label: seriesLabel(name), color: modelColor(name, i) }))} />}
          >
            <ResponsiveContainer>
              <AreaChart data={series.data?.points ?? []} margin={{ top: 4, right: 4, bottom: 0, left: -4 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="t" tickFormatter={xFmt} {...AXIS_PROPS} />
                <YAxis tickFormatter={(v) => formatCost(v, { compact: true })} {...AXIS_PROPS} width={56} />
                <Tooltip content={<ChartTooltip labelFormatter={tipFmt} format={(v) => formatCost(v)} />} />
                {seriesNames.map((name, i) => (
                  <Area
                    key={name}
                    dataKey={name}
                    name={seriesLabel(name)}
                    stackId="cost"
                    type="monotone"
                    stroke={modelColor(name, i)}
                    fill={modelColor(name, i)}
                    fillOpacity={0.22}
                    strokeWidth={1.5}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="grid gap-4 xl:grid-cols-2">
            <Card className="overflow-hidden">
              <CardHeader title="By model" description="Calls, tokens (input / output) and cost" />
              {loading ? (
                <div className="p-5">
                  <Skeleton className="h-40" />
                </div>
              ) : (
                <CostTable rows={byModel} label="Model" total={c?.total_cost_usd ?? 0} colorOf={modelColor} />
              )}
            </Card>

            <Card className="overflow-hidden">
              <CardHeader title="By agent" description="LLM call cost attributed to the agent that made the calls" />
              {loading ? (
                <div className="p-5">
                  <Skeleton className="h-40" />
                </div>
              ) : (
                <>
                  {byAgent.length > 0 && (
                    <div className="h-44 px-3">
                      <ResponsiveContainer>
                        <BarChart data={byAgent.slice(0, 8)} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
                          <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                          <XAxis type="number" tickFormatter={(v) => formatCost(v, { compact: true })} {...AXIS_PROPS} />
                          <YAxis type="category" dataKey="name" width={110} {...AXIS_PROPS} minTickGap={0} />
                          <Tooltip
                            cursor={{ fill: "var(--muted)", opacity: 0.5 }}
                            content={<ChartTooltip format={(v) => formatCost(v)} />}
                          />
                          <Bar dataKey="cost_usd" name="Cost" fill="var(--chart-1)" radius={[0, 3, 3, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                  <CostTable rows={byAgent} label="Agent" total={c?.total_cost_usd ?? 0} colorOf={() => "var(--chart-1)"} />
                </>
              )}
            </Card>
          </div>

          <Card>
            <CardHeader title="Most expensive traces" description={`Over ${RANGE_LABEL[range]}`} />
            <TraceTable traces={c?.top_traces} loading={loading} compact emptyLabel="No billed traces." />
          </Card>
        </>
      )}
    </div>
  );
}
