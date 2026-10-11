import type { ReactNode } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
  "var(--chart-6)",
  "var(--muted-foreground)",
];

export const AXIS_PROPS = {
  tickLine: false,
  axisLine: false,
  tickMargin: 8,
  minTickGap: 24,
} as const;

export function ChartCard({
  title,
  description,
  action,
  loading,
  height = 240,
  children,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  loading?: boolean;
  height?: number;
  children: ReactNode;
}) {
  return (
    <Card>
      <CardHeader title={title} description={description} action={action} />
      <CardContent>
        <div style={{ height }}>{loading ? <Skeleton className="h-full w-full" /> : children}</div>
      </CardContent>
    </Card>
  );
}

interface TooltipEntry {
  name?: string | number;
  value?: number | string | null;
  color?: string;
  dataKey?: string | number;
}

/** Tooltip content shared by every chart; `format` renders each value. */
export function ChartTooltip({
  active,
  payload,
  label,
  labelFormatter,
  format,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string;
  labelFormatter?: (label: string) => string;
  format: (value: number, key: string) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-popover text-popover-foreground min-w-40 rounded-lg border px-3 py-2 text-xs shadow-lg">
      {label !== undefined && <div className="mb-1.5 font-medium">{labelFormatter ? labelFormatter(label) : label}</div>}
      <div className="space-y-1">
        {payload.map((entry) => (
          <div key={String(entry.dataKey)} className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ background: entry.color }} />
              {entry.name}
            </span>
            <span className="font-medium tabular-nums">
              {entry.value === null || entry.value === undefined ? "—" : format(Number(entry.value), String(entry.dataKey))}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      {items.map((i) => (
        <span key={i.label} className="text-muted-foreground flex items-center gap-1.5 text-xs">
          <span className="size-2 rounded-full" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
