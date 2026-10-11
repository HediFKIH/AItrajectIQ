import type { ReactNode } from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { formatChange } from "@/lib/format";
import { cn } from "@/lib/utils";

export function KpiCard({
  label,
  value,
  change,
  higherIsBetter = true,
  hint,
  spark,
  color = "var(--chart-1)",
  loading,
}: {
  label: string;
  value: ReactNode;
  change?: number | null;
  higherIsBetter?: boolean;
  hint?: ReactNode;
  spark?: number[];
  color?: string;
  loading?: boolean;
}) {
  const hasChange = change !== null && change !== undefined && Math.abs(change) >= 0.0005;
  const good = hasChange && (change! > 0) === higherIsBetter;
  const Icon = !hasChange ? Minus : change! > 0 ? ArrowUpRight : ArrowDownRight;
  const id = `spark-${label.replace(/\W/g, "")}`;

  return (
    <Card className="relative overflow-hidden p-4">
      <div className="text-muted-foreground text-xs font-medium">{label}</div>
      {loading ? (
        <Skeleton className="mt-2 h-7 w-24" />
      ) : (
        <div className="mt-1.5 text-2xl font-semibold tracking-tight tabular-nums">{value}</div>
      )}
      <div className="mt-1 flex items-center gap-1.5 text-xs">
        {change !== undefined && (
          <span
            className={cn(
              "inline-flex items-center gap-0.5 font-medium",
              !hasChange ? "text-muted-foreground" : good ? "text-success" : "text-destructive",
            )}
          >
            <Icon className="size-3.5" />
            {formatChange(change)}
          </span>
        )}
        {hint && <span className="text-muted-foreground truncate">{hint}</span>}
      </div>
      {spark && spark.some((v) => v > 0) && (
        <div className="pointer-events-none absolute right-0 bottom-0 left-0 h-12 opacity-70">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={spark.map((v, i) => ({ i, v }))} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} fill={`url(#${id})`} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}
