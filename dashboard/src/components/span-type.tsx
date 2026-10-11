import { Bot, Database, Layers, Sparkles, Waypoints, Wrench } from "lucide-react";
import type { SpanType } from "@/lib/types";
import { cn } from "@/lib/utils";

export const SPAN_TYPES: Record<SpanType, { label: string; icon: typeof Bot; color: string; bar: string }> = {
  agent: { label: "Agent", icon: Bot, color: "text-chart-1 bg-chart-1/12", bar: "bg-chart-1" },
  llm: { label: "LLM", icon: Sparkles, color: "text-chart-2 bg-chart-2/12", bar: "bg-chart-2" },
  tool: { label: "Tool", icon: Wrench, color: "text-chart-4 bg-chart-4/15", bar: "bg-chart-4" },
  chain: { label: "Chain", icon: Waypoints, color: "text-chart-6 bg-chart-6/12", bar: "bg-chart-6" },
  retriever: { label: "Retriever", icon: Database, color: "text-chart-3 bg-chart-3/12", bar: "bg-chart-3" },
  embedding: { label: "Embedding", icon: Layers, color: "text-chart-5 bg-chart-5/12", bar: "bg-chart-5" },
};

export function SpanTypeIcon({ type, className }: { type: SpanType; className?: string }) {
  const meta = SPAN_TYPES[type] ?? SPAN_TYPES.chain;
  const Icon = meta.icon;
  return (
    <span title={meta.label} className={cn("inline-flex size-6 shrink-0 items-center justify-center rounded-md", meta.color, className)}>
      <Icon className="size-3.5" />
    </span>
  );
}
