import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Minus, Plus } from "lucide-react";
import { SpanTypeIcon } from "@/components/span-type";
import { Button } from "@/components/ui/button";
import { formatCompact, formatCost, formatDuration } from "@/lib/format";
import type { SpanNode } from "@/lib/types";
import { cn } from "@/lib/utils";
import { buildGraph, layoutGraph, NODE_H, NODE_W, subtreeTotals, type GraphNode, type PlacedNode } from "./graph";

const PADDING = 24;
const MIN_SCALE = 0.4;
const MAX_SCALE = 1.5;

function edgePath(from: PlacedNode, to: PlacedNode): string {
  const x1 = from.x + NODE_W / 2 + PADDING;
  const y1 = from.y + NODE_H + PADDING;
  const x2 = to.x + NODE_W / 2 + PADDING;
  const y2 = to.y + PADDING;
  const mid = (y1 + y2) / 2;
  return `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;
}

function NodeCard({
  node,
  active,
  selected,
  onClick,
}: {
  node: GraphNode;
  active: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  const totals = subtreeTotals(node);
  const icon = node.kind === "tool" ? "tool" : node.kind === "root" ? "chain" : "agent";
  const counts = [
    node.runs > 1 && `×${node.runs}`,
    totals.llmCalls > 0 && `${totals.llmCalls} LLM`,
    node.kind !== "tool" && totals.toolCalls > 0 && `${totals.toolCalls} tool${totals.toolCalls > 1 ? "s" : ""}`,
    node.kind === "tool" && `${node.toolCalls} call${node.toolCalls > 1 ? "s" : ""}`,
  ].filter(Boolean);
  const duration = node.lastEnd - node.firstStart;

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "bg-card animate-node-in flex flex-col justify-between rounded-xl border p-2.5 text-left shadow-sm transition-[box-shadow,border-color]",
        "hover:border-foreground/25",
        node.kind === "root" && "bg-muted/60",
        totals.errors > 0 && "border-destructive/50",
        selected && "border-primary ring-primary/30 ring-2",
        active && !selected && "border-chart-2 ring-chart-2/30 ring-2",
      )}
      style={{ width: NODE_W, height: NODE_H }}
    >
      <div className="flex min-w-0 items-center gap-2">
        <SpanTypeIcon type={icon} />
        <span className="min-w-0 flex-1 truncate text-sm font-semibold" title={node.label}>
          {node.label}
        </span>
        {node.parallel && (
          <span
            className="bg-chart-4/15 text-chart-4 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium"
            title="Runs at the same time as another agent"
          >
            ∥ parallel
          </span>
        )}
        {active && (
          <span className="relative flex size-2.5 shrink-0">
            <span className="bg-chart-2 absolute inline-flex size-full animate-ping rounded-full opacity-75" />
            <span className="bg-chart-2 relative inline-flex size-2.5 rounded-full" />
          </span>
        )}
      </div>
      <div className="text-muted-foreground flex items-center gap-1.5 truncate text-[11px]">
        {counts.join(" · ") || "—"}
        {node.models.length > 0 && <span className="truncate">· {node.models.join(", ")}</span>}
      </div>
      <div className="text-muted-foreground flex items-center justify-between text-[11px] tabular-nums">
        <span>{node.running ? "running…" : duration > 0 ? formatDuration(duration) : ""}</span>
        <span className="flex gap-2">
          {totals.tokens > 0 && <span>{formatCompact(totals.tokens)} tok</span>}
          {totals.costUsd > 0 && <span>{formatCost(totals.costUsd)}</span>}
          {totals.errors > 0 && <span className="text-destructive">{totals.errors} err.</span>}
        </span>
      </div>
    </button>
  );
}

export function GraphView({
  spans,
  rootLabel,
  running,
  freshIds,
  selectedId,
  onSelect,
}: {
  spans: SpanNode[];
  rootLabel: string;
  running: boolean;
  freshIds: Set<string>;
  selectedId: string | null;
  onSelect: (spanId: string) => void;
}) {
  const root = useMemo(() => buildGraph(spans, rootLabel), [spans, rootLabel]);
  const { placed, width, height } = useMemo(() => layoutGraph(root), [root]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [autoFit, setAutoFit] = useState(true);

  const canvasW = width + PADDING * 2;
  const canvasH = height + PADDING * 2;

  useLayoutEffect(() => {
    if (!autoFit || !containerRef.current) return;
    const available = containerRef.current.clientWidth;
    setScale(Math.max(MIN_SCALE, Math.min(1, available / canvasW)));
  }, [autoFit, canvasW]);

  // The latest node to have finished work is where the run currently is.
  const latest = useMemo(() => {
    let best: GraphNode | null = null;
    for (const p of placed) {
      if (p.node.kind === "root" || !p.node.spanIds.length) continue;
      if (!best || p.node.lastEnd > best.lastEnd) best = p.node;
    }
    return best;
  }, [placed]);

  const isActive = (n: GraphNode) =>
    running && n.kind !== "root" && (n === latest || n.spanIds.some((id) => freshIds.has(id)));

  const zoom = (factor: number) => {
    setAutoFit(false);
    setScale((s) => Math.max(MIN_SCALE, Math.min(MAX_SCALE, s * factor)));
  };

  return (
    <div className="relative">
      <div className="absolute top-2 right-3 z-10 flex gap-1">
        <Button variant="outline" size="icon" onClick={() => zoom(1 / 1.2)} title="Zoom out">
          <Minus />
        </Button>
        <Button variant="outline" size="icon" onClick={() => zoom(1.2)} title="Zoom in">
          <Plus />
        </Button>
        <Button variant="outline" size="icon" onClick={() => setAutoFit(true)} title="Fit to view">
          <Maximize2 />
        </Button>
      </div>
      <div
        ref={containerRef}
        className="graph-grid scrollbar-thin overflow-auto"
        style={{ height: Math.min(Math.max(canvasH * scale, 220), 620) }}
      >
        <div style={{ width: canvasW * scale, height: canvasH * scale }}>
          <div
            className="relative origin-top-left"
            style={{ width: canvasW, height: canvasH, transform: `scale(${scale})` }}
          >
            <svg className="pointer-events-none absolute inset-0" width={canvasW} height={canvasH}>
              {placed
                .filter((p) => p.parent)
                .map((p) => {
                  const active = isActive(p.node);
                  return (
                    <path
                      key={p.node.key}
                      d={edgePath(p.parent!, p)}
                      fill="none"
                      strokeWidth={active ? 2 : 1.5}
                      strokeDasharray={p.node.parallel && !active ? "6 4" : undefined}
                      className={cn(active ? "edge-flow stroke-chart-2" : "stroke-border", p.node.errors > 0 && "stroke-destructive/60")}
                    />
                  );
                })}
            </svg>
            {placed.map((p) => {
              const ids = p.node.spanIds;
              return (
                <div key={p.node.key} className="absolute" style={{ left: p.x + PADDING, top: p.y + PADDING }}>
                  <NodeCard
                    node={p.node}
                    active={isActive(p.node)}
                    selected={!!selectedId && ids.includes(selectedId)}
                    onClick={() => {
                      if (ids.length) onSelect(ids[0]);
                    }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
