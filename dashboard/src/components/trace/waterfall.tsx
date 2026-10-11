import { ChevronRight } from "lucide-react";
import { SPAN_TYPES, SpanTypeIcon } from "@/components/span-type";
import { formatCost, formatDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import { spanHasError, spanLabel, type TreeRow } from "./tree";

const TICKS = [0, 0.25, 0.5, 0.75, 1];
const INDENT_PX = 16;

export function Waterfall({
  rows,
  totalMs,
  selectedId,
  collapsed,
  freshIds,
  onSelect,
  onToggle,
}: {
  rows: TreeRow[];
  totalMs: number;
  selectedId: string | null;
  collapsed: Set<string>;
  freshIds: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
}) {
  const total = Math.max(totalMs, 1);

  return (
    <div className="text-sm">
      <div className="text-muted-foreground grid grid-cols-[minmax(220px,42%)_1fr] border-b text-[11px] font-medium">
        <div className="px-4 py-2 uppercase tracking-wide">Step</div>
        <div className="relative mr-4 py-2">
          {TICKS.map((t) => (
            <span
              key={t}
              className="absolute top-2 -translate-x-1/2 tabular-nums first:translate-x-0 last:-translate-x-full"
              style={{ left: `${t * 100}%` }}
            >
              {formatDuration(t * total)}
            </span>
          ))}
          &nbsp;
        </div>
      </div>

      <div role="tree">
        {rows.map(({ span, depth, hasChildren }) => {
          const left = (span.start_offset_ms / total) * 100;
          const width = Math.max((span.duration_ms / total) * 100, 0.4);
          const selected = span.id === selectedId;
          const failed = spanHasError(span);
          const meta = SPAN_TYPES[span.span_type] ?? SPAN_TYPES.chain;
          return (
            <div
              key={span.id}
              role="treeitem"
              aria-selected={selected}
              aria-expanded={hasChildren ? !collapsed.has(span.id) : undefined}
              onClick={() => onSelect(span.id)}
              className={cn(
                "group grid cursor-pointer grid-cols-[minmax(220px,42%)_1fr] items-center border-b last:border-0",
                selected ? "bg-accent/70" : "hover:bg-muted/50",
                freshIds.has(span.id) && "animate-flash",
              )}
            >
              <div className="flex min-w-0 items-center gap-1.5 py-1.5 pr-2 pl-3" style={{ paddingLeft: 12 + depth * INDENT_PX }}>
                {hasChildren ? (
                  <button
                    type="button"
                    aria-label={collapsed.has(span.id) ? "Expand" : "Collapse"}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggle(span.id);
                    }}
                    className="text-muted-foreground hover:text-foreground rounded p-0.5"
                  >
                    <ChevronRight className={cn("size-3.5 transition-transform", !collapsed.has(span.id) && "rotate-90")} />
                  </button>
                ) : (
                  <span className="w-[18px] shrink-0" />
                )}
                <SpanTypeIcon type={span.span_type} />
                <div className="min-w-0 flex-1">
                  <div className={cn("truncate font-medium", failed && "text-destructive")}>{spanLabel(span)}</div>
                  <div className="text-muted-foreground flex gap-2 truncate text-[11px]">
                    <span>{meta.label}</span>
                    {span.model && span.span_type === "llm" && <span className="truncate">{span.model}</span>}
                    {span.cost_usd ? <span className="tabular-nums">{formatCost(span.cost_usd)}</span> : null}
                  </div>
                </div>
              </div>
              <div className="relative mr-4 h-9">
                {TICKS.slice(1, -1).map((t) => (
                  <span key={t} className="bg-border absolute top-0 bottom-0 w-px" style={{ left: `${t * 100}%` }} />
                ))}
                <div
                  className={cn(
                    "absolute top-1/2 h-4 -translate-y-1/2 rounded-sm transition-opacity",
                    failed ? "bg-destructive" : meta.bar,
                    selected ? "opacity-100 ring-2 ring-ring/40" : "opacity-80 group-hover:opacity-100",
                  )}
                  style={{ left: `${Math.min(left, 99.6)}%`, width: `${Math.min(width, 100 - Math.min(left, 99.6))}%` }}
                  title={`${spanLabel(span)} · ${formatDuration(span.duration_ms)}`}
                />
                <span
                  className="text-muted-foreground absolute top-1/2 -translate-y-1/2 text-[11px] tabular-nums whitespace-nowrap"
                  style={
                    left + width > 80
                      ? { right: `${100 - left}%`, paddingRight: 6 }
                      : { left: `${left + width}%`, paddingLeft: 6 }
                  }
                >
                  {formatDuration(span.duration_ms)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
