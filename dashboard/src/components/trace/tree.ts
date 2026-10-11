import type { SpanNode } from "@/lib/types";

export interface TreeRow {
  span: SpanNode;
  depth: number;
  hasChildren: boolean;
}

const byStart = (a: SpanNode, b: SpanNode) => a.start_offset_ms - b.start_offset_ms || a.step_index - b.step_index;

/** Flatten spans into display rows (depth-first, children by start time), skipping
 *  the descendants of collapsed spans. Spans whose parent isn't stored become roots. */
export function buildRows(spans: SpanNode[], collapsed: Set<string>): TreeRow[] {
  const byId = new Map<string, SpanNode>();
  for (const s of spans) if (s.sdk_span_id) byId.set(s.sdk_span_id, s);

  const children = new Map<string, SpanNode[]>();
  const roots: SpanNode[] = [];
  for (const s of spans) {
    const parent = s.parent_span_id;
    if (parent && parent !== s.sdk_span_id && byId.has(parent)) {
      const list = children.get(parent) ?? [];
      list.push(s);
      children.set(parent, list);
    } else {
      roots.push(s);
    }
  }

  const rows: TreeRow[] = [];
  const seen = new Set<string>();
  const visit = (span: SpanNode, depth: number) => {
    if (seen.has(span.id)) return;
    seen.add(span.id);
    const kids = span.sdk_span_id ? (children.get(span.sdk_span_id) ?? []) : [];
    rows.push({ span, depth, hasChildren: kids.length > 0 });
    if (collapsed.has(span.id)) return;
    [...kids].sort(byStart).forEach((k) => visit(k, depth + 1));
  };
  [...roots].sort(byStart).forEach((r) => visit(r, 0));
  // Spans caught in a parent cycle are never reached from a root: show them at the top level.
  spans.filter((s) => !seen.has(s.id)).sort(byStart).forEach((s) => visit(s, 0));
  return rows;
}

export function spanLabel(span: SpanNode): string {
  return span.name || span.tool_name || span.agent_name || span.model || span.span_type;
}

export function spanHasError(span: SpanNode): boolean {
  return Boolean(span.error || span.tool_error || span.status === "failed" || span.status === "error");
}
