import type { SpanNode } from "@/lib/types";
import { spanHasError, spanLabel } from "./tree";

export type GraphNodeKind = "root" | "agent" | "tool";

export interface GraphNode {
  key: string;
  label: string;
  kind: GraphNodeKind;
  depth: number;
  children: GraphNode[];
  /** Spans folded into this node, by start time. */
  spanIds: string[];
  /** Agent spans merged into the node (0 for nodes derived from the call stack). */
  runs: number;
  llmCalls: number;
  toolCalls: number;
  tokensIn: number;
  tokensOut: number;
  costUsd: number;
  errors: number;
  models: string[];
  firstStart: number;
  lastEnd: number;
  /** Agent run at the same time as a sibling agent. */
  parallel: boolean;
  /** An agent span of the node is still open. */
  running: boolean;
}

const AGENT_TYPES = new Set(["agent", "chain"]);
const TOOL_TYPES = new Set(["tool", "retriever", "embedding"]);

function newNode(key: string, label: string, kind: GraphNodeKind, depth: number): GraphNode {
  return {
    key,
    label,
    kind,
    depth,
    children: [],
    spanIds: [],
    runs: 0,
    llmCalls: 0,
    toolCalls: 0,
    tokensIn: 0,
    tokensOut: 0,
    costUsd: 0,
    errors: 0,
    models: [],
    firstStart: Number.POSITIVE_INFINITY,
    lastEnd: 0,
    parallel: false,
    running: false,
  };
}

/** Flag sibling agents whose runs overlap in time (or that the source reports as parallel). */
function markParallel(siblings: GraphNode[]): void {
  const agents = siblings.filter((n) => n.kind === "agent");
  for (const a of agents) {
    for (const b of agents) {
      if (a === b) continue;
      const aEnd = a.running ? Number.POSITIVE_INFINITY : a.lastEnd;
      const bEnd = b.running ? Number.POSITIVE_INFINITY : b.lastEnd;
      if (a.firstStart < bEnd && b.firstStart < aEnd) a.parallel = b.parallel = true;
    }
  }
}

/** One level of an agent path: `key` identifies the node, `label` is what is shown. */
interface Segment {
  key: string;
  label: string;
}

const spanEnd = (s: SpanNode) => (s.status === "running" ? Number.POSITIVE_INFINITY : s.start_offset_ms + s.duration_ms);

/**
 * Agent paths of a trace. Successive runs of the same agent under the same parent share a
 * node (loops show as run counts); runs that overlap in time get their own node, so
 * parallel agents with the same name stay side by side.
 */
function agentPaths(byId: Map<string, SpanNode>) {
  const pathOf = new Map<string, Segment[]>();
  const slotEnds = new Map<string, number[]>();

  const ancestors = (span: SpanNode, seen: Set<string>): Segment[] => {
    let parentId = span.parent_span_id;
    while (parentId && !seen.has(parentId)) {
      seen.add(parentId);
      const parent = byId.get(parentId);
      if (!parent) break;
      if (AGENT_TYPES.has(parent.span_type)) return own(parent, seen);
      parentId = parent.parent_span_id;
    }
    return [];
  };

  const own = (agent: SpanNode, seen: Set<string> = new Set()): Segment[] => {
    const cached = pathOf.get(agent.id);
    if (cached) return cached;
    const parentPath = ancestors(agent, seen);
    const label = spanLabel(agent);
    const group = `${parentPath.map((s) => s.key).join("/")}/${label}`;
    const ends = slotEnds.get(group) ?? [];
    let slot = ends.findIndex((end) => end <= agent.start_offset_ms);
    if (slot < 0) slot = ends.push(0) - 1;
    ends[slot] = spanEnd(agent);
    slotEnds.set(group, ends);
    const segment = slot === 0 ? { key: label, label } : { key: `${label}~${slot}`, label: `${label} (${slot + 1})` };
    const path = [...parentPath, segment];
    pathOf.set(agent.id, path);
    return path;
  };

  return { own, ancestors: (span: SpanNode) => ancestors(span, new Set()) };
}

/**
 * Execution graph of a trace, LangGraph style: one node per agent (agents with the same
 * path are merged, so loops show up as run counts), a child node per tool, and LLM calls
 * summed on the agent that made them.
 *
 * Agents come from agent/chain spans when the trace has some; otherwise from the
 * application call stack captured by the SDK (`caller_chain`).
 */
export function buildGraph(spans: SpanNode[], rootLabel: string): GraphNode {
  const root = newNode("", rootLabel, "root", 0);
  const nodes = new Map<string, GraphNode>([["", root]]);
  const byId = new Map<string, SpanNode>();
  for (const s of spans) if (s.sdk_span_id) byId.set(s.sdk_span_id, s);
  const useHierarchy = spans.some((s) => AGENT_TYPES.has(s.span_type));

  const ensurePath = (segments: Segment[]): GraphNode => {
    let node = root;
    segments.forEach((segment, i) => {
      const key = `${node.key}/${segment.key}`;
      let child = nodes.get(key);
      if (!child) {
        child = newNode(key, segment.label, "agent", i + 1);
        nodes.set(key, child);
        node.children.push(child);
      }
      node = child;
    });
    return node;
  };

  const add = (node: GraphNode, span: SpanNode) => {
    node.spanIds.push(span.id);
    node.firstStart = Math.min(node.firstStart, span.start_offset_ms);
    node.lastEnd = Math.max(node.lastEnd, span.start_offset_ms + span.duration_ms);
    if (spanHasError(span)) node.errors += 1;
  };

  // Per node, how much each model weighs: the main model is listed first, not side
  // calls that happen to come first (e.g. Claude Code's Haiku title/summary calls).
  const modelWeight = new Map<GraphNode, Map<string, number>>();

  const ordered = [...spans].sort((a, b) => a.start_offset_ms - b.start_offset_ms || a.step_index - b.step_index);
  const paths = agentPaths(byId);
  // Agents first, by start time, so parallel runs are told apart in start order.
  for (const span of ordered) if (useHierarchy && AGENT_TYPES.has(span.span_type)) paths.own(span);

  for (const span of ordered) {
    const path = useHierarchy
      ? paths.ancestors(span)
      : (span.caller_chain ?? []).map((name) => ({ key: name, label: name }));
    const owner = ensurePath(path);

    if (AGENT_TYPES.has(span.span_type) && useHierarchy) {
      const node = ensurePath(paths.own(span));
      node.runs += 1;
      if (span.metadata?.parallel === true) node.parallel = true;
      if (span.status === "running") node.running = true;
      add(node, span);
      continue;
    }
    if (TOOL_TYPES.has(span.span_type)) {
      const name = span.tool_name || spanLabel(span);
      const key = `${owner.key}#tool:${name}`;
      let tool = nodes.get(key);
      if (!tool) {
        tool = newNode(key, name, "tool", owner.depth + 1);
        nodes.set(key, tool);
        owner.children.push(tool);
      }
      tool.toolCalls += 1;
      add(tool, span);
      continue;
    }
    owner.llmCalls += 1;
    owner.tokensIn += span.tokens_in ?? 0;
    owner.tokensOut += span.tokens_out ?? 0;
    owner.costUsd += span.cost_usd ?? 0;
    if (span.model) {
      const weights = modelWeight.get(owner) ?? new Map<string, number>();
      const weight = (span.cost_usd ?? 0) * 1e6 + (span.tokens_in ?? 0) + (span.tokens_out ?? 0) + 1;
      weights.set(span.model, (weights.get(span.model) ?? 0) + weight);
      modelWeight.set(owner, weights);
    }
    add(owner, span);
  }

  // Bubble totals up so every node shows what happened inside it, then order by start.
  const finish = (node: GraphNode): void => {
    const weights = modelWeight.get(node);
    if (weights) node.models = [...weights.entries()].sort((a, b) => b[1] - a[1]).map(([m]) => m);
    node.children.forEach(finish);
    node.children.sort((a, b) => a.firstStart - b.firstStart);
    markParallel(node.children);
    for (const c of node.children) {
      node.firstStart = Math.min(node.firstStart, c.firstStart);
      node.lastEnd = Math.max(node.lastEnd, c.lastEnd);
    }
  };
  finish(root);
  if (!Number.isFinite(root.firstStart)) root.firstStart = 0;
  return root;
}

export interface NodeTotals {
  llmCalls: number;
  toolCalls: number;
  tokens: number;
  costUsd: number;
  errors: number;
}

/** Totals of a node and everything below it. */
export function subtreeTotals(node: GraphNode): NodeTotals {
  const totals: NodeTotals = {
    llmCalls: node.llmCalls,
    toolCalls: node.kind === "tool" ? node.toolCalls : 0,
    tokens: node.tokensIn + node.tokensOut,
    costUsd: node.costUsd,
    errors: node.errors,
  };
  for (const c of node.children) {
    const t = subtreeTotals(c);
    totals.llmCalls += t.llmCalls;
    totals.toolCalls += t.toolCalls;
    totals.tokens += t.tokens;
    totals.costUsd += t.costUsd;
    totals.errors += t.errors;
  }
  return totals;
}

export interface PlacedNode {
  node: GraphNode;
  x: number;
  y: number;
  parent: PlacedNode | null;
}

export const NODE_W = 216;
export const NODE_H = 84;
const H_GAP = 28;
const V_GAP = 56;

/** Top-down tidy layout: each subtree gets the width of its leaves, parents are centered. */
export function layoutGraph(root: GraphNode): { placed: PlacedNode[]; width: number; height: number } {
  const widths = new Map<GraphNode, number>();
  const measure = (n: GraphNode): number => {
    const childrenWidth = n.children.reduce((sum, c) => sum + measure(c), 0) + H_GAP * Math.max(0, n.children.length - 1);
    const w = Math.max(NODE_W, childrenWidth);
    widths.set(n, w);
    return w;
  };
  const width = measure(root);

  const placed: PlacedNode[] = [];
  let maxDepth = 0;
  const place = (n: GraphNode, left: number, depth: number, parent: PlacedNode | null) => {
    const w = widths.get(n)!;
    const p: PlacedNode = { node: n, x: left + (w - NODE_W) / 2, y: depth * (NODE_H + V_GAP), parent };
    placed.push(p);
    maxDepth = Math.max(maxDepth, depth);
    const childrenWidth = n.children.reduce((sum, c) => sum + widths.get(c)!, 0) + H_GAP * Math.max(0, n.children.length - 1);
    let x = left + (w - childrenWidth) / 2;
    for (const c of n.children) {
      place(c, x, depth + 1, p);
      x += widths.get(c)! + H_GAP;
    }
  };
  place(root, 0, 0, null);
  return { placed, width, height: (maxDepth + 1) * NODE_H + maxDepth * V_GAP };
}
