import { useSyncExternalStore } from "react";

export type LiveStatus = "idle" | "connecting" | "live" | "offline" | "revoked";

export interface LiveTraceEvent {
  traceId: string;
  kind: "started" | "completed" | "span";
  name?: string | null;
  status?: string;
  at: number;
}

interface LiveState {
  status: LiveStatus;
  events: LiveTraceEvent[];
  /** Trace ids that changed recently, for highlighting rows (id -> time). */
  touched: Record<string, number>;
}

const MAX_EVENTS = 30;
const TOUCH_MS = 4000;

let state: LiveState = { status: "idle", events: [], touched: {} };
const listeners = new Set<() => void>();

function emit(next: Partial<LiveState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export const live = {
  setStatus(status: LiveStatus) {
    if (state.status !== status) emit({ status });
  },
  push(event: LiveTraceEvent) {
    const now = Date.now();
    const touched: Record<string, number> = {};
    for (const [id, at] of Object.entries(state.touched)) if (now - at < TOUCH_MS) touched[id] = at;
    touched[event.traceId] = now;
    emit({ events: [event, ...state.events].slice(0, MAX_EVENTS), touched });
  },
  reset() {
    emit({ status: "idle", events: [], touched: {} });
  },
};

export function useLiveStatus(): LiveStatus {
  return useSyncExternalStore(subscribe, () => state.status);
}

export function useLiveEventsFeed(): LiveTraceEvent[] {
  return useSyncExternalStore(subscribe, () => state.events);
}

export function useTouched(): Record<string, number> {
  return useSyncExternalStore(subscribe, () => state.touched);
}

export const TOUCH_DURATION_MS = TOUCH_MS;
