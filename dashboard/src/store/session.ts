import { useSyncExternalStore } from "react";
import type { RangeKey } from "@/lib/types";

export interface SavedProject {
  id: string;
  name: string;
  key: string;
  keyPrefix: string;
}

interface SessionState {
  projects: SavedProject[];
  currentId: string | null;
  range: RangeKey;
}

const STORAGE_KEY = "aitrajectiq.session";
const RANGES: RangeKey[] = ["1h", "24h", "7d", "30d"];

function load(): SessionState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<SessionState>;
      const projects = Array.isArray(parsed.projects) ? parsed.projects : [];
      return {
        projects,
        currentId: projects.some((p) => p.id === parsed.currentId) ? parsed.currentId! : (projects[0]?.id ?? null),
        range: RANGES.includes(parsed.range as RangeKey) ? (parsed.range as RangeKey) : "24h",
      };
    }
  } catch {
    /* corrupted storage: start fresh */
  }
  return { projects: [], currentId: null, range: "24h" };
}

let state: SessionState = load();
const listeners = new Set<() => void>();

function setState(next: Partial<SessionState>) {
  state = { ...state, ...next };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export const session = {
  get: () => state,
  addProject(project: SavedProject) {
    const projects = [...state.projects.filter((p) => p.id !== project.id), project];
    setState({ projects, currentId: project.id });
  },
  removeProject(id: string) {
    const projects = state.projects.filter((p) => p.id !== id);
    setState({ projects, currentId: state.currentId === id ? (projects[0]?.id ?? null) : state.currentId });
  },
  selectProject(id: string) {
    if (state.projects.some((p) => p.id === id)) setState({ currentId: id });
  },
  setRange(range: RangeKey) {
    setState({ range });
  },
};

export function useSession<T>(selector: (s: SessionState) => T): T {
  return useSyncExternalStore(subscribe, () => selector(state));
}

export function useCurrentProject(): SavedProject | null {
  return useSession((s) => s.projects.find((p) => p.id === s.currentId) ?? null);
}

export function useRange(): RangeKey {
  return useSession((s) => s.range);
}
