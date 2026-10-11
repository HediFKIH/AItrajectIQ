import { useSyncExternalStore } from "react";

export type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "aitrajectiq.theme";
const media = window.matchMedia("(prefers-color-scheme: dark)");
const listeners = new Set<() => void>();

let theme: Theme = (localStorage.getItem(STORAGE_KEY) as Theme) || "system";

function isDark(t: Theme) {
  return t === "dark" || (t === "system" && media.matches);
}

function apply() {
  document.documentElement.classList.toggle("dark", isDark(theme));
  listeners.forEach((l) => l());
}

media.addEventListener("change", () => theme === "system" && apply());
apply();

export function setTheme(next: Theme) {
  theme = next;
  localStorage.setItem(STORAGE_KEY, next);
  apply();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTheme(): { theme: Theme; dark: boolean } {
  const current = useSyncExternalStore(subscribe, () => theme);
  return { theme: current, dark: isDark(current) };
}
