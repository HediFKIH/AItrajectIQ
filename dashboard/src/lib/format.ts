import type { RangeKey } from "./types";

const LOCALE = "en-US";

/** Backend datetimes are UTC; older routes send them without a zone suffix. */
export function parseDate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const hasZone = /[zZ]|[+-]\d{2}:?\d{2}$/.test(value);
  const date = new Date(hasZone ? value : `${value}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function formatCost(value: number | null | undefined, opts: { compact?: boolean } = {}): string {
  const v = value ?? 0;
  if (v === 0) return "$0";
  const abs = Math.abs(v);
  const digits = opts.compact ? (abs >= 100 ? 0 : abs >= 1 ? 2 : abs >= 0.01 ? 3 : 4) : abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6;
  return `$${v.toLocaleString(LOCALE, { minimumFractionDigits: 0, maximumFractionDigits: digits })}`;
}

export function formatNumber(value: number | null | undefined): string {
  return (value ?? 0).toLocaleString(LOCALE);
}

export function formatCompact(value: number | null | undefined): string {
  return new Intl.NumberFormat(LOCALE, { notation: "compact", maximumFractionDigits: 1 }).format(value ?? 0);
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes} min ${seconds.toString().padStart(2, "0")} s`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toLocaleString(LOCALE, { maximumFractionDigits: digits })}%`;
}

export function formatChange(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toLocaleString(LOCALE, { maximumFractionDigits: 1 })}%`;
}

export function formatDateTime(value: string | null | undefined): string {
  const d = parseDate(value);
  if (!d) return "—";
  return d.toLocaleString(LOCALE, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const rtf = new Intl.RelativeTimeFormat(LOCALE, { numeric: "auto" });

export function formatRelative(value: string | null | undefined, now = Date.now()): string {
  const d = parseDate(value);
  if (!d) return "—";
  const seconds = Math.round((d.getTime() - now) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 45) return "just now";
  if (abs < 3600) return rtf.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(seconds / 3600), "hour");
  return rtf.format(Math.round(seconds / 86400), "day");
}

export function axisTimeFormatter(range: RangeKey) {
  const options: Intl.DateTimeFormatOptions =
    range === "1h" || range === "24h"
      ? { hour: "2-digit", minute: "2-digit" }
      : range === "7d"
        ? { weekday: "short", hour: "2-digit" }
        : { day: "2-digit", month: "short" };
  const fmt = new Intl.DateTimeFormat(LOCALE, options);
  return (value: string) => {
    const d = parseDate(value);
    return d ? fmt.format(d) : "";
  };
}

export function tooltipTimeFormatter(range: RangeKey) {
  const options: Intl.DateTimeFormatOptions =
    range === "30d"
      ? { weekday: "short", day: "2-digit", month: "short" }
      : { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" };
  const fmt = new Intl.DateTimeFormat(LOCALE, options);
  return (value: string) => {
    const d = parseDate(value);
    return d ? fmt.format(d) : "";
  };
}

export function shortId(id: string | null | undefined, size = 8): string {
  if (!id) return "—";
  return id.length > size ? `${id.slice(0, size)}…` : id;
}
