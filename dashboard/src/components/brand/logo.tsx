import { useId } from "react";
import { cn } from "@/lib/utils";

export const BRAND = {
  name: "AItrajectIQ",
  category: "Agentic workflow observability",
  tagline: "Trace every path. Understand every action.",
  features: ["Agent traces", "Workflow visualization", "Execution analytics", "Root-cause analysis"],
} as const;

/** Rounded-square mark: one rising trace through two steps to a target.
 * Three shapes only, thick strokes — readable at 16px (favicon) and 32px (sidebar). */
export function LogoMark({ className }: { className?: string }) {
  const id = useId();
  return (
    <svg viewBox="0 0 32 32" className={cn("size-8 shrink-0", className)} role="img" aria-label={BRAND.name}>
      <defs>
        <linearGradient id={`${id}-bg`} x1="4" y1="2" x2="28" y2="30" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#a78bfa" />
          <stop offset="1" stopColor="#4c1d95" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill={`url(#${id}-bg)`} />
      <path
        d="M8 23C12 23 13 16.5 16.5 16.5S21 9.5 24 9.5"
        fill="none"
        stroke="#fff"
        strokeWidth="2.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="8" cy="23" r="2.7" fill="#fff" />
      <circle cx="16.5" cy="16.5" r="2.5" fill="#fff" />
      <circle cx="24" cy="9.5" r="4.35" fill="none" stroke="#fff" strokeWidth="2" />
      <circle cx="24" cy="9.5" r="1.7" fill="#fff" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("font-semibold tracking-tight", className)}>
      AItraject<span className="text-primary">IQ</span>
    </span>
  );
}

/** Mark + name. Pass `compact={false}` to also show the category line. */
export function Logo({ compact = true }: { compact?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <LogoMark />
      <div className="min-w-0 leading-tight">
        <Wordmark className="block text-[16px]" />
        {!compact && (
          <div className="text-muted-foreground truncate text-[10px] font-medium tracking-[0.12em] uppercase" title={BRAND.tagline}>
            {BRAND.category}
          </div>
        )}
      </div>
    </div>
  );
}
