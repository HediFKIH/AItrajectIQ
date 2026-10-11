import { useLiveStatus, type LiveStatus } from "@/store/live";
import { cn } from "@/lib/utils";

const LABELS: Record<LiveStatus, { text: string; dot: string; title: string }> = {
  idle: { text: "Offline", dot: "bg-muted-foreground", title: "No project connected" },
  connecting: { text: "Connecting…", dot: "bg-warning", title: "Connecting to the real-time stream" },
  live: { text: "Live", dot: "bg-success", title: "New data shows up live" },
  offline: { text: "Reconnecting…", dot: "bg-warning", title: "Stream interrupted, retrying in a few seconds" },
  revoked: { text: "Key rejected", dot: "bg-destructive", title: "The API key was revoked or doesn't have the read scope" },
};

export function LiveBadge() {
  const status = useLiveStatus();
  const { text, dot, title } = LABELS[status];
  return (
    <div title={title} className="bg-card flex h-8 items-center gap-2 rounded-full border px-3 text-xs font-medium">
      <span className="relative flex size-2">
        {status === "live" && <span className={cn("absolute inline-flex size-full animate-ping rounded-full opacity-60", dot)} />}
        <span className={cn("relative inline-flex size-2 rounded-full", dot)} />
      </span>
      {text}
    </div>
  );
}
