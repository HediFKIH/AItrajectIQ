import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { Radio } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLiveEventsFeed } from "@/store/live";

const STORAGE_KEY = "aitrajectiq.follow";

/** When enabled, open every trace that starts while the page is displayed. */
function useFollowLive(): [boolean, (value: boolean) => void] {
  const [follow, setFollowState] = useState(() => localStorage.getItem(STORAGE_KEY) === "1");
  const events = useLiveEventsFeed();
  const navigate = useNavigate();
  const lastSeen = useRef(events[0]?.at ?? 0);

  useEffect(() => {
    const fresh = events.filter((e) => e.kind === "started" && e.at > lastSeen.current);
    if (events[0]) lastSeen.current = Math.max(lastSeen.current, events[0].at);
    if (follow && fresh.length) navigate(`/traces/${fresh[0].traceId}`);
  }, [events, follow, navigate]);

  const setFollow = (value: boolean) => {
    localStorage.setItem(STORAGE_KEY, value ? "1" : "0");
    setFollowState(value);
  };
  return [follow, setFollow];
}

export function FollowLiveToggle({ className }: { className?: string }) {
  const [follow, setFollow] = useFollowLive();
  return (
    <button
      type="button"
      role="switch"
      aria-checked={follow}
      onClick={() => setFollow(!follow)}
      title="Automatically open each new trace"
      className={cn(
        "inline-flex h-9 shrink-0 items-center gap-2 rounded-md border px-3 text-xs font-medium transition-colors",
        follow ? "border-chart-2/40 bg-chart-2/10 text-chart-2" : "bg-card text-muted-foreground hover:text-foreground",
        className,
      )}
    >
      <Radio className={cn("size-3.5", follow && "animate-pulse")} />
      Follow live
    </button>
  );
}
