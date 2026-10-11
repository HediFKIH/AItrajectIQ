import { Navigate, Outlet, useMatches } from "react-router";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { useCurrentProject } from "@/store/session";
import { LiveBadge } from "./live-badge";
import { RangePicker } from "./range-picker";
import { MobileNav, Sidebar } from "./sidebar";
import { ThemeToggle } from "./theme-toggle";

export interface RouteHandle {
  title: string;
  description?: string;
  showRange?: boolean;
}

export function AppShell() {
  const project = useCurrentProject();
  useLiveEvents(project);
  const matches = useMatches();
  const handle = [...matches].reverse().find((m) => (m.handle as RouteHandle | undefined)?.title)?.handle as
    | RouteHandle
    | undefined;

  if (!project) return <Navigate to="/connect" replace />;

  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="bg-background/80 sticky top-0 z-20 flex h-14 shrink-0 items-center gap-3 border-b px-4 backdrop-blur md:px-6">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold tracking-tight">{handle?.title}</h1>
          </div>
          {handle?.showRange && <RangePicker />}
          <LiveBadge />
          <ThemeToggle />
        </header>
        <MobileNav />
        <main className="scrollbar-thin flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1400px] p-4 md:p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
