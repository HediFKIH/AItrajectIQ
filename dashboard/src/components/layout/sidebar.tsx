import { NavLink, useMatch, useSearchParams } from "react-router";
import { ChevronsUpDown, CircleDollarSign, Gauge, KeyRound, LayoutDashboard, Lightbulb, ListTree } from "lucide-react";
import { Logo } from "@/components/brand/logo";
import { cn } from "@/lib/utils";
import { session, useCurrentProject, useSession } from "@/store/session";

export { Logo };

const NAV = [
  { to: "/overview", label: "Overview", icon: LayoutDashboard },
  { to: "/traces", label: "Traces", icon: ListTree },
  { to: "/costs", label: "Costs", icon: CircleDollarSign },
  { to: "/metrics", label: "Metrics", icon: Gauge },
  { to: "/recommendations", label: "Recommendations", icon: Lightbulb },
];

const TRACE_SCOPED = new Set(["/metrics", "/recommendations"]);

/** Nav links, with "Metrics" and "Recommendations" scoped to the trace being viewed. */
function useNav() {
  const [params] = useSearchParams();
  const onTrace = useMatch("/traces/:traceId")?.params.traceId;
  const onMetrics = useMatch("/metrics");
  const onRecommendations = useMatch("/recommendations");
  const traceId = onTrace ?? (onMetrics || onRecommendations ? params.get("trace") : null);
  return NAV.map((item) => (TRACE_SCOPED.has(item.to) && traceId ? { ...item, to: `${item.to}?trace=${traceId}` } : item));
}

function ProjectSwitcher() {
  const projects = useSession((s) => s.projects);
  const current = useCurrentProject();
  return (
    <div className="relative">
      <label className="text-muted-foreground mb-1 block px-1 text-[11px] font-medium tracking-wide uppercase">Project</label>
      <div className="bg-card hover:bg-muted relative flex items-center rounded-lg border transition-colors">
        <div className="bg-accent text-accent-foreground ml-2 flex size-6 shrink-0 items-center justify-center rounded-md text-xs font-semibold">
          {(current?.name ?? "?").slice(0, 1).toUpperCase()}
        </div>
        <select
          aria-label="Project"
          value={current?.id ?? ""}
          onChange={(e) => session.selectProject(e.target.value)}
          className="w-full cursor-pointer appearance-none truncate bg-transparent py-2 pr-8 pl-2 text-sm font-medium outline-none"
        >
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        <ChevronsUpDown className="text-muted-foreground pointer-events-none absolute right-2 size-4" />
      </div>
    </div>
  );
}

export function Sidebar() {
  const nav = useNav();
  return (
    <aside className="bg-sidebar hidden w-60 shrink-0 flex-col border-r md:flex">
      <div className="flex h-14 items-center border-b px-4">
        <Logo />
      </div>
      <div className="px-3 pt-4">
        <ProjectSwitcher />
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 px-3 pt-5">
        {nav.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors",
                isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )
            }
          >
            <Icon className="size-4" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t p-3">
        <NavLink
          to="/connect"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors",
              isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )
          }
        >
          <KeyRound className="size-4" />
          Connect an agent
        </NavLink>
      </div>
    </aside>
  );
}

export function MobileNav() {
  const nav = useNav();
  return (
    <nav className="bg-sidebar flex gap-1 overflow-x-auto border-b px-3 py-2 md:hidden">
      {nav.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium whitespace-nowrap",
              isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground",
            )
          }
        >
          <Icon className="size-3.5" />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}
