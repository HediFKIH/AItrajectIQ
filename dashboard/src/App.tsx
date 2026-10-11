import { createBrowserRouter, Navigate, RouterProvider } from "react-router";
import { AppShell, type RouteHandle } from "@/components/layout/app-shell";
import { ConnectPage } from "@/pages/connect";
import { CostsPage } from "@/pages/costs";
import { MetricsPage } from "@/pages/metrics";
import { NotFoundPage } from "@/pages/not-found";
import { OverviewPage } from "@/pages/overview";
import { RecommendationsPage } from "@/pages/recommendations";
import { TraceDetailPage } from "@/pages/trace-detail";
import { TracesPage } from "@/pages/traces";

const handle = (h: RouteHandle) => h;

const router = createBrowserRouter([
  { path: "/connect", element: <ConnectPage /> },
  {
    element: <AppShell />,
    children: [
      { index: true, element: <Navigate to="/overview" replace /> },
      { path: "overview", element: <OverviewPage />, handle: handle({ title: "Overview", showRange: true }) },
      { path: "traces", element: <TracesPage />, handle: handle({ title: "Traces", showRange: true }) },
      { path: "traces/:traceId", element: <TraceDetailPage />, handle: handle({ title: "Trace details" }) },
      { path: "costs", element: <CostsPage />, handle: handle({ title: "Costs", showRange: true }) },
      { path: "metrics", element: <MetricsPage />, handle: handle({ title: "Evaluation metrics", showRange: true }) },
      {
        path: "recommendations",
        element: <RecommendationsPage />,
        handle: handle({ title: "Recommendations", showRange: true }),
      },
      { path: "*", element: <NotFoundPage />, handle: handle({ title: "Page not found" }) },
    ],
  },
]);

export function App() {
  return <RouterProvider router={router} />;
}
