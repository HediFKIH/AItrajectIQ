# AItrajectIQ Dashboard

Real-time React dashboard for AItrajectIQ: overview, traces (span tree and waterfall), costs and evaluation metrics.

Stack: Vite, React 19, TypeScript, Tailwind CSS v4, TanStack Query, Recharts. Live updates come from the
backend's Server-Sent Events stream (`GET /api/v1/stream`).

## Run

```powershell
cd dashboard
npm install
npm run dev        # http://localhost:5173
```

The backend URL defaults to `http://localhost:8000`; override it with `VITE_API_URL` in `dashboard/.env`
(see `.env.example`).

Open the app and paste a project API key (`atq_...`) with the `read` scope. Keys are stored in the
browser's local storage and sent in the `X-API-Key` header, never in URLs. Several projects can be saved and
switched from the sidebar.

## Build

```powershell
npm run build      # type check + production bundle in dist/
npm run preview
```

## Notes

- Run the backend with a single uvicorn worker: live events are published in-process.
- Traces, costs and metrics refresh automatically when the stream reports new activity; a running trace's
  detail page also polls every 5 s as a fallback.
