import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router";
import { Activity, ArrowRight, Coins, Eye, EyeOff, KeyRound, Lightbulb, Loader2, Plus, Trash2 } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { BRAND } from "@/components/brand/logo";
import { Logo } from "@/components/layout/sidebar";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { IntegrationGuide } from "@/components/setup/integration-guide";
import { Badge } from "@/components/ui/badge";
import { Button, buttonClass } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/empty-state";
import { Input } from "@/components/ui/input";
import { Segmented } from "@/components/ui/segmented";
import { API_URL, ApiError, apiGet, apiPost } from "@/lib/api";
import type { ProjectInfo } from "@/lib/types";
import { session, useCurrentProject, useSession } from "@/store/session";

type Mode = "new" | "key";

interface QuickstartProject {
  project_id: string;
  project_name: string;
  key: string;
  prefix: string;
}

const PROMISES = [
  { icon: Activity, title: "Every run, live", text: "Prompts, tools, subagents and answers, step by step." },
  { icon: Coins, title: "The real cost", text: "Per model, per trace, cache included, and what you could save." },
  { icon: Lightbulb, title: "What to fix", text: "Quality scored by a judge, plus rules ready to paste into Cursor or Claude Code." },
];

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function NewProjectForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function create(e: FormEvent) {
    e.preventDefault();
    const value = name.trim();
    if (!value) return;
    setPending(true);
    setError(null);
    try {
      const created = await apiPost<QuickstartProject>("/projects/quickstart", null, { name: value });
      session.addProject({ id: created.project_id, name: created.project_name, key: created.key, keyPrefix: created.prefix });
      setName("");
      onCreated();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setPending(false);
    }
  }

  return (
    <form onSubmit={create} className="space-y-3">
      <label htmlFor="project-name" className="text-sm font-medium">
        Project name
      </label>
      <Input
        id="project-name"
        autoComplete="off"
        placeholder="My agent"
        maxLength={80}
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      {error && <ErrorState error={error} />}
      <Button type="submit" className="w-full" disabled={pending || !name.trim()}>
        {pending ? <Loader2 className="animate-spin" /> : <Plus />}
        Create project
      </Button>
      <p className="text-muted-foreground text-xs">
        An API key is created with the project and stored in this browser. Setup instructions are shown next.
      </p>
    </form>
  );
}

function ExistingKeyForm() {
  const navigate = useNavigate();
  const client = useQueryClient();
  const [key, setKey] = useState("");
  const [show, setShow] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function connect(e: FormEvent) {
    e.preventDefault();
    const raw = key.trim();
    if (!raw) return;
    setPending(true);
    setError(null);
    try {
      const info = await apiGet<ProjectInfo>("/api/v1/project", raw);
      session.addProject({ id: info.id, name: info.name, key: raw, keyPrefix: info.key.prefix });
      client.setQueryData(["p", info.id, "project"], info);
      setKey("");
      navigate("/overview");
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setError("Invalid or revoked key.");
      else if (err instanceof ApiError && err.status === 403)
        setError('This key doesn\'t have the "read" scope. Use a key with the ingest and read scopes.');
      else setError(errorText(err));
    } finally {
      setPending(false);
    }
  }

  return (
    <form onSubmit={connect} className="space-y-3">
      <label htmlFor="api-key" className="text-sm font-medium">
        API key
      </label>
      <div className="relative">
        <KeyRound className="text-muted-foreground absolute top-1/2 left-3 size-4 -translate-y-1/2" />
        <Input
          id="api-key"
          type={show ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          placeholder="atq_..."
          value={key}
          onChange={(e) => setKey(e.target.value)}
          className="pr-10 pl-9 font-mono"
        />
        <button
          type="button"
          onClick={() => setShow((v) => !v)}
          className="text-muted-foreground hover:text-foreground absolute top-1/2 right-3 -translate-y-1/2"
          aria-label={show ? "Hide key" : "Show key"}
        >
          {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
        </button>
      </div>
      {error && <ErrorState error={error} />}
      <Button type="submit" className="w-full" disabled={pending || !key.trim()}>
        {pending && <Loader2 className="animate-spin" />}
        Connect
      </Button>
    </form>
  );
}

export function ConnectPage() {
  const projects = useSession((s) => s.projects);
  const current = useCurrentProject();
  const navigate = useNavigate();
  const client = useQueryClient();
  const [mode, setMode] = useState<Mode>(projects.length ? "key" : "new");
  const [justCreated, setJustCreated] = useState(false);

  return (
    <div className="min-h-full">
      <header className="flex h-14 items-center justify-between border-b px-4 md:px-6">
        <Logo />
        <div className="flex items-center gap-2">
          {projects.length > 0 && (
            <Link to="/overview" className={buttonClass("outline", "sm")}>
              Open dashboard
              <ArrowRight />
            </Link>
          )}
          <ThemeToggle />
        </div>
      </header>

      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-10 md:px-6 lg:grid-cols-[1fr_1.25fr]">
        <div className="space-y-6">
          <div>
            <p className="text-primary text-[11px] font-semibold tracking-[0.14em] uppercase">{BRAND.category}</p>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight">{BRAND.tagline}</h1>
            <p className="text-muted-foreground mt-2 text-sm">
              See what your AI agents do, and what they cost. Connect Cursor, Claude Code or your own app in two
              minutes. No code changes needed for coding agents.
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {BRAND.features.map((feature) => (
                <Badge key={feature}>{feature}</Badge>
              ))}
            </div>
          </div>

          <ul className="space-y-3">
            {PROMISES.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3">
                <div className="bg-accent text-accent-foreground flex size-8 shrink-0 items-center justify-center rounded-lg">
                  <Icon className="size-4" />
                </div>
                <div>
                  <p className="text-sm font-medium">{title}</p>
                  <p className="text-muted-foreground text-xs">{text}</p>
                </div>
              </li>
            ))}
          </ul>

          <Card>
            <CardContent className="space-y-4 pt-5">
              <Segmented<Mode>
                value={mode}
                onChange={setMode}
                options={[
                  { value: "new", label: "New project" },
                  { value: "key", label: "I already have a key" },
                ]}
              />
              {mode === "new" ? <NewProjectForm onCreated={() => setJustCreated(true)} /> : <ExistingKeyForm />}
              <p className="text-muted-foreground text-xs">
                Backend: <span className="font-mono">{API_URL}</span>
              </p>
            </CardContent>
          </Card>

          {projects.length > 0 && (
            <Card>
              <CardHeader title="Your projects" description="Switch between them from the sidebar." />
              <CardContent className="space-y-2">
                {projects.map((p) => (
                  <div key={p.id} className="flex items-center gap-3 rounded-lg border px-3 py-2.5">
                    <div className="bg-accent text-accent-foreground flex size-8 items-center justify-center rounded-md text-sm font-semibold">
                      {p.name.slice(0, 1).toUpperCase()}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        {p.name}
                        {p.id === current?.id && <Badge tone="primary">active</Badge>}
                      </div>
                      <div className="text-muted-foreground font-mono text-xs">{p.keyPrefix}…</div>
                    </div>
                    {p.id !== current?.id && (
                      <Button variant="ghost" size="sm" onClick={() => session.selectProject(p.id)}>
                        Select
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Remove ${p.name}`}
                      onClick={() => {
                        session.removeProject(p.id);
                        client.removeQueries({ queryKey: ["p", p.id] });
                      }}
                    >
                      <Trash2 className="text-muted-foreground" />
                    </Button>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>

        <Card id="setup" className="h-fit">
          {current ? (
            <>
              <CardHeader
                title={`Connect an agent to "${current.name}"`}
                description={
                  justCreated
                    ? "Project created. Copy the command for your tool: traces stream in live."
                    : "Copy the command for your tool: traces stream into the dashboard live."
                }
                action={
                  <Button size="sm" onClick={() => navigate("/overview")}>
                    Dashboard
                    <ArrowRight />
                  </Button>
                }
              />
              <CardContent>
                <IntegrationGuide key={current.id} apiKey={current.key} />
              </CardContent>
            </>
          ) : (
            <>
              <CardHeader title="How it works" />
              <CardContent>
                <ol className="space-y-3 text-sm">
                  {[
                    "Create a project: an API key is generated for it.",
                    "Copy the command for your tool (Cursor, Claude Code, Python, OpenTelemetry).",
                    "Use your agent as usual: traces, costs, evaluations and recommendations stream in live.",
                  ].map((text, i) => (
                    <li key={text} className="flex gap-3">
                      <span className="bg-primary/12 text-primary flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold">
                        {i + 1}
                      </span>
                      <span>{text}</span>
                    </li>
                  ))}
                </ol>
              </CardContent>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
