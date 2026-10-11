import { useState, type ReactNode } from "react";
import { Eye, EyeOff } from "lucide-react";
import { TextToCopy } from "@/components/recommendations/fix-block";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/segmented";
import { API_URL } from "@/lib/api";

type Tool = "cursor" | "claude" | "python";
type Os = "windows" | "unix";

const HOOK_EVENTS = [
  "beforeSubmitPrompt",
  "postToolUse",
  "postToolUseFailure",
  "subagentStart",
  "subagentStop",
  "afterAgentResponse",
  "stop",
];

function cursorWindows(key: string): string {
  return [
    `$dir = "$HOME\\.aitrajectiq"`,
    `New-Item -ItemType Directory -Force $dir, "$HOME\\.cursor" | Out-Null`,
    `Invoke-WebRequest "${API_URL}/integrations/cursor-hook.py" -OutFile "$dir\\cursor_hook.py"`,
    `$config = @{ api_key = "${key}"; base_url = "${API_URL}" } | ConvertTo-Json`,
    `[IO.File]::WriteAllText("$dir\\config.json", $config)`,
    `$hooksFile = "$HOME\\.cursor\\hooks.json"`,
    `if (Test-Path $hooksFile) { Copy-Item $hooksFile "$hooksFile.bak" -Force }`,
    `$hook = @(@{ command = "py -3 ""$dir\\cursor_hook.py"""; timeout = 5 })`,
    `$hooks = [ordered]@{}`,
    `foreach ($e in ${HOOK_EVENTS.map((e) => `"${e}"`).join(", ")}) { $hooks[$e] = $hook }`,
    `[IO.File]::WriteAllText($hooksFile, (@{ version = 1; hooks = $hooks } | ConvertTo-Json -Depth 6))`,
  ].join("\n");
}

function cursorUnix(key: string): string {
  return [
    "mkdir -p ~/.aitrajectiq ~/.cursor",
    `curl -fsSL "${API_URL}/integrations/cursor-hook.py" -o ~/.aitrajectiq/cursor_hook.py`,
    `printf '{"api_key": "%s", "base_url": "%s"}\\n' "${key}" "${API_URL}" > ~/.aitrajectiq/config.json`,
    "[ -f ~/.cursor/hooks.json ] && cp ~/.cursor/hooks.json ~/.cursor/hooks.json.bak",
    "python3 - <<'EOF'",
    "import json, os",
    'cmd = "python3 " + os.path.expanduser("~/.aitrajectiq/cursor_hook.py")',
    `events = [${HOOK_EVENTS.map((e) => `"${e}"`).join(", ")}]`,
    'with open(os.path.expanduser("~/.cursor/hooks.json"), "w") as f:',
    '    json.dump({"version": 1, "hooks": {e: [{"command": cmd, "timeout": 5}] for e in events}}, f, indent=2)',
    "EOF",
  ].join("\n");
}

function claudeWindows(key: string): string {
  return [
    `$setup = "$env:TEMP\\aitrajectiq_claude_setup.py"`,
    `Invoke-WebRequest "${API_URL}/integrations/claude-code-setup.py" -OutFile $setup`,
    `py -3 $setup "${API_URL}" "${key}"`,
  ].join("\n");
}

function claudeUnix(key: string): string {
  return `curl -fsSL "${API_URL}/integrations/claude-code-setup.py" | python3 - "${API_URL}" "${key}"`;
}

function pythonEnvWindows(key: string): string {
  return [`$env:AITRAJECTIQ_API_KEY = "${key}"`, `$env:AITRAJECTIQ_BASE_URL = "${API_URL}"`].join("\n");
}

function pythonEnvUnix(key: string): string {
  return [`export AITRAJECTIQ_API_KEY="${key}"`, `export AITRAJECTIQ_BASE_URL="${API_URL}"`].join("\n");
}

function pythonCode(key: string): string {
  return ["import aitrajectiq", "", `aitrajectiq.init(api_key="${key}", base_url="${API_URL}")`, "aitrajectiq.patch()"].join("\n");
}

function mcpCursor(key: string): string {
  return JSON.stringify({ mcpServers: { aitrajectiq: { url: `${API_URL}/mcp`, headers: { "X-API-Key": key } } } }, null, 2);
}

function mcpClaude(key: string): string {
  return `claude mcp add --transport http aitrajectiq ${API_URL}/mcp --header "X-API-Key: ${key}"`;
}

function Step({ n, children }: { n: number; children: ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="bg-primary/12 text-primary flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold">
        {n}
      </span>
      <div className="min-w-0 flex-1 space-y-2 text-sm">{children}</div>
    </li>
  );
}

/** Setup instructions for the active project's key, ready to paste. */
export function IntegrationGuide({ apiKey }: { apiKey: string }) {
  const [tool, setTool] = useState<Tool>("cursor");
  const [os, setOs] = useState<Os>(navigator.userAgent.includes("Windows") ? "windows" : "unix");
  const [reveal, setReveal] = useState(false);
  const masked = `${apiKey.slice(0, 8)}…`;
  const shown = (text: string) => (reveal ? undefined : text.split(apiKey).join(masked));
  const copy = (label: string, content: string, mono = true) => (
    <TextToCopy label={label} content={content} display={shown(content)} mono={mono} maxHeight={280} />
  );
  const osPicker = (
    <Segmented<Os>
      value={os}
      onChange={setOs}
      options={[
        { value: "windows", label: "Windows" },
        { value: "unix", label: "macOS / Linux" },
      ]}
    />
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Segmented<Tool>
          value={tool}
          onChange={setTool}
          options={[
            { value: "cursor", label: "Cursor" },
            { value: "claude", label: "Claude Code" },
            { value: "python", label: "Python app" },
          ]}
        />
        <Button variant="ghost" size="sm" onClick={() => setReveal(!reveal)}>
          {reveal ? <EyeOff /> : <Eye />}
          {reveal ? "Hide key" : "Show key"}
        </Button>
      </div>

      {tool === "cursor" && (
        <ol className="space-y-4">
          <Step n={1}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p>Install the hook from a terminal (requires Python 3).</p>
              {osPicker}
            </div>
            {os === "windows" ? copy("PowerShell", cursorWindows(apiKey)) : copy("Terminal", cursorUnix(apiKey))}
            <p className="text-muted-foreground text-xs">
              Replaces <code className="bg-muted rounded px-1">~/.cursor/hooks.json</code> (the old one is saved as{" "}
              <code className="bg-muted rounded px-1">hooks.json.bak</code>). If something goes wrong, the hook logs its errors to{" "}
              <code className="bg-muted rounded px-1">~/.aitrajectiq/cursor_hook.log</code> and never blocks the agent.
            </p>
          </Step>
          <Step n={2}>
            <p>Restart Cursor and ask the agent something: the trace shows up live in Traces.</p>
          </Step>
        </ol>
      )}

      {tool === "claude" && (
        <ol className="space-y-4">
          <Step n={1}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p>Run this command in a terminal (requires Python 3).</p>
              {osPicker}
            </div>
            {os === "windows" ? copy("PowerShell", claudeWindows(apiKey)) : copy("Terminal", claudeUnix(apiKey))}
            <p className="text-muted-foreground text-xs">
              It enables Claude Code telemetry and installs an end-of-response hook in{" "}
              <code className="bg-muted rounded px-1">~/.claude/settings.json</code>, without touching your other settings
              (the old file is saved as <code className="bg-muted rounded px-1">settings.json.bak</code>). The hook sends the
              final answer and the files read, which are needed to judge whether the task was completed.
            </p>
          </Step>
          <Step n={2}>
            <p>
              Restart <code className="bg-muted rounded px-1">claude</code> and send a request: each prompt becomes a trace,
              evaluated as soon as Claude finishes responding.
            </p>
          </Step>
        </ol>
      )}

      {tool === "python" && (
        <>
          <ol className="space-y-4">
            <Step n={1}>
              <p>Install the SDK from the AItrajectIQ repository.</p>
              {copy("Terminal", "pip install -e ./sdk")}
            </Step>
            <Step n={2}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p>Set the project key in your terminal.</p>
                {osPicker}
              </div>
              {os === "windows" ? copy("PowerShell", pythonEnvWindows(apiKey)) : copy("Terminal", pythonEnvUnix(apiKey))}
            </Step>
            <Step n={3}>
              <p>
                Run your app by prefixing your usual command with{" "}
                <code className="bg-muted rounded px-1">aitrajectiq</code>. No code changes needed: OpenAI, Anthropic,
                Mistral, LangChain… are traced automatically.
              </p>
              {copy("Terminal", "aitrajectiq python main.py")}
              <p className="text-muted-foreground text-xs">
                Also works with <code className="bg-muted rounded px-1">aitrajectiq uvicorn app:app --port 7000</code>.
              </p>
            </Step>
          </ol>
          <details className="text-muted-foreground text-xs">
            <summary className="cursor-pointer">Prefer to initialize it in code</summary>
            <div className="mt-2">{copy("Python", pythonCode(apiKey))}</div>
          </details>
          <p className="text-muted-foreground text-xs">
            Another language, or already on OpenTelemetry? Send OTLP over HTTP to{" "}
            <code className="bg-muted rounded px-1">{API_URL}/otel/v1/traces</code> with the{" "}
            <code className="bg-muted rounded px-1">x-api-key</code> header.
          </p>
        </>
      )}

      <section className="space-y-3 border-t pt-4">
        <div>
          <p className="text-sm font-medium">Optional: query your traces from the agent (MCP)</p>
          <p className="text-muted-foreground text-xs">
            The tabs above send traces to AItrajectIQ. MCP works the other way around: it lets Cursor or Claude Code read your
            traces, costs and recommendations, for example "why did my last trace fail?".
          </p>
        </div>
        <ol className="space-y-4">
          <Step n={1}>
            <p>
              In Cursor: <code className="bg-muted rounded px-1">.cursor/mcp.json</code>
            </p>
            {copy(".cursor/mcp.json", mcpCursor(apiKey))}
          </Step>
          <Step n={2}>
            <p>In Claude Code:</p>
            {copy("Terminal", mcpClaude(apiKey))}
          </Step>
        </ol>
      </section>
    </div>
  );
}
