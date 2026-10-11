import { useState } from "react";
import { Check, Sparkles, Wand2 } from "lucide-react";
import { TextToCopy } from "@/components/recommendations/fix-block";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";
import { useRewritePrompt } from "@/hooks/queries";
import { formatNumber } from "@/lib/format";
import type { RangeKey, RecommendedPrompt } from "@/lib/types";

type View = "proposed" | "current";

function subtitle(prompt: RecommendedPrompt): string {
  if (prompt.target === "cursor") return "Cursor reads this file on every request: paste it as is.";
  if (prompt.target === "claude-code") return "Claude Code reads CLAUDE.md at every session: add this section to it.";
  if (prompt.model && prompt.calls) {
    return `The most expensive system prompt of the period, used by ${formatNumber(prompt.calls)} call(s) to ${prompt.model}.`;
  }
  return "Add this to your application's system prompt.";
}

/** A recommended prompt or rule file: what changes, the text to copy, and an AI rewrite on demand. */
export function PromptPanel({
  prompt,
  range,
  traceId,
}: {
  prompt: RecommendedPrompt;
  range?: RangeKey;
  traceId?: string;
}) {
  const rewrite = useRewritePrompt();
  const [view, setView] = useState<View>("proposed");
  const shown = rewrite.data ?? prompt;
  const ai = shown.source === "ai";
  const copyLabel = shown.file ? `Paste into ${shown.file}` : prompt.target === "system_prompt" ? "Suggested system prompt" : "Suggested text";

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="bg-accent text-accent-foreground flex size-8 shrink-0 items-center justify-center rounded-lg">
            <Sparkles className="size-4" />
          </div>
          <div className="min-w-0">
            <h3 className="flex flex-wrap items-center gap-2 text-sm font-semibold tracking-tight">
              {prompt.title}
              {prompt.baseline && !ai && <Badge>Starter rule</Badge>}
              {ai && <Badge tone="info">AI version</Badge>}
            </h3>
            <p className="text-muted-foreground mt-0.5 text-xs">{subtitle(prompt)}</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={rewrite.isPending || ai}
          onClick={() => rewrite.mutate({ prompt_id: prompt.id, range, trace_id: traceId })}
          title="The evaluation model rewrites this text based on the measured issues"
        >
          <Wand2 />
          {rewrite.isPending ? "Rewriting…" : ai ? "Rewritten by AI" : "Improve with AI"}
        </Button>
      </div>

      {shown.changes.length > 0 ? (
        <ul className="space-y-1 text-xs">
          {shown.changes.map((change) => (
            <li key={change} className="flex gap-2">
              <Check className="text-success mt-0.5 size-3.5 shrink-0" />
              <span>{change}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-muted-foreground text-xs">
          {prompt.baseline
            ? "Follow instructions, verify claims, check the result before answering, read and fix tool errors. It will be replaced by a targeted rule as soon as an issue is measured."
            : "No issues measured on this prompt. AI can suggest a clearer, shorter version."}
        </p>
      )}

      {rewrite.error && <p className="text-destructive text-xs">{(rewrite.error as Error).message}</p>}

      {prompt.current && (
        <Segmented<View>
          value={view}
          onChange={setView}
          options={[
            { value: "proposed", label: "Suggested" },
            { value: "current", label: "Current" },
          ]}
        />
      )}
      {view === "current" && prompt.current ? (
        <TextToCopy label="Current system prompt" content={prompt.current} />
      ) : (
        <TextToCopy label={copyLabel} content={shown.content} maxHeight={320} />
      )}
    </div>
  );
}

export function RecommendedPromptCard(props: { prompt: RecommendedPrompt; range?: RangeKey; traceId?: string }) {
  return (
    <Card className="p-4">
      <PromptPanel {...props} />
    </Card>
  );
}
