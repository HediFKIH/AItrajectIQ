import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

export function tryParseJson(text: string | null | undefined): unknown | undefined {
  if (!text) return undefined;
  const trimmed = text.trim();
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return undefined;
  try {
    return JSON.parse(trimmed);
  } catch {
    return undefined;
  }
}

export function CopyButton({ text, className }: { text: string; className?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      title="Copy"
      onClick={(e) => {
        e.stopPropagation();
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1200);
        });
      }}
      className={cn("text-muted-foreground hover:text-foreground inline-flex items-center", className)}
    >
      {done ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
    </button>
  );
}

/** Text block for prompts, responses and JSON payloads, with a copy button. */
export function CodeBlock({ value, className, maxHeight = 320 }: { value: unknown; className?: string; maxHeight?: number }) {
  const parsed = typeof value === "string" ? tryParseJson(value) : undefined;
  const text =
    typeof value === "string" && parsed === undefined ? value : JSON.stringify(parsed ?? value, null, 2);
  const isJson = typeof value !== "string" || parsed !== undefined;
  return (
    <div className={cn("group bg-muted/60 relative rounded-lg border", className)}>
      <CopyButton text={text} className="absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100" />
      <pre
        className={cn(
          "scrollbar-thin overflow-auto p-3 text-xs leading-relaxed break-words whitespace-pre-wrap",
          isJson && "font-mono",
        )}
        style={{ maxHeight }}
      >
        {text}
      </pre>
    </div>
  );
}
