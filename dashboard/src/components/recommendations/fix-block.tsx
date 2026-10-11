import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Fix, Recommendation } from "@/lib/types";
import { cn } from "@/lib/utils";

const MONO_KINDS = new Set<Fix["kind"]>(["code", "config", "file"]);

export function CopyTextButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={(e) => {
        e.stopPropagation();
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        });
      }}
    >
      {done ? <Check /> : <Copy />}
      {done ? "Copied" : label}
    </Button>
  );
}

/** A ready-to-paste text (instruction, rule file, code) with its copy button. `display`
 * replaces the text on screen only (a masked API key), the copy keeps `content`. */
export function TextToCopy({
  label,
  content,
  display,
  mono,
  maxHeight = 224,
}: {
  label: string;
  content: string;
  display?: string;
  mono?: boolean;
  maxHeight?: number;
}) {
  return (
    <div className="bg-muted/40 rounded-lg border">
      <div className="flex items-center justify-between gap-2 border-b py-0.5 pr-1 pl-3">
        <span className="text-muted-foreground truncate text-[11px] font-medium">{label}</span>
        <CopyTextButton text={content} />
      </div>
      <pre
        className={cn(
          "scrollbar-thin overflow-auto p-3 text-xs leading-relaxed break-words whitespace-pre-wrap",
          mono && "font-mono",
        )}
        style={{ maxHeight }}
      >
        {display ?? content}
      </pre>
    </div>
  );
}

/** The text to copy of a recommendation, then the other files of the same fix (a hook's script). */
export function FixFiles({ rec, maxHeight }: { rec: Recommendation; maxHeight?: number }) {
  const files = [rec.fix, ...(rec.extra_fixes ?? [])].filter((f): f is Fix => !!f);
  if (files.length === 0) return null;
  return (
    <div className="space-y-2">
      {files.map((f, i) => (
        <TextToCopy
          key={`${f.file ?? f.label}-${i}`}
          label={f.label}
          content={f.content}
          mono={MONO_KINDS.has(f.kind)}
          maxHeight={maxHeight}
        />
      ))}
    </div>
  );
}
