import { Segmented } from "@/components/ui/segmented";
import type { RangeKey } from "@/lib/types";
import { session, useRange } from "@/store/session";

export const RANGE_OPTIONS: { value: RangeKey; label: string }[] = [
  { value: "1h", label: "1h" },
  { value: "24h", label: "24h" },
  { value: "7d", label: "7d" },
  { value: "30d", label: "30d" },
];

export const RANGE_LABEL: Record<RangeKey, string> = {
  "1h": "the last hour",
  "24h": "the last 24 hours",
  "7d": "the last 7 days",
  "30d": "the last 30 days",
};

export function RangePicker() {
  const range = useRange();
  return <Segmented value={range} options={RANGE_OPTIONS} onChange={session.setRange} />;
}
