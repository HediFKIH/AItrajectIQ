import { useState } from "react";
import { ChevronLeft, ChevronRight, Loader2, Search } from "lucide-react";
import { FollowLiveToggle } from "@/components/follow-live";
import { RANGE_LABEL } from "@/components/layout/range-picker";
import { TraceTable } from "@/components/trace-table";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/empty-state";
import { Input, Select } from "@/components/ui/input";
import { useDebounced } from "@/hooks/useDebounced";
import { useTraces } from "@/hooks/queries";
import { formatNumber } from "@/lib/format";
import type { TraceFilters, TraceOrigin, TraceStatus } from "@/lib/types";
import { useRange } from "@/store/session";

const PAGE_SIZE = 50;

export function TracesPage() {
  const range = useRange();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<TraceStatus | "">("");
  const [source, setSource] = useState<TraceOrigin | "">("");
  const [page, setPage] = useState(1);
  const q = useDebounced(search.trim(), 300);

  const filters: TraceFilters = { q, status, source, page, pageSize: PAGE_SIZE };
  const traces = useTraces(range, filters);
  const total = traces.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const resetPage = <T,>(setter: (v: T) => void) => (v: T) => {
    setter(v);
    setPage(1);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center">
        <div className="relative flex-1">
          <Search className="text-muted-foreground absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          <Input
            value={search}
            onChange={(e) => resetPage(setSearch)(e.target.value)}
            placeholder="Search by name, trace ID or OpenTelemetry ID…"
            className="pl-9"
          />
        </div>
        <div className="flex gap-2">
          <Select value={status} onChange={(e) => resetPage(setStatus)(e.target.value as TraceStatus | "")} className="w-40">
            <option value="">All statuses</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="timeout">Timeout</option>
          </Select>
          <Select value={source} onChange={(e) => resetPage(setSource)(e.target.value as TraceOrigin | "")} className="w-40">
            <option value="">All sources</option>
            <option value="cursor">Cursor</option>
            <option value="claude-code">Claude Code</option>
            <option value="sdk">SDK</option>
            <option value="otel">OpenTelemetry</option>
          </Select>
          <FollowLiveToggle />
        </div>
      </div>

      {traces.error ? (
        <ErrorState error={traces.error} />
      ) : (
        <Card className="overflow-hidden">
          <div className="text-muted-foreground flex items-center justify-between border-b px-4 py-2.5 text-xs">
            <span className="flex items-center gap-2">
              {formatNumber(total)} trace{total !== 1 ? "s" : ""} in {RANGE_LABEL[range]}
              {traces.isFetching && <Loader2 className="size-3.5 animate-spin" />}
            </span>
            <span>
              Page {page} / {pages}
            </span>
          </div>
          <TraceTable
            traces={traces.data?.traces}
            loading={traces.isLoading}
            emptyLabel={q || status || source ? "No traces match these filters." : "No traces in this period."}
          />
          {pages > 1 && (
            <div className="flex items-center justify-end gap-2 border-t px-4 py-2.5">
              <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                <ChevronLeft />
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
                Next
                <ChevronRight />
              </Button>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
