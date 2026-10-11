import { fetchEventSource } from "@microsoft/fetch-event-source";
import { buildUrl } from "./api";

export interface StreamEvent {
  type: string;
  data: Record<string, unknown>;
}

export interface StreamHandlers {
  onOpen: () => void;
  onEvent: (event: StreamEvent) => void;
  onDisconnect: () => void;
  onUnauthorized: () => void;
}

class FatalStreamError extends Error {}

const RETRY_MS = 3000;

/** Connect to GET /api/v1/stream with the key in a header (never in the URL).
 *  Reconnects automatically until `signal` is aborted or the key is rejected. */
export function connectStream(apiKey: string, handlers: StreamHandlers, signal: AbortSignal): Promise<void> {
  return fetchEventSource(buildUrl("/api/v1/stream"), {
    headers: { "X-API-Key": apiKey },
    signal,
    openWhenHidden: true,
    async onopen(res) {
      if (res.status === 401 || res.status === 403) {
        handlers.onUnauthorized();
        throw new FatalStreamError("unauthorized");
      }
      if (!res.ok || !res.headers.get("content-type")?.includes("text/event-stream")) {
        throw new Error(`stream unavailable (${res.status})`);
      }
      handlers.onOpen();
    },
    onmessage(msg) {
      if (!msg.event) return;
      if (msg.event === "revoked") {
        handlers.onUnauthorized();
        throw new FatalStreamError("revoked");
      }
      let data: Record<string, unknown> = {};
      try {
        data = msg.data ? JSON.parse(msg.data) : {};
      } catch {
        return;
      }
      handlers.onEvent({ type: msg.event, data });
    },
    onclose() {
      handlers.onDisconnect();
      throw new Error("closed");
    },
    onerror(err) {
      if (err instanceof FatalStreamError) throw err;
      handlers.onDisconnect();
      return RETRY_MS;
    },
  }).catch((err) => {
    if (!(err instanceof FatalStreamError) && !signal.aborted) throw err;
  });
}
