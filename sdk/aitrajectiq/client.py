"""
Client HTTP vers le backend AItrajectIQ.
Les spans partent d'une file d'envoi traitee par un thread de fond : zero impact sur la
latence de l'application. `flush()` attend que la file soit vide ; il est appele avant
de fermer une trace et a l'arret du programme, pour ne perdre aucun span.

Mode offline : si AITRAJECTIQ_OFFLINE=1 ou si la cle API est absente,
tous les appels HTTP sont silencieusement ignores. Les spans sont toujours captures
en memoire locale pour la generation du graphe. Le project_id est optionnel :
le backend le deduit de la cle API.
"""

import atexit
import os
import queue
import threading
import time
import logging
import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger("aitrajectiq")

_QUEUE_MAX = 10_000
_SENDER_THREADS = 4
_EXIT_FLUSH_TIMEOUT_S = 10.0

_queue: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
_sender_lock = threading.Lock()
_senders: list = []
_error_count = 0

_config = {
    "base_url": "http://localhost:8000",
    "api_key": None,
    "project_id": None,
    "enabled": True,
}


def configure(base_url: str, api_key: str, project_id: str):
    _config["base_url"] = base_url.rstrip("/")
    _config["api_key"] = api_key
    _config["project_id"] = project_id


def is_offline() -> bool:
    """Returns True when backend calls should be skipped entirely."""
    if os.environ.get("AITRAJECTIQ_OFFLINE", "").strip() == "1":
        return True
    if not _config.get("api_key"):
        return True
    return False


def _report_error(endpoint: str, detail: str):
    """Warn about the first failures, then every 50th, so a broken setup is visible
    without flooding the application's logs."""
    global _error_count
    _error_count += 1
    if _error_count <= 5 or _error_count % 50 == 0:
        logger.warning("AItrajectIQ: envoi %s refuse (%d echec(s)) : %s", endpoint, _error_count, detail)


def _post(endpoint: str, payload: dict) -> Optional[dict]:
    url = f"{_config['base_url']}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": _config["api_key"] or "",
    }
    data = json.dumps(payload, default=str).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body
    except HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:500]
        except Exception:
            detail = str(e)
        _report_error(endpoint, f"HTTP {e.code} {detail}")
        return None
    except URLError as e:
        _report_error(endpoint, f"backend injoignable ({_config['base_url']}): {getattr(e, 'reason', e)}")
        return None
    except (OSError, json.JSONDecodeError) as e:
        _report_error(endpoint, str(e))
        return None


def _sender_loop():
    while True:
        endpoint, payload = _queue.get()
        try:
            _post(endpoint, payload)
        except Exception as e:
            logger.debug("AItrajectIQ: envoi %s impossible: %s", endpoint, e)
        finally:
            _queue.task_done()


def _ensure_senders():
    if _senders:
        return
    with _sender_lock:
        if _senders:
            return
        for i in range(_SENDER_THREADS):
            t = threading.Thread(target=_sender_loop, name=f"aitrajectiq-sender-{i}", daemon=True)
            t.start()
            _senders.append(t)


def flush(timeout: float = 5.0) -> bool:
    """Wait until every queued span has been sent. Returns False on timeout."""
    deadline = time.monotonic() + timeout
    while _queue.unfinished_tasks:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def post_sync(endpoint: str, payload: dict) -> Optional[dict]:
    """Blocking POST — used for trace start/end where we need the response."""
    if not _config["enabled"] or is_offline():
        return None
    return _post(endpoint, payload)


def post_async(endpoint: str, payload: dict):
    """Queue a POST for the background senders."""
    if not _config["enabled"] or is_offline():
        return
    _ensure_senders()
    try:
        _queue.put_nowait((endpoint, payload))
    except queue.Full:
        _report_error(endpoint, "file d'envoi pleine, span abandonne")


atexit.register(lambda: flush(_EXIT_FLUSH_TIMEOUT_S))


def send_span(span_data: dict):
    span_data.setdefault("project_id", _config["project_id"])
    post_async("/ingest/span", span_data)


def send_log(log_data: dict):
    #Fire-and-forget conversational log ingestion.
    log_data.setdefault("project_id", _config["project_id"])
    post_async("/ingest/log", log_data)



def start_trace(name: str, metadata: Optional[dict] = None) -> Optional[str]:
    payload = {
        "project_id": _config["project_id"],
        "name": name,
        "metadata": metadata or {},
    }
    result = post_sync("/ingest/trace/start", payload)
    if result:
        return result.get("trace_id")
    return None


def end_trace(trace_id: str, status: str = "completed", ended_at: Optional[float] = None) -> Optional[dict]:
    """Close the trace once its spans have been delivered, so the trace evaluation
    sees all of them. `ended_at` (epoch seconds) defaults to now on the server."""
    flush()
    payload = {"trace_id": trace_id, "status": status}
    if ended_at:
        from datetime import datetime, timezone
        payload["ended_at"] = datetime.fromtimestamp(ended_at, timezone.utc).isoformat()
    return post_sync("/ingest/trace/end", payload)
