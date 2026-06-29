"""
Client HTTP fire-and-forget vers le backend LLMInsight.
Chaque envoi se fait dans un thread daemon — zéro impact sur la latence de l'application.

Mode offline : si LLMINSIGHT_OFFLINE=1 ou si project_id/api_key sont absents,
tous les appels HTTP sont silencieusement ignores. Les spans sont toujours captures
en memoire locale pour la generation du graphe.
"""

import os
import threading
import logging
import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("llminsight")

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
    if os.environ.get("LLMINSIGHT_OFFLINE", "").strip() == "1":
        return True
    if not _config.get("api_key") or not _config.get("project_id"):
        return True
    return False


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
    except URLError as e:
        if hasattr(e, "read"):
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
        else:
            detail = str(e.reason) if hasattr(e, "reason") else str(e)
        logger.debug("LLMInsight ERREUR %s: %s", endpoint, detail)
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("LLMInsight ERREUR %s: %s", endpoint, e)
        return None


def post_sync(endpoint: str, payload: dict) -> Optional[dict]:
    """Blocking POST — used for trace start/end where we need the response."""
    if not _config["enabled"] or is_offline():
        return None
    return _post(endpoint, payload)


def post_async(endpoint: str, payload: dict):
    """Fire-and-forget POST in a daemon thread."""
    if not _config["enabled"] or is_offline():
        return
    t = threading.Thread(target=_post, args=(endpoint, payload), daemon=True)
    t.start()


def send_span(span_data: dict):
    span_data.setdefault("project_id", _config["project_id"])
    post_async("/ingest/span", span_data)


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


def end_trace(trace_id: str, status: str = "completed") -> Optional[dict]:
    payload = {"trace_id": trace_id, "status": status}
    return post_sync("/ingest/trace/end", payload)
