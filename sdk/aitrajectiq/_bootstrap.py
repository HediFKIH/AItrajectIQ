"""
Bootstrap automatique — chargé via PYTHONSTARTUP par aitrajectiq-run.
Initialise et patche AItrajectIQ sans toucher au code de l'application.
"""

import os

_project_id = os.environ.get("AITRAJECTIQ_PROJECT_ID")
_api_key = os.environ.get("AITRAJECTIQ_API_KEY")
_base_url = os.environ.get("AITRAJECTIQ_BASE_URL", "http://localhost:8000")

if _api_key:
    try:
        import aitrajectiq

        aitrajectiq.init(project_id=_project_id, api_key=_api_key, base_url=_base_url)
        aitrajectiq.patch()
    except Exception as _e:
        import sys

        print(f"[AItrajectIQ] Bootstrap warning: {_e}", file=sys.stderr)
