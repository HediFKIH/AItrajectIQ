"""
Bootstrap automatique — chargé via PYTHONSTARTUP par llminsight-run.
Initialise et patche LLMInsight sans toucher au code de l'application.
"""

import os

_project_id = os.environ.get("LLMINSIGHT_PROJECT_ID")
_api_key = os.environ.get("LLMINSIGHT_API_KEY")
_base_url = os.environ.get("LLMINSIGHT_BASE_URL", "http://localhost:8085")

if _project_id and _api_key:
    try:
        import llminsight

        llminsight.init(project_id=_project_id, api_key=_api_key, base_url=_base_url)
        llminsight.patch()
    except Exception as _e:
        import sys

        print(f"[LLMInsight] Bootstrap warning: {_e}", file=sys.stderr)
