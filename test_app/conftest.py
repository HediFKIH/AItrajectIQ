"""
Shared fixtures for the integration tests: they run against a live backend and are
skipped when AITRAJECTIQ_API_KEY is unset or the backend does not answer.

    $env:AITRAJECTIQ_API_KEY = "atq_..."
    pytest test_app -m integration
"""

import json
import os
import urllib.error
import urllib.request

import pytest

collect_ignore = ["chatbot.py", "demo_agent_hierarchy.py"]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a running AItrajectIQ backend and AITRAJECTIQ_API_KEY")


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("AITRAJECTIQ_BASE_URL", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.environ.get("AITRAJECTIQ_API_KEY")
    if not key:
        pytest.skip("AITRAJECTIQ_API_KEY is not set")
    return key


@pytest.fixture(scope="session")
def project(base_url, api_key) -> dict:
    """The project of the key; skips when the backend is down, fails when it rejects the key."""
    request = urllib.request.Request(f"{base_url}/auth/whoami", headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        pytest.fail(f"{base_url}/auth/whoami answered {e.code}: check AITRAJECTIQ_API_KEY")
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"backend not reachable at {base_url}: {e}")
