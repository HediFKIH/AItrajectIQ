import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Manual scripts that call real providers when imported: run them with `python tests/<file>`.
collect_ignore = ["test_simple.py", "test_no_trace.py", "test_agent.py", "mock_backend.py"]
