"""
llminsight-run : lance N'IMPORTE quelle application Python
avec l'instrumentation LLMInsight activée — ZERO modification de code.

Usage:
    llminsight-run python main.py --port 7000
    llminsight-run uvicorn app:main --host 0.0.0.0 --port 7000

Configuration via variables d'environnement :
    LLMINSIGHT_PROJECT_ID   (obligatoire)
    LLMINSIGHT_API_KEY      (obligatoire)
    LLMINSIGHT_BASE_URL     (defaut: http://localhost:8085)
"""

import os
import sys
import runpy
import logging

logger = logging.getLogger("llminsight")


def main():
    project_id = os.environ.get("LLMINSIGHT_PROJECT_ID")
    api_key = os.environ.get("LLMINSIGHT_API_KEY")
    base_url = os.environ.get("LLMINSIGHT_BASE_URL", "http://localhost:8085")

    if not project_id or not api_key:
        print(
            "ERREUR: variables d'environnement requises:\n"
            "  LLMINSIGHT_PROJECT_ID\n"
            "  LLMINSIGHT_API_KEY\n\n"
            "Usage:\n"
            '  set LLMINSIGHT_PROJECT_ID=<id>\n'
            '  set LLMINSIGHT_API_KEY=<key>\n'
            "  llminsight-run python main.py --port 7000",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print(
            "Usage: llminsight-run <commande> [args...]\n"
            "  Exemple: llminsight-run python main.py --port 7000",
            file=sys.stderr,
        )
        sys.exit(1)

    os.environ["LLMINSIGHT_PROJECT_ID"] = project_id
    os.environ["LLMINSIGHT_API_KEY"] = api_key
    os.environ["LLMINSIGHT_BASE_URL"] = base_url

    import llminsight

    llminsight.init(project_id=project_id, api_key=api_key, base_url=base_url)
    llminsight.patch()
    print(f"[LLMInsight] Instrumentation active (project={project_id}, backend={base_url})")

    target_args = sys.argv[1:]

    # Si la commande est "python script.py ...", on exécute le script
    # dans le même process pour que le monkey-patching soit effectif.
    if target_args[0] in ("python", "python3", "py", sys.executable):
        script = target_args[1]
        sys.argv = target_args[1:]  # le script voit ses propres args

        if script == "-m":
            module = target_args[2]
            sys.argv = target_args[2:]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            script_dir = os.path.abspath(os.path.dirname(script))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            os.chdir(script_dir)
            runpy.run_path(script, run_name="__main__")
    else:
        # Pour uvicorn, gunicorn, etc. on injecte via subprocess
        # avec PYTHONSTARTUP pour que le patching se fasse dans le child.
        bootstrap = os.path.join(os.path.dirname(__file__), "_bootstrap.py")
        env = os.environ.copy()
        env["PYTHONSTARTUP"] = bootstrap
        env["LLMINSIGHT_PROJECT_ID"] = project_id
        env["LLMINSIGHT_API_KEY"] = api_key
        env["LLMINSIGHT_BASE_URL"] = base_url

        import subprocess

        proc = subprocess.run(target_args, env=env)
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
