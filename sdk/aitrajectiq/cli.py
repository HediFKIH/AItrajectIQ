"""
aitrajectiq-run : lance N'IMPORTE quelle application Python
avec l'instrumentation AItrajectIQ activée — ZERO modification de code.

Usage:
    aitrajectiq-run python main.py --port 7000
    aitrajectiq-run uvicorn app:main --host 0.0.0.0 --port 7000

Configuration via variables d'environnement :
    AITRAJECTIQ_API_KEY      (obligatoire — identifie le projet)
    AITRAJECTIQ_PROJECT_ID   (optionnel)
    AITRAJECTIQ_BASE_URL     (defaut: http://localhost:8000)
"""

import os
import sys
import runpy
import logging

logger = logging.getLogger("aitrajectiq")


def main():
    project_id = os.environ.get("AITRAJECTIQ_PROJECT_ID")
    api_key = os.environ.get("AITRAJECTIQ_API_KEY")
    base_url = os.environ.get("AITRAJECTIQ_BASE_URL", "http://localhost:8000")

    if not api_key:
        print(
            "ERREUR: variable d'environnement requise:\n"
            "  AITRAJECTIQ_API_KEY   (cle atq_... du projet)\n\n"
            "Usage:\n"
            '  set AITRAJECTIQ_API_KEY=<key>\n'
            "  aitrajectiq-run python main.py --port 7000",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        print(
            "Usage: aitrajectiq-run <commande> [args...]\n"
            "  Exemple: aitrajectiq-run python main.py --port 7000",
            file=sys.stderr,
        )
        sys.exit(1)

    os.environ["AITRAJECTIQ_API_KEY"] = api_key
    os.environ["AITRAJECTIQ_BASE_URL"] = base_url

    import aitrajectiq

    aitrajectiq.init(project_id=project_id, api_key=api_key, base_url=base_url)
    aitrajectiq.patch()
    print(
        f"[AItrajectIQ] Instrumentation active "
        f"(project={project_id or '<from API key>'}, backend={base_url})"
    )

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
        env["AITRAJECTIQ_API_KEY"] = api_key
        env["AITRAJECTIQ_BASE_URL"] = base_url

        import subprocess

        proc = subprocess.run(target_args, env=env)
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
