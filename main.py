# main.py
"""CLI entry point for the SonarQube AI Auto-Fix Agent."""
import argparse
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()

from db.sqlite import checkpointer
from orchestrator.supervisor import build_supervisor
from state import AgentState

supervisor = build_supervisor(checkpointer=checkpointer)


def run(project_key: str, branch: str, max_rounds: int,
        github_repo: str, thread_id: str) -> str:
    import os
    os.environ.setdefault("GITHUB_REPO", github_repo)

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(
        project_key=project_key,
        branch=branch,
        issues=[],
        fixes=[],
        validation_result=None,
        round_number=0,
        max_rounds=max_rounds,
        pr_url=None,
        messages=[],
    )

    print(f"[agent] Starting run — thread_id={thread_id}")
    final = supervisor.invoke(initial_state, config=config)
    return final.get("pr_url", "")


def resume(thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[agent] Resuming run — thread_id={thread_id}")
    final = supervisor.invoke({}, config=config)
    return final.get("pr_url", "")


def main():
    parser = argparse.ArgumentParser(
        description="SonarQube AI Auto-Fix Agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new fix run")
    run_parser.add_argument("--project", required=True, help="SonarQube project key")
    run_parser.add_argument("--branch", default="main")
    run_parser.add_argument("--max-rounds", type=int, default=3)
    run_parser.add_argument("--github-repo", required=True,
                            help="GitHub repo e.g. myorg/payment-service")
    run_parser.add_argument("--thread-id", default=None,
                            help="Optional: reuse a specific thread ID")

    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted run")
    resume_parser.add_argument("--thread-id", required=True)

    args = parser.parse_args()

    if args.command == "run":
        thread_id = args.thread_id or str(uuid.uuid4())
        print(f"[agent] thread_id: {thread_id}  (use --thread-id {thread_id} to resume)")
        pr_url = run(
            project_key=args.project,
            branch=args.branch,
            max_rounds=args.max_rounds,
            github_repo=args.github_repo,
            thread_id=thread_id,
        )
        print(f"[agent] Done. PR: {pr_url}")
        sys.exit(0)

    if args.command == "resume":
        pr_url = resume(args.thread_id)
        print(f"[agent] Resumed. PR: {pr_url}")
        sys.exit(0)


if __name__ == "__main__":
    main()
