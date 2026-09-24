"""Download and verify public model checkpoints — ORCS's, or a consumer's (``package``)."""

from __future__ import annotations

import argparse

from orcs.release import ensure_released_model, released_model_ids


def main(package: str = "orcs") -> None:
    parser = argparse.ArgumentParser(
        description=f"Download and verify public {package} model checkpoints."
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        metavar="TASK",
        help="exact task ID (downloads every released model when omitted)",
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even when the cache is valid"
    )
    parser.add_argument(
        "--list", action="store_true", help="list released task IDs without downloading"
    )
    args = parser.parse_args()

    available = released_model_ids(package)
    if args.list:
        print("\n".join(available))
        return

    tasks = args.tasks or available
    for task_id in tasks:
        ensure_released_model(task_id, force=args.force, package=package)


if __name__ == "__main__":
    main()
