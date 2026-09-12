"""Download and verify public ORCS model checkpoints."""

from __future__ import annotations

import argparse

from orcs.release import ensure_released_model, released_model_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        metavar="TASK",
        help="exact Orcs task ID (downloads every released model when omitted)",
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even when the cache is valid"
    )
    parser.add_argument(
        "--list", action="store_true", help="list released task IDs without downloading"
    )
    args = parser.parse_args()

    available = released_model_ids()
    if args.list:
        print("\n".join(available))
        return

    tasks = args.tasks or available
    for task_id in tasks:
        ensure_released_model(task_id, force=args.force)


if __name__ == "__main__":
    main()
