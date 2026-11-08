"""Benchmark a policy's complete reference repertoire, then serve its report."""

from __future__ import annotations

import argparse
import collections
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", nargs="?", help="Registered ORCS task ID")
    p.add_argument("--agent", choices=("trained", "initial", "release"), default="trained")
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--checkpoint-file")
    sources.add_argument("--wandb-run-path", help="entity/project/run-id (read only)")
    p.add_argument("--wandb-checkpoint-name")
    p.add_argument("--agent-config", help="Saved agent.yaml; adjacent params/agent.yaml is used automatically")
    p.add_argument("--num-envs", type=positive, default=4096, help="Maximum concurrent trials (default: 4096)")
    p.add_argument("--trials-per-motion", type=positive, default=65)
    p.add_argument("--nominal", action="store_true", help="Remove robustness noise, retain task events and failures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=positive, default=20, help="Normalized-time bins, plus a separate hold bin")
    p.add_argument("--device", help="For example cuda:0 (default: CUDA if available)")
    p.add_argument("--cpu-threads", type=positive, default=4)
    p.add_argument("--output", help="New result directory (default: logs/eval/<task>/<source>/<checkpoint>/<time>)")
    p.add_argument("--report-only", metavar="DIRECTORY", help="Regenerate saved trial facts without a GPU")
    p.add_argument("--no-serve", action="store_true", help="Save artifacts and exit")
    p.add_argument("--port", type=int, default=0, help="Local report port (default: choose a free port)")
    p.add_argument("--_collect", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None) -> None:
    p = parser()
    args = p.parse_args(argv)
    if not 0 <= args.port <= 65535:
        p.error("--port must be between 0 and 65535")
    if args.seed < 0:
        p.error("--seed must be nonnegative for a reproducible benchmark")
    if args.report_only:
        if args.task or args.checkpoint_file or args.wandb_run_path or args.output:
            p.error("--report-only takes an existing directory, without a task/checkpoint/output")
        output = Path(args.report_only).expanduser().resolve()
        code = 0
    else:
        if not args.task:
            p.error("a task is required for collection")
        if args.agent == "trained" and not (args.checkpoint_file or args.wandb_run_path):
            p.error("provide --checkpoint-file or --wandb-run-path (or --agent initial/release)")
        if args.agent != "trained" and (args.checkpoint_file or args.wandb_run_path):
            p.error("checkpoint sources require --agent trained")
        if args.wandb_checkpoint_name and not args.wandb_run_path:
            p.error("--wandb-checkpoint-name requires --wandb-run-path")
        if args._collect:
            from orcs.core.motion_benchmark.balanced_rollouts import collect
            collect(args)
            return
        source = (args.wandb_run_path.split("/")[-1] if args.wandb_run_path
                  else Path(args.checkpoint_file).resolve().parent.name if args.checkpoint_file else args.agent)
        checkpoint = (Path(args.checkpoint_file).stem if args.checkpoint_file
                      else Path(args.wandb_checkpoint_name).stem if args.wandb_checkpoint_name else args.agent)
        # CLI values are labels here, never path traversal components.
        def safe(s):
            return "".join(c if c.isalnum() or c in "._-" else "_" for c in s).strip(".") or "source"
        output = Path(args.output).expanduser().resolve() if args.output else (
            Path("logs/eval") / safe(args.task) / safe(source) / safe(checkpoint)
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")).resolve()
        output.mkdir(parents=True, exist_ok=False)
        print(f"Results: {output}\nCollecting inference rollouts; details in collection.log", flush=True)
        # A separate process releases the entire CUDA context before CPU report
        # serving, including simulator caches that empty_cache() cannot release.
        command = [sys.executable, "-u", "-m", "orcs.cli.bench_motion_repertoire",
                   *(sys.argv[1:] if argv is None else argv), "--_collect", "--output", str(output)]
        tail = collections.deque(maxlen=30)
        with (output / "collection.log").open("w") as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True, env=os.environ.copy())
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    tail.append(line)
                    if line.startswith("BENCH"):
                        print(line, end="", flush=True)
                code = child.wait()
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
                code = 130
            finally:
                child.stdout.close()
        if code:
            print("Collection stopped; saved records will be reported as partial.\n" + "".join(tail), file=sys.stderr)
            if not (output / "trials.jsonl").is_file():
                raise SystemExit(code)
    from orcs.core.motion_benchmark.report import render, serve
    summary = render(output)
    counts = summary["macro"]
    print(f"Completed {counts['completed']}/{counts['expected']}; failed {counts['failed']}; "
          f"invalid {counts['invalid']}; censored {counts['censored']}; pending {counts['pending']}")
    print(f"Report: {(output / 'report.html').as_uri()}")
    if not args.no_serve and code == 0:
        serve(output, args.port)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
