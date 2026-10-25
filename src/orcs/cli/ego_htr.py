"""Prepare, retarget and review EgoHTR motions with an optional external UMR."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, help="Default: <ORCS root>/logs/ego_htr")
    commands = p.add_subparsers(dest="command", required=True)

    def command(name, help):
        sub = commands.add_parser(name, help=help)
        sub.add_argument(
            "--json",
            action="store_true",
            help="Machine-readable result (default for non-view operations)",
        )
        return sub

    c = command("configure", "Record local optional dependencies once")
    for field in ("data-root", "umr-root", "umr-python", "smplx-model"):
        c.add_argument("--" + field, type=Path)
    c = command("doctor", "Inspect local inputs and UMR patch compatibility")
    c.add_argument("--apply-patch", action="store_true")
    c = command("import-legacy", "Import the reviewed experiment without changing it")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--ids", nargs="+", default=["03", "04", "05", "06"])
    c = command("inspect", "Inspect local sequences or prepared clips")
    c.add_argument("--sequence")
    c.add_argument("--clip")
    c = command("prepare", "Prepare an EgoHTR clip or a UMR correspondence stage")
    c.add_argument("--recipe", type=Path)
    c.add_argument("--preset")
    c.add_argument("--clip")
    c.add_argument("--run", default="latest")
    c.add_argument(
        "--stage",
        choices=["source", "collision", "build", "train", "all"],
        default="all",
    )
    c = command("retarget", "Create a separate candidate, solve and audit it")
    c.add_argument("--clip", required=True)
    c.add_argument("--parent", default="accepted")
    c.add_argument("--label", default="candidate")
    c.add_argument("--overrides", type=Path)
    c.add_argument("--max-frames", type=int, default=0)
    c = command("audit", "Recheck a motion, optionally compare a parent")
    c.add_argument("--clip", required=True)
    c.add_argument("--run", default="latest")
    c.add_argument("--compare")
    c.add_argument("--debug", action="store_true")
    c = command("report", "Generate optional comparison images/videos/gallery")
    c.add_argument("--clip", required=True)
    c.add_argument("--run", default="latest")
    c.add_argument("--compare", default="accepted")
    c.add_argument("--videos", action="store_true")
    c.add_argument("--human", action="store_true")
    c = command("stage", "Export reviewed clips for PerLoco training (no UMR)")
    c.add_argument("--clips", nargs="+", required=True, help="Viewer labels or clip IDs")
    c.add_argument("--run", default="accepted", help="Pin this revision for each clip")
    c.add_argument("--out", type=Path, help="Default: data/terrain_motions/egohtr")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--batch", type=int, default=256)
    c = command("view", "Launch the Viser review interface")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8085)
    c.add_argument("--clip")
    c.add_argument("--time", type=float, default=0.0)
    c = command("notes", "Read comments or append a disposition")
    c.add_argument("--clip", required=True)
    c.add_argument("--pending", action="store_true")
    c.add_argument("--resolve")
    c.add_argument(
        "--status", choices=["addressed", "resolved", "dismissed"], default="addressed"
    )
    c.add_argument("--run")
    c.add_argument("--text", default="")
    c = command("comment", "Record a frame comment through the same contract as Viser")
    c.add_argument("--clip", required=True)
    c.add_argument("--run", default="latest")
    c.add_argument("--frame", type=int, required=True)
    c.add_argument("--text", required=True)
    c.add_argument("--surface", default="Scan")
    c = command("accept", "Record user acceptance of a completed candidate")
    c.add_argument("--clip", required=True)
    c.add_argument("--run", required=True)
    c = command("prune", "Preview or remove selected inactive run outputs")
    c.add_argument("--clip", required=True)
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--apply", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c = command("remote", "Inspect/extract bounded NPZ members from Hugging Face")
    c.add_argument("--sequence", required=True)
    c.add_argument("--revision", required=True)
    c.add_argument("--extract", nargs="*", default=[])
    c.add_argument("--prefix", nargs="*", default=[])
    c = command("memory-probe", "Optional synthetic simulator-memory diagnostic")
    c.add_argument("--worlds", type=int, default=128)
    c.add_argument("--terrain", choices=["plane", "hfield", "convex"], default="hfield")
    c.add_argument("--ccd", type=int, default=50)
    c.add_argument("--nccd", type=int, default=64)
    c = commands.add_parser("_targets")
    c.add_argument("--request", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("_surface-audit")
    c.add_argument("--request", type=Path, required=True)
    c.add_argument("--probes", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("_prepare")
    c.add_argument("--request", type=Path, required=True)
    return p


def dispatch(a):
    from orcs.ego_htr.workspace import Workspace, read_json

    ws = Workspace(a.workspace)
    if a.command == "configure":
        return ws.configure(
            data_root=a.data_root,
            umr_root=a.umr_root,
            umr_python=a.umr_python,
            smplx_model=a.smplx_model,
        )
    if a.command == "doctor":
        from orcs.ego_htr.umr import doctor

        return doctor(ws, a.apply_patch)
    if a.command == "import-legacy":
        from orcs.ego_htr.legacy import import_legacy

        return import_legacy(ws, a.root, a.ids)
    if a.command == "inspect":
        if a.clip:
            return dict(
                clip=ws.clip(a.clip),
                runs=ws.runs(a.clip),
                pending_notes=ws.notes(a.clip, True),
            )
        from orcs.ego_htr.source import inspect_source

        return inspect_source(ws, a.sequence) if a.sequence else dict(clips=ws.clips())
    if a.command == "prepare":
        from orcs.ego_htr.source import prepare

        return prepare(ws, a.recipe, a.preset, a.clip, a.run, a.stage)
    if a.command == "retarget":
        from orcs.ego_htr.umr import solve

        return solve(
            ws,
            a.clip,
            a.parent,
            a.label,
            read_json(a.overrides) if a.overrides else None,
            a.max_frames,
        )
    if a.command == "audit":
        from orcs.ego_htr.audit import audit

        result = dict(run=a.run, checks=audit(ws, ws.run(a.clip, a.run), debug=a.debug))
        if a.compare:
            result["comparison"] = audit(ws, ws.run(a.clip, a.compare))
        return result
    if a.command == "report":
        from orcs.ego_htr.report import report

        return report(ws, a.clip, a.run, a.compare, a.videos, a.human)
    if a.command == "stage":
        from orcs.cli._stage_egohtr import stage

        return stage(ws, a.clips, a.run, a.out, a.device, a.batch)
    if a.command == "view":
        from orcs.ego_htr.viewer import Viewer

        Viewer(ws, a.host, a.port, a.clip, a.time).run()
        return None
    if a.command == "notes":
        if a.resolve:
            comments = {n["id"]: n for n in ws.notes(a.clip) if n["type"] == "comment"}
            if a.resolve not in comments:
                raise ValueError("Unknown note ID")
            if a.run:
                ws.run(a.clip, a.run)
            return ws.append_note(
                a.clip,
                dict(
                    type="disposition",
                    note_id=a.resolve,
                    status=a.status,
                    run=a.run,
                    text=a.text,
                ),
            )
        return ws.notes(a.clip, a.pending)
    if a.command == "comment":
        return ws.comment(ws.run(a.clip, a.run), a.frame, a.text, a.surface)
    if a.command == "accept":
        return dict(accepted=ws.accept(a.clip, a.run))
    if a.command == "prune":
        if a.apply and a.dry_run:
            raise ValueError("Choose --apply or --dry-run")
        return ws.prune(a.clip, a.runs, a.apply)
    if a.command == "remote":
        from orcs.ego_htr.remote import inspect_remote

        return inspect_remote(ws, a.sequence, a.revision, a.extract, a.prefix)
    if a.command == "memory-probe":
        from orcs.ego_htr.memory import probe

        return probe(a.worlds, a.terrain, a.ccd, a.nccd)
    if a.command == "_targets":
        from orcs.ego_htr.umr import prepare_targets

        prepare_targets(a.request, a.output)
    elif a.command == "_surface-audit":
        from orcs.ego_htr.audit import surface_audit

        surface_audit(a.request, a.probes, a.output)
    elif a.command == "_prepare":
        from orcs.ego_htr.source import prepare_worker

        prepare_worker(a.request)
    return None


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        # Keep stdout a single JSON result even when optional libraries print.
        with (
            contextlib.redirect_stdout(sys.stderr)
            if args.command != "view"
            else contextlib.nullcontext()
        ):
            result = dispatch(args)
        if result is not None:
            print(json.dumps(result, indent=2, allow_nan=False))
        if isinstance(result, dict) and result.get("status") in (
            "failed",
            "failed_checks",
        ):
            return 2
        return 0
    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        subprocess.CalledProcessError,
        ImportError,
    ) as error:
        print(
            json.dumps(dict(error=str(error), type=type(error).__name__)),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
