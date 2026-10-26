"""Isolated external UMR bridge. No UMR import or mandatory dependency."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

from .robot import normalize_motion
from .workspace import digest, merge, now, read_json, safe_name, write_json

UMR_COMMIT = "d1cdaedaf67a827198da71c5fa831f48b2e51f91"


def isolated_command(python, *arguments):
    # Load only ORCS into the isolated interpreter. Adding the main environment's
    # site-packages to PYTHONPATH would shadow UMR's NumPy/MuJoCo versions.
    bootstrap = (
        "import pathlib, runpy, sys, tempfile; "
        "p=pathlib.Path(sys.argv[1]); "
        'bridge=tempfile.TemporaryDirectory(prefix="egohtr-import-"); '
        '(pathlib.Path(bridge.name)/"orcs").symlink_to(p,target_is_directory=True); '
        "sys.path.insert(0,bridge.name); "
        'sys.argv=["orcs-egohtr",*sys.argv[2:]]; runpy.run_module("orcs.cli.ego_htr",run_name="__main__")'
    )
    return [python, "-c", bootstrap, str(files("orcs")), *arguments]


def patch_path():
    return Path(str(files("orcs.ego_htr").joinpath("patches/umr_egohtr.patch")))


def patch_status(root, apply=False):
    root = Path(root)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UMR_COMMIT:
        raise ValueError(f"UMR revision {revision}; expected {UMR_COMMIT}")
    patch = patch_path()
    done = subprocess.run(
        ["git", "-C", str(root), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
    )
    if done.returncode == 0:
        return dict(revision=revision, patch="applied", patch_sha256=digest(patch))
    clean = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", str(patch)], capture_output=True
    )
    if clean.returncode:
        raise ValueError(
            "UMR integration patch conflicts with this checkout; inspect local edits"
        )
    if apply:
        subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
    return dict(
        revision=revision,
        patch="applied" if apply else "needed",
        patch_sha256=digest(patch),
    )


def doctor(ws, apply_patch=False):
    import importlib.util

    settings = ws.settings()
    checks = {
        name: bool(importlib.util.find_spec(name))
        for name in ("numpy", "mujoco", "viser", "mjviser", "trimesh")
    }
    checks.update(
        {
            name: Path(settings[name]).exists() if settings[name] else False
            for name in ("data_root", "umr_root", "umr_python", "smplx_model")
        }
    )
    patch = None
    if checks["umr_root"]:
        try:
            patch = patch_status(settings["umr_root"], apply=apply_patch)
        except (ValueError, subprocess.CalledProcessError) as error:
            patch = dict(error=str(error))
    return dict(settings=settings, checks=checks, umr=patch, clips=ws.clips())


def run_process(command, log, cwd=None):
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        MUJOCO_GL="egl",
    )
    env.setdefault("MPLCONFIGDIR", "/tmp/orcs-egohtr-mpl")
    with Path(log).open("a") as stream:
        stream.write(f"\n[{now()}] {json.dumps([str(x) for x in command])}\n")
        stream.flush()
        subprocess.run(
            [str(x) for x in command],
            check=True,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def runtime_config(ws, run, scratch):
    import numpy as np

    settings = ws.settings()
    cfg = merge(run["config"], {})
    source = ws.asset(run, "source")
    # UMR's archive carries a source directory. Rebind it per invocation so
    # neither old log paths nor the original workspace location are required.
    with np.load(ws.asset(run, "motion_source"), allow_pickle=True) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    data["source_data_dir"] = np.array(str(source))
    objects = [
        dict(
            name="terrain",
            obj=str(ws.asset(run, "scan")),
            prop=str(source / "prop_terrain.csv"),
            xml=str(ws.asset(run, "terrain")),
        )
    ]
    data["samp_objects"] = np.array(json.dumps(objects))
    motion_name = safe_name(cfg.get("motion", {}).get("seq_key", "egohtr"))
    motion_path = scratch / (motion_name + ".npz")
    np.savez_compressed(motion_path, **data)
    cfg.setdefault("motion", {})["data"] = str(motion_path)
    # UMR writes a floating-root XML beside its input. Keep that scratch output
    # outside the immutable model cache, with explicit mesh references.
    from .robot import mesh_files

    tree, meshes = mesh_files(ws.asset(run, "robot"))
    for node, path in meshes:
        node.set("file", str(path))
    tree.write(scratch / "robot.xml", encoding="unicode")
    cfg.setdefault("robot", {})["xml"] = str(scratch / "robot.xml")
    cfg["smplx_model_dir"] = settings["smplx_model"]
    cfg.setdefault("retarget", {})["out"] = str(scratch / "result.npz")
    hsi = cfg.setdefault("hsi_hoi", {})
    hsi.update(source_sequence_dir=str(source), source_objects=objects)
    corr = cfg.setdefault("correspondence", {})
    if "correspondence" not in run["assets"]:
        raise ValueError(
            "Correspondence is not prepared; run the build/train stages first"
        )
    prepared = ws.asset(run, "correspondence")
    corr.setdefault("dataset", {}).update(
        out=str(prepared / "dataset.npz"),
        smpl_models=[
            dict(
                dir=settings["smplx_model"],
                genders=[run.get("gender", "male")],
                type="smplx",
            )
        ],
    )
    corr.setdefault("train", {})["out_dir"] = str(prepared / "trained")
    corr["slots"] = str(prepared / "trained/correspondence_slots_final.npz")
    if "targets" in run["assets"]:
        cfg.setdefault("solver", {})["egohtr_review_targets"] = str(
            ws.asset(run, "targets")
        )
        # UMR's optional native-terrain cache also writes a temporary combined
        # XML beside this input. Keep all solver writes outside immutable assets.
        tree, meshes = mesh_files(ws.asset(run, "orcs_robot"))
        for node, path in meshes:
            node.set("file", str(path))
        native_xml = scratch / "orcs_robot.xml"
        tree.write(native_xml, encoding="unicode")
        cfg["solver"]["egohtr_review_orcs_xml"] = str(native_xml)
    return cfg


def solve(ws, clip, parent="accepted", label="candidate", overrides=None, max_frames=0):
    settings = ws.settings()
    if not settings["umr_python"] or not Path(settings["umr_python"]).is_file():
        raise ValueError(
            "Configure an isolated UMR interpreter with configure --umr-python"
        )
    status = patch_status(settings["umr_root"])
    if status["patch"] != "applied":
        raise ValueError(
            "Apply the integration patch with doctor --apply-patch before solving"
        )
    baseline = ws.run(clip, parent)
    fields = {k: baseline[k] for k in ("assets", "config", "source", "preparation")}
    fields["assets"] = dict(fields["assets"])
    overrides = overrides or {}
    if max_frames < 0:
        raise ValueError("--max-frames must be nonnegative")
    unknown = set(overrides) - {
        "config",
        "targets",
        "hand_reference",
        "note_ids",
        "summary",
    }
    if unknown:
        raise ValueError(f"Unknown override keys: {sorted(unknown)}")
    fields["config"] = merge(fields["config"], overrides.get("config", {}))
    defaults = json.loads(
        files("orcs.ego_htr").joinpath("presets/review_defaults.json").read_text()
    )
    fields["config"]["solver"] = merge(defaults, fields["config"].get("solver", {}))
    if max_frames:
        fields["config"].setdefault("motion", {})["max_frames"] = max_frames
    if (
        "targets" in fields["assets"]
        and fields["config"].get("motion", {}).get("stride", 1) != 1
    ):
        raise ValueError("Review velocity constraints require 30 Hz, stride 1")
    run = ws.new_run(
        clip,
        label,
        parent=baseline["id"],
        **fields,
        overrides=overrides,
        summary=overrides.get("summary", "Retarget with the saved recipe."),
        note_ids=overrides.get("note_ids", []),
        provenance=dict(umr=status, interpreter=settings["umr_python"]),
    )
    directory = ws.run_dir(run)
    try:
        with tempfile.TemporaryDirectory(prefix="egohtr-") as temp:
            scratch = Path(temp)
            if "targets" in overrides or "hand_reference" in overrides:
                # Geometry libraries for target planning live in the isolated environment.
                target_request = scratch / "targets.json"
                write_json(
                    target_request,
                    dict(
                        workspace=str(ws.root),
                        run=run,
                        options=overrides.get("targets"),
                        hand_reference=overrides.get("hand_reference"),
                    ),
                )
                run_process(
                    isolated_command(
                        settings["umr_python"],
                        "_targets",
                        "--request",
                        target_request,
                        "--output",
                        scratch / "targets.npz",
                    ),
                    directory / "run.log",
                )
                from .workspace import cache_files

                cached = cache_files(
                    ws, {"targets.npz": scratch / "targets.npz"}, "targets", overrides
                )
                run["assets"]["targets"] = ws.ref(cached / "targets.npz")
                write_json(directory / "run.json", run)
            cfg = runtime_config(ws, run, scratch)
            if not Path(cfg["correspondence"]["slots"]).exists():
                raise ValueError(
                    "Correspondence is not prepared; run the build/train stages first"
                )
            write_json(scratch / "runtime.json", cfg)
            run_process(
                [
                    settings["umr_python"],
                    Path(settings["umr_root"])
                    / "scripts/humanoid_retarget_pipeline.py",
                    "--config",
                    scratch / "runtime.json",
                    "--stage",
                    "retarget",
                    "--skip-view",
                    "--force-retarget",
                ],
                directory / "run.log",
            )
            normalize_motion(
                scratch / "result.npz",
                directory / "motion.npz",
                run["source"]["start_frame"],
                run["source"]["fps"],
            )
        from .audit import audit

        checks = audit(ws, run)
        status = "ready" if checks["passed"] else "failed_checks"
        return ws.finish(run, checks, status)
    except BaseException as error:
        ws.finish(run, dict(error=f"{type(error).__name__}: {error}"), "failed")
        raise


def prepare_targets(request, output):
    from . import targets
    from .robot import load_motion
    from .workspace import Workspace

    record = read_json(request)
    ws = Workspace(record["workspace"])
    run = record["run"]
    parent = ws.run(run["clip"], run["parent"])
    scale = float(load_motion(ws.run_dir(parent) / "motion.npz")["smpl_scale"].item())
    if record["options"] is not None:
        targets.build(
            ws.asset(run, "source"),
            ws.asset(run, "terrain"),
            scale,
            output,
            record["options"],
        )
        # Foot corrections must retain an existing accepted hand reference.
        if "targets" in run["assets"] and not record["hand_reference"]:
            import numpy as np

            with (
                np.load(output) as current,
                np.load(ws.asset(run, "targets")) as previous,
            ):
                data = {k: current[k].copy() for k in current.files}
                for name in ("hand_points", "hand_local_point"):
                    if name in previous:
                        data[name] = previous[name].copy()
            np.savez_compressed(output, **data)
    else:
        import shutil

        shutil.copy2(ws.asset(run, "targets"), output)
    if record["hand_reference"]:
        reference = ws.run(run["clip"], record["hand_reference"])
        targets.add_hands(
            ws.run_dir(reference) / "motion.npz", output, output, ws.asset(run, "robot")
        )
