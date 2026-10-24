"""One-time import of the reviewed research workspace; never a runtime dependency."""

from __future__ import annotations

import json
from pathlib import Path

from .robot import cache_xml, normalize_motion
from .workspace import cache_files, digest, write_json


def portable_config(config):
    cfg = json.loads(json.dumps(config))
    for section, names in {
        "robot": ["xml"],
        "motion": ["data"],
        "retarget": ["out"],
        "hsi_hoi": ["source_objects", "source_sequence_dir"],
        "solver": ["egohtr_review_targets", "egohtr_review_orcs_xml"],
    }.items():
        for name in names:
            cfg.get(section, {}).pop(name, None)
    cfg.pop("smplx_model_dir", None)
    cfg.pop("soma_usd_path", None)
    cfg.pop("view", None)
    cfg.get("correspondence", {}).get("dataset", {}).pop("out", None)
    cfg.get("correspondence", {}).get("dataset", {}).pop("smpl_models", None)
    cfg.get("correspondence", {}).get("train", {}).pop("out_dir", None)
    cfg.get("correspondence", {}).pop("slots", None)
    # Unused furniture preparation locations must not survive the import.
    cfg.get("hsi_hoi", {}).get("object", {}).pop("grail_mjcf_dir", None)
    return cfg


def import_legacy(ws, root, ids=("03", "04", "05", "06")):
    import numpy as np

    root = Path(root).resolve()
    first = json.loads((root / f"candidate_{ids[0]}/runtime.json").read_text())
    settings = ws.configure(
        umr_root=Path(first["robot"]["xml"]).parents[2],
        umr_python=root / "venv/bin/python",
        smplx_model=first["smplx_model_dir"],
    )
    robot = cache_xml(ws, first["robot"]["xml"])
    orcs_robot = cache_xml(ws, root / "orcs_g1_collision.xml", "orcs-robot")
    imported = []
    for cid in ids:
        old = root / f"candidate_{cid}"
        prep = json.loads((old / "preparation.json").read_text())
        source = dict(
            sequence=prep["source"],
            start_frame=prep["start_frame"],
            end_frame=prep["end_frame_exclusive"],
            fps=30.0,
        )
        ident = f"{prep['source']}__{prep['start']:g}-{prep['end']:g}s"
        clip = ws.create_clip(ident, cid, source, prep["title"])
        if ws.runs(ident):
            imported.append(dict(clip=ident, skipped="already imported"))
            continue
        config = json.loads((old / "runtime.json").read_text())
        source_files = {p.name: p for p in (old / "source").glob("*.npy")}
        source_files["terrain.obj"] = old / "source/terrain.obj"
        source_files["prop_terrain.csv"] = old / "source/prop_terrain.csv"
        source_files["motion.npz"] = Path(config["motion"]["data"])
        # Recover sparse annotation flags into the new source cache.
        metadata = root.parent / "shortlibt/source_metadata" / prep["source"]
        for side in ("Left", "Right"):
            path = metadata / f"egomc_footlock_{side}Foot.npy"
            if path.exists():
                source_files[path.name] = path
        cache = cache_files(ws, source_files, "source", source)
        corr = config["correspondence"]
        trained = Path(corr["train"]["out_dir"])
        corr_files = {
            "dataset.npz": Path(corr["dataset"]["out"]),
            "trained/correspondence_slots_final.npz": trained
            / "correspondence_slots_final.npz",
        }
        if (trained / "checkpoint-final.pt").exists():
            corr_files["trained/checkpoint-final.pt"] = trained / "checkpoint-final.pt"
        correspondence = cache_files(ws, corr_files, "correspondence")
        common = dict(
            source=ws.ref(cache),
            scan=ws.ref(cache / "terrain.obj"),
            motion_source=ws.ref(cache / "motion.npz"),
            robot=ws.ref(robot),
            orcs_robot=ws.ref(orcs_robot),
            correspondence=ws.ref(correspondence),
        )
        # Current reviewed result becomes 0000; keep exact historical versions too.
        variants = [
            ("accepted", old),
            ("before-feedback", old / "before_feedback"),
            ("before-posture", old / "before_knee_review"),
            ("initial", old / "initial_pass"),
        ]
        mapping = {}
        for label, directory in variants:
            if not (directory / "g1_motion.npz").exists():
                continue
            cfg = (
                json.loads((directory / "runtime.json").read_text())
                if (directory / "runtime.json").exists()
                else config
            )
            terrain = (
                directory / "terrain.xml"
                if directory != old
                else old / "source/terrain.xml"
            )
            if not terrain.exists():
                terrain = old / "source/terrain.xml"
            assets = dict(common, terrain=ws.ref(cache_xml(ws, terrain, "terrain")))
            target = cfg.get("solver", {}).get("egohtr_review_targets")
            if target and Path(target).exists():
                target_cache = cache_files(ws, {"targets.npz": Path(target)}, "targets")
                assets["targets"] = ws.ref(target_cache / "targets.npz")
            run = ws.new_run(
                ident,
                label,
                assets=assets,
                config=portable_config(cfg),
                source=source,
                preparation={k: v for k, v in prep.items() if k != "model_file"},
                provenance=dict(
                    imported_from=str(directory),
                    original_motion_sha256=digest(directory / "g1_motion.npz"),
                    umr_commit="d1cdaedaf67a827198da71c5fa831f48b2e51f91",
                ),
                summary="Reviewed reference imported without changing qpos or terrain.",
            )
            dest = ws.run_dir(run)
            normalize_motion(
                directory / "g1_motion.npz", dest / "motion.npz", source["start_frame"]
            )
            with (
                np.load(directory / "g1_motion.npz", allow_pickle=True) as original,
                np.load(dest / "motion.npz") as copied,
            ):
                if not np.array_equal(original["qpos"], copied["qpos"]):
                    raise ValueError("Legacy import changed joint trajectory")
            checks = {}
            for name in (
                "metrics",
                "feedback_audit",
                "feedback_support_audit",
                "orcs_self_audit",
                "knee_smoothness_audit",
            ):
                path = directory / f"{name}.json"
                if path.exists():
                    checks[name] = json.loads(path.read_text())
            ws.finish(run, checks, status="imported")
            mapping[label] = run
        clip.update(
            accepted=mapping["accepted"]["id"], latest=mapping["accepted"]["id"]
        )
        write_json(ws.clip_dir(ident) / "clip.json", clip)
        notes = root / "review_notes.jsonl"
        if notes.exists():
            for line in notes.read_text().splitlines():
                note = json.loads(line)
                if note["clip"] != cid:
                    continue
                # Original Latest notes predate contact fixes, hence before-feedback.
                label = {
                    "Latest": "before-feedback",
                    "Before feedback": "before-feedback",
                    "Initial pass": "initial",
                    "Before posture review": "before-posture",
                }.get(note["variant"])
                if label not in mapping:
                    raise ValueError(f"Cannot bind historical note: {note}")
                record = mapping[label]
                event = ws.append_note(
                    ident,
                    dict(
                        type="comment",
                        run=record["id"],
                        motion_sha256=record["motion_sha256"],
                        frame=note["frame"],
                        source_frame=note["source_frame"],
                        source_time_s=note["source_frame"] / 30,
                        clip_time_s=note["clip_time_s"],
                        surface=note["surface"],
                        assets=record["assets"],
                        text=note["note"],
                        at=note["at"],
                        legacy=True,
                        legacy_record=note,
                    ),
                )
                ws.append_note(
                    ident,
                    dict(
                        type="disposition",
                        note_id=event["id"],
                        status="resolved",
                        run=mapping["accepted"]["id"],
                        text="Historical feedback incorporated in the user-reviewed reference before migration.",
                    ),
                )
        imported.append(
            dict(clip=ident, versions={k: v["id"] for k, v in mapping.items()})
        )
    return dict(imported=imported, settings=settings)
