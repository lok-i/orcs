"""Small, file-backed workspace shared by the viewer and a repository agent."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orcs.core.paths import DATA_ROOT, DEPS_ROOT, REPO_ROOT, smplx_dir

SCHEMA = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    """Replace metadata atomically; readers never observe partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def cache_files(ws, files, category, identity=None):
    identity = dict(
        inputs={name: digest(path) for name, path in files.items()},
        recipe=identity or {},
    )
    folder = ws.root / "_cache" / category / key(identity)
    if not (folder / "asset.json").exists():
        folder.mkdir(parents=True, exist_ok=True)
        for name, path in files.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        write_json(folder / "asset.json", identity)
    return folder


def safe_name(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value) or value in (".", ".."):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def merge(base, updates):
    """Merge explicit settings; lists replace lists (not append)."""
    result = json.loads(json.dumps(base))
    for name, value in updates.items():
        result[name] = (
            merge(result.get(name, {}), value) if isinstance(value, dict) else value
        )
    return result


@contextlib.contextmanager
def locked(path):
    # Lock the existing metadata inode's directory, not a replaceable JSON inode.
    fd = os.open(Path(path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class Workspace:
    def __init__(self, root=None):
        self.root = Path(root or REPO_ROOT / "logs/ego_htr").expanduser().resolve()

    def settings(self):
        defaults = dict(
            schema=SCHEMA,
            data_root=str(DATA_ROOT / "EgoHTR"),
            umr_root=str(DEPS_ROOT / "UMR"),
            umr_python="",
            smplx_model=str(smplx_dir() / "smplx/SMPLX_MALE.npz"),
        )
        path = self.root / "workspace.json"
        return merge(defaults, read_json(path)) if path.exists() else defaults

    def configure(self, **values):
        self.root.mkdir(parents=True, exist_ok=True)
        settings = self.settings()
        for name, value in values.items():
            if value is not None:
                # Resolving a venv's python symlink loses its pyvenv.cfg context.
                settings[name] = (
                    os.path.abspath(Path(value).expanduser())
                    if name == "umr_python"
                    else str(Path(value).expanduser().resolve())
                )
        write_json(self.root / "workspace.json", settings)
        return settings

    def ref(self, path):
        return str(Path(path).resolve().relative_to(self.root))

    def path(self, reference):
        path = (self.root / reference).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Workspace reference escapes its root: {reference}")
        return path

    def clips(self):
        return [read_json(p) for p in sorted(self.root.glob("*/clip.json"))]

    def clip(self, name):
        found = [c for c in self.clips() if name in (c["id"], c.get("label"))]
        if len(found) != 1:
            raise ValueError(f"Clip {name!r}: expected one match, got {len(found)}")
        return found[0]

    def clip_dir(self, name):
        return self.path(self.clip(name)["id"])

    def runs(self, name):
        return [read_json(p) for p in sorted(self.clip_dir(name).glob("*/run.json"))]

    def run(self, name, revision="latest"):
        clip = self.clip(name)
        revision = (
            clip.get(revision) if revision in ("latest", "accepted") else revision
        )
        if not revision:
            raise ValueError(f"Clip {name} has no selected revision")
        return read_json(self.path(clip["id"]) / safe_name(revision) / "run.json")

    def run_dir(self, run):
        return self.path(run["clip"]) / safe_name(run["id"])

    def asset(self, run, name):
        return self.path(run["assets"][name])

    def create_clip(self, ident, label, source, title=None):
        safe_name(ident)
        folder = self.path(ident)
        folder.mkdir(parents=True, exist_ok=True)
        with locked(folder):
            if (folder / "clip.json").exists():
                return self.clip(ident)
            record = dict(
                schema=SCHEMA,
                id=ident,
                label=label,
                title=title or ident,
                source=source,
                latest=None,
                accepted=None,
            )
            write_json(folder / "clip.json", record)
            return record

    def new_run(self, name, label, parent=None, **fields):
        folder = self.clip_dir(name)
        safe_name(label)
        with locked(folder):
            numbers = [
                int(p.name.split("_", 1)[0])
                for p in folder.iterdir()
                if p.is_dir() and re.match(r"^\d+_", p.name)
            ]
            ident = f"{max(numbers, default=-1) + 1:04d}_{label}"
            target = folder / ident
            target.mkdir()
            record = dict(
                schema=SCHEMA,
                clip=folder.name,
                id=ident,
                parent=parent,
                created_at=now(),
                status="running",
                **fields,
            )
            write_json(target / "run.json", record)
            (target / "run.log").touch()
            return record

    def finish(self, run, checks, status="ready"):
        target = self.run_dir(run)
        if read_json(target / "run.json")["status"] != "running":
            raise ValueError("Completed runs are immutable; allocate a new revision")
        run.update(status=status, completed_at=now(), checks=checks)
        if (target / "motion.npz").exists():
            run["motion_sha256"] = digest(target / "motion.npz")
        write_json(target / "run.json", run)
        if status == "ready":
            folder = target.parent
            with locked(folder):
                clip = read_json(folder / "clip.json")
                # A slow earlier solve must not replace a newer candidate.
                if not clip.get("latest") or run["id"] > clip["latest"]:
                    clip["latest"] = run["id"]
                    write_json(folder / "clip.json", clip)
        return run

    def notes(self, name, pending=False):
        path = self.clip_dir(name) / "notes.jsonl"
        if not path.exists():
            return []
        with path.open() as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            events = [json.loads(line) for line in stream if line.strip()]
        if not pending:
            return events
        resolved = {
            n["note_id"]
            for n in events
            if n["type"] == "disposition" and n["status"] in ("resolved", "dismissed")
        }
        return [n for n in events if n["type"] == "comment" and n["id"] not in resolved]

    def append_note(self, name, record):
        folder = self.clip_dir(name)
        event = dict(
            record, id=record.get("id", uuid.uuid4().hex), at=record.get("at", now())
        )
        with (folder / "notes.jsonl").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def comment(self, run, frame, text, surface="Scan", motion_hash=None):
        import numpy as np

        path = self.run_dir(run) / "motion.npz"
        actual = digest(path)
        if motion_hash and actual != motion_hash:
            raise ValueError("Loaded motion changed; reload before annotating")
        with np.load(path) as motion:
            if not 0 <= frame < len(motion["qpos"]):
                raise ValueError("Frame outside motion")
            source_frame = int(motion["source_frames"][frame])
            fps = float(motion["fps"].item())
        if not text.strip():
            raise ValueError("Empty comment")
        return self.append_note(
            run["clip"],
            dict(
                type="comment",
                run=run["id"],
                motion_sha256=actual,
                frame=frame,
                clip_time_s=frame / fps,
                source_frame=source_frame,
                source_time_s=source_frame / run["source"]["fps"],
                surface=surface,
                assets=run["assets"],
                text=text.strip(),
            ),
        )

    def accept(self, name, revision):
        run = self.run(name, revision)
        if run["status"] not in ("ready", "imported"):
            raise ValueError("Only a completed, reviewed run can be accepted")
        if digest(self.run_dir(run) / "motion.npz") != run["motion_sha256"]:
            raise ValueError("Motion hash changed")
        folder = self.clip_dir(name)
        with locked(folder):
            clip = read_json(folder / "clip.json")
            clip["accepted"] = run["id"]
            write_json(folder / "clip.json", clip)
        self.append_note(
            name,
            dict(type="acceptance", run=run["id"], motion_sha256=run["motion_sha256"]),
        )
        return run["id"]

    def prune(self, name, revisions, apply=False):
        folder = self.clip_dir(name)
        with locked(folder):
            clip = read_json(folder / "clip.json")
            selected = []
            for revision in revisions:
                run = self.run(name, revision)
                revision = run["id"]
                if revision in (clip.get("latest"), clip.get("accepted")):
                    raise ValueError(f"Cannot prune active/accepted run {revision}")
                if run["status"] == "running":
                    raise ValueError(f"Cannot prune running revision {revision}")
                path = self.run_dir(run)
                files = [
                    p for p in path.rglob("*") if p.is_file() and p.name != "run.json"
                ]
                selected.append(
                    dict(
                        run=revision,
                        bytes=sum(p.stat().st_size for p in files),
                        files=[self.ref(p) for p in files],
                    )
                )
            if apply:
                for item in selected:
                    path = folder / item["run"]
                    for child in path.iterdir():
                        if child.name == "run.json":
                            continue
                        if child.is_dir() and not child.is_symlink():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                    run = read_json(path / "run.json")
                    run.update(status="pruned", pruned_at=now())
                    write_json(path / "run.json", run)
            return dict(applied=apply, selected=selected, cache_untouched=True)
