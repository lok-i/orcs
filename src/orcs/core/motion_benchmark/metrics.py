"""CPU-only aggregation of versioned trial facts; no simulator imports."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean

SCHEMA_VERSION = 1
METRICS = {
    "global_position_m": "Global point error (m)",
    "relative_position_m": "Root-relative point error (m)",
    "root_position_m": "Root position error (m)",
    "root_orientation_rad": "Root orientation error (rad)",
    "point_velocity_m_s": "Point velocity error (m/s)",
}
STATUSES = ("completed", "failed", "invalid", "censored")


def balanced_batches(motions: int, trials: int, capacity: int):
    """Global trial IDs retain identity across fixed-size execution batches."""
    if min(motions, trials, capacity) < 1:
        raise ValueError("motions, trials and capacity must be positive")
    for start in range(0, motions * trials, capacity):
        yield [(j, j % motions, j // motions)
               for j in range(start, min(start + capacity, motions * trials))]


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p, zz = successes / total, z * z
    center = (p + zz / (2 * total)) / (1 + zz / total)
    radius = z * math.sqrt(p * (1 - p) / total + zz / (4 * total**2)) / (1 + zz / total)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _average(values):
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def aggregate(receipt: dict, trials: list[dict]) -> dict:
    """Trial → motion → macro averaging; absent/invalid trials stay visible."""
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark schema; use its matching ORCS version")
    catalog, k, bins = receipt["motions"], receipt["trials_per_motion"], receipt["bins"]
    groups = [[] for _ in catalog]
    seen = set()
    for trial in trials:
        identity = trial["trial_id"]
        if identity in seen:
            raise ValueError(f"Duplicate trial ID: {identity}")
        seen.add(identity)
        idx = trial["motion_index"]
        if not 0 <= idx < len(catalog) or identity % len(catalog) != idx:
            raise ValueError("Trial does not match the repertoire")
        if not 0 <= identity < len(catalog) * k or trial["status"] not in STATUSES:
            raise ValueError("Invalid trial identity or status")
        groups[idx].append(trial)
    rows = []
    for motion, group in zip(catalog, groups, strict=True):
        counts = Counter(t["status"] for t in group)
        valid = [t for t in group if t["status"] in ("completed", "failed")]
        completed = [t for t in group if t["status"] == "completed"]
        failed = [t for t in group if t["status"] == "failed"]
        row = dict(motion_index=motion["index"], motion=motion["id"], expected=k,
                   recorded=len(group), pending=k-len(group),
                   **{s: counts[s] for s in STATUSES})
        # Unfinished runs provide a lower bound, never a complete-looking score.
        row["completion_rate"] = counts["completed"] / k
        row["failure_rate"] = counts["failed"] / k
        row["completion_ci95"] = (
            wilson(counts["completed"], k)
            if not receipt["nominal"] and len(valid) == k else None)
        row["mean_progress"] = _average(t["progress"] for t in valid)
        row["failure_time_s"] = _average(t["elapsed_s"] for t in failed)
        row["terminations"] = dict(Counter(tag for t in failed for tag in t["terminations"]))
        for name in METRICS:
            row[name] = _average(t["metrics"].get(name) for t in valid)
            row["completed_" + name] = _average(t["metrics"].get(name) for t in completed)
        row["entered_bins"] = [sum(t["bin_samples"][b] > 0 for t in valid) for b in range(bins + 1)]
        row["failed_bins"] = [sum(t["end_bin"] == b for t in failed) for b in range(bins + 1)]
        row["bin_failure_rate"] = [f / n if n else None for f, n in zip(
            row["failed_bins"], row["entered_bins"], strict=True)]
        row["survival"] = [sum(t["progress"] >= b / bins for t in valid) / k
                           for b in range(bins + 1)]
        rows.append(row)
    total = Counter(t["status"] for t in trials)
    macro = {s: total[s] for s in STATUSES}
    macro.update(expected=len(catalog)*k, recorded=len(trials), pending=len(catalog)*k-len(trials),
                 motions=len(catalog), covered_motions=sum(bool(g) for g in groups),
                 completion_rate=mean(r["completion_rate"] for r in rows),
                 mean_progress=_average(r["mean_progress"] for r in rows))
    for name in METRICS:
        macro[name] = _average(r[name] for r in rows)
    return {"macro": macro, "per_motion": rows}
