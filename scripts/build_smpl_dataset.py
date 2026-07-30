"""Build the persistent SMPL dataset for Orcs-Uolm-AdptSonic-Smpl.

Converts a directory of SONIC smpl pkls into the flat layout the UOLM motion
command flat-scans (one sample dir per clip):

  data/smpl_motions/<clip>/sample0/{motion,smpl_motion,object_motion,contact_matrix}.npz

Object motion is a static nominal placeholder unless a matching object npz is
found (no smpl+object clips exist yet). Once built:

  play Orcs-Uolm-AdptSonic-Smpl --agent initial --viewer native   # multi-clip rollout

Usage:
  python scripts/build_smpl_dataset.py --src <dir-of-pkls> [--object-dir <dir>]
  python scripts/build_smpl_dataset.py --src dependencies/GR00T-WholeBodyControl/sample_data/smpl_filtered
"""

from __future__ import annotations

import argparse
from pathlib import Path

from orcs.core.paths import DATA_ROOT
from orcs.tasks.uolm.smpl_data import load_smpl_clip, stage_clip

_DEFAULT_OUT = DATA_ROOT / "smpl_motions"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="directory of SONIC smpl .pkl clips")
    p.add_argument("--out", default=str(_DEFAULT_OUT), help="dataset root to write")
    p.add_argument("--z-up", action="store_true",
                   help="pkls are already z-up (default: y-up, converted)")
    p.add_argument("--object-dir", default=None,
                   help="dir of <clip>.npz object motions (matched by clip stem)")
    p.add_argument("--limit", type=int, default=None, help="cap number of clips")
    args = p.parse_args()

    pkls = sorted(Path(args.src).glob("*.pkl"))
    if args.limit:
        pkls = pkls[: args.limit]
    if not pkls:
        raise SystemExit(f"no .pkl clips under {args.src}")

    out = Path(args.out)
    obj_dir = Path(args.object_dir) if args.object_dir else None
    print(f"[build] {len(pkls)} clips -> {out}")

    for pkl in pkls:
        stem = pkl.stem
        joints, root_quat, joints_viz = load_smpl_clip(str(pkl), args.z_up)
        obj_npz = None
        if obj_dir is not None and (obj_dir / f"{stem}.npz").exists():
            obj_npz = str(obj_dir / f"{stem}.npz")
        stage_clip(out / stem / "sample0", joints, root_quat, joints_viz, obj_npz)
        print(f"[build]   {stem}: {joints.shape[0]} frames"
              f"{' (+object)' if obj_npz else ''}")

    print("[build] done — play Orcs-Uolm-AdptSonic-Smpl --agent initial")


if __name__ == "__main__":
    main()
