"""Generate the nominal-stand reference clip.

    orcs-make-nominal            # -> <DATA_ROOT>/nominal_motions/nominal/stand/sample1/

A motion command that says "stand there". Every frame is the SAME pose — the
robot entity's own `init_state`, which is also the offset the joint-position
action term works from (`use_default_offset=True`) and what
`reset_scene_to_default` writes. That identity is the point: **a zero adapter
action means the robot holds the reference exactly**, so any departure from the
nominal pose is the adapter's doing and is measurable as such.

Why a real clip on disk instead of a synthetic command class: nothing downstream
needs to know. `ConcatMotionLoader` loads it, RSI samples it, the SONIC
tokenizer reads a future window of it, and the tracking rewards score against it
— all unchanged. A "constant reference" command would be a second code path
through every one of those.

Why `--seconds` covers a whole episode rather than one frame: the frames are
identical, so phase carries no information, but the machinery still advances the
timeline. A clip longer than the episode means no env ever reaches the end, so
`exceeded_motion_by_eps` and the last-frame freeze stay dormant instead of
needing a special case. Pair it with `alpha_phase_init=0.0` in the command cfg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from orcs.cli._motion_npz import Fk, default_stand, il_body_names
from orcs.core.paths import DATA_ROOT

DATASET = "nominal"
MOTION = "stand"
"""`<root>/<dataset>/<motion>/<sampleN>/motion.npz` — the layout `scan` walks."""


def build(seconds: float, fps: float, device: str) -> tuple[dict, dict, dict, list[str]]:
    """(state, vel, bodies, il_joint_names) for a held nominal stand."""
    joint_pos, root_quat, root_z = default_stand(device)
    n = max(int(round(seconds * fps)), 1)

    state = {
        "root_pos": torch.tensor([0.0, 0.0, root_z], device=device).expand(n, 3).clone(),
        "root_quat": root_quat.to(device).expand(n, 4).clone(),
        "joint_pos": joint_pos.to(device).expand(n, -1).clone(),
    }
    # A held pose has no velocity anywhere. Not differenced from the positions:
    # a central difference of a constant is exactly zero, so writing zeros says
    # the same thing without pretending a derivative was measured.
    vel = {
        "root_lin_vel": torch.zeros(n, 3, device=device),
        "root_ang_vel": torch.zeros(n, 3, device=device),
        "joint_vel": torch.zeros_like(state["joint_pos"]),
    }
    fk = Fk(batch=min(n, 256), fps=fps, device=device)
    return state, vel, fk(state, vel), fk.il_names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DATA_ROOT / "nominal_motions"))
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="clip length; must exceed the consuming task's episode")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    state, vel, bodies, il_names = build(args.seconds, args.fps, args.device)
    n = state["joint_pos"].shape[0]

    sample_dir = Path(args.out) / DATASET / MOTION / "sample1"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sample_dir / "motion.npz",
        fps=np.array([int(args.fps)]),
        joint_pos=state["joint_pos"].cpu().numpy().astype(np.float32),
        joint_vel=vel["joint_vel"].cpu().numpy().astype(np.float32),
        joint_names=np.array(il_names),
        body_names=np.array(il_body_names()),
        **bodies,
    )
    (sample_dir / "metadata.json").write_text(json.dumps({
        "clip": MOTION,
        "frames": n,
        "fps": args.fps,
        "source": "orcs.cli.make_nominal_motion — robot init_state, held",
    }, indent=1) + "\n")

    pelvis_z = float(state["root_pos"][0, 2])
    feet = [bodies["body_pos_w"][0, i, 2] for name, i in
            (("left", 21), ("right", 22))]  # IL rows of the ankle_roll links
    print(f"[orcs] nominal stand: {n} frames @ {args.fps:g} Hz -> {sample_dir}")
    print(f"       pelvis z={pelvis_z:.3f} m, ankle_roll z={feet[0]:.3f}/{feet[1]:.3f} m")


if __name__ == "__main__":
    main()
