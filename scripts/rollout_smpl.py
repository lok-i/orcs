"""Roll the frozen SONIC base (zero-adapt) on ONE smpl + object clip.

Phase-2a smoke path: no dataset infra — the given clip is staged into a
scratch single-clip dataset (flat layout) and Sortr-OmniObj-Smpl's env cfg is
pointed at it. Rewards are nullified in the -Smpl cfg; this is rollout only.

Inputs (all optional — omitted -> synthetic static-pose smoke clip):
  --smpl    SONIC-format pkl (pose_aa (T,72), smpl_joints (T,24,3), transl,
            fps; y-up unless --z-up) OR a ready npz (smpl_joints RAW (T,24,3),
            smpl_root_quat_w (T,4) z-up/wxyz/base-rot-removed
            [, smpl_joints_viz_w (T,24,3) z-up world — ghost only]).
  --object  npz with obj_pos_w (T,3), obj_quat_w (T,4) [, obj_*_vel_w] —
            omitted -> static nominal pose at (1.2, 0, z).
  --wrists  npz/pkl field is not needed separately: G1 wrist refs ride
            motion.npz joint_pos; synthetic staging writes zeros (valid,
            degraded wrist orientation only).

Usage:
  python scripts/rollout_smpl.py                     # synthetic smoke clip
  python scripts/rollout_smpl.py --smpl clip.pkl --steps 300 --viewer native
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_SMPL_BASE_ROT_CONJ = np.array([0.5, -0.5, -0.5, -0.5])  # conj([.5,.5,.5,.5])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz hamilton product, (N,4)x(N,4)->(N,4)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """axis-angle (N,3) -> wxyz quat (N,4)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-8), 0.0)
    half = 0.5 * angle
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def load_smpl_clip(
    path: str | None, z_up: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (smpl_joints RAW (T,24,3), smpl_root_quat_w (T,4), joints_viz_w (T,24,3)).

    smpl_joints stay RAW pkl values (y-up, root-centered, no transl) — that is
    exactly what SONIC's smpl encoder consumed in training (motion lib never
    converts joints; only the root quat gets y->z up + base-rot removal).
    joints_viz_w = (joints + transl) converted to z-up — ghost/RSI only."""
    if path is None:
        # synthetic static skeleton (~1.7 m human, T-ish pose), 5 s @ 50 fps
        T = 250
        j = np.zeros((24, 3), dtype=np.float32)
        z = {0: 0.95, 1: 0.85, 2: 0.85, 3: 1.05, 4: 0.50, 5: 0.50, 6: 1.15,
             7: 0.10, 8: 0.10, 9: 1.25, 10: 0.05, 11: 0.05, 12: 1.45,
             13: 1.35, 14: 1.35, 15: 1.60, 16: 1.35, 17: 1.35, 18: 1.05,
             19: 1.05, 20: 0.80, 21: 0.80, 22: 0.72, 23: 0.72}
        y = {1: 0.10, 2: -0.10, 4: 0.11, 5: -0.11, 7: 0.12, 8: -0.12,
             10: 0.12, 11: -0.12, 13: 0.08, 14: -0.08, 16: 0.20, 17: -0.20,
             18: 0.24, 19: -0.24, 20: 0.26, 21: -0.26, 22: 0.27, 23: -0.27}
        for k, v in z.items():
            j[k, 2] = v
        for k, v in y.items():
            j[k, 1] = v
        joints = np.repeat(j[None], T, axis=0)
        quat = np.zeros((T, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        return joints, quat, joints  # synthetic is authored z-up world

    if path.endswith(".npz"):
        d = np.load(path)
        joints = d["smpl_joints"].astype(np.float32)
        viz = (d["smpl_joints_viz_w"] if "smpl_joints_viz_w" in d
               else d["smpl_joints"]).astype(np.float32)
        return joints, d["smpl_root_quat_w"].astype(np.float32), viz

    import joblib
    d = joblib.load(path)  # SONIC smpl pkl (pose_aa, transl, smpl_joints, fps)
    joints = np.asarray(d["smpl_joints"], dtype=np.float32)  # RAW — encoder-exact
    transl = np.asarray(d.get("transl", np.zeros((len(joints), 3))), dtype=np.float32)
    root_q = _aa_to_quat(np.asarray(d["pose_aa"], dtype=np.float32)[:, :3])
    # Convention split (gear_sonic): smpl_joints ship ALREADY z-up; only
    # pose_aa's root (SMPL-native) and transl ("y is the z for smpl") are
    # y-up and need converting.
    if not z_up:
        transl = np.stack(
            [transl[..., 0], -transl[..., 2], transl[..., 1]], axis=-1)
        rx90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        root_q = _quat_mul(np.broadcast_to(rx90, root_q.shape), root_q)
    root_q = _quat_mul(root_q, np.broadcast_to(_SMPL_BASE_ROT_CONJ, root_q.shape))
    world = joints + transl[:, None, :]
    world[..., 2] -= world[..., 2].min()  # rest on the ground plane
    return joints, root_q.astype(np.float32), world.astype(np.float32)


def stage_dataset(
    root: Path, joints: np.ndarray, root_quat: np.ndarray,
    joints_viz: np.ndarray, object_npz: str | None,
) -> None:
    """Write the flat single-clip layout: <root>/clip/sample0/*.npz."""
    T = joints.shape[0]
    sample = root / "clip" / "sample0"
    sample.mkdir(parents=True)

    # motion.npz: placeholder robot ref (IL order) — RSI seed only. Root
    # follows the smpl root xy at standing height; joints/vels zero (wrist
    # refs zero -> valid, degraded wrist orientation only).
    nb = 35  # covers all IL tracked-body indices
    body_pos = np.zeros((T, nb, 3), dtype=np.float32)
    body_pos[:, :, :2] = joints_viz[:, :1, :2]  # follow the smpl WORLD path
    body_pos[:, 0, 2] = 0.793  # pelvis standing height
    body_quat = np.zeros((T, nb, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        sample / "motion.npz",
        joint_pos=np.zeros((T, 29), dtype=np.float32),
        joint_vel=np.zeros((T, 29), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((T, nb, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((T, nb, 3), dtype=np.float32),
    )

    np.savez(sample / "smpl_motion.npz",
             smpl_joints=joints,               # RAW (encoder-exact)
             smpl_root_quat_w=root_quat,       # z-up, wxyz, base rot removed
             smpl_joints_viz_w=joints_viz)     # z-up world (ghost only)

    if object_npz is not None:
        d = np.load(object_npz)
        np.savez(sample / "object_motion.npz", **{k: d[k] for k in d.files})
    else:  # static nominal pose in front of the human
        obj_pos = np.zeros((T, 3), dtype=np.float32)
        obj_pos[:] = (1.2, 0.0, 0.3)
        obj_quat = np.zeros((T, 4), dtype=np.float32)
        obj_quat[:, 0] = 1.0
        np.savez(sample / "object_motion.npz",
                 obj_pos_w=obj_pos, obj_quat_w=obj_quat)

    # zeros contact matrix (establishes the ContactSchedule legend)
    from sortr.env_cfg import _CONTACT_GRAPH_BODY_NAMES
    names = list(_CONTACT_GRAPH_BODY_NAMES) + ["object", "world"]
    np.savez(sample / "contact_matrix.npz",
             body_names=np.array(names),
             matrix=np.zeros((T, len(names), len(names)), dtype=np.int8))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smpl", default=None, help="smpl clip (.pkl SONIC / .npz ours)")
    p.add_argument("--z-up", action="store_true",
                   help="pkl input is already z-up (default: y-up, converted)")
    p.add_argument("--object", default=None, help="object_motion npz")
    p.add_argument("--steps", type=int, default=200, help="headless steps (viewer none)")
    p.add_argument("--viewer", default="none", choices=("none", "native", "viser"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import sortr  # noqa: F401 — task registration + mjlab compat

    joints, root_quat, joints_viz = load_smpl_clip(args.smpl, args.z_up)
    T = joints.shape[0]
    scratch = Path(tempfile.mkdtemp(prefix="sortr_smpl_"))
    stage_dataset(scratch, joints, root_quat, joints_viz, args.object)
    print(f"[rollout] staged {T}-frame clip -> {scratch}")

    cfg = load_env_cfg("Sortr-OmniObj-Smpl", play=True)
    cfg.scene.num_envs = 1
    mc = cfg.commands["motion"]
    mc.dataset_dir = str(scratch)
    mc.ordered_object_names = None
    mc.exclude_motions = None
    mc.motion_file = str(scratch / "clip/sample0/motion.npz")
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = T * step_dt + 2.0

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_cfg = load_rl_cfg("Sortr-OmniObj-Smpl")
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    policy = runner.get_inference_policy(device=device)  # NO ckpt: zero-adapt base

    if args.viewer != "none":
        from mjlab.scripts import play as P
        if args.viewer == "native":
            P.NativeMujocoViewer(env, policy).run()
        else:
            P.ViserPlayViewer(env, policy, checkpoint_manager=None).run()
        env.close()
        return

    obs = env.get_observations()
    tok = obs["tokenizer"]
    print(f"[rollout] tokenizer obs: {tuple(tok.shape)} (expect (1, 840))")
    for i in range(args.steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = env.step(act)
        if i % 50 == 0:
            root_h = env.unwrapped.scene["robot"].data.root_link_pos_w[0, 2]
            print(f"[rollout] step {i:4d} | root z {root_h:.3f}")
    print("[rollout] done — frozen base rolled the smpl clip.")
    env.close()


if __name__ == "__main__":
    main()
