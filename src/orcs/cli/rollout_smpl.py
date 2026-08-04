"""Roll the frozen SONIC base (zero-adapt) on ONE smpl + object clip — quicktest.

Stages the given clip into a scratch single-clip dataset (flat layout) and
points Orcs-Uolm-AdaptSonic-Smpl's play cfg at it. Rewards are nullified in the -Smpl cfg;
this is rollout only. For a persistent multi-clip dataset use
scripts/build_smpl_dataset.py + `play Orcs-Uolm-AdaptSonic-Smpl`.

  --smpl    SONIC smpl pkl (pose_aa (T,72), transl, smpl_joints (T,24,3), fps;
            y-up unless --z-up) OR a ready npz (see orcs.tasks.uolm.smpl_data).
            Omitted -> synthetic static-pose smoke clip.
  --object  npz with obj_pos_w (T,3), obj_quat_w (T,4) — omitted -> static
            nominal pose (no smpl+object clips exist yet; placeholder).

Usage:
  python scripts/rollout_smpl.py                       # synthetic smoke clip
  python scripts/rollout_smpl.py --smpl clip.pkl --viewer native
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch


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

    import orcs  # noqa: F401 — task registration + mjlab compat
    from orcs.tasks.uolm.smpl_data import load_smpl_clip, stage_clip

    joints, root_quat, joints_viz = load_smpl_clip(args.smpl, args.z_up)
    T = joints.shape[0]
    scratch = Path(tempfile.mkdtemp(prefix="orcs_smpl_"))
    stage_clip(scratch / "clip" / "sample0", joints, root_quat, joints_viz, args.object)
    print(f"[rollout] staged {T}-frame clip -> {scratch}")

    cfg = load_env_cfg("Orcs-Uolm-AdaptSonic-Smpl", play=True)
    cfg.scene.num_envs = 1
    mc = cfg.commands["motion"]
    mc.dataset_dir = str(scratch)
    mc.ordered_object_names = None
    mc.exclude_motions = None
    mc.motion_file = str(scratch / "clip/sample0/motion.npz")
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = T * step_dt + 2.0

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_cfg = load_rl_cfg("Orcs-Uolm-AdaptSonic-Smpl")
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
    print(f"[rollout] tokenizer obs: {tuple(obs['tokenizer'].shape)} (expect (1, 840))")
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
