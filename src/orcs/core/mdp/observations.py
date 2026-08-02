"""Robot-only observation terms — nothing here names an object or a terrain.

Two families:

  privileged robot state   `robot_root_pos_env` — the odometry channel
  sys1 command stream      `robot_root_{lin,ang}_vel_cmd` — SUGAR's {v,w}_cmd_t

plus `unweighted_reward_vector`, which is critic conditioning and reads only
the reward manager, so it is task-blind by construction.

The command accessors need a motion command exposing mjlab's `MotionCommand`
anchor API (`anchor_quat_w`, `anchor_lin_vel_w`, `anchor_ang_vel_w`) — that is
the contract, not a concrete class.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

__all__ = [
    "robot_root_pos_env",
    "robot_root_lin_vel_cmd",
    "robot_root_ang_vel_cmd",
    "unweighted_reward_vector",
]


def robot_root_pos_env(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Robot root position in env frame (world - env_origin) -> (B, 3).

    The odometry channel that makes an env-frame goal actionable: without it an
    env-frame goal and an env-frame state are two absolutes the network can
    difference, but neither is reachable relative to the robot. Privileged (no
    state estimator on hw) — augmentation/critic only.
    """
    robot = env.scene[robot_cfg.name]
    return robot.data.root_link_pos_w - env.scene.env_origins


def robot_root_lin_vel_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """v_cmd_t: reference anchor linear velocity in the REF anchor's own frame
    -> (B, 3). Pure function of the reference stream (no live-state coupling),
    SUGAR ref_anchor_lin_vel_b convention — sys1 can emit it open-loop."""
    cmd = env.command_manager.get_term(command_name)
    return quat_apply_inverse(cmd.anchor_quat_w, cmd.anchor_lin_vel_w)


def robot_root_ang_vel_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """w_cmd_t: reference anchor angular velocity in the REF anchor's own frame
    -> (B, 3). Same convention as robot_root_lin_vel_cmd."""
    cmd = env.command_manager.get_term(command_name)
    return quat_apply_inverse(cmd.anchor_quat_w, cmd.anchor_ang_vel_w)


def unweighted_reward_vector(
    env: ManagerBasedRlEnv, enabled: bool = False, terms: list[str] | None = None
) -> torch.Tensor:
    """(B, K) per-term UNWEIGHTED reward rates — critic-only conditioning,
    V(concat(s, r)). Reads the reward manager's step cache (already computed
    this step from the SAME post-step state the obs describe; zero recompute).

    enabled=False -> zeros of the same shape: critic architecture stays
    byte-identical across the with/without A/B, only information flips.
    Toggle: --env.observations.critic.terms.reward_vec.params.enabled True
    terms=[...] -> only those reward terms, in the given order (e.g. the
    task-layer subset as a MuZero-style aux prediction target).

    Notes:
      - Obs-manager dim probe runs before the reward manager exists -> zeros
        fallback sized from cfg.rewards / terms.
      - First step after reset reads the pre-reset cache (one-frame stale).
      - Zero-weight terms read 0 (reward manager skips them).
    """
    n = len(terms) if terms else len(env.cfg.rewards)
    if not enabled or not hasattr(env, "reward_manager"):
        return torch.zeros(env.num_envs, n, device=env.device)
    rm = env.reward_manager
    w = getattr(env, "_reward_vec_inv_w", None)
    if w is None:
        # _step_reward stores raw*weight -> divide weights back out
        w = torch.tensor(
            [c.weight if c.weight != 0.0 else 1.0 for c in rm._term_cfgs],
            device=env.device,
        )
        env._reward_vec_inv_w = w
    vec = rm._step_reward / w
    if terms:
        sel = getattr(env, "_reward_vec_sel", None)
        if sel is None:
            sel = env._reward_vec_sel = {}
        key = tuple(terms)
        if key not in sel:
            sel[key] = torch.tensor(
                [rm._term_names.index(t) for t in key], device=env.device)
        vec = vec[:, sel[key]]
    return vec
