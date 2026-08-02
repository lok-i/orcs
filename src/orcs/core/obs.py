"""Observation-group plumbing and the robot-only term bundles.

Every orcs task assembles obs the same way — uniform group construction, then
atoms composed into named groups. The uniformity is the point: an experiment
becomes a re-pick of groups, never a re-plumb.

What lives here is what a task cannot disagree about: what proprio means, what
the robot's own root state is, and what the sys1 root-twist command is. What
each task feeds its adapter is the task's business.
"""

from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from orcs.core import mdp

__all__ = [
    "T", "grp", "proprio_terms", "robot_root_state_terms",
    "robot_root_twist_cmd_terms",
]


def T(func, params: dict | None = None) -> ObservationTermCfg:
    """One observation term; `params=None` means the term takes none."""
    return ObservationTermCfg(func=func) if params is None else ObservationTermCfg(
        func=func, params=params)


def grp(terms: dict, *, concat: bool = True, corrupt: bool = False
        ) -> ObservationGroupCfg:
    """Uniform group: nan-sanitized per term. concat=False -> dict group."""
    return ObservationGroupCfg(
        terms=terms, concatenate_terms=concat,
        enable_corruption=corrupt, nan_policy="sanitize", nan_check_per_term=True,
    )


def proprio_terms() -> dict:
    """Robot self-state (deployable): gravity dir + base ang vel + joint pos/vel.

    `base_lin_vel` is NOT here — there is no state estimator on hw (1Aug2026).
    A critic that wants it adds it explicitly; it never deploys.
    """
    return {
        "projected_gravity": T(mdp.projected_gravity),
        "base_ang_vel": T(mdp.base_ang_vel),
        "joint_pos": T(mdp.joint_pos_rel),
        "joint_vel": T(mdp.joint_vel_rel),
    }


def robot_root_state_terms() -> dict:
    """Privileged robot-root state: env-frame position + body-frame twist.

    Orientation is already in `proprio_terms`/the frozen base's stream, so only
    the odometry half lives here.
    """
    return {
        "robot_root_pos_env": T(mdp.robot_root_pos_env),
        "robot_root_lin_vel_b": T(mdp.base_lin_vel),
        "robot_root_ang_vel_b": T(mdp.base_ang_vel),
    }


def robot_root_twist_cmd_terms(p: dict) -> dict:
    """sys1's root-twist command — {v,w}_cmd_t in the REF anchor's own frame.

    Robot-state language only, so it is frame-clean for any task. A task that
    also has a per-body contact schedule prepends it; the twist half is shared.
    """
    return {
        "robot_root_lin_vel_cmd": T(mdp.robot_root_lin_vel_cmd, p),
        "robot_root_ang_vel_cmd": T(mdp.robot_root_ang_vel_cmd, p),
    }
