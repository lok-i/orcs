"""Virtual-force curriculum for stabilised early-stage training.

Decaying virtual-force curriculum (ResMimic arXiv:2510.05070, DexMachina
arXiv:2505.24853), ported from fcrl to mjlab: every env step a PD controller
writes a world-frame wrench (``xfrc_applied`` via
``entity.write_external_wrench_to_sim``) pushing the object toward its
motion-command reference. A dimensionless gain ``α(t): 1 → terminal_scale``
decays with ``env.policy_update_count`` (written by ``PolicyUpdateCounter``)
so the policy gradually takes over.

Single-rigid-body (SRB) dynamics of the object, world frame:

    linear:   m ẍ = f
    angular:  I ω̇ + ω × (I ω) = τ

We assume the gyroscopic term ω × (I ω) ≈ 0 (slow/near-symmetric object), so
the angular channel reduces to the same 1st-order-in-I linear form as the
linear one — one critically-damped (ζ = 1) PD solution serves both, scaled by
m and I respectively:

    f = m ( ωₙ² (x* − x)      − 2 ωₙ ẋ )
    τ = I ( ωₙ² (θ* ⊖ θ)      − 2 ωₙ ω )

i.e. per-DOF kp = ωₙ², kd = 2 ωₙ (natural frequency ωₙ, damping ζ = 1); the
only knob is ωₙ. Gains are then multiplied by α(t):

  decay_mode=None:          α = 1 until t*, then snap to terminal_scale
  decay_mode="linear":      α(t) = 1 + (terminal_scale − 1) * t/t*
  decay_mode="exponential": α(t) = terminal_scale ** (t/t*)

Past t* (=``decay_by_policy_iterations``) α goes to exactly 0 and the wrench is
cleared once. m and I (mean principal moment) are read per-world from the model
at init (stays correct under mass DR, which writes per-world).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_conjugate, quat_mul

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.manager_term_config import ManagerTermBaseCfg

__all__ = ["VirtualObjectForceCurriculum"]


def _quat_error_world(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """Rotation-vector error (axis*angle) from q_current -> q_target, world frame."""
    q_err = quat_mul(q_target, quat_conjugate(q_current))
    q_err = torch.where(q_err[:, 0:1] < 0, -q_err, q_err)
    return 2.0 * q_err[:, 1:4]


class VirtualObjectForceCurriculum(ManagerTermBase):
    """Event term (mode="interval", interval_range_s=(0,0) → every env step)
    applying a decaying, critically-damped PD virtual wrench on the object
    toward the motion command's reference pose. See module docstring."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        p = cfg.params
        self._wn: float = p.get("natural_frequency", 10.0)
        self._terminal_scale: float = p.get("terminal_scale", 1e-6)
        self._decay_mode: str | None = p.get("decay_mode", None)
        self._decay_iters: int = p.get("decay_by_policy_iterations", 5_000)
        self._exp_ratio: float = self._terminal_scale ** (1.0 / max(1, self._decay_iters))
        self.alpha: float = 1.0  # dimensionless decay multiplier, 1 → terminal_scale
        self._last_policy_update: int | None = None
        self._wrench_cleared = False

        self._object = env.scene[p.get("object_cfg", SceneEntityCfg("object")).name]
        self._command_name: str = p.get("command_name", "motion")


        idx = self._object.data.indexing.body_ids
        model = self._object.data.model
        self._mass = model.body_mass[:, idx].sum(-1, keepdim=True).to(env.device)
        self._inertia = (
            model.body_inertia[:, idx].mean(-1).sum(-1, keepdim=True).to(env.device)
        )

        self._kp_lin = self._mass * self._wn**2
        self._kd_lin = 2.0 * self._mass * self._wn
        self._kp_ang = self._inertia * self._wn**2
        self._kd_ang = 2.0 * self._inertia * self._wn

        print(
            f"[VirtualObjectForceCurriculum] wn={self._wn}, "
            f"terminal_scale={self._terminal_scale}, decay_mode={self._decay_mode}, "
            f"decay_iters={self._decay_iters}, "
            f"mass range=[{self._mass.min().item():.3f}, {self._mass.max().item():.3f}] kg, "
            f"inertia range=[{self._inertia.min().item():.2e}, {self._inertia.max().item():.2e}] kg·m²"
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        **_: object,  # cfg params consumed in __init__
    ) -> None:
        # --- recompute decay multiplier from current policy iteration (resume-safe) ---
        t: int = getattr(env, "policy_update_count", 0)
        if t != self._last_policy_update:
            self._last_policy_update = t
            if t > self._decay_iters:
                self.alpha = 0.0
            elif t == self._decay_iters:
                self.alpha = self._terminal_scale
            elif self._decay_mode == "exponential":
                self.alpha = self._exp_ratio**t
            elif self._decay_mode == "linear":
                self.alpha = 1.0 + (self._terminal_scale - 1.0) * (t / self._decay_iters)
            # decay_mode=None: constant α=1 until boundary snap

        env.extras.setdefault("log", {})
        env.extras["log"]["VirtualObjectForce/alpha"] = self.alpha

        obj = self._object
        if self.alpha == 0.0:
            if not self._wrench_cleared:  # xfrc_applied persists — clear once
                zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
                obj.write_external_wrench_to_sim(zeros, zeros)
                self._wrench_cleared = True
            return

        cmd = env.command_manager.get_term(self._command_name)
        force_w = self.alpha * (
            self._kp_lin * (cmd.object_pos_w - obj.data.root_link_pos_w)
            - self._kd_lin * obj.data.root_link_lin_vel_w
        )
        ori_err_w = _quat_error_world(obj.data.root_link_quat_w, cmd.object_quat_w)
        torque_w = self.alpha * (
            self._kp_ang * ori_err_w - self._kd_ang * obj.data.root_link_ang_vel_w
        )

        obj.write_external_wrench_to_sim(
            force_w.unsqueeze(1), torque_w.unsqueeze(1)
        )
