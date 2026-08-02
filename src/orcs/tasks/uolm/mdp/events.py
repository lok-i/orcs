"""Event terms — contact-gated object perturbation.

RSI lives in ObjectMotionCommand (wired via the command cfg fields); generic
resets/pushes come from mjlab stock mdp; `PolicyUpdateCounter` is task-blind
and lives in :mod:`orcs.core.mdp.events`, re-exported here.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from orcs.core.mdp.events import PolicyUpdateCounter  # noqa: F401 — moved to core

__all__ = [
    "PolicyUpdateCounter",
    "PerturbObjectInRobotContact",
]


# ---------------------------------------------------------------------------
# Contact-gated object perturbation (state variation, robustness domain)
# ---------------------------------------------------------------------------

class PerturbObjectInRobotContact(ManagerTermBase):
    """Kick the object's root velocity on EVERY step the robot's hands touch it
    (control-authority gate -> the kick is recoverable, not a fling; root-link
    contact is not meaningful control).

    - mode="interval", interval_range_s=(0.0, 0.0) -> ticks every env step, so
      the perturbation is a continuous shake for as long as contact holds.
    - stateless: no once-per-episode latch and no random watch-start (both
      dropped 2026-08-01) — contact alone gates firing, ``reset`` is a no-op.
    - live contact from the one multi-primary object contact-graph sensor
      (columns are MODEL order — resolved by name via ``sensor.primary_names``).
    """

    def __init__(self, cfg, env):
        super().__init__(env)

        self._cols: list[int] | None = None  # sensor built after cfg time

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)


    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids,
        velocity_range: dict[str, tuple[float, float]],
        object_cfg: SceneEntityCfg,
        sensor_name: str,
        contact_body_names: tuple[str, ...],
        contact_force_threshold: float = 0.1,
    ):
        del env_ids  # interval mode ticks all envs; we gate ourselves
        sensor = env.scene.sensors[sensor_name]
        if self._cols is None:
            self._cols = [sensor.primary_names.index(b)
                          for b in contact_body_names]
        force = torch.norm(sensor.data.force, dim=-1)  # (N, K) model order
        in_contact = (force[:, self._cols] > contact_force_threshold).any(dim=-1)

        envs_to_pertueb = in_contact .nonzero(as_tuple=False).squeeze(-1)
        if envs_to_pertueb.numel() == 0:
            return
        push_by_setting_velocity(
                                 env, 
                                 envs_to_pertueb, 
                                 velocity_range=velocity_range, 
                                 asset_cfg=object_cfg
                                 )
