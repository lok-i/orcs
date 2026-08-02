"""Task-blind event terms."""

from __future__ import annotations

from mjlab.managers.manager_base import ManagerTermBase

__all__ = ["PolicyUpdateCounter"]


class PolicyUpdateCounter(ManagerTermBase):
    """Derives ``env.policy_update_count`` from ``env.common_step_counter``.

    Fires every env step (interval 0) and writes a single int on the env
    instance so any manager term can read ``env.policy_update_count`` — the
    clock every anneal/curriculum in orcs is written against.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self._num_steps_per_env: int = cfg.params["num_steps_per_env"]
        self._init_count: int = cfg.params.get("init_policy_update_count", 0)
        env.policy_update_count = self._init_count

    def __call__(
        self, env, env_ids,
        num_steps_per_env: int = 0,
        init_policy_update_count: int = 0,
    ):
        del num_steps_per_env, init_policy_update_count
        env.policy_update_count = (
            self._init_count + env.common_step_counter // self._num_steps_per_env
        )

    def reset(self, env_ids=None):
        pass
