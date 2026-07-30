"""Register the UOLM (Uni-Object Loco-Manipulation) tasks with mjlab.

  Orcs-Uolm        frozen SONIC base + LoRA adapter, robot command space.
  Orcs-Uolm-TaRa   tabula rasa from-scratch MLP — the no-frozen-base floor.
  Orcs-Uolm-Smpl   human SMPL command space — SONIC smpl encoder; rollout-only
                   for now (rewards + RSI unsupported, PR pending).

Registration is skipped (with a warning) when local assets/motions are missing:
object XMLs are machine-generated (assets/.../make_object_models.py) and not
tracked, so a fresh checkout must not break ``import orcs``.
"""

from mjlab.tasks.registry import register_mjlab_task

from orcs.tasks.uolm.env_cfg import uolm_env_cfg
from orcs.tasks.uolm.rl_cfg import _SMPL_CKPT, sonic_agent_cfg, tara_agent_cfg

try:
    register_mjlab_task(
        task_id="Orcs-Uolm",
        env_cfg=uolm_env_cfg(),
        play_env_cfg=uolm_env_cfg(play=True),
        rl_cfg=sonic_agent_cfg(),
    )
    register_mjlab_task(
        task_id="Orcs-Uolm-TaRa",
        env_cfg=uolm_env_cfg(agent="tara"),
        play_env_cfg=uolm_env_cfg(agent="tara", play=True),
        rl_cfg=tara_agent_cfg(),
    )
    # SMPL command space: same object plumbing, smpl tokenizer + encoder.
    # Rollout-only for now (rewards nullified; see scripts/rollout_smpl.py).
    register_mjlab_task(
        task_id="Orcs-Uolm-Smpl",
        env_cfg=uolm_env_cfg(command_space="smpl"),
        play_env_cfg=uolm_env_cfg(command_space="smpl", play=True),
        rl_cfg=sonic_agent_cfg("orcs_uolm_smpl", base_checkpoint=_SMPL_CKPT),
    )
except FileNotFoundError as e:
    print(f"[orcs.tasks.uolm] skipping task registration: {e}")
