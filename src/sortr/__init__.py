"""sortr — SONIC REtarget & REfine. Registers Sortr-OmniObj (+ -Smpl).

Registration is skipped (with a warning) when local assets/motions are
missing — object XMLs are machine-generated (make_object_models.py) and not
tracked, so a fresh checkout must not break ``import sortr``.
"""

from mjlab.tasks.registry import register_mjlab_task

from sortr.env_cfg import sortr_omni_obj_env_cfg
from sortr.rl_cfg import _SMPL_CKPT, sonic_agent_cfg

try:
    register_mjlab_task(
        task_id="Sortr-OmniObj",
        env_cfg=sortr_omni_obj_env_cfg(),
        play_env_cfg=sortr_omni_obj_env_cfg(play=True),
        rl_cfg=sonic_agent_cfg(),
    )
    # SMPL command space: same object plumbing, smpl tokenizer + encoder.
    # Rollout-only for now (rewards nullified; see scripts/rollout_smpl.py).
    register_mjlab_task(
        task_id="Sortr-OmniObj-Smpl",
        env_cfg=sortr_omni_obj_env_cfg(command_space="smpl"),
        play_env_cfg=sortr_omni_obj_env_cfg(command_space="smpl", play=True),
        rl_cfg=sonic_agent_cfg("sortr_omni_obj_smpl", base_checkpoint=_SMPL_CKPT),
    )
except FileNotFoundError as e:
    print(f"[sortr] skipping Sortr-OmniObj task registration: {e}")

from sortr._mjlab_compat import apply as _apply_mjlab_compat

_apply_mjlab_compat()  # let mjlab train/play tolerate the multi-clip "motion" command
