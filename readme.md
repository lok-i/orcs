# sortr — SONIC REtarget & REfine

omni-object locomanipulation: frozen SONIC WBC base + LoRA adapter (PPO), ObjKin obs.

## Install
```bash
scripts/setup/sync_dependencies.sh   # assets, retargeted_motions, mocke, rsl_rl
pip install -e .
# object XMLs are machine-generated (not tracked):
# python dependencies/assets/source/omni_objects/make_object_models.py
```

## Play / Train
```bash
play  Sortr-OmniObj --agent initial   # frozen base, no ckpt; also zero|random|trained
train Sortr-OmniObj --num_envs 4096
```

## SMPL rollout (Sortr-OmniObj-Smpl, phase 2)
```bash
# one-time: port the smpl encoder (in dependencies/mocke)
python dependencies/mocke/scripts/port_sonic_checkpoint.py --smpl
# roll frozen base on a clip (omit --smpl for a synthetic smoke clip)
python scripts/rollout_smpl.py --smpl clip.pkl --viewer native
```
smpl_motion.npz contract: `smpl_joints` (T,24,3) + `smpl_root_quat_w` (T,4) — z-up, wxyz, SMPL base rot removed. Wrist refs ride motion.npz `joint_pos` (zeros OK).

## Layout
```
src/sortr/
  __init__.py       # registers Sortr-OmniObj + mjlab compat shim
  env_cfg.py        # THE env factory (sonic wiring, 3-stream obs, MoTr rewards)
  rl_cfg.py         # PPO runner + sonic LoRA-adapter agent
  robustness.py     # training domain (state + param variations)
  assets.py         # flat-hand G1 + omni-object variant entity
  mdp/              # object-manip term library
```
