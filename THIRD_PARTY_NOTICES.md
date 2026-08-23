# Third-party notices

The BSD-3-Clause license at the repository root applies only to original ORCS
material. Third-party material retains its upstream terms.

## Adapted code

`src/orcs/tasks/dodge/mdp/rewards.py::ball_clearance` adapts MimicKit's
`compute_dodge_reward` from its SMP dodgeball environment.

- Source: [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit)
- License: Apache-2.0; see `LICENSES/Apache-2.0.txt`
- Citation: Xue Bin Peng, *MimicKit: A Reinforcement Learning Framework for
  Motion Imitation and Control*, arXiv:2510.13794, 2025.

## Research provenance

ORCS's virtual-assistance implementation and SMPL seed-state pipeline were
written for this repository. They build on ideas described by the following
works; no source code from them is bundled here:

- Mandi Zhao, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, and
  Shuran Song, *DexMachina: Functional Retargeting for Bimanual Dexterous
  Manipulation*, arXiv:2505.24853, 2025.
- Siheng Zhao, Yanjie Ze, Yue Wang, C. Karen Liu, Pieter Abbeel, Guanya Shi,
  and Rocky Duan, *ResMimic: From General Motion Tracking to Humanoid
  Whole-body Loco-Manipulation via Residual Learning*, arXiv:2510.05070, 2025.

## External dependencies and data

The following projects are resolved separately and are not relicensed or
bundled in the ORCS Python distribution:

| Dependency or data | Use | Upstream terms |
|---|---|---|
| [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) via Mocke | Frozen motion prior and compatible contracts | Apache-2.0 code; NVIDIA Open Model License weights |
| [GRAIL](https://github.com/NVlabs/GRAIL) | Terrain-motion source data and staging tools | NVIDIA's upstream repository and data terms |
| [OmniRetarget Dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset) | Retargeted terrain and object motions | MIT as declared by its dataset card |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Manually supplied body-model files | SMPL-X model license |
| [mjlab](https://github.com/mujocolab/mjlab), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), and the installed assets package | Simulation, learning, and model assets | Their respective distributions and licenses |

Users are responsible for obtaining restricted datasets and body models and
for complying with their terms. A citation or acknowledgement does not replace
a license.
