"""Command-line entry points — the only layer allowed to import everything.

    orcs-stage-terrain    (terrain, motion) dataset -> the orcs-native layout
    orcs-view-terrain     inspect + curate a staged source, in a browser
    orcs-build-smpl       SONIC smpl pkls -> data/smpl_motions
    orcs-rollout-smpl     roll one smpl clip through the frozen base
    orcs-pseudo-retarget  SMPL -> assisted kinematic retarget for RSI
    orcs-view-seeds       inspect SMPL + G1 seed + assistance in a browser

These live INSIDE the package, not in `scripts/`, because `scripts/` does not
ship in a wheel — a consumer that pip-installs orcs would otherwise get the
runtime and no way to produce data for it. `scripts/*.py` survive as thin
wrappers so the paths in readme.md and muscle memory keep working.

Layer note: `cli` sits above `tasks` (staging imports the source readers), so
it is exempt from the no-upward-imports rule the same way `orcs.registration`
is. Nothing may import FROM here.
"""
