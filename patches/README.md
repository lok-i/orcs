# mjlab consumer compatibility patch

`mjlab-main.patch` applies to `d872bfd2b048da657345c07669172fa284b38abf`, the
exact upstream SHA in `deps.lock`. It adds pinned mujoco-warp main compatibility, explicit camera backend
selection and a separate OptiX sensing path outside CUDA capture. Physics graphs
remain enabled. Geometry/site rotations now use scalar-first quaternions.
The metadata keeps the consumer's rsl_rl 5.4.1 fork and identifies the patched
framework as `1.6.0+vibe.20261126` (upstream main reports 1.6.0); this is not a
published package release. MuJoCo Core 4.0.1 support is now upstream.

`SimulationCfg.deterministic=False` enables all MuJoCo Warp determinism modes
when requested, before compilation/capture in both initialization paths. Overflow
warnings remain enabled. Play and tracking evaluation accept
`--env.sim.deterministic True`, as training does. Renderer contexts receive the
same flag for flex BVHs. The physics flag alone does not guarantee deterministic
PyTorch learning, cross-GPU results or results across environment counts.

The old world-frame velocity reward functions remain under their original import
paths and semantics for existing tasks and saved configurations. Upstream's new
relative-frame rewards remain available separately. G1 model preservation lives
in ORCS's asset factory, documented in [G1 compatibility](../docs/g1-compatibility.md).

`sync_deps.sh` applies this patch before resolving every editable dependency and
extra together. A consumer such as ViBe points its own lock at this file. Do not
run the dependency's own `uv sync`, install extras independently, or edit a
materialized patch and expect setup to discard it. The helper records the exact
owned diff in Git metadata and refuses to overwrite additional edits. Review
changes to the patch, then regenerate it with `git diff --binary HEAD` in the
upstream checkout. Keep that checkout's HEAD at the pinned upstream commit.

OptiX is optional (`sync_deps.sh --optix`), requires Python 3.12 and an R590+
NVIDIA driver with the pinned bindings, and rejects shared raycast sensors.
PerLoco uses Warp only. Actual OptiX rendering remains unvalidated on the local
R580 host; no driver changes are part of this build.

The patch contains modified excerpts from mjlab, Copyright 2025 The mjlab
Developers, under Apache-2.0. The license is preserved at
[`LICENSES/mjlab-Apache-2.0.txt`](../LICENSES/mjlab-Apache-2.0.txt).
