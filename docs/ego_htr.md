# EgoHTR motion preparation

Scene-aware G1 retargeting with an optional UMR bridge and a Viser review loop.

## Use

```bash
.venv/bin/python -m orcs.cli.ego_htr view --port 8085
```

Open `http://localhost:8085`. If another viewer occupies that port, use `8086`.

Start an agent chat with:

> Read `src/orcs/ego_htr/SKILL.md`. Review my EgoHTR frame comments and continue the retargeting workflow.

1. Select **Motion → 03** and **Version → Latest**.
2. Play, pause or step to the frame; write feedback and click **Save note**.
3. Ask the agent to revise the comments.
4. Click **Reload files → Latest**; compare with **Accepted**.
5. Tell the agent when to accept a result.

## Viewer

| Control | Use |
|---|---|
| Motion / Version | Clip, latest candidate, accepted reference and historical revisions |
| Frame / Clip time / Previous / Next | Exact pose review; source timestamp shown |
| Play / Speed / Loop | 0.1–2× playback; looping starts off |
| Scan / Collision / Overlay / Hidden | Inspect measured terrain and the convex proxy |
| Opacity / Wireframe / Show G1 / Root path | Inspect occluded contacts and motion path |
| Fit whole clip / Focus on G1 | Reset camera framing; drag/pan/scroll otherwise |
| Save note | Binds feedback to the displayed frame, version, motion hash and terrain |

- Leave the viewer running while the agent creates a revision; reload to see it.
- Only the selected clip's terrain stays resident.
- Connected viewers share playback/selection; each has its own camera.
- Remote machine: forward the chosen localhost port through SSH or the IDE.

## Included workflow

| Feature | Agent handles |
|---|---|
| Source selection | Inspect local/remote NPZ members; fetch selected arrays; choose short intervals |
| Terrain | Full-body crop, clustered scan, bounded whole/tiled CoACD proxies |
| Retarget | Exact SMPL-X source geometry, isolated UMR, saved configuration and caches |
| Corrections | Foot support, separation, native collision, root continuity, knee/elbow posture, preserved palms |
| Verification | All-frame geometric/timing/smoothness checks and comparisons |
| Optional exports | Scan/proxy/human videos, contact sheets and annotated before/after poses |
| History | Independent iterations, accepted-reference protection, note status and safe pruning |

## Files and setup

- Code: `src/orcs/ego_htr/`; one CLI: `orcs.cli.ego_htr` (`orcs-egohtr` after installation).
- Local runs: `logs/ego_htr/<sequence>__<start>-<end>s/<iteration>/`.
- Each completed motion run: **`run.json`, `motion.npz`, `run.log`**.
- Source/robot/terrain/targets/correspondence are shared under `_cache/`.
- Comments: one `notes.jsonl` per clip. Optional `reports/` and `debug/` are generated on request.
- `egohtr` is an optional Python extra for viewer/report/download tools.
- UMR remains an external checkout and isolated interpreter, configured once by
  the agent. No mandatory UMR dependency or change to the training environment.
- Large assets and licensed body models stay local. A fresh checkout needs those
  inputs prepared before clips appear; the empty viewer supports **Reload files**.
- Full command/schema/feature ledger: [agent tools](../src/orcs/ego_htr/agent_tools.md).

## Current boundary

- EgoHTR, G1, stationary terrain and 30 Hz review constraints.
- Kinematic checks and playback do not establish controller stability.
- The synthetic GPU-memory diagnostic does not certify real-scene PPO memory.
- 03's reviewed hunch/roughness is parked; 04/05/06 remain preserved references.
- Policy-task integration and training follow separately.
